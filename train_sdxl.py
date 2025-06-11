import argparse
import os
import datetime
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# SDXLへの変更点: StableDiffusionXLPipeline をインポート
from diffusers import StableDiffusionXLPipeline
from torchvision import transforms
from PIL import Image
# U-Netのチェックポイントがsafetensors形式の場合に必要
from safetensors.torch import load_file

# 定数設定
LEARNING_RATE = 1e-6 # SDXLの学習率は低めが一般的
DEVICE = "cuda:0"
DTYPE = torch.float16 # A100やRTX 30/40シリーズ以降が必要。それ以外のGPUでは torch.float16 を検討してください。
TOKEN = os.environ.get("HF_TOKEN") # Hugging Faceの認証トークン。設定されていない場合でもエラーにならないように .get() を使う
NUM_STEPS = 50
POSITIVE = "(masterpiece), best quality, best composition, "
NEGATIVE = "low quality, bad anatomy, nsfw, many human, credit, sign, cap, lowres, text, error, missing fingers, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, missing fingers"

# データセットの定義
class ImageFolderDataset(Dataset):
    """
    指定されたディレクトリの画像とそのディレクトリ名から抽出したプロンプトを返すデータセットクラス。
    """
    def __init__(self, root_dir, size=1024):
        self.image_paths = []
        self.prompts = []
        self.size = size
        
        # ディレクトリを走査し、画像パスとプロンプトを収集
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    cells = []
                    # データセットのパス構造からプロンプトを生成するロジック
                    # 例: images/category1/category2/image.png -> prompt = "category1, category2"
                    dir_parts = dirpath.split(os.sep) # OSに依存しないパス区切り文字を使用
                    
                    # root_dir より下の階層のディレクトリ名をプロンプトとして抽出
                    # root_dir の最も深い部分がプロンプトに含まれないように調整
                    relative_dir_parts = dir_parts[len(root_dir.split(os.sep)):]
                    
                    for cell in relative_dir_parts:
                        if cell.startswith("__"): # '__' で始まるディレクトリは無視
                            cells = [] 
                            break
                        elif cell.startswith("_"): # '_' で始まるディレクトリはスキップ
                            continue
                        if len(cell) > 0: # 空のセルを除外
                            cells.append(cell.replace('_', ' ')) # アンダースコアをスペースに置換

                    if len(cells) == 0:
                        continue # プロンプトが抽出できなかった画像はスキップ

                    self.image_paths.append(os.path.join(dirpath, file))
                    self.prompts.append(", ".join(cells))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"指定されたディレクトリに画像ファイルが見つかりませんでした: {root_dir}")

        print(f"データセットの画像数: {len(self.image_paths)}, 画像サイズ: {size}")
        print(f"プロンプトの一例 (上位5件): {list(set(self.prompts))[:5]} ... (ユニークなプロンプト数: {len(set(self.prompts))})", flush=True)
        
        self.transform = transforms.Compose([
            transforms.Resize(self.size), # 短い辺をsizeにリサイズ
            transforms.CenterCrop(self.size), # 中央クロップで size x size にする
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image).to(DTYPE) # 画像をDTYPEに変換
        text = self.prompts[idx]
        
        # SDXLではtokenizer処理はトレーニングループで行うため、ここでは文字列と画像のみを返す
        return {"image": image, "text": text}

def train(args):
    """
    Stable Diffusion XL U-Net のファインチューニングを行う関数。
    """
    # SDXLモデルのロード
    model_id = args.load_model if args.load_model else "stabilityai/stable-diffusion-xl-base-1.0"
    
    print(f"SDXLパイプラインをロード中: {model_id}...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=DTYPE, 
        use_auth_token=TOKEN, # Hugging Face認証トークンを使用
        #variant="fp16", # fp16バリアントが存在すればメモリ効率が良い
    )
    pipeline.to(DEVICE)
    pipeline.text_encoder.to(DTYPE)
    pipeline.text_encoder_2.to(DTYPE)
    pipeline.unet.to(DTYPE)
    print("SDXLパイプラインのロードが完了しました。")

    # VAEとText Encodersのパラメータを凍結する (学習対象外とする)
    print("VAEとText Encodersを凍結中...")
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)

    # U-Net のみ勾配計算を有効にし、学習モードにする
    unet = pipeline.unet
    unet.train()
    
    # もし指定されたU-Netのチェックポイントがあればロード
    if args.unet:
        print(f"U-Netの重みをロード中: {args.unet}...")
        # safetensors 形式のファイルロードを推奨
        unet.load_state_dict(load_file(args.unet, device=DEVICE), strict=False)
        print("U-Netの重みのロードが完了しました。")

    # フォルダからデータセットをロード
    size = args.image_size # SDXLは1024x1024が基本
    dataset = ImageFolderDataset(args.images, size=size)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # オプティマイザとスケジューラ
    # U-Net のパラメータのみを最適化対象とする
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    # 学習率スケジューラ: エポックごとに少しずつ学習率を減衰
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999) 

    # VAEのスケールファクター (潜在空間へのエンコード時に使用)
    vae_scaling_factor = pipeline.vae.config.scaling_factor
    print(f"VAEスケールファクター: {vae_scaling_factor}")

    start_time = time.time()
    for epoch in range(args.epoch):
        sum_loss = 0
        
        for step, batch in enumerate(dataloader):
            images = batch["image"].to(DEVICE) # (batch_size, 3, size, size)
            prompts = batch["text"] # プロンプト文字列のリスト

            optimizer.zero_grad() # 勾配をゼロクリア

            # タイムスタンプ (ノイズレベル) をランダムに生成
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (images.shape[0],), device=DEVICE).long()

            # VAEで画像を潜在空間にエンコード
            latents = pipeline.vae.encode(images).latent_dist.sample()
            latents = latents * vae_scaling_factor # SDXLのVAEに合わせたスケール

            # ノイズを生成して潜在表現に加える
            noise = torch.randn_like(latents, dtype=DTYPE).to(DEVICE)
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            # SDXLのプロンプトエンコーディング
            # `pipeline.encode_prompt` を用いて、プロンプト埋め込みとプールされたプロンプト埋め込みを取得
            # トレーニング時は do_classifier_free_guidance=False にする (CLSを使わない)
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                pipeline.encode_prompt(
                    prompt=prompts,
                    device=DEVICE,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False, # トレーニング時はFalse
                )

            # SDXLのU-Netに必要な追加のコンディショニング情報 (time_ids) を生成
            # original_size, crops_coords_top_left, target_size は画像の処理方法に合わせて設定
            original_size = (size, size)
            crops_coords_top_left = (0, 0) # 中央クロップなので (0,0)
            target_size = (size, size)

            add_text_embeds = pooled_prompt_embeds # OpenCLIPのプールされた出力
            add_time_ids = pipeline._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
            ).to(DEVICE)
            
            # U-Netに渡す追加のコンディショニング情報を辞書にまとめる
            passed_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            # U-Netでノイズ予測
            # SDXLのU-Netは `encoder_hidden_states` (CLIP L), `pooled_embeddings` (OpenCLIP pooled), `added_cond_kwargs` を受け取る
            noise_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states=prompt_embeds,
                #pooled_embeddings=pooled_prompt_embeds,
                added_cond_kwargs=passed_added_cond_kwargs,
            )["sample"]

            # 損失計算 (MSE Loss): 精度はfloatで計算して誤差を小さくする
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") 
            sum_loss += loss.item()

            # 勾配降下
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 10 == 0: # 10ステップごとに進捗を表示
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.epoch}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Avg Loss: {sum_loss/(step+1):.4f}, LR: {current_lr:.6f}", flush=True)

        passed = (time.time() - start_time) / 60
        avg_loss_epoch = sum_loss / len(dataloader)
        print(f" === Epoch {epoch+1} 終了. 平均損失: {avg_loss_epoch:.4f}, 現在の学習率: {scheduler.get_last_lr()[0]:.6f}, 経過時間: {passed:.1f}分 ===", flush=True)
        scheduler.step() # エポック終了時に学習率を更新

    # 学習済み U-Net をパイプラインに統合して SDXL モデルとして保存
    save_path = args.save_model
    pipeline.unet = unet # 学習されたU-Netをパイプラインに設定
    # VAE, Text Encoder は凍結されているため、元のpipelineのものそのまま。
    # pipeline全体を保存することで、生成時に再度ロードできる
    pipeline.save_pretrained(save_path)
    print(f"ファインチューニングされたSDXLモデルは以下に保存されました: {save_path}")

def read_prompts(path) -> list:
    """プロンプトファイルからプロンプトのリストを読み込む"""
    prompt_list = []
    with open(path, 'r', encoding='utf-8') as f: # エンコーディングを指定
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"): # 空行やコメント行はスキップ
                continue
            prompt_list.append(line)
    return prompt_list

def save_images(images: list, output_dir: str, prompt_idx: int):
    """生成画像を保存する"""
    os.makedirs(output_dir, exist_ok=True) # 出力ディレクトリが存在しない場合は作成
    now = datetime.datetime.now()
    header = now.strftime("%y%m%d-%H%M%S") 
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{header}_{prompt_idx:03d}_{i:02d}.png")
        image.save(output_path)
        print(f"生成画像を保存しました: {output_path}")

def generate_image(args):
    """
    学習済みモデルで画像を生成する関数。
    """
    if args.prompt:
        prompt_list = [args.prompt]
    elif args.prompt_file:
        prompt_list = read_prompts(args.prompt_file)
    else:
        raise Exception("画像生成にはプロンプト (--prompt または --prompt_file) を指定してください！")
    
    size = args.image_size # SDXLのデフォルトは1024x1024

    # モデルをロード (推論時の安全チェックは無効にする場合が多い)
    print(f"画像生成のためSDXLパイプラインをロード中: {args.load_model}...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.load_model, 
        torch_dtype=DTYPE, 
        use_auth_token=TOKEN,
        safety_checker=None # 推論時の安全フィルターを無効化（任意）
    ).to(DEVICE)
    pipeline.enable_vae_slicing() # VAEのメモリ使用量を削減
    print("SDXLパイプラインのロードが完了しました。")

    if args.init_image: # img2img 生成の場合
        init_image = Image.open(args.init_image).convert("RGB").resize((size, size))
        # バッチサイズ分だけ初期画像を複製
        init_images_batch = [init_image for _ in range(args.batch)]
        
        for p_idx, prompt_str in enumerate(prompt_list):
            prompts = [POSITIVE + prompt_str for _ in range(args.batch)]
            negatives = [NEGATIVE for _ in range(args.batch)]
            
            with torch.no_grad(): # 推論時は勾配計算を無効にする
                with torch.autocast(DEVICE, dtype=DTYPE): # mixed precision
                    images = pipeline(
                        prompt=prompts,
                        negative_prompt=negatives,
                        image=init_images_batch, # 初期画像
                        strength=0.8, # init_imageを使う場合のノイズレベル (0.0-1.0)
                        num_inference_steps=NUM_STEPS,
                        guidance_scale=7.0, 
                        height=size,
                        width=size,
                    ).images
            save_images(images, args.output, p_idx)

    else: # Text-to-image 生成の場合
        for p_idx, prompt_str in enumerate(prompt_list):
            prompts = [POSITIVE + prompt_str for _ in range(args.batch)]
            negatives = [NEGATIVE for _ in range(args.batch)]
            
            with torch.no_grad(): # 推論時は勾配計算を無効にする
                with torch.autocast(DEVICE, dtype=DTYPE): # mixed precision
                    images = pipeline(
                        prompt=prompts, 
                        negative_prompt=negatives, 
                        height=size, 
                        width=size, 
                        num_inference_steps=NUM_STEPS, 
                        guidance_scale=7.0 
                    ).images
            save_images(images, args.output, p_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion XL のU-Netのみに対するファインチューニングスクリプト")
    parser.add_argument('--image_size', default=1024, type=int, help="画像サイズ (SDXLのデフォルトは1024)")
    parser.add_argument('--unet', default=None, help="U-Netチェックポイントのパス (オプション: 引き続き学習する場合や特定のU-Netをロードする場合に利用)")
    parser.add_argument('--images', help="学習対象イメージがあるディレクトリのパス (例: 'data/my_images')")
    parser.add_argument('--epoch', default=5, type=int, help="学習エポック数")
    parser.add_argument('--batch', default=4, type=int, help="学習/生成時のバッチサイズ")
    parser.add_argument('--save_model', default="./tuned/sdxl_unet_tuned", help="ファインチューニングされたSDXLモデルの保存先ディレクトリ")
    parser.add_argument('--load_model', default=None, help="事前に学習されたSDXLモデルのパス、またはHugging Face ID (デフォルト: stabilityai/stable-diffusion-xl-base-1.0)")
    parser.add_argument('--init_image', default=None, help="img2img 生成用の初期画像のパス")
    parser.add_argument('--prompt', default=None, help="画像生成用の単一プロンプト")
    parser.add_argument('--prompt_file', default=None, help="画像生成用の複数プロンプトが記述されたファイルのパス (各行に1つのプロンプト)")
    parser.add_argument('--output', default="./output", help="生成された画像の保存先ディレクトリ")
    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    if args.images:
        print("--- 学習を開始します ---")
        train(args)
    elif args.load_model and (args.prompt or args.prompt_file):
        print("--- 画像生成を開始します ---")
        generate_image(args)
    else:
        print("使い方を誤っています。学習には --images 、生成には --load_model と --prompt/--prompt_file を指定してください。")
        parser.print_help()
