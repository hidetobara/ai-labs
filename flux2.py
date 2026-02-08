import torch
from diffusers import Flux2Pipeline, AutoModel
from diffusers import Flux2KleinPipeline
from transformers import Mistral3ForConditionalGeneration
from diffusers.utils import load_image
import argparse
from pathlib import Path
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class TrainingDataset(Dataset):
    """学習用データセット"""
    def __init__(self, training_data, image_size=1024):
        self.training_data = training_data
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        data = self.training_data[idx]
        image = Image.open(data['image_path']).convert('RGB')
        image = self.transform(image)
        return {
            'image': image,
            'prompt': data['prompt'],
            'image_path': str(data['image_path'])
        }

def collect_training_data(train_dir):
    """
    学習データを収集する関数
    フォルダ名をプロンプトとして使用
    再帰的にフォルダを探索
    """
    train_dir = Path(train_dir)
    training_data = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # 再帰的にフォルダを探索
    for root, dirs, files in os.walk(train_dir):
        root_path = Path(root)
        
        # 画像ファイルを収集
        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
        
        if image_files:
            # フォルダ名をプロンプトとして使用
            # train_dirからの相対パスを取得
            rel_path = root_path.relative_to(train_dir)
            
            # フォルダ名の各部分を結合してプロンプトを作成
            if str(rel_path) == '.':
                # ルートディレクトリの場合はディレクトリ名を使用
                prompt = train_dir.name
            else:
                # サブディレクトリの場合はパスを結合
                prompt = ', '.join(rel_path.parts)
            
            for img_file in image_files:
                img_path = root_path / img_file
                training_data.append({
                    'image_path': img_path,
                    'prompt': prompt
                })
    
    return training_data

def finetune_model(pipe, train_dir, output_model_path, epochs=5, learning_rate=1e-5, batch_size=1):
    """
    モデルを直接ファインチューニングする関数
    """
    print("Collecting training data...")
    training_data = collect_training_data(train_dir)
    
    if not training_data:
        print(f"No training data found in {train_dir}")
        return None
    
    print(f"Found {len(training_data)} training images across {len(set([d['prompt'] for d in training_data]))} categories")
    
    # プロンプトごとの画像数を表示
    from collections import Counter
    prompt_counts = Counter([d['prompt'] for d in training_data])
    print("\nTraining data distribution:")
    for prompt, count in prompt_counts.items():
        print(f"  - '{prompt}': {count} images")
    
    # データセットとデータローダーを作成
    dataset = TrainingDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # transformerを訓練モードに設定
    pipe.transformer.train()
    pipe.vae.eval()  # VAEは固定
    pipe.text_encoder.eval()  # Text encoderも固定
    
    # transformerのパラメータのみを訓練可能に
    for param in pipe.transformer.parameters():
        param.requires_grad = True
    
    for param in pipe.vae.parameters():
        param.requires_grad = False
    
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    
    # オプティマイザー設定
    optimizer = torch.optim.AdamW(
        pipe.transformer.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # 学習率スケジューラー
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    
    device = pipe.transformer.device
    dtype = pipe.transformer.dtype  # bfloat16を取得
    
    # VAEのscaling_factorを取得（複数の方法を試す）
    vae_scaling_factor = None
    if hasattr(pipe.vae.config, 'scaling_factor'):
        vae_scaling_factor = pipe.vae.config.scaling_factor
    elif hasattr(pipe.vae, 'scaling_factor'):
        vae_scaling_factor = pipe.vae.scaling_factor
    elif 'scaling_factor' in pipe.vae.config:
        vae_scaling_factor = pipe.vae.config['scaling_factor']
    else:
        # デフォルト値を使用（FLUX.2の場合は通常0.13025）
        vae_scaling_factor = 0.13025
        print(f"Warning: Could not find scaling_factor in VAE config. Using default value: {vae_scaling_factor}")
    
    print(f"VAE scaling factor: {vae_scaling_factor}")
    
    # トレーニングループ
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Total trainable parameters: {sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad):,}")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 画像をbfloat16に変換してからVAEに渡す
                images = batch['image'].to(device, dtype=dtype)
                prompts = batch['prompt']
                
                # VAEで画像をエンコード
                with torch.no_grad():
                    latents = pipe.vae.encode(images).latent_dist.sample()
                    latents = latents * vae_scaling_factor
                
                # テキストエンコーディング
                with torch.no_grad():
                    prompt_outputs = pipe.encode_prompt(
                        prompts,
                        device=device,
                        num_images_per_prompt=1,
                    )

                    if isinstance(prompt_outputs, tuple):
                        prompt_embeds = prompt_outputs[0]
                    else:
                        prompt_embeds = prompt_outputs

                # ノイズを追加
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()

                # Flow Matchingのノイズスケジューリング
                sigmas = timesteps.float() / pipe.scheduler.config.num_train_timesteps
                sigmas = sigmas.view(-1, 1, 1, 1)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # img_idsを準備
                with torch.no_grad():
                    height = latents.shape[2] * pipe.vae_scale_factor
                    width = latents.shape[3] * pipe.vae_scale_factor

                    latent_image_ids = pipe._prepare_latent_ids(
                        latents.shape[0],
                        height // pipe.vae_scale_factor,
                        width // pipe.vae_scale_factor,
                        device,
                        dtype,
                    )

                # デバッグ
                #print("Transformer forward signature:")
                #import inspect
                #print(inspect.signature(pipe.transformer.forward))

                # モデルの予測
                model_pred = pipe.transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    img_ids=latent_image_ids,
                    txt_ids=None,
                    guidance=None,
                    return_dict=False,
                )[0]

                # Flow Matchingの損失（velocity matching）
                target = noise - latents
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")                
                
                # バックプロパゲーション
                optimizer.zero_grad()
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(pipe.transformer.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                # プログレスバーの更新
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                print(f"\n  Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        # 定期的にチェックポイントを保存
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = Path(output_model_path) / f"checkpoint-epoch-{epoch+1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving checkpoint to {checkpoint_path}...")
            pipe.save_pretrained(str(checkpoint_path))
    
    # 最終モデルを保存
    print(f"\nSaving fine-tuned model to {output_model_path}...")
    output_path = Path(output_model_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pipe.save_pretrained(str(output_path))
    
    print("\nFine-tuning complete!")
    print(f"Model saved to: {output_path}")
    return pipe

def inference(pipe, input_dir, output_dir, prompt, reference=None, start=0, limit=20):
    """
    推論モード：入力画像に対してプロンプトを適用して画像を生成

    Args:
        pipe: FLUX2パイプライン
        input_dir (str): 入力画像のディレクトリ
        output_dir (str): 出力画像のディレクトリ
        prompt (str): 生成プロンプト
        reference (str): 参照画像
        start (int): 開始インデックス
        limit (int): 処理する画像数の上限
    """
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    # Get all image files from input folder
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    if len(image_files) >= limit:
        image_files = image_files[start: start+limit]

    ref_image = load_image(reference) if reference and os.path.isfile(reference) else None

    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        print(f"Processing {idx}/{len(image_files)}: {image_file.name}")

        try:
            # Load input image
            images = []
            images.append(load_image(str(image_file)))
            if ref_image:
                images.append(ref_image)

            # Generate output image
            output_image = pipe(
                prompt=prompt,
                image=images,
                num_inference_steps=15,
                guidance_scale=4,
            ).images[0]

            # Save with original filename (keeping extension)
            output_filename = f"{image_file.stem}.flux2.jpg"
            output_file_path = output_path / output_filename
            output_image.save(str(output_file_path))
            print(f"Saved: {output_file_path}")

        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue

    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Process images with FLUX.2-dev or fine-tune the model')
    parser.add_argument('--input', help='Input folder containing images for inference')
    parser.add_argument('--output', help='Output folder for processed images')
    parser.add_argument('--prompt', help='Prompt for image generation (inference mode)')
    parser.add_argument('--ref', help='Reference image')
    parser.add_argument('--model', help='Model path')
    parser.add_argument('--start', default=0, type=int, help='Start index (optional)')
    parser.add_argument('--limit', default=20, type=int, help='Limit number of images (optional)')
    
    # ファインチューニング用オプション
    parser.add_argument('--finetune', action='store_true', help='Enable fine-tuning mode')
    parser.add_argument('--train_dir', help='Training directory containing images organized by folder names')
    parser.add_argument('--output_model', help='Output path for fine-tuned model')
    parser.add_argument('--epochs', default=5, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate (default: 1e-5)')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
    
    args = parser.parse_args()

    device = "cuda:0"
    torch_dtype = torch.bfloat16

    # Load model from specified path or download from repo
    print("Loading model...")
    if args.model:
        pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)
    else:
        pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch_dtype)
    
    pipe.to(device)

    # ファインチューニングモード
    if args.finetune:
        if not args.train_dir:
            print("Error: --train_dir is required for fine-tuning mode")
            return
        if not args.output_model:
            print("Error: --output_model is required for fine-tuning mode")
            return
        
        finetune_model(pipe, args.train_dir, args.output_model, args.epochs, args.lr, args.batch_size)
        return

    # 推論モード
    if not args.input or not args.output or not args.prompt:
        print("Error: --input, --output, and --prompt are required for inference mode")
        return

    inference(pipe, args.input, args.output, args.prompt, args.ref, args.start, args.limit)

if __name__ == "__main__":
    main()

# 使用例:
# 
# ファインチューニング:
# python3 flux2.py --finetune --train_dir ./images/_v2/girl/ --output_model ./tuned/flux2 --epochs 10 --lr 1e-5 --batch_size 2
#
# 推論:
# python3 flux2.py --input ./images/_crop/_s1002/ --ref ./images/_ref/style_spo_a.png --output ./output --prompt "change image1 into a realistic photo using Bokeh, she wears Japanese gym uniform in image2" --limit 3
#
# ファインチューニング済みモデルで推論:
# python3 flux2.py --input ./images/__tmp/ --output ./output --prompt "..." --model ./finetuned_model --limit 10