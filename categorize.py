import os
import argparse
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy

def extract_keywords(caption, nlp, max_keywords=5):
    """
    キャプションから名詞を抽出してキーワードを取得
    
    Args:
        caption (str): 生成されたキャプション
        nlp: spaCyの言語モデル
        max_keywords (int): 抽出する最大キーワード数
    
    Returns:
        list: 抽出されたキーワードのリスト
    """
    doc = nlp(caption)
    
    # 名詞を抽出（固有名詞と普通名詞）
    nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    
    # 重複を削除しつつ順序を保持
    seen = set()
    unique_nouns = []
    for noun in nouns:
        if noun.lower() not in seen:
            seen.add(noun.lower())
            unique_nouns.append(noun)
    
    # 最大キーワード数まで返す
    return unique_nouns[:max_keywords]

def generate_captions(folder_path, min_count=1, output_dir=None, organize=False, save_caption=False):
    """
    指定フォルダ内の全画像にキャプションを生成する
    
    Args:
        folder_path (str): 画像フォルダのパス
        min_count (int): フォルダを作成する最小キーワード出現回数
        output_dir (str): 出力先ディレクトリ（Noneの場合は元フォルダ内に作成）
        organize (bool): 画像を整理してコピーするかどうか
        save_caption (bool): キャプションを.capファイルとして保存するかどうか
    """
    # モデルとプロセッサの読み込み
    print("モデルを読み込んでいます...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # spaCyの英語モデルを読み込み
    print("spaCyモデルを読み込んでいます...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("エラー: spaCyの英語モデルがインストールされていません")
        print("以下のコマンドでインストールしてください:")
        print("  python -m spacy download en_core_web_sm")
        return
    
    # サポートする画像拡張子
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # フォルダ内の画像ファイルを取得
    folder = Path(folder_path)
    if not folder.exists():
        print(f"エラー: フォルダ '{folder_path}' が見つかりません")
        return
    
    image_files = [f for f in folder.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"フォルダ '{folder_path}' 内に画像ファイルが見つかりませんでした")
        return
    
    print(f"\n{len(image_files)} 個の画像が見つかりました\n")
    print("=" * 80)
    
    # 画像ごとのキャプションとキーワードを保存
    image_data = {}
    all_keywords = []
    
    # 各画像にキャプションを生成
    for i, image_path in enumerate(image_files, 1):
        try:
            # 画像を開く
            image = Image.open(image_path).convert('RGB')
            
            # 画像を処理してキャプション生成
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs, max_length=50)
            caption = processor.decode(output[0], skip_special_tokens=True)
            
            # キーワードを抽出
            keywords = extract_keywords(caption, nlp, max_keywords=5)
            
            # キャプションとキーワードを保存
            image_data[image_path] = {
                'caption': caption,
                'keywords': keywords
            }
            all_keywords.extend(keywords)
            
            # 結果を表示
            print(f"[{i}/{len(image_files)}] {image_path.name}")
            print(f"キャプション: {caption}")
            print(f"キーワード: {', '.join(keywords)}")
            print("-" * 80)
            
        except Exception as e:
            print(f"[{i}/{len(image_files)}] {image_path.name}")
            print(f"エラー: {str(e)}")
            print("-" * 80)
    
    print("\n処理が完了しました")
    
    # 画像を整理する場合
    if organize and image_data:
        organize_images(image_data, all_keywords, folder_path, min_count, output_dir, save_caption)

def organize_images(image_data, all_keywords, source_folder, min_count, output_dir, save_caption):
    """
    キーワードに基づいて画像を階層的に整理する
    
    Args:
        image_data (dict): 画像パスとデータ（キャプション、キーワード）のマッピング
        all_keywords (list): 全キーワードのリスト
        source_folder (str): 元のフォルダパス
        min_count (int): フォルダを作成する最小キーワード出現回数
        output_dir (str): 出力先ディレクトリ
        save_caption (bool): キャプションを.capファイルとして保存するかどうか
    """
    print("\n" + "=" * 80)
    print("画像を整理しています...")
    
    # キーワードの出現回数をカウント
    keyword_counter = Counter(all_keywords)
    
    # 出現回数がmin_count以上のキーワードを取得
    frequent_keywords = {kw for kw, count in keyword_counter.items() if count >= min_count}
    
    print(f"\n出現回数 {min_count} 回以上のキーワード: {len(frequent_keywords)} 個")
    for kw in sorted(frequent_keywords):
        count = keyword_counter[kw]
        print(f"  - {kw}: {count}回")
    
    if not frequent_keywords:
        print(f"\n出現回数 {min_count} 回以上のキーワードが見つかりませんでした")
        return
    
    # 出力ディレクトリの設定
    if output_dir:
        base_output = Path(output_dir)
    else:
        base_output = Path(source_folder) / "organized"
    
    base_output.mkdir(parents=True, exist_ok=True)
    
    # 画像をコピー
    copied_count = 0
    for image_path, data in image_data.items():
        keywords = data['keywords']
        caption = data['caption']
        
        # 頻出キーワードのみをフィルタ
        valid_keywords = [kw for kw in keywords if kw in frequent_keywords]
        
        if not valid_keywords:
            continue
        
        # 階層的なパスを作成（最大階層）
        keyword_path = base_output
        for kw in valid_keywords[:5]:  # 最大階層
            keyword_path = keyword_path / kw
        
        # フォルダを作成
        keyword_path.mkdir(parents=True, exist_ok=True)
        
        # 画像をコピー
        dest_path = keyword_path / image_path.name
        
        # 同名ファイルが存在する場合は番号を付ける
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = keyword_path / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.copy2(image_path, dest_path)
        copied_count += 1
        print(f"コピー: {image_path.name} -> {keyword_path.relative_to(base_output)}/")
        
        # キャプションファイルを保存
        if save_caption:
            cap_path = dest_path.with_suffix('.cap')
            with open(cap_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            print(f"  キャプション保存: {cap_path.name}")
    
    print(f"\n{copied_count} 個の画像を '{base_output}' に整理しました")
    if save_caption:
        print(f"{copied_count} 個のキャプションファイル(.cap)を保存しました")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='フォルダ内の画像にBLIPモデルでキャプションを生成し、キーワードで整理します'
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='画像が含まれるフォルダのパス'
    )
    parser.add_argument(
        '--min-count',
        type=int,
        default=1,
        help='フォルダを作成する最小キーワード出現回数（デフォルト: 1）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='整理した画像の出力先ディレクトリ（指定しない場合は元フォルダ内に"organized"フォルダを作成）'
    )
    parser.add_argument(
        '--organize',
        action='store_true',
        help='このフラグを指定すると、画像をキーワードフォルダに整理してコピーします'
    )
    parser.add_argument(
        '--caption',
        action='store_true',
        help='このフラグを指定すると、生成したキャプションを.capファイルとして保存します'
    )
    
    args = parser.parse_args()
    generate_captions(args.folder_path, args.min_count, args.output_dir, args.organize, args.caption)

# 例
# python3 categorize.py /app/images/_v1/girl/junior_idol/ --organize --min-count 2 --output-dir /app/images/_v1/girl/__junior_idol --caption
# python3 categorize.py /app/images/__tmp/ --organize --min-count 2 --output-dir /app/images/_auto/_1043 --caption
