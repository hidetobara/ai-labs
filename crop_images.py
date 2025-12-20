import argparse
from pathlib import Path
from PIL import Image

def crop_and_save_images(input_folder: str, output_folder: str):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in input_path.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) // 2
                    top = (height - min_dim) // 2
                    right = left + min_dim
                    bottom = top + min_dim
                    
                    cropped_img = img.crop((left, top, right, bottom)).resize((512, 512))
                    
                    output_file = output_path / img_path.name
                    cropped_img.save(output_file)
                    print(f"Processed: {img_path.name}")
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine Tuning SDXL")
    parser.add_argument('--inputs', nargs='*', help="input path")
    parser.add_argument('--output', default="images/tmp/", help="output path")
    args = parser.parse_args()

    if args.inputs:
        for i in args.inputs:
            crop_and_save_images(i, args.output)
