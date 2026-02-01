# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Stable Diffusion and FLUX experimentation repository for fine-tuning and generating images using diffusion models. The codebase supports training custom models on domain-specific image datasets and running inference through CLI scripts or a web UI.

## Environment Setup

### Docker Environment
All development is done within a Docker container with NVIDIA GPU support:

```bash
# Start the Docker environment with GPU access
docker compose run --rm --service-ports labs /bin/bash
```

The container mounts:
- `./` → `/app/` (working directory)
- `/d/obara/myPictures` → `/pictures/`
- `/home/baraoto/MyPictures/` → `/pictures2/`

### Dependencies
Base image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

Key packages:
- PyTorch 2.6.0 with CUDA 12.4
- Hugging Face Diffusers (installed from git main branch)
- transformers, accelerate
- gradio (for web UI)
- BLIP for image captioning
- spaCy for NLP

## Dataset Structure and Training Data Organization

### Directory-Based Prompts
Training datasets use a hierarchical directory structure where folder names automatically become training prompts:

```
images/
├── _v2/
│   └── girl/
│       ├── long_hair/
│       │   └── image1.jpg  → prompt: "girl, long hair"
│       └── short_hair/
│           └── image2.jpg  → prompt: "girl, short hair"
```

**Important Rules**:
- Folders starting with `__` (double underscore): Skip this directory and all subdirectories
- Folders starting with `_` (single underscore): Ignore this level but process subdirectories
- Folder names with underscores are converted to spaces: `long_hair` → `"long hair"`
- Path segments are joined with commas to create the prompt

### Caption Files
Images can have companion `.cap` files with the same basename:
- `image.jpg` + `image.cap` → Uses caption from `.cap` file
- Lines starting with `#` in `.cap` files are comments
- First non-empty, non-comment line is used as the prompt

## Training Commands

### Stable Diffusion 1.5 Fine-tuning

```bash
# Basic training (512x512 images, 20 epochs)
python3 train_sd.py --image_size 512 --images images/ --epoch 20 > train.out &

# Training from a checkpoint with custom UNET weights
python3 train_sd.py --image_size 768 --images images/ --epoch 50 \
    --load_model ./tuned/sd15d --unet ./models/chilloutmix.safetensors

# Continue training from a previous checkpoint
python3 train_sd.py --image_size 768 --images images/_v2/ --epoch 15 \
    --load_model ./tuned/sd15_768/
```

**Key Parameters**:
- `--image_size`: Training resolution (512, 768, 1024)
- `--batch`: Batch size (default: 4)
- `--epoch`: Number of training epochs
- `--load_model`: Base model or checkpoint to start from
- `--unet`: Load custom UNET weights (e.g., from CivitAI models)
- `--save_model`: Output directory for trained model (default: `./tuned/sd15`)

Training only fine-tunes the UNET while freezing VAE and text encoder.

### FLUX2 Fine-tuning

```bash
# Fine-tune FLUX2 model
python3 flux2.py --finetune --train_dir ./images/_v2/girl/ \
    --output_model ./tuned/flux2 --epochs 10 --lr 1e-5 --batch_size 2
```

**Key Parameters**:
- `--train_dir`: Directory with training images
- `--output_model`: Output path for fine-tuned model
- `--epochs`: Number of training epochs (default: 5)
- `--lr`: Learning rate (default: 1e-5)
- `--batch_size`: Batch size (default: 1)

FLUX2 training freezes VAE and text encoder, only training the transformer component.

### VAE and Transformer Training

```bash
# Train custom VAE
python3 train_vae.py [args]

# Train transformer
python3 train_transformer.py [args]
```

## Inference Commands

### Stable Diffusion 1.5 Generation

```bash
# Generate from prompt file
python3 train_sd.py --load_model ./tuned/sd15/ \
    --prompt_file ./data/test_prompts.txt --image_size 512

# Generate from single prompt
python3 train_sd.py --load_model ./tuned/sd15/ \
    --prompt "masterpiece, best quality, girl, anime style, lalafell"

# Image-to-image generation
python3 train_sd.py --load_model ./tuned/sd15/ \
    --init_image ./input.jpg --prompt "water color style" --image_size 768
```

**Generation Parameters**:
- Inference steps: 50 (hardcoded in `NUM_STEPS`)
- Guidance scale: 7.0 (text-to-image) or 5.0 (img2img)
- Strength: 0.8 (img2img only)
- Positive prompt prefix: `"(masterpiece), best quality, best composition, "`
- Output: `./output/sd_{timestamp}_{index}.png`

### FLUX2 Generation

```bash
# Inference with FLUX2
python3 flux2.py --input ./images/__tmp/ --output ./output \
    --prompt "Change the image to a realistic photo, she is a Japanese girl" \
    --limit 10

# Use fine-tuned model
python3 flux2.py --input ./images/__tmp/ --output ./output \
    --prompt "..." --model ./tuned/flux2 --limit 10
```

**Parameters**:
- `--start`: Start index (default: 0)
- `--limit`: Number of images to process (default: 20)
- `--model`: Path to custom/fine-tuned model
- Guidance scale: 4 (hardcoded)
- Inference steps: 15 (hardcoded)

### Web UI

```bash
# Launch Gradio web interface for SD1.5 with ControlNet
python3 web_sd.py --model ./tuned/sd15/ --unet ./models/custom.safetensors

# Default model (no args)
python3 web_sd.py
```

The web UI runs on port 7860 (mapped through Docker) and provides:
- Image-to-image generation with ControlNet (Canny edge detection)
- Interactive sliders for ControlNet scale, Canny threshold, and strength
- Automatic PNG export to `./output/tmp/sd_{timestamp}.png`

## Utility Scripts

### Image Categorization

```bash
# Generate BLIP captions and organize images by keywords
python3 categorize.py /app/images/_v1/girl/junior_idol/ --organize \
    --min-count 2 --output-dir /app/images/_v1/girl/__junior_idol --caption
```

**Functionality**:
- Uses BLIP model to generate captions
- Extracts nouns/keywords with spaCy
- Creates hierarchical folder structure based on keywords
- `--caption`: Saves `.cap` files alongside images
- `--min-count`: Minimum keyword frequency to create a folder

### Image Cropping

```bash
# Crop images (utility script)
python3 crop_images.py [args]
```

### Sample Generation

```bash
# Generate test samples
python3 make_samples.py [args]
```

## Model Conversion

```bash
# Convert Diffusers format to original SD checkpoint
python3 convert_diffusers_to_original_stable_diffusion.py \
    --model_path ./tuned/sd15_768b/ \
    --checkpoint_path ./tuned/sd15_spo.safetensors \
    --half --use_safetensors
```

Converts Hugging Face Diffusers format models to standalone `.safetensors` files compatible with other SD tools.

## GPU Power Management

On Windows with NVIDIA GPUs:

```powershell
# Set power limit for GPU 1 to 240W
nvidia-smi.exe -pl 240 -i 1
```

## Code Architecture

### Training Pipeline (train_sd.py, flux2.py)

1. **Dataset Loading**: `ImageFolderDataset` class walks directories, creates prompts from folder names
2. **Model Loading**: Loads pretrained model from Hugging Face or local checkpoint
3. **Freezing**: VAE and text encoder are frozen, only UNET/transformer is trained
4. **Training Loop**:
   - Encode images to latents via VAE
   - Add noise at random timesteps
   - Predict noise with UNET/transformer
   - Compute MSE loss between predicted and actual noise
   - Backprop and optimize
5. **Checkpoint Saving**: Saves full pipeline (UNET + VAE + text encoder) in Diffusers format

### Inference Pipeline

1. **Model Loading**: Load trained pipeline from checkpoint
2. **Prompt Processing**: Add quality prefix, apply negative prompts
3. **Generation**:
   - Text-to-image: Start from pure noise
   - Image-to-image: Encode init image to latents, add partial noise
4. **Denoising**: Run diffusion sampling with scheduler
5. **Decoding**: VAE decodes latents to images
6. **Saving**: Export to timestamped PNG files

### Web UI (web_sd.py)

- **ControlNet Integration**: Uses Canny edge detection for structural guidance
- **Latent Encoding**: Custom VAE encoding path for img2img with ControlNet
- **Gradio Interface**: Real-time parameter adjustment with sliders

## Important Constants and Defaults

### Stable Diffusion 1.5
- Learning rate: `1e-4`
- Device: `cuda:0`
- Dtype: `torch.bfloat16`
- VAE scaling factor: `0.18215`
- Timestep range: 0-700 (training), max 1000
- Inference steps: 50

### FLUX2
- Learning rate: `1e-5` (default)
- Dtype: `torch.bfloat16`
- VAE scaling factor: `0.13025`
- Inference steps: 15
- Guidance scale: 4

## Common Workflows

### Training a Custom Model from Scratch

1. Organize training images in directory structure
2. Start Docker container
3. Run training: `python3 train_sd.py --images images/ --epoch 20`
4. Monitor progress via output logs
5. Test inference: `python3 train_sd.py --load_model ./tuned/sd15 --prompt "test prompt"`

### Fine-tuning from a Community Model

1. Download model weights to `./models/`
2. Train with UNET initialization: `python3 train_sd.py --unet ./models/model.safetensors --images images/ --epoch 10`
3. Generate samples to evaluate quality
4. Continue training if needed

### Automatic Dataset Organization

1. Run categorization: `python3 categorize.py ./images/raw/ --organize --caption --min-count 3`
2. Review generated folder structure
3. Manually curate if needed
4. Use organized dataset for training

## Notes on Model Formats

- **Diffusers format**: Multi-file directory structure (used by Hugging Face, this repo)
- **Safetensors format**: Single-file checkpoint (use conversion script for compatibility)
- Models are saved in Diffusers format by default
- Use `convert_diffusers_to_original_stable_diffusion.py` to export for external tools
