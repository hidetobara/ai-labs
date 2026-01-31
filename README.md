Stable Diffusion, Flux の実験場

# 環境
docker compose run --rm --service-ports labs /bin/bash

# Power Limit
PowerShell上で
$ nvidia-smi.exe -pl 240 -i 1

# 変換
python3 convert_diffusers_to_original_stable_diffusion.py --model_path ./tuned/sd15_768b/ --checkpoint_path ./tuned/sd15_spo.safetensors --half --use_safetensors
