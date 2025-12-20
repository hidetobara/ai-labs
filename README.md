Transformer は特定のlatent画像を覚えることができるか？

# 環境
docker compose run --rm --service-ports labs /bin/bash

# Power Limit
PowerShell上で
$ nvidia-smi.exe -pl 240 -i 1

# 変換
python3 convert_diffusers_original_stable_diffusion.py --model_path ./tuned/sd15 --checkpoint_path ./tuned/my_sd15.safetensors --half --use_safetensors
