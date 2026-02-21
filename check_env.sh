#!/bin/bash
cd ~/cosmos-predict2.5
source .venv/bin/activate
echo "=== Python Check ==="
python -c "import cosmos_predict2; print('cosmos_predict2: OK')"
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader
echo "=== HF Hub Models ==="
ls ~/.cache/huggingface/hub/ 2>/dev/null || echo "No HF hub cache"
echo "=== Done ==="
