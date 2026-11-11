"""Download the fixed official checkpoint; no remote Python code is executed."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parents[1]
plan=json.loads((root/'config/aligned-v1.json').read_text())['ttm']
snapshot_download(plan['model_id'], revision=plan['revision'],
    local_dir=str(root/'data/models/ttm-3442406'),
    allow_patterns=['config.json','model.safetensors','README.md','LICENSE'])
print('Pinned TTM checkpoint ready:',plan['revision'])
