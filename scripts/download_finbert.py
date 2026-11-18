"""Download the pinned academic encoder/tokenizer; never execute remote model code."""
from pathlib import Path
import json

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
plan = json.loads((ROOT / "config/finbert-v1.json").read_text())
snapshot_download(plan["model_id"], revision=plan["revision"],
    local_dir=str(ROOT / f"data/models/finbert-{plan['revision'][:7]}"),
    allow_patterns=["config.json", "model.safetensors", "tokenizer_config.json", "special_tokens_map.json",
                    "vocab.txt", "tokenizer.json", "README.md", "LICENSE*"])
print("Pinned FinBERT local checkpoint ready:", plan["revision"])
