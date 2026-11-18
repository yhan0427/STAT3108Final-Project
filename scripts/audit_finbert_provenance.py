"""Compare frozen safetensors to original 2020 weights, using weights-only loading."""
from pathlib import Path
import sys
import urllib.request

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    metadata_path = ROOT / 'data/models/finbert-provenance-tree.json'
    metadata = read_json(metadata_path)
    records = {r['path']: r for r in metadata['records']}
    original = records['pytorch_model.bin']
    revision = original['lastCommit']['id']
    url = f'https://huggingface.co/ProsusAI/finbert/resolve/{revision}/pytorch_model.bin'
    temporary = ROOT / 'data/models/finbert-original-audit/pytorch_model.bin'
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if not temporary.exists() or sha256(temporary) != original['lfs']['oid']:
        urllib.request.urlretrieve(url, temporary)
    if sha256(temporary) != original['lfs']['oid']:
        raise ValueError('Original binary hash mismatch')
    old = torch.load(temporary, map_location='cpu', weights_only=True)
    cfg = read_json(ROOT / 'config/finbert-v1.json')
    current = ROOT / f"data/models/finbert-{cfg['revision'][:7]}/model.safetensors"
    if sha256(current) != records['model.safetensors']['lfs']['oid']:
        raise ValueError('Current weights differ from frozen provenance metadata')
    new = load_file(current)
    shared = sorted(set(old) & set(new))
    mismatches = [k for k in shared if not torch.equal(old[k], new[k])]
    old_only, new_only = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    report = {'completed_utc': utc_now(), 'source_sha256': sha256(Path(__file__)),
        'metadata_sha256': sha256(metadata_path), 'metadata_url': metadata['url'],
        'original_url': url, 'original_commit': original['lastCommit'],
        'original_sha256': sha256(temporary), 'safetensors_sha256': sha256(current),
        'safetensors_commit': records['model.safetensors']['lastCommit'],
        'shared_tensors': len(shared), 'different_tensors': mismatches,
        'original_only_keys': old_only, 'safetensors_only_keys': new_only,
        'shared_parameters_bitwise_equal': not mismatches,
        'limitations': 'Repository timestamps and exact parameter equality support earlier parameter availability; not exhaustive pretraining-corpus or tokenizer historical provenance certification. Not evidence of historically deployable full data pipeline.',
        'market_outcomes_2026_accessed': False}
    write_json(ROOT / 'results/FINBERT_WEIGHT_PROVENANCE_V1.json', report)
    if not mismatches and not new_only:
        temporary.unlink()
        temporary.parent.rmdir()
    print({k: report[k] for k in ['shared_tensors', 'different_tensors', 'original_only_keys', 'safetensors_only_keys']}, flush=True)


if __name__ == '__main__':
    main()
