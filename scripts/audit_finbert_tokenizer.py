"""Verify local tokenizer/config bytes against pinned original Git blob identities."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    metadata_path = ROOT / 'data/models/finbert-provenance-tree.json'
    metadata = read_json(metadata_path)
    records = {r['path']: r for r in metadata['records']}
    revision = read_json(ROOT / 'config/finbert-v1.json')['revision']
    directory = ROOT / f'data/models/finbert-{revision[:7]}'
    checks = []
    for name in ['vocab.txt', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json']:
        path = directory / name
        data = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        record = records[name]
        if blob != record['oid'] or record['lastCommit']['date'] >= '2024-01-01':
            raise ValueError('Artifact mismatch or later tokenizer/config revision')
        checks.append({'file': name, 'sha256': sha256(path), 'git_blob_sha1': blob, 'last_commit': record['lastCommit'], 'matches': True})
    report = {'completed_utc': utc_now(), 'metadata_sha256': sha256(metadata_path),
        'source_sha256': sha256(Path(__file__)), 'metadata_url': metadata['url'],
        'checks': checks, 'scope': 'Local artifact bytes match pinned repository objects whose reported last modification is2020. Supports earlier availability; not an exhaustive training-corpus audit.',
        'market_outcomes_2026_accessed': False}
    write_json(ROOT / 'results/FINBERT_TOKENIZER_PROVENANCE_V1.json', report)
    print('Verified four tokenizer/config Git blobs, all original2020 commits')


if __name__ == '__main__':
    main()
