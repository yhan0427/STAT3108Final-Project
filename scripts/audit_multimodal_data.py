"""Independent full-matrix checks of source IDs, past-only joins and PCA fitting set."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    destination = ROOT / 'data/processed/multimodal-100-v1'
    manifest = read_json(destination / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha256(destination / name) != expected:
            raise ValueError('Dataset artifact changed: ' + name)
    panel = pd.read_pickle(destination / 'panel.pkl')
    documents = pd.read_pickle(destination / 'documents.pkl')
    vectors = np.load(destination / 'document_vectors.npz')
    ids = panel.text_document_id.to_numpy(dtype=int)
    available = ids >= 0
    if available.sum() == 0 or ids[available].max() >= len(documents):
        raise ValueError('Invalid document IDs')
    referenced = documents.iloc[ids[available]].reset_index(drop=True)
    origins = panel.loc[available].reset_index(drop=True)
    if not np.array_equal(referenced.issuer_id, origins.issuer_id) or not np.array_equal(referenced.role, origins.role):
        raise ValueError('Cross-company or role text assignment')
    age = (origins.date - pd.to_datetime(referenced.filing_date)).dt.days
    if not age.between(1, 550).all() or not np.array_equal(age.to_numpy(), origins.text_age_days.to_numpy()):
        raise ValueError('Future/same-day/stale text or inconsistent age')
    boundary = pd.to_datetime(documents.entity_start, errors='coerce')
    if (boundary.notna() & pd.to_datetime(documents.filing_date).lt(boundary)).any():
        raise ValueError('Pre-spell text survived')
    quality = read_json(ROOT / 'results/TEXT_CORPUS_REVIEW_100_V1.json')
    excluded = {r['accession'] for r in quality['documents'] if r['decision'] == 'exclude'}
    if set(documents.accession) & excluded:
        raise ValueError('Rejected text reached document matrix')
    training = panel.role.eq('training_company') & panel.split.eq('train')
    if panel.loc[training, 'target_end'].max() >= pd.Timestamp('2022-01-01'):
        raise ValueError('Unpurged training target')
    expected_ids = np.unique(ids[training & available])
    if not np.array_equal(expected_ids, vectors['training_document_ids']):
        raise ValueError('PCA fitting document set mismatch')
    if not documents.iloc[expected_ids].role.eq('training_company').all():
        raise ValueError('Held company in reducer')
    reducer = joblib.load(destination / 'training_document_pca.joblib')
    if not np.allclose(reducer.mean_, vectors['embedding'][expected_ids].astype(float).mean(axis=0), atol=1e-12, rtol=0):
        raise ValueError('PCA mean not unique training-document mean')
    recomputed = reducer.transform(vectors['embedding'].astype(float)).astype(np.float32)
    difference = float(np.max(np.abs(recomputed - vectors['reduced'][:, :reducer.n_components_])))
    if difference > 1e-6:
        raise ValueError('Stored PCA transformation mismatch')
    if panel.date.max() >= pd.Timestamp('2026-01-01') or pd.to_datetime(documents.filing_date).max() >= pd.Timestamp('2026-01-01'):
        raise ValueError('Sealed-stage boundary violation')
    report = {'completed_utc': utc_now(), 'dataset_manifest_sha256': sha256(destination / 'manifest.json'),
        'source_sha256': sha256(Path(__file__)), 'market_rows': len(panel),
        'available_daily_document_assignments_checked': int(available.sum()),
        'document_vectors_checked': len(documents), 'unique_training_pca_documents_checked': len(expected_ids),
        'pca_recomputed_max_absolute_difference': difference,
        'past_only_join_and_550day_staleness': True, 'issuer_and_role_alignment': True,
        'excluded_sections_absent': True, 'pre_spell_sections_absent': True,
        'all_dataset_hashes_match': True, 'market_outcomes_2026_accessed': False,
        'limitations': 'Structural/temporal consistency checks do not certify every unreviewed section, all corporate histories or original SEC fact correctness.'}
    write_json(ROOT / 'results/MULTIMODAL_DATA_CHECKS_100_V1.json', report)
    print(report)


if __name__ == '__main__':
    main()
