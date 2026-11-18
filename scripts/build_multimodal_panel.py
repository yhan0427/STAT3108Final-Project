"""Matched, identity-guarded cohort with training-document-only FinBERT PCA."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now
from volrisk.text_features import align_documents, fit_training_document_pca


def main():
    destination = ROOT / 'data/processed/multimodal-100-v1'
    if (destination / 'manifest.json').exists():
        raise FileExistsError('Preserve completed matched dataset')
    review_path = ROOT / 'results/TEXT_CORPUS_REVIEW_100_V1.json'
    review = read_json(review_path)
    corpus_path = ROOT / 'data/processed/text-corpus-100-v1/manifest.json'
    if not review['review_complete'] or sha256(corpus_path) != review['manifest_sha256']:
        raise ValueError('Missing frozen text quality gate')
    corpus = read_json(corpus_path)
    excluded = {r['accession'] for r in review['documents'] if r['decision'] == 'exclude'}
    accepted = {r['accession'] for r in review['documents'] if r['decision'] == 'accept'}
    eligible = [r for r in corpus['documents'] if r['status'] == 'candidate_needs_review'
                and r['accession'] not in excluded and (not r.get('quality_flags') or r['accession'] in accepted)]
    parent = ROOT / 'data/processed/financial-entity-v1/panel.pkl'
    if sha256(parent) != read_json(parent.parent / 'manifest.json')['panel_sha256']:
        raise ValueError('Parent dataset changed')
    roster_path = ROOT / 'config/multimodal-100-v1.csv'
    roster = pd.read_csv(roster_path)
    panel = pd.read_pickle(parent)
    panel = panel.loc[panel.ticker.isin(roster.ticker)].copy().reset_index(drop=True)
    if panel.ticker.nunique() != 100 or panel.date.max() >= pd.Timestamp('2026-01-01'):
        raise ValueError('Unexpected cohort or sealed data')
    for r in roster.itertuples():
        group = panel.loc[panel.ticker.eq(r.ticker)]
        if not group.role.eq(r.role).all() or not group.issuer_id.eq(r.issuer_id).all():
            raise ValueError('Cohort role/issuer mismatch')
    spells = read_json(ROOT / 'config/entity-spells-v1.json')['spells']
    records, vectors = [], []
    pre_spell = 0
    for r in sorted(eligible, key=lambda r: (r['issuer_id'], r['filing_date'], r['accession'])):
        start = spells.get(r['ticker'], {}).get('start')
        if start and r['filing_date'] < start:
            pre_spell += 1
            continue
        metadata = ROOT / f"data/processed/finbert-corpus-100-v1/{r['accession']}.json"
        encoded = read_json(metadata)
        path = ROOT / encoded['embedding_path']
        if encoded['text_sha256'] != r['text_sha256'] or sha256(path) != encoded['embedding_sha256']:
            raise ValueError('Encoding/source identity changed')
        vector = np.load(path)['embedding']
        if vector.shape != (768,) or not np.isfinite(vector).all():
            raise ValueError('Invalid representation')
        records.append({**encoded, 'document_id': len(records), 'entity_start': start})
        vectors.append(vector)
    documents = pd.DataFrame(records)
    matrix = np.stack(vectors)
    join_columns = ['text_age_days', 'text_log_tokens', 'text_available', 'newer_original_unavailable',
                    'text_document_id', 'source_document_id']
    original = pd.DataFrame(corpus['documents'])
    for ticker, group in panel.groupby('ticker', sort=False):
        docs = documents.loc[documents.ticker.eq(ticker)]
        filings = original.loc[original.ticker.eq(ticker), 'filing_date']
        aligned = align_documents(group.date, docs, filings, stale_days=550,
                                  entity_start=spells.get(ticker, {}).get('start'))
        if not np.array_equal(group.date.to_numpy(), aligned.date.to_numpy()):
            raise ValueError('Join reordered market histories')
        panel.loc[group.index, join_columns] = aligned[join_columns].to_numpy()
    panel[['text_document_id', 'source_document_id']] = panel[['text_document_id', 'source_document_id']].astype(int)
    fitting = panel.split.eq('train') & panel.role.eq('training_company')
    reducer, reduced, used = fit_training_document_pca(matrix, documents, panel,
        panel.text_document_id.to_numpy(), fitting.to_numpy(), width=32)
    counts = []
    for (role, split), group in panel.loc[panel.split.ne('excluded')].groupby(['role', 'split']):
        common = group if split == 'train' else group.loc[group.context_ready.eq(True)]
        counts.append({'role': role, 'split': split, 'companies': common.ticker.nunique(),
            'origins': len(common), 'text_available_fraction': float(common.text_available.mean()),
            'companies_with_any_text': common.loc[common.text_available.eq(1), 'ticker'].nunique()})
    destination.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(destination / 'panel.pkl')
    documents.to_pickle(destination / 'documents.pkl')
    np.savez(destination / 'document_vectors.npz', embedding=matrix, reduced=reduced, training_document_ids=used)
    joblib.dump(reducer, destination / 'training_document_pca.joblib')
    manifest = {'built_utc': utc_now(), 'parent_panel_sha256': sha256(parent),
        'roster_sha256': sha256(roster_path), 'quality_review_sha256': sha256(review_path),
        'entity_policy_sha256': sha256(ROOT / 'config/entity-spells-v1.json'),
        'source_sha256': sha256(Path(__file__)), 'text_features_sha256': sha256(ROOT / 'src/volrisk/text_features.py'),
        'files': {p.name: sha256(p) for p in destination.iterdir() if p.is_file()},
        'companies': int(panel.ticker.nunique()), 'market_rows': len(panel), 'document_vectors': len(documents),
        'pre_spell_documents_excluded': pre_spell, 'training_unique_documents_for_pca': len(used),
        'pca_components': int(reducer.n_components_), 'pca_explained_variance_fraction': float(reducer.explained_variance_ratio_.sum()),
        'counts': counts, 'scope': 'Fixed 80 training /20 held reference issuers; no text-availability deletion/replacement. Sampled structural QA, not exhaustive certification. Strict next-trading-date join,550-day staleness,identity-spell guards. PCA fitted on unique documents actually joined to fitting origins only.',
        'market_outcomes_2026_accessed': False}
    write_json(destination / 'manifest.json', manifest)
    write_json(ROOT / 'results/MULTIMODAL_DATA_100_V1.json', manifest)
    print(pd.DataFrame(counts).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
