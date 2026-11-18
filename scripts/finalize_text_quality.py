"""Seal sampled structural QA and the explicitly inspected attestation exclusions."""
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    local = ROOT / 'data/processed/text-quality-100-v1'
    sample = read_json(local / 'review_excerpts.json')
    partial = read_json(local / 'partial_decisions.json')
    if not partial['review_complete'] or partial['review_excerpts_sha256'] != sha256(local / 'review_excerpts.json'):
        raise ValueError('Incomplete or changed inspected sample')
    corpus_path = ROOT / 'data/processed/text-corpus-100-v1/manifest.json'
    corpus = read_json(corpus_path)
    candidates = {r['accession']: r for r in corpus['documents'] if r['status'] == 'candidate_needs_review'}
    decisions = {}
    if len(partial['documents']) != len(sample['documents']) or len({r['index'] for r in partial['documents']}) != len(sample['documents']):
        raise ValueError('Missing or duplicate sample decisions')
    for entry in partial['documents']:
        source = sample['documents'][entry['index']]
        if entry['accession'] != source['accession'] or entry['text_sha256'] != source['text_sha256']:
            raise ValueError('Decision/source mismatch')
        decisions[entry['accession']] = {**entry, 'decision': 'exclude' if entry['decision'] == 'reject' else entry['decision'],
                                        'selection_reason': source['selection_reason'], 'ticker': source['ticker']}
    screened = []
    for accession, row in candidates.items():
        path = ROOT / row['text_path']
        if sha256(path) != row['text_sha256']:
            raise ValueError('Changed frozen section')
        text = path.read_text().lower()
        if 'report of independent' in text and 'we have audited' in text:
            # All 26 contexts were inspected before this narrowly bounded rule was sealed.
            if row['ticker'] not in {'BSX', 'PEP'}:
                raise ValueError('Unreviewed supplementary attestation issuer')
            screened.append(accession)
            previous = decisions.get(accession)
            if previous and previous['decision'] != 'exclude':
                raise ValueError('Conflicting sample decision')
            if not previous:
                decisions[accession] = {'accession': accession, 'ticker': row['ticker'],
                    'decision': 'exclude', 'text_sha256': row['text_sha256'],
                    'selection_reason': 'supplementary_attestation_scope_review',
                    'reason': 'Agent inspected formal independent auditor/ICFR attestation embedded inside extracted span; outside primary MD&A scope.'}
    if len(screened) != 26:
        raise ValueError('Supplementary screen no longer matches inspected evidence')
    flagged = [r for r in candidates.values() if r.get('quality_flags')]
    if any(r['accession'] not in decisions for r in flagged):
        raise ValueError('Flagged document remains undecided')
    counts = Counter((r['selection_reason'], r['decision']) for r in decisions.values())
    output = {'created_utc': utc_now(), 'review_complete': True, 'reviewer': 'agent-assisted boundary/interior review; not independent human certification',
        'manifest_sha256': sha256(corpus_path), 'sample_sha256': sha256(local / 'review_excerpts.json'),
        'decision_ledger_sha256': sha256(local / 'partial_decisions.json'), 'source_sha256': sha256(Path(__file__)),
        'candidate_documents': len(candidates), 'sample_documents': len(sample['documents']),
        'decision_counts': [{'selection_reason': a, 'decision': b, 'count': n} for (a, b), n in sorted(counts.items())],
        'supplementary_attestation_screen': {'documents': len(screened), 'scope': 'Added after an ordinary sample revealed ICFR contamination, before any text forecast fitting; all matching contexts inspected.'},
        'training_eligibility': 'All flags explicitly reviewed; ordinary candidates eligible under sampled QA unless explicitly excluded. Unreviewed ordinary documents are not individually certified. Entity-spell restrictions apply separately.',
        'limitations': 'Sampling detected extraction errors. No universal extraction accuracy or independent reviewer agreement claimed. No parser repairs or outcome-based company replacements.',
        'documents': sorted(decisions.values(), key=lambda r: r['accession']), 'market_outcomes_2026_accessed': False}
    write_json(ROOT / 'results/TEXT_CORPUS_REVIEW_100_V1.json', output)
    print(output['decision_counts'])


if __name__ == '__main__':
    main()
