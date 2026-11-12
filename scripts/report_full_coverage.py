"""Publish actual coverage counts without opening any model outcomes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volrisk.io import read_json
x=read_json(ROOT/'results/FULL_COHORT_AUDIT_V2.json')
lines=['# Full-cohort coverage audit','',f"Frozen candidates: {x['candidate_issuers']} CIK-defined issuers. Retained raw histories: {x['retained_raw_histories']}. Unique retained company-date price rows: {x['raw_company_date_rows_retained']:,}. Reference date: 2025-12-31. These are actual filtered counts, not a target filled by synthetic data.",'',
'## Model-ready origins','', '| Role | Period | Firms: 60-day history | Origins: 60-day history | Firms: 512-rv20 history | Origins: 512-rv20 history |','|---|---|---:|---:|---:|---:|']
for r in x['model_ready_counts']:lines.append(f"| {r['role']} | {r['split']} | {r['firms_context60']} | {r['origins_context60']} | {r['firms_context512']} | {r['origins_context512']} |")
lines+=['','Train/validation rows listed for held companies are available historical labels, not fitting inputs. Unified training explicitly excludes all held-company labels from fitting, early stopping, hyperparameter selection and calibration. The held role excludes every company already scored in the pilot. Original pilot held firms now belong to development.','',
'## Unavailable histories','', '| Reference ticker | Role | Acquisition/validation failure |','|---|---|---|']
for r in x['failed']:lines.append(f"| {r['ticker']} | {r['role']} | {r.get('reason','').replace('|','/')} |")
lines+=['','## Recovered security identities','',
'BK and MMC were recovered via verified symbol-only changes to BNY and MRSH. Candidate issuers/roles were not replaced. Reference symbols, source announcements, SEC identifiers and provider aliases remain separate. Other mapping candidates involving mergers or entity changes were not automatically stitched.','',
'## Boundaries','',
'Frozen cohort reconstructed from a dated historical page plus explicit official December corrections; full independent official list validation remains qualified. Reference-date selection and current-provider availability can create survival/selection bias for earlier history. This is not historical membership at every date. No price forward fill, missing terminal return fabrication, or count-based padding was used. Raw history need not provide enough context for all methods. 2026 local price outcomes remain sealed.','',
'See docs/FULL_COHORT.md for acquisition, role selection and unified training details. Dataset v2 and model artifacts remain local; only provenance, coverage and aggregate results are versioned. The existing demo/poster are preliminary application artifacts and do not automatically reflect this larger research panel.']
(ROOT/'results/FULL_COHORT_COVERAGE_V2.md').write_text('\n'.join(lines)+'\n')
