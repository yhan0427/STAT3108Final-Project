# Full-cohort coverage audit

Frozen candidates: 500 CIK-defined issuers. Retained raw histories: 491. Unique retained company-date price rows: 1,428,780. Reference date: 2025-12-31. These are actual filtered counts, not a target filled by synthetic data.

## Model-ready origins

| Role | Period | Firms: 60-day history | Origins: 60-day history | Firms: 512-rv20 history | Origins: 512-rv20 history |
|---|---|---:|---:|---:|---:|
| held_company | test | 99 | 47042 | 97 | 46657 |
| held_company | train | 97 | 161686 | 95 | 134028 |
| held_company | validation | 97 | 46560 | 97 | 46217 |
| training_company | test | 391 | 187836 | 390 | 186809 |
| training_company | train | 386 | 650683 | 380 | 541061 |
| training_company | validation | 389 | 186221 | 387 | 183780 |

Train/validation rows listed for held companies are available historical labels, not fitting inputs. Unified training explicitly excludes all held-company labels from fitting, early stopping, hyperparameter selection and calibration. The held role excludes every company already scored in the pilot. Original pilot held firms now belong to development.

## Unavailable histories

| Reference ticker | Role | Acquisition/validation failure |
|---|---|---|
| AVB | training_company | $AVB: Data doesn't exist for startDate = 1388552400, endDate = 1767243600 |
| CTRA | training_company | $CTRA: Data doesn't exist for startDate = 1388552400, endDate = 1767243600 |
| DAY | training_company | $DAY: possibly delisted; no timezone found |
| EA | training_company | $EA: Data doesn't exist for startDate = 1388552400, endDate = 1767243600 |
| EQR | training_company | $EQR: Data doesn't exist for startDate = 1388552400, endDate = 1767243600 |
| HOLX | training_company | $HOLX: possibly delisted; no timezone found |
| KVUE | training_company | $KVUE: Data doesn't exist for startDate = 1388552400, endDate = 1767243600 |
| PSKY | held_company | $PSKY: Data doesn't exist for startDate = 1388552400, endDate = 1767243600 |
| WBD | training_company | $WBD: Data doesn't exist for startDate = 1388552400, endDate = 1767243600 |

## Recovered security identities

BK and MMC were recovered via verified symbol-only changes to BNY and MRSH. Candidate issuers/roles were not replaced. Reference symbols, source announcements, SEC identifiers and provider aliases remain separate. Other mapping candidates involving mergers or entity changes were not automatically stitched.

## Boundaries

Frozen cohort reconstructed from a dated historical page plus explicit official December corrections; full independent official list validation remains qualified. Reference-date selection and current-provider availability can create survival/selection bias for earlier history. This is not historical membership at every date. No price forward fill, missing terminal return fabrication, or count-based padding was used. Raw history need not provide enough context for all methods. 2026 local price outcomes remain sealed.

See docs/FULL_COHORT.md for acquisition, role selection and unified training details. Dataset v2 and model artifacts remain local; only provenance, coverage and aggregate results are versioned. The existing demo/poster are preliminary application artifacts and do not automatically reflect this larger research panel.
