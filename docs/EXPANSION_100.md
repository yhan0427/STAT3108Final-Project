# First expanded training cohort

The 100-company candidate roster was frozen before new-company price acquisition in `config/expansion-100-v1.csv`. It keeps the original thirty companies and selects seventy additions by sector-proportional largest remainders and a fixed issuer hash. Alternating the new firms' hash ranks assigns thirty-five to fitting and thirty-five to held-company evaluation. No performance or price-coverage outcome selected the additions.

## Source reconstruction

The base is the [Wikipedia revision dated 2025-12-23](https://en.wikipedia.org/w/index.php?oldid=1329075976), retrieved via the revision API with cutoff 2025-12-31. Its incomplete December table had 501 securities. Apply the [official effective-2025-12-22 announcement](https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20251205-1480992/1480992_december2025rebalance.pdf): add CRH/CVNA/FIX, remove SOLS; LKQ and MHK were already absent. Names/sector follow the official additions; CIK mappings come from the preserved SEC ticker file. Both original rejected sources remain marked rejected; the reconstruction is a distinct derived registry with an explicit change record.

This produces 503 securities and 500 CIK-defined issuers. Those counts reconcile the share classes; they do not constitute independent full official verification. The cohort is a reconstructed fixed reference-date research universe, not historical membership at every backfilled date. Wikipedia's data attribution remains applicable; source metadata is not original MIT-licensed project code. The page source, hash, source date and every correction are preserved locally/in the configuration. Issuer identifiers may need separate continuity review around reorganizations.

## Collection and realized coverage

100 candidates yielded 99 usable pre-2026 price histories and 291,416 unique company-date raw rows. WBD history was unavailable from the provider and was not replaced. This failure is retained in EXPANSION_100_COVERAGE.json. Successful raw collection does not imply enough context for every model/split. Q has insufficient pre-2022 history for training: 64 companies actually provide fitting and validation labels, within the 65-company fitting-role cohort. The 34 retained held-company histories supply no training, selection or calibration labels.

Shorter histories are kept after IPO/listing rather than forward-filled. Missing calendars fail the check; valid contexts must remain within one issuer. Targets are the twenty returns after next-close execution (t+2 through t+21), purged at every period boundary. The 2026 outcome holdout remains sealed.

## Models and evaluation

Train-only preprocessing and common sixty-session market inputs for pooled Ridge and LSTM. Ridge alpha and LSTM epochs use only non-held validation firms in 2022–2023. Average the three declared LSTM seeds. Timestral uses the same pinned pretrained Transformer as the prior experiment, with 512 rv20 observations and the declared quantile-RMS point. Its context/information differs, and pretraining overlap is unresolved. Report both raw and validation-calibrated versions of every procedure; held-company labels fit neither step.

Evaluate the original, new fitting-role and held-company groups separately on eligible 2024–2025 origins. These calendar periods were already inspected on the original cohort; new-company evaluation is not an untouched new time period. Retain all company scores and date-block intervals. The expansion is a test of whether earlier patterns extend, not an effort to force an LSTM/Ridge finding.

Commands: prepare_expansion.py freezes the reconstructed registry/roles, build_expansion.py collects and builds the panel, train_expansion.py fits and evaluates. Full row-level data and fitted artifacts stay local under ignored data/results paths; configurations, coverage and aggregate research results are versioned. Missing/short histories, reference-date selection and surviving provider coverage remain limitations. No 500-company or million-observation result is claimed.
