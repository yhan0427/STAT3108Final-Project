"""Audit existing trusted local artifacts; never retrain or reevaluate a model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from volrisk.io import read_json, sha256, write_json
from volrisk.dataset import MARKET_FEATURES, HAR_FEATURES
from volrisk.fundamentals import FUND_FEATURES, make_snapshots
from volrisk.models import TrainOnlyScaler
from volrisk.metrics import ranking_by_date, block_bootstrap

root = Path(__file__).resolve().parents[1]
run = root / 'results/runs/main-v1'
bundle = joblib.load(run / 'models.joblib')
frozen = read_json(run / 'frozen_run.json')
panel = pd.read_pickle(root / bundle['panel_path'])
checks = {}
checks['panel_hash_matches_fitted_data'] = sha256(root / bundle['panel_path']) == bundle['panel_sha256']
checks['weights_unchanged'] = all(sha256(run / n) == h for n,h in frozen['weights_sha256'].items())
manifest = frozen['dataset_manifest']
checks['raw_sources_unchanged'] = all(sha256(root / n) == h for n,h in manifest['sources'].items())
for split, stop in [('train','2022-01-01'),('validation','2024-01-01'),('test','2026-01-01')]:
    group = panel.loc[panel.split == split]
    checks[split+'_target_ends_before_boundary'] = bool((group.target_end < pd.Timestamp(stop)).all())
    checks[split+'_filings_strictly_before_origin'] = bool((group.filing_date < group.date).all())
    idx = group.index.to_numpy()
    checks[split+'_sequences_stay_within_company'] = bool(np.array_equal(panel.loc[idx,'ticker'].to_numpy(),panel.loc[idx-59,'ticker'].to_numpy()))
train = panel.loc[panel.split == 'train']
for key, fields in [('market_scaler',MARKET_FEATURES),('tab_scaler',MARKET_FEATURES+HAR_FEATURES),('har_scaler',HAR_FEATURES),('fund_scaler',FUND_FEATURES)]:
    expected = TrainOnlyScaler().fit(train[fields])
    checks[key+'_equals_training_only_fit'] = all(np.array_equal(getattr(expected,k),getattr(bundle[key],k)) for k in ['median','low','high','mean','std'])
log_y = np.log(train.target.to_numpy())
checks['target_scaling_training_only'] = bool(np.isclose(log_y.mean(),bundle['target_mean']) and np.isclose(log_y.std(),bundle['target_std']))
checks['test_prediction_hash_matches_original_evaluation'] = sha256(run/'test_predictions.csv') == read_json(run/'TEST_EVALUATION_COMPLETED.json')['prediction_sha256']

# Stronger than checking timestamp columns: actually delete every later filing
# from a raw input and require identical historical snapshots at selected cutoffs.
entities = read_json(root/'data/raw/sec/resolved_entities.json')
for ticker in ['AAPL','MSFT','JNJ','XOM','CAT']:
    full = read_json(root/entities[ticker]['path'])
    for cutoff in ['2018-06-30','2021-06-30']:
        stripped = read_json(root/entities[ticker]['path'])
        for taxonomy in stripped['facts'].values():
            for concept in taxonomy.values():
                for unit, records in concept.get('units',{}).items():
                    concept['units'][unit] = [r for r in records if r.get('filed','9999') <= cutoff]
        a = make_snapshots(full,latest=cutoff).reset_index(drop=True)
        b = make_snapshots(stripped,latest=cutoff).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
        checks[f'{ticker}_{cutoff}_later_filing_deletion_invariant'] = True

pred = pd.read_csv(run/'test_predictions.csv')
ranking = {m:ranking_by_date(pred,m,5).set_index('date').top_k_overlap for m in frozen['models']}
intervals = {}
for model,baseline in [('LSTM_market','Persistence'),('LSTM_fundamentals','LSTM_market')]:
    diff = ranking[model] - ranking[baseline]
    intervals[model+' minus '+baseline] = {str(b):block_bootstrap(diff.to_numpy(),b,2000) for b in [20,60]}
write_json(root/'results/AUDIT.json',{'checks':checks,'all_checks_pass':all(checks.values()),'ranking_difference_intervals':intervals,'note':'Post-hoc audit of existing predictions. Positive ranking differences favor first model. Not a fresh holdout or paper replication. These checks cannot certify the completeness of vendor point-in-time archives.'})
print(f'{sum(checks.values())}/{len(checks)} artifact and timing checks passed')
print(intervals)
assert all(checks.values())
