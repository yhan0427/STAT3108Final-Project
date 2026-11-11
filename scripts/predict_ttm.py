"""Isolated optional dependency worker: no target labels are passed to this process."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from volrisk.aligned import restore_ttm_prediction
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    plan = read_json(Path(__file__).resolve().parents[1] / 'config/aligned-v1.json')['ttm']
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    model = TinyTimeMixerForPrediction.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True)
    model.eval()
    if model.config.context_length != 512 or model.config.prediction_length < plan['prediction_step']:
        raise ValueError('Checkpoint context or horizon mismatch')
    contexts = np.load(args.input, allow_pickle=False)
    if contexts.ndim != 2 or contexts.shape[1] != 512 or not np.isfinite(contexts).all():
        raise ValueError('Expected finite 512-step contexts')
    predictions, floored = [], 0
    for start in range(0, len(contexts), plan['batch_size']):
        x = contexts[start:start+plan['batch_size']].astype(np.float64)
        means, scales = x.mean(axis=1), x.std(axis=1)
        scales = np.maximum(scales, 1e-8)
        values = torch.tensor(((x-means[:,None])/scales[:,None])[...,None], dtype=torch.float32)
        kwargs = {}
        if model.config.resolution_prefix_tuning:
            kwargs['freq_token'] = torch.full((len(x),1), plan['frequency_token'], dtype=torch.long)
        with torch.inference_mode():
            output = model(past_values=values, **kwargs).prediction_outputs.numpy()
        prediction, count = restore_ttm_prediction(output, means, scales, plan['prediction_step'], plan['positive_floor'])
        predictions.extend(prediction.tolist()); floored += count
        if start % 4096 == 0:
            print(f'TTM {start}/{len(contexts)}', flush=True)
    np.save(args.output, np.asarray(predictions))
    write_json(str(args.output)+'.json', {'created_utc':utc_now(), 'model':plan, 'parameter_count':sum(p.numel() for p in model.parameters()),
        'contexts':len(contexts), 'floored_predictions':floored,
        'input_sha256':sha256(args.input), 'output_sha256':sha256(args.output),
        'model_files':{p.name:sha256(p) for p in Path(args.model_dir).iterdir() if p.is_file()},
        'packages':{name:importlib.metadata.version(name) for name in ['granite-tsfm','torch','transformers','numpy']}})

if __name__ == '__main__':
    main()
