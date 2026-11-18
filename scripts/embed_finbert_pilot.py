"""Frozen local FinBERT representations, gated by a hash-bound extraction review."""
import argparse
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volrisk.io import read_json, sha256, utc_now, write_json
from volrisk.text_embeddings import chunk_token_ids, weighted_document_mean


def encode(text, tokenizer, model, batch_size=4):
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    chunks = chunk_token_ids(ids)
    if sum(map(len, chunks)) != len(ids):
        raise ValueError("Content tokens lost or duplicated")
    means, probabilities, counts = [], [], []
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            width = max(map(len, batch)) + 2
            input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            content = torch.zeros_like(input_ids, dtype=torch.bool)
            for i, chunk in enumerate(batch):
                built = tokenizer.build_inputs_with_special_tokens(chunk)
                if len(built) != len(chunk) + 2 or built[1:-1] != chunk:
                    raise ValueError("Unexpected BERT special-token layout")
                input_ids[i, :len(built)] = torch.tensor(built)
                attention[i, :len(built)] = 1
                content[i, 1:len(chunk) + 1] = True
            output = model.base_model(input_ids=input_ids, attention_mask=attention,
                                      token_type_ids=torch.zeros_like(input_ids), return_dict=True)
            masked = output.last_hidden_state * content.unsqueeze(-1)
            mean = masked.sum(dim=1) / content.sum(dim=1, keepdim=True)
            probs = torch.softmax(model.classifier(output.pooler_output), dim=-1)
            means.extend(mean.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            counts.extend(map(len, batch))
    return weighted_document_mean(means, counts), weighted_document_mean(probabilities, counts), {
        "content_tokens": len(ids), "chunks": len(chunks), "content_tokens_encoded": sum(counts),
        "first_chunk_tokens": counts[0], "last_chunk_tokens": counts[-1], "overlap_tokens": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default="results/MDA_REVIEW_PILOT_V5.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Positive batch size required")
    review_path = ROOT / args.review
    review = read_json(review_path)
    if not review.get("review_complete"):
        raise ValueError("An explicit completed extraction review is required")
    approved = [r for r in review["documents"] if r["decision"] == "accept"]
    if not approved:
        raise ValueError("No approved MD&A text")
    config_path = ROOT / "config/finbert-v1.json"
    cfg = read_json(config_path)
    model_path = ROOT / f"data/models/finbert-{cfg['revision'][:7]}"
    artifacts = {p.name: sha256(p) for p in model_path.iterdir() if p.is_file()}
    dest = ROOT / "data/processed/finbert-pilot-v1"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"review_sha256": sha256(review_path), "config_sha256": sha256(config_path),
            "model_artifacts": artifacts, "model_revision": cfg["revision"],
            "source_sha256": sha256(Path(__file__)),
            "pooling_source_sha256": sha256(ROOT / "src/volrisk/text_embeddings.py"),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "device": "cpu", "threads": 4,
            "batch_size": args.batch_size, "frozen": True, "fitted_reduction": False}
    plan_path = dest / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Embedding plan changed; version this run")
    write_json(plan_path, plan)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    model.requires_grad_(False)
    model.eval()
    if model.config.hidden_size != 768 or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError("Unexpected pinned FinBERT architecture/tokenizer")
    ledger = []
    for row in approved[:args.limit] if args.limit else approved:
        text_path = ROOT / row["text_path"]
        if sha256(text_path) != row["text_sha256"]:
            raise ValueError("Reviewed text changed")
        if row["filing_date"] > "2025-12-31":
            raise ValueError("Disallowed disclosure date")
        path = dest / f"{row['accession']}.npz"
        meta_path = dest / f"{row['accession']}.json"
        if path.exists():
            record = read_json(meta_path)
            if sha256(path) != record["embedding_sha256"] or record["text_sha256"] != row["text_sha256"]:
                raise ValueError("Cached embedding changed")
        else:
            begun = time.monotonic()
            vector, sentiment, detail = encode(text_path.read_text(), tokenizer, model, args.batch_size)
            if vector.shape != (768,) or sentiment.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("Invalid representation shape/values")
            np.savez(path, embedding=vector, sentiment_probabilities=sentiment)
            record = {**row, **detail, "seconds": time.monotonic() - begun,
                      "embedding_path": str(path.relative_to(ROOT)), "embedding_sha256": sha256(path),
                      "hidden_size": len(vector), "sentiment_labels": model.config.id2label,
                      "embedding_l2_norm": float(np.linalg.norm(vector))}
            write_json(meta_path, record)
        ledger.append(record)
        write_json(ROOT / "results/FINBERT_PILOT_V1.json", {
            "updated_utc": utc_now(), "plan": plan, "documents": ledger,
            "complete": len(ledger) == len(approved), "embeddings_computed": len(ledger),
            "market_outcomes_2026_accessed": False, "forecast_models_fitted": False,
            "limits": "Frozen document representations only, not demonstrated forecast value. Full source/extraction/review hashes retained; no dimensional reduction fitted. Sentiment probabilities are separate from 768-dimensional hidden-state embeddings. Pretraining document-level overlap remains unverified.",
        })
        print(len(ledger), row["ticker"], row["filing_date"], record["content_tokens"], record["chunks"], round(record["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
