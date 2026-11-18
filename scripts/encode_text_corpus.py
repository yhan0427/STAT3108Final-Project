"""Frozen, resumable document encoding while corpus collection proceeds; never fits PCA/models."""
import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embed_finbert_pilot import encode
from volrisk.io import read_json, sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    corpus_dir = ROOT / "data/processed/text-corpus-100-v1"
    corpus_plan = corpus_dir / "plan.json"
    cfg_path = ROOT / "config/finbert-v1.json"
    cfg = read_json(cfg_path)
    model_path = ROOT / f"data/models/finbert-{cfg['revision'][:7]}"
    dest = ROOT / "data/processed/finbert-corpus-100-v1"
    dest.mkdir(parents=True, exist_ok=True)
    plan = {"registered_utc": utc_now(), "corpus_plan_sha256": sha256(corpus_plan),
            "encoder_config_sha256": sha256(cfg_path), "revision": cfg["revision"],
            "model_artifacts": {p.name: sha256(p) for p in model_path.iterdir() if p.is_file()},
            "encoder_backend_sha256": sha256(ROOT / "scripts/embed_finbert_pilot.py"),
            "pooling_sha256": sha256(ROOT / "src/volrisk/text_embeddings.py"),
            "runner_sha256": sha256(Path(__file__)), "torch": torch.__version__,
            "transformers": transformers.__version__, "threads": 4, "batch_size": 4,
            "device": "cpu", "frozen": True,
            "scope": "Stateless representations of frozen-parser candidates; flag-dependent review gates supervised fitting, not a claim of individually validated bulk text. No PCA or forecast fit, no 2026 market data."}
    plan_path = dest / "plan.json"
    if plan_path.exists():
        old = read_json(plan_path)
        if {k: v for k, v in old.items() if k != "registered_utc"} != {k: v for k, v in plan.items() if k != "registered_utc"}:
            raise ValueError("Representation contract changed")
        plan = old
    else:
        write_json(plan_path, plan)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, use_safetensors=True)
    model.requires_grad_(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    pilot = read_json(ROOT / "results/FINBERT_PILOT_V1.json")
    reusable = {r["text_sha256"]: r for r in pilot["documents"]}
    if pilot["plan"]["model_revision"] != cfg["revision"]:
        reusable = {}
    last_print = 0
    while True:
        if sha256(corpus_plan) != plan["corpus_plan_sha256"]:
            raise ValueError("Corpus source rules changed while encoding")
        corpus = read_json(corpus_dir / "manifest.json")
        status = read_json(ROOT / "results/TEXT_CORPUS_100_AUDIT_V1.json")
        review_path = ROOT / "results/TEXT_CORPUS_REVIEW_100_V1.json"
        reviewed = {r["accession"]: r["decision"] for r in read_json(review_path)["documents"]} if review_path.exists() else {}
        ledger, skipped = [], []
        candidates = [r for r in corpus["documents"] if r["status"] == "candidate_needs_review"]
        for row in candidates:
            if row["filing_date"] > "2025-12-31":
                raise ValueError("Disallowed filing date")
            if reviewed.get(row["accession"]) == "exclude" or (row.get("quality_flags") and reviewed.get(row["accession"]) != "accept"):
                skipped.append(row["accession"])
                continue
            path = dest / f"{row['accession']}.npz"
            metadata_path = dest / f"{row['accession']}.json"
            if path.exists():
                record = read_json(metadata_path)
                if sha256(path) != record["embedding_sha256"] or record["text_sha256"] != row["text_sha256"]:
                    raise ValueError("Cached representation/source changed")
            else:
                text_path = ROOT / row["text_path"]
                if sha256(text_path) != row["text_sha256"]:
                    raise ValueError("Source text changed")
                begun = time.monotonic()
                previous = reusable.get(row["text_sha256"])
                if previous:
                    old_path = ROOT / previous["embedding_path"]
                    if sha256(old_path) != previous["embedding_sha256"]:
                        raise ValueError("Reusable pilot vector changed")
                    shutil.copyfile(old_path, path)
                    detail = {k: previous[k] for k in ["content_tokens", "chunks", "content_tokens_encoded", "first_chunk_tokens", "last_chunk_tokens", "overlap_tokens"]}
                else:
                    vector, probabilities, detail = encode(text_path.read_text(), tokenizer, model, 4)
                    if vector.shape != (768,) or not np.isfinite(vector).all():
                        raise ValueError("Invalid hidden-state vector")
                    np.savez(path, embedding=vector, sentiment_probabilities=probabilities)
                record = {k: row[k] for k in ["ticker", "cik", "issuer_id", "role", "sector", "accession", "filing_date", "report_period", "document_url", "document_sha256", "text_path", "text_sha256", "quality_flags", "quality_review_status"]}
                record.update(detail, embedding_path=str(path.relative_to(ROOT)), embedding_sha256=sha256(path),
                              encoder_revision=cfg["revision"], hidden_size=768,
                              reused_pilot_vector=previous is not None, seconds=time.monotonic() - begun)
                write_json(metadata_path, record)
            ledger.append(record)
            if len(ledger) % 10 == 0 and len(ledger) > last_print:
                print("ENCODED", len(ledger), "of currently available", len(candidates), flush=True)
                last_print = len(ledger)
        write_json(dest / "manifest.json", {"plan": plan, "documents": ledger, "skipped_for_review": skipped})
        write_json(ROOT / "results/FINBERT_CORPUS_100_V1.json", {
            "updated_utc": utc_now(), "plan": plan, "embeddings_computed": len(ledger),
            "documents_currently_available": len(candidates), "skipped_for_review": len(skipped),
            "source_collection_complete": status["complete"],
            "all_candidates_encoded": status["complete"] and not skipped and len(ledger) == len(candidates),
            "local_manifest_sha256": sha256(dest / "manifest.json"),
            "content_tokens_encoded": sum(r["content_tokens_encoded"] for r in ledger),
            "pca_fitted": False, "forecast_models_fitted": False, "market_outcomes_2026_accessed": False,
            "quality_scope": "Unflagged frozen-parser candidates can be stateless-encoded before bulk quality review; this does not approve them for supervised use or certify individual boundaries. Flagged candidates await explicit decisions.",
        })
        if not args.follow or status["complete"]:
            print("ENCODING QUEUE EXHAUSTED", len(ledger), "vectors;", len(skipped), "await/excluded by review", flush=True)
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
