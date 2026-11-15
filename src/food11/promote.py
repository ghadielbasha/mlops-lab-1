"""Quality-gated MLflow model alias promotion for MLOps Lab 5."""

import argparse
import json
import math
import os
from pathlib import Path

from mlflow import MlflowClient

MODEL_NAME = "food11"
METRIC_NAME = "test_accuracy"
DEFAULT_EVALUATION = Path("data/food11_processed_mini_clean/evaluation")


def quality_gate(candidate_accuracy: float, champion_accuracy: float, n: int):
    """Return (promote, minimum accuracy, 2 standard errors)."""
    if n <= 0:
        raise ValueError("Test set size must be positive")
    if not (0 <= champion_accuracy <= 1 and 0 <= candidate_accuracy <= 1):
        raise ValueError("Accuracies must fall between zero and one")
    two_se = 2 * math.sqrt(champion_accuracy * (1 - champion_accuracy) / n)
    required_accuracy = champion_accuracy + two_se
    return candidate_accuracy >= required_accuracy, required_accuracy, two_se


def evaluate_version(version: int, n: int, tracking_uri: str):
    """Read MLflow metrics, apply gate, update aliases and tag candidate."""
    client = MlflowClient(tracking_uri=tracking_uri)
    candidate = client.get_model_version(MODEL_NAME, str(version))
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if int(candidate.version) == int(champion.version):
        raise ValueError("Candidate is already champion; nothing to compare")
    if not candidate.run_id or not champion.run_id:
        raise ValueError("Both model versions must reference an MLflow training run")

    candidate_metrics = client.get_run(candidate.run_id).data.metrics
    champion_metrics = client.get_run(champion.run_id).data.metrics
    if METRIC_NAME not in candidate_metrics or METRIC_NAME not in champion_metrics:
        raise ValueError("Missing test_accuracy in candidate or champion MLflow run")
    acc_candidate = float(candidate_metrics[METRIC_NAME])
    acc_champion = float(champion_metrics[METRIC_NAME])
    promote, required, two_se = quality_gate(acc_candidate, acc_champion, n)
    decision = "champion" if promote else "challenger"

    # Tag the candidate for future audit, even when its accuracy is insufficient.
    tags = {
        "gate_decision": decision,
        "gate_candidate_accuracy": f"{acc_candidate:.8f}",
        "gate_champion_accuracy": f"{acc_champion:.8f}",
        "gate_required_accuracy": f"{required:.8f}",
        "gate_two_standard_errors": f"{two_se:.8f}",
        "gate_test_set_size": str(n),
        "gate_previous_champion_version": str(champion.version),
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, str(version), key, value)
    client.set_registered_model_alias(MODEL_NAME, decision, str(version))

    result = {
        "candidate_version": int(candidate.version),
        "candidate_run_id": candidate.run_id,
        "champion_before": int(champion.version),
        "champion_accuracy": acc_champion,
        "candidate_accuracy": acc_candidate,
        "test_images": n,
        "two_standard_errors": two_se,
        "required_accuracy": required,
        "decision": decision,
        "champion_after": int(client.get_model_version_by_alias(MODEL_NAME, "champion").version),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="MLflow model promotion quality gate")
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--result-file", type=Path, default=None)
    args = parser.parse_args()
    if args.test_size is None:
        if not args.evaluation.is_dir():
            parser.error(f"Evaluation dataset not found: {args.evaluation}")
        n = sum(
            1 for path in args.evaluation.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        n = args.test_size

    result = evaluate_version(
        args.version, n, os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    print(json.dumps(result, indent=2))
    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
