from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SEMANTIC_TYPE_MAP = {
    "global": "global",
    "obj": "object",
    "cat": "category",
    "attr": "attribute",
    "rel": "relation",
}
EXPECTED_SEMANTIC_TYPES = ("global", "object", "category", "attribute", "relation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze GQA semantic-category accuracy from a model result jsonl.")
    parser.add_argument(
        "--questions-json",
        type=str,
        required=True,
        help="Path to GQA testdev_balanced_questions.json.",
    )
    parser.add_argument(
        "--results-jsonl",
        type=str,
        required=True,
        help="Path to model result jsonl.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        required=True,
        help="Path to write per-sample semantic analysis jsonl.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        required=True,
        help="Path to write semantic summary json.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="QRouter",
        help="Optional model name to store in summary/output rows.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=12578,
        help="Expected number of test questions/results for validation.",
    )
    return parser.parse_args()


def read_questions(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Expected a non-empty JSON object mapping question_id to payload.")
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    question_id = str(row.get("question_id", "")).strip()
    image_id = str(row.get("image_id", "")).strip()
    prediction = row.get("prediction", row.get("pred", ""))
    ground_truth = row.get("ground_truth", row.get("gt", ""))
    is_correct_raw = row.get("is_correct", False)
    if isinstance(is_correct_raw, bool):
        is_correct = is_correct_raw
    elif isinstance(is_correct_raw, (int, float)):
        is_correct = bool(is_correct_raw)
    else:
        is_correct = str(is_correct_raw).strip().lower() in {"1", "true", "yes"}
    return {
        "image_id": image_id,
        "question_id": question_id,
        "prediction": str(prediction).strip(),
        "ground_truth": str(ground_truth).strip(),
        "is_correct": is_correct,
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_semantic_type(payload: dict[str, Any]) -> str:
    raw_semantic = str(payload.get("types", {}).get("semantic", "")).strip().lower()
    return SEMANTIC_TYPE_MAP.get(raw_semantic, raw_semantic or "__missing__")


def main() -> None:
    args = parse_args()
    questions = read_questions(args.questions_json)
    results = read_jsonl(args.results_jsonl)

    question_type_map: dict[str, str] = {}
    question_type_counts = Counter()
    unknown_question_types = Counter()

    for question_id, payload in questions.items():
        semantic_type = normalize_semantic_type(payload)
        question_type_map[str(question_id)] = semantic_type
        if semantic_type in EXPECTED_SEMANTIC_TYPES:
            question_type_counts[semantic_type] += 1
        else:
            unknown_question_types[semantic_type] += 1

    semantic_total = sum(question_type_counts.values())
    num_questions = len(questions)
    question_total_matches_expected = num_questions == args.expected_total
    semantic_total_matches_expected = semantic_total == args.expected_total

    per_type_correct = Counter()
    per_type_total = Counter()
    missing_question_ids: list[str] = []
    enriched_rows: list[dict[str, Any]] = []

    for raw_row in results:
        row = normalize_result_row(raw_row)
        question_id = row["question_id"]
        semantic_type = question_type_map.get(question_id)
        if semantic_type is None:
            semantic_type = "__missing__"
            missing_question_ids.append(question_id)

        enriched = {
            "model_name": args.model_name,
            "image_id": row["image_id"],
            "question_id": question_id,
            "type": semantic_type,
            "prediction": row["prediction"],
            "ground_truth": row["ground_truth"],
            "is_correct": row["is_correct"],
        }
        enriched_rows.append(enriched)

        per_type_total[semantic_type] += 1
        per_type_correct[semantic_type] += int(row["is_correct"])

    result_total_matches_expected = len(results) == args.expected_total
    missing_question_ids = sorted(set(missing_question_ids))

    per_type_summary: dict[str, dict[str, Any]] = {}
    for semantic_type in EXPECTED_SEMANTIC_TYPES:
        num_samples = per_type_total[semantic_type]
        num_correct = per_type_correct[semantic_type]
        accuracy = (num_correct / num_samples) if num_samples > 0 else None
        per_type_summary[semantic_type] = {
            "num_questions": question_type_counts[semantic_type],
            "num_samples": num_samples,
            "num_correct": num_correct,
            "accuracy": accuracy,
        }

    extra_type_summary = {}
    for semantic_type in sorted(set(per_type_total.keys()) - set(EXPECTED_SEMANTIC_TYPES)):
        num_samples = per_type_total[semantic_type]
        num_correct = per_type_correct[semantic_type]
        extra_type_summary[semantic_type] = {
            "num_samples": num_samples,
            "num_correct": num_correct,
            "accuracy": (num_correct / num_samples) if num_samples > 0 else None,
        }

    summary = {
        "model_name": args.model_name,
        "questions_json": str(Path(args.questions_json)),
        "results_jsonl": str(Path(args.results_jsonl)),
        "num_questions": num_questions,
        "num_results": len(results),
        "expected_total": args.expected_total,
        "question_total_matches_12578": question_total_matches_expected,
        "semantic_total_matches_12578": semantic_total_matches_expected,
        "result_total_matches_12578": result_total_matches_expected,
        "question_type_counts": {key: question_type_counts[key] for key in EXPECTED_SEMANTIC_TYPES},
        "unknown_question_type_counts": dict(unknown_question_types),
        "per_type_accuracy": per_type_summary,
        "extra_type_accuracy": extra_type_summary,
        "missing_question_id_count": len(missing_question_ids),
        "missing_question_ids": missing_question_ids[:50],
    }

    write_jsonl(args.out_jsonl, enriched_rows)
    write_json(args.summary_json, summary)

    print(f"model_name: {args.model_name}")
    print(f"questions_json: {args.questions_json}")
    print(f"results_jsonl: {args.results_jsonl}")
    print(f"num_questions: {num_questions}")
    print(f"num_results: {len(results)}")
    print("question_type_counts:")
    for semantic_type in EXPECTED_SEMANTIC_TYPES:
        print(f"  {semantic_type}: {question_type_counts[semantic_type]}")
    print(f"question_total_matches_12578: {question_total_matches_expected}")
    print(f"semantic_total_matches_12578: {semantic_total_matches_expected}")
    print(f"result_total_matches_12578: {result_total_matches_expected}")
    if unknown_question_types:
        print(f"unknown_question_type_counts: {dict(unknown_question_types)}")
    if missing_question_ids:
        print(f"missing_question_id_count: {len(missing_question_ids)}")
    print("per_type_accuracy:")
    for semantic_type in EXPECTED_SEMANTIC_TYPES:
        payload = per_type_summary[semantic_type]
        print(
            f"  {semantic_type}: "
            f"num_samples={payload['num_samples']} "
            f"num_correct={payload['num_correct']} "
            f"accuracy={payload['accuracy']}"
        )
    print(f"out_jsonl: {args.out_jsonl}")
    print(f"summary_json: {args.summary_json}")


if __name__ == "__main__":
    main()
