"""Aggregate the manual audit label files in results/audit/ into results/audit_summary.json."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results" / "audit"


def read(name: str) -> list[dict]:
    with (AUDIT / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


summary: dict = {"coherence": {}, "metadata": {}, "grounding": {}}

by_strategy = defaultdict(list)
for r in read("coherence_labels.csv"):
    by_strategy[r["strategy"]].append(int(r["score_0_2"]))
for strategy, scores in by_strategy.items():
    summary["coherence"][strategy] = {"n_positions": len(scores), "sum": sum(scores),
                                      "mean_of_2": round(sum(scores) / len(scores), 2),
                                      "counts": dict(sorted(Counter(scores).items()))}

trace_filter = {}
for strategy in "ABC":
    for line in (ROOT / "results" / "traces" / f"{strategy}.jsonl").open(encoding="utf-8"):
        t = json.loads(line)
        if t["split"] == "test" and t["question_filter"]:
            trace_filter[(strategy, t["id"])] = t["top3_respects_filter"]

meta = defaultdict(list)
for r in read("metadata_labels.csv"):
    meta[r["strategy"]].append(r)
for strategy, rows in meta.items():
    filtered = {k: v for k, v in trace_filter.items() if k[0] == strategy}
    summary["metadata"][strategy] = {
        "records_checked": len(rows),
        "complete_and_correct": sum(r["metadata_complete_and_correct"] == "yes" for r in rows),
        "subsection_ok": sum(r["subsection_ok"] == "yes" for r in rows),
        "filter_compliance_test": f"{sum(filtered.values())}/{len(filtered)}" if filtered else "N/A",
    }

grounding = defaultdict(list)
for r in read("grounding_labels.csv"):
    grounding[r["strategy"]].append(int(r["grounding_0_2"]))
for strategy, scores in grounding.items():
    counts = Counter(scores)
    summary["grounding"][strategy] = {"n_questions": len(scores),
                                      "count_0": counts[0], "count_1": counts[1], "count_2": counts[2],
                                      "mean_of_2": round(sum(scores) / len(scores), 2)}

out = ROOT / "results" / "audit_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
