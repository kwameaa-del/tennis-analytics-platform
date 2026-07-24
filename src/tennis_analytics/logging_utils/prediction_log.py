from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

FIELDNAMES = [
    "prediction_id", "timestamp_utc", "tour", "player_1", "player_2",
    "surface", "best_of", "model_p1_prob", "model_p2_prob",
    "benchmark_p1_prob", "benchmark_p2_prob", "status", "winner",
]


def append_prediction(path: Path, row: dict[str, object]) -> str:
    """Append an immutable pre-match forecast record and return its ID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_id = str(row.get("prediction_id") or f"pred-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}")
    output = {
        "prediction_id": prediction_id,
        "timestamp_utc": row.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tour": row["tour"], "player_1": row["player_1"], "player_2": row["player_2"],
        "surface": row["surface"], "best_of": row["best_of"],
        "model_p1_prob": f"{float(row['model_p1_prob']):.6f}",
        "model_p2_prob": f"{1.0 - float(row['model_p1_prob']):.6f}",
        "benchmark_p1_prob": f"{float(row['benchmark_p1_prob']):.6f}",
        "benchmark_p2_prob": f"{1.0 - float(row['benchmark_p1_prob']):.6f}",
        "status": row.get("status", "pending"), "winner": row.get("winner", ""),
    }
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(output)
    return prediction_id
