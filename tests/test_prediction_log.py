import csv

from tennis_analytics.logging_utils.prediction_log import append_prediction


def test_append_prediction(tmp_path):
    path = tmp_path / "predictions.csv"
    prediction_id = append_prediction(path, {
        "tour": "atp", "player_1": "A", "player_2": "B", "surface": "Hard",
        "best_of": 3, "model_p1_prob": 0.6, "benchmark_p1_prob": 0.58,
    })
    assert prediction_id.startswith("pred-")
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["model_p2_prob"] == "0.400000"
