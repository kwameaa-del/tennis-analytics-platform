import pandas as pd

from tennis_analytics.features.build import build_features


def test_feature_builder_uses_pre_match_state_and_balanced_framing(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = []
    for i in range(24):
        rows.append({
            "Date": f"2024-01-{(i % 28) + 1:02d}",
            "Comment": "Completed",
            "Winner": "A" if i % 2 == 0 else "B",
            "Loser": "B" if i % 2 == 0 else "A",
            "Surface": "Hard",
            "Best of": 3,
            "WRank": 10,
            "LRank": 20,
            "Year": 2024,
            "PSW": 1.5,
            "PSL": 2.5,
            "AvgW": 1.6,
            "AvgL": 2.4,
        })
    pd.DataFrame(rows).to_csv(raw, index=False)
    output = tmp_path / "features.csv"
    result = build_features(raw, output, form_window=2, random_seed=42)
    assert output.exists()
    assert {"D_Elo", "D_SurfElo", "D_Form", "D_LogRank", "BenchmarkProb_P1"} <= set(result.columns)
    assert 0.25 < result["P1_Won"].mean() < 0.75
