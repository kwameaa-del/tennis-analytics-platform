import pandas as pd

from tennis_analytics.evaluation.walk_forward import evaluate_walk_forward


def test_walk_forward_returns_one_row_per_test_year(tmp_path):
    rows = []
    for year in (2021, 2022, 2023, 2024):
        for i in range(30):
            rows.append({
                "Date": f"{year}-01-{(i % 28) + 1:02d}", "Year": year,
                "D_Elo": i - 15, "D_SurfElo": (i - 15) / 2, "D_Form": (i % 10) / 10 - 0.5,
                "D_LogRank": (i - 15) / 20, "BestOf": 3,
                "BenchmarkProb_P1": 0.55 if i % 2 == 0 else 0.45,
                "P1_Won": 1 if i % 2 == 0 else 0,
            })
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = evaluate_walk_forward(path)
    assert list(result["test_year"]) == [2023, 2024]
    assert (result["n"] == 30).all()
