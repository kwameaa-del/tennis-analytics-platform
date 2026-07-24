import joblib
import pandas as pd

from tennis_analytics.models.train import train_tennis_model


def test_train_model_persists_artifact(tmp_path):
    rows = []
    for i in range(40):
        rows.append({
            "Date": f"2024-01-{(i % 28) + 1:02d}",
            "D_Elo": i - 20,
            "D_SurfElo": i / 2 - 10,
            "D_Form": (i % 10) / 10 - 0.5,
            "D_LogRank": (i - 20) / 20,
            "BestOf": 3,
            "P1_Won": i % 2,
        })
    features = tmp_path / "features.csv"
    model_path = tmp_path / "model.joblib"
    pd.DataFrame(rows).to_csv(features, index=False)
    artifact = train_tennis_model(features, model_path)
    assert artifact.trained_rows == 40
    assert model_path.exists()
    loaded = joblib.load(model_path)
    assert loaded.features[0] == "D_Elo"
