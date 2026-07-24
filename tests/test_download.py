import pandas as pd

from tennis_analytics.data.download import download_tour_data


def test_download_uses_existing_cache(tmp_path):
    pd.DataFrame([{"Winner": "A", "Loser": "B", "Year": 2024}]).to_csv(
        tmp_path / "atp_raw_2024.csv", index=False
    )
    path = download_tour_data("atp", [2024], tmp_path, retries=1, retry_base_seconds=0)
    assert path.exists()
    result = pd.read_csv(path)
    assert len(result) == 1
