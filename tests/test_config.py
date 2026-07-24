import json

import pytest

from tennis_analytics.config import load_settings
from tennis_analytics.exceptions import ConfigurationError


def test_load_settings_validates_values(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "years": [2021, 2022],
        "base_elo": 1500,
        "standard_k": 32,
        "provisional_k": 48,
        "provisional_match_limit": 30,
        "recent_form_window": 10,
        "random_seed": 42,
        "logistic_regression_c": 0.1,
    }))
    with pytest.raises(ConfigurationError, match="At least three years"):
        load_settings(path)
