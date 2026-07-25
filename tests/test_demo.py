from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tennis_analytics.cli import commands
from tennis_analytics.cli import main as cli_main
from tennis_analytics.data.demo import generate_demo_data


def test_generate_demo_data_creates_deterministic_dataset(
    tmp_path: Path,
) -> None:
    """Demo generation should create reproducible model-ready raw data."""

    first_output = tmp_path / "first" / "atp_raw.csv"
    second_output = tmp_path / "second" / "atp_raw.csv"

    first_path = generate_demo_data(
        "atp",
        first_output,
        random_seed=123,
        start_year=2021,
        number_of_years=3,
        matches_per_year=20,
    )
    second_path = generate_demo_data(
        "ATP",
        second_output,
        random_seed=123,
        start_year=2021,
        number_of_years=3,
        matches_per_year=20,
    )

    assert first_path == first_output
    assert second_path == second_output
    assert first_output.exists()
    assert second_output.exists()

    first_frame = pd.read_csv(first_output)
    second_frame = pd.read_csv(second_output)

    pd.testing.assert_frame_equal(
        first_frame,
        second_frame,
    )

    assert len(first_frame) == 60

    assert set(first_frame.columns) == {
        "Date",
        "Comment",
        "Winner",
        "Loser",
        "Surface",
        "Best of",
        "WRank",
        "LRank",
        "Year",
        "AvgW",
        "AvgL",
    }

    assert set(first_frame["Comment"]) == {"Completed"}
    assert set(first_frame["Best of"]) == {3}
    assert set(first_frame["Surface"]) == {
        "Hard",
        "Clay",
        "Grass",
    }

    assert (
        first_frame["Winner"] != first_frame["Loser"]
    ).all()

    assert first_frame["AvgW"].gt(1.0).all()
    assert first_frame["AvgL"].gt(1.0).all()


@pytest.mark.parametrize(
    ("number_of_years", "matches_per_year", "message"),
    (
        (
            2,
            20,
            "number_of_years must be at least 3",
        ),
        (
            3,
            19,
            "matches_per_year must be at least 20",
        ),
    ),
)
def test_generate_demo_data_rejects_invalid_sizes(
    tmp_path: Path,
    number_of_years: int,
    matches_per_year: int,
    message: str,
) -> None:
    """Demo generation should reject datasets too small for evaluation."""

    with pytest.raises(
        ValueError,
        match=message,
    ):
        generate_demo_data(
            "atp",
            tmp_path / "atp_raw.csv",
            number_of_years=number_of_years,
            matches_per_year=matches_per_year,
        )


def test_run_demo_executes_complete_offline_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The demo command should generate data and run all pipeline stages."""

    raw_file = tmp_path / "data" / "raw" / "atp_raw.csv"
    features_file = (
        tmp_path / "data" / "processed" / "atp_features.csv"
    )
    evaluation_file = (
        tmp_path / "reports" / "atp_walk_forward.csv"
    )
    model_file = tmp_path / "models" / "atp_model.joblib"

    paths = {
        "raw_directory": raw_file.parent,
        "raw_file": raw_file,
        "features_file": features_file,
        "evaluation_report": evaluation_file,
        "model_file": model_file,
    }

    settings = SimpleNamespace(random_seed=99)

    monkeypatch.setattr(
        commands,
        "_context",
        lambda tour: (
            "atp",
            tmp_path,
            settings,
            paths,
        ),
    )

    generated_calls: list[tuple[str, Path, int]] = []

    def fake_generate_demo_data(
        tour: str,
        output_path: Path,
        *,
        random_seed: int,
    ) -> Path:
        generated_calls.append(
            (
                tour,
                output_path,
                random_seed,
            )
        )
        return output_path

    monkeypatch.setattr(
        commands,
        "generate_demo_data",
        fake_generate_demo_data,
    )
    monkeypatch.setattr(
        commands,
        "run_build",
        lambda tour: features_file,
    )
    monkeypatch.setattr(
        commands,
        "run_evaluate",
        lambda tour: evaluation_file,
    )
    monkeypatch.setattr(
        commands,
        "run_train",
        lambda tour: model_file,
    )

    outputs = commands.run_demo("ATP")

    assert generated_calls == [
        (
            "atp",
            raw_file,
            99,
        )
    ]

    assert outputs == {
        "raw_data": raw_file,
        "features": features_file,
        "evaluation": evaluation_file,
        "model": model_file,
    }


def test_cli_demo_command_prints_generated_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should expose and execute the offline demo command."""

    outputs = {
        "raw_data": tmp_path / "data" / "raw" / "atp_raw.csv",
        "features": (
            tmp_path
            / "data"
            / "processed"
            / "atp_features.csv"
        ),
        "evaluation": (
            tmp_path
            / "reports"
            / "atp_walk_forward.csv"
        ),
        "model": (
            tmp_path
            / "models"
            / "atp_model.joblib"
        ),
    }

    monkeypatch.setattr(
        cli_main,
        "_configure_application_logging",
        lambda: None,
    )
    monkeypatch.setattr(
        cli_main,
        "run_demo",
        lambda tour: outputs,
    )

    exit_code = cli_main.main(
        [
            "demo",
            "atp",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Offline demo completed successfully." in captured.out

    for name, path in outputs.items():
        assert f"{name}: {path}" in captured.out


def test_cli_returns_failure_when_demo_raises_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should convert command failures into a nonzero exit code."""

    monkeypatch.setattr(
        cli_main,
        "_configure_application_logging",
        lambda: None,
    )

    def fail_demo(tour: str) -> dict[str, Path]:
        raise RuntimeError("demo failure")

    monkeypatch.setattr(
        cli_main,
        "run_demo",
        fail_demo,
    )

    exit_code = cli_main.main(
        [
            "demo",
            "atp",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: demo failure" in captured.out