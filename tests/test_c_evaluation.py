import csv

import pytest

from scripts.evaluate_recommender import export, summarize


def test_evaluation_export_does_not_invent_ratings(c_database, tmp_path):
    assert export(tmp_path) == 50
    summary = summarize(tmp_path / "ratings.csv")
    assert summary["rated"] == 0
    assert summary["mean_precision_at_5"] is None
    assert (tmp_path / "reference_analyses.json").is_file()
    with pytest.raises(FileExistsError):
        export(tmp_path)


def test_evaluation_summary_requires_all_ratings(c_database, tmp_path):
    export(tmp_path)
    path = tmp_path / "ratings.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["relevant"] = "1" if row["rank"] != "5" else "0"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(path)
    assert summary["rated"] == 50
    assert summary["mean_precision_at_5"] == pytest.approx(0.8)
