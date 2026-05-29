import json

from lagunavision.eval.visual_overfit import QUESTION, write_visual_overfit_dataset


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_visual_overfit_dataset_writes_image_controls(tmp_path):
    write_visual_overfit_dataset(tmp_path, train_count=32, eval_count=16)

    train = _read_jsonl(tmp_path / "train.jsonl")
    eval_rows = _read_jsonl(tmp_path / "eval.jsonl")
    wrong = _read_jsonl(tmp_path / "wrong.jsonl")
    blank = _read_jsonl(tmp_path / "blank.jsonl")

    assert len(train) == 32
    assert len(eval_rows) == 16
    assert len(wrong) == 16
    assert len(blank) == 16
    assert all(row["question"] == QUESTION for row in eval_rows)
    assert all(row["ocr_text"] == "" for row in eval_rows)
    assert (tmp_path / "images" / "blank.png").exists()

    for index, row in enumerate(eval_rows):
        assert row["answer"] in row["must_include"]
        assert row["image"] != wrong[index]["image"]
        assert row["image"] != blank[index]["image"]
        assert row["question"] == wrong[index]["question"] == blank[index]["question"]
        assert row["answer"] == wrong[index]["answer"] == blank[index]["answer"]
