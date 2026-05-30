from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from lagunavision.eval.endpoint_eval import EndpointEvalConfig, _extract_answer, _extract_raw_answer, run_endpoint_eval
from lagunavision.eval.score_eval import score_answer
from lagunavision.types import EvalManifestItem


def test_score_answer_supports_strict_all_required_terms(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not-a-real-image")
    item = EvalManifestItem(
        id="strict",
        image=image,
        question="What error is shown?",
        ocr_text="",
        rubric="vqa",
        must_include=("forkpty", "Resource temporarily unavailable"),
        accepted_fix_terms=(),
        must_not_include=(),
        must_include_mode="all",
    )

    assert not score_answer(item, "forkpty failed").passed
    assert score_answer(item, "forkpty: Resource temporarily unavailable").passed


def test_endpoint_eval_scores_and_groups_categories(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not-a-real-image")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "id": "shape",
                "image": image.name,
                "question": "What colored shape is shown?",
                "answer": "red circle",
                "rubric": "vqa",
                "must_include": ["red circle"],
                "accepted_fix_terms": [],
                "must_not_include": [],
                "category": "expected_pass/coarse_color_shape",
                "expected_result": "expected_pass",
            },
            {
                "id": "ocr",
                "image": image.name,
                "question": "What exact terminal error is shown?",
                "answer": "forkpty: Resource temporarily unavailable",
                "rubric": "vqa",
                "must_include": ["forkpty", "Resource temporarily unavailable"],
                "must_include_mode": "all",
                "accepted_fix_terms": [],
                "must_not_include": [],
                "category": "known_failure/tiny_ocr",
                "expected_result": "known_failure",
            },
        ],
    )

    server = _EndpointServer()
    try:
        output = tmp_path / "answers.jsonl"
        summary = run_endpoint_eval(
            EndpointEvalConfig(
                endpoint=server.url,
                manifest=manifest,
                output=output,
                max_new_tokens=16,
                timeout=5,
            )
        )
    finally:
        server.close()

    assert summary["passed"] == 1
    assert summary["total"] == 2
    assert summary["by_category"]["expected_pass/coarse_color_shape"]["passed"] == 1
    assert summary["by_category"]["known_failure/tiny_ocr"]["passed"] == 0
    assert (tmp_path / "answers.summary.json").exists()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["answer"] == "red circle"
    assert rows[0]["raw_answer"] == "red circle"
    assert rows[0]["raw_response"] == {"answer": "red circle"}
    assert rows[1]["must_include_mode"] == "all"


def test_extract_answer_uses_final_content_after_reasoning_trace() -> None:
    assert _extract_answer({"content": "<think>green? maybe</think>\nFinal answer: red"}) == "red"
    assert _extract_answer({"answer": "Reasoning...\nAnswer: blue square"}) == "blue square"
    assert _extract_answer({"message": {"content": "<think>list</think>\ncircle"}}) == "circle"
    assert _extract_answer({"answer": "circle\n</assistant>"}) == "circle"
    assert _extract_answer({"answer": "Answer: No\n\n<no text>\n<no text>"}) == "No"
    assert _extract_answer({"answer": "left, right, both, or unclear?\n\nAnswer: left\n\nunanswerable: no"}) == "left"
    assert _extract_answer({"answer": "green circle\n<no answer>\n<no answer>"}) == "green circle"
    assert _extract_answer({"answer": "pink\n<no answer>\n<no"}) == "pink"
    assert _extract_answer({"answer": "green circle\n<no answer>\n<"}) == "green circle"
    payload = {"choices": [{"message": {"reasoning_content": "maybe green", "content": "Final answer: blue"}}]}
    assert _extract_answer(payload) == "blue"
    assert _extract_raw_answer(payload) == "Final answer: blue"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class _EndpointServer:
    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        question = payload["inputs"]["question"]
        answer = "red circle" if "colored shape" in question else "terminal error"
        body = json.dumps({"answer": answer}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return
