from lagunavision.eval.web_probe import DEFAULT_WEB_PROBES


def test_default_web_probes_are_description_only() -> None:
    assert len(DEFAULT_WEB_PROBES) >= 5
    assert all(case.url.startswith("https://") for case in DEFAULT_WEB_PROBES)
    assert all(case.must_include for case in DEFAULT_WEB_PROBES)
    assert "example.com" not in {case.url.replace("https://", "").rstrip("/") for case in DEFAULT_WEB_PROBES}
