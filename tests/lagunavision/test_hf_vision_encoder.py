import sys
from pathlib import Path
from types import SimpleNamespace

from lagunavision.encoders import hf_vision
from lagunavision.encoders.hf_vision import HfVisionEncoder


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> None:
        self.requires_grad = value


class _Model:
    def __init__(self) -> None:
        self.parameter = _Parameter()
        self.device = None
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def to(self, device: str) -> None:
        self.device = device

    def parameters(self):
        return [self.parameter]


class _TransformersLogging:
    def __init__(self) -> None:
        self.verbosity = 20
        self.history = []

    def get_verbosity(self):
        self.history.append(("get", self.verbosity))
        return self.verbosity

    def set_verbosity_error(self):
        self.history.append(("error", None))
        self.verbosity = 40

    def set_verbosity(self, value):
        self.history.append(("set", value))
        self.verbosity = value


def test_siglip_encoder_loads_vision_model_only(monkeypatch, tmp_path: Path) -> None:
    calls = []
    model = _Model()
    model_dir = tmp_path / "siglip-so400m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"vision_config": {"hidden_size": 1152, "num_hidden_layers": 1}}',
        encoding="utf-8",
    )
    transformer_logging = _TransformersLogging()

    class AutoImageProcessor:
        @staticmethod
        def from_pretrained(model_id):
            calls.append(("processor", model_id))
            return object()

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id):
            calls.append(("auto_model", model_id))
            raise AssertionError("SigLIP should load SiglipVisionModel, not full AutoModel")

    class SiglipVisionConfig:
        def __init__(self, **kwargs) -> None:
            calls.append(("siglip_vision_config", kwargs))

    class SiglipVisionModel:
        @staticmethod
        def from_pretrained(model_id, config=None):
            calls.append(("siglip_vision", model_id, type(config).__name__))
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=AutoImageProcessor,
            AutoModel=AutoModel,
            SiglipVisionConfig=SiglipVisionConfig,
            SiglipVisionModel=SiglipVisionModel,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.utils",
        SimpleNamespace(logging=transformer_logging),
    )
    monkeypatch.setattr(hf_vision, "resolve_torch_device", lambda device: "cpu")

    encoder = HfVisionEncoder(model_id=str(model_dir))
    encoder._load_sync()

    assert encoder._model is model
    assert model.eval_called
    assert model.device == "cpu"
    assert model.parameter.requires_grad is False
    assert calls == [
        ("processor", str(model_dir)),
        ("siglip_vision_config", {"hidden_size": 1152, "num_hidden_layers": 1}),
        ("siglip_vision", str(model_dir), "SiglipVisionConfig"),
    ]
    assert transformer_logging.history == [("get", 20), ("error", None), ("set", 20)]


def test_non_siglip_encoder_uses_auto_model(monkeypatch) -> None:
    calls = []
    model = _Model()

    class AutoImageProcessor:
        @staticmethod
        def from_pretrained(model_id):
            calls.append(("processor", model_id))
            return object()

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id):
            calls.append(("auto_model", model_id))
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=AutoImageProcessor,
            AutoModel=AutoModel,
        ),
    )
    monkeypatch.setattr(hf_vision, "resolve_torch_device", lambda device: "cpu")

    encoder = HfVisionEncoder(model_id="openai/clip-vit-base-patch32")
    encoder._load_sync()

    assert encoder._model is model
    assert calls == [
        ("processor", "openai/clip-vit-base-patch32"),
        ("auto_model", "openai/clip-vit-base-patch32"),
    ]
