from transformers import LlamaConfig, LlamaForCausalLM

from lagunavision.backbones.base import LoraSettings
from lagunavision.backbones.hf_causal import HfCausalBackbone

_SETTINGS = LoraSettings(rank=4, alpha=8, dropout=0.0, targets=("q_proj", "v_proj"))


def _tiny_backbone() -> HfCausalBackbone:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    backbone = HfCausalBackbone("tiny-random-llama")
    backbone._model = LlamaForCausalLM(config)
    backbone._tokenizer = object()
    return backbone


def test_enable_lora_trains_only_the_adapter() -> None:
    backbone = _tiny_backbone()
    backbone.freeze()

    adapter_params = backbone.enable_lora(_SETTINGS)

    assert adapter_params
    assert all(p.requires_grad for p in adapter_params)
    base = [p for name, p in backbone.torch_module.named_parameters() if "lora_" not in name]
    assert all(not p.requires_grad for p in base)


def test_adapter_disabled_is_a_live_context_only_with_lora() -> None:
    backbone = _tiny_backbone()
    backbone.freeze()
    # no adapter yet: null context
    with backbone.adapter_disabled():
        pass

    backbone.enable_lora(_SETTINGS)
    with backbone.adapter_disabled():
        pass


def test_save_and_load_adapter_roundtrip(tmp_path) -> None:
    backbone = _tiny_backbone()
    backbone.freeze()
    backbone.enable_lora(_SETTINGS)

    backbone.save_adapter(tmp_path / "lora")
    assert (tmp_path / "lora" / "adapter_config.json").exists()

    reloaded = _tiny_backbone()
    reloaded.load_adapter(tmp_path / "lora")
    assert reloaded._lora_enabled
