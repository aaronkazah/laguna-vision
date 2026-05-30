"""Startup compatibility patches for Hugging Face's endpoint wrapper."""

try:
    import transformers
    import transformers.file_utils as file_utils
    import transformers.utils as transformers_utils

    try:
        from transformers.utils.import_utils import is_torch_available
    except Exception:
        from transformers.utils import is_torch_available

    if not hasattr(file_utils, "is_tf_available"):
        file_utils.is_tf_available = lambda: False
    if not hasattr(file_utils, "is_torch_available"):
        file_utils.is_torch_available = is_torch_available
    for name, value in {
        "CONFIG_NAME": "config.json",
        "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
        "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
        "SAFE_WEIGHTS_NAME": "model.safetensors",
        "TF2_WEIGHTS_NAME": "tf_model.h5",
        "TF_WEIGHTS_NAME": "model.ckpt",
        "WEIGHTS_INDEX_NAME": "pytorch_model.bin.index.json",
        "WEIGHTS_NAME": "pytorch_model.bin",
    }.items():
        for module in (file_utils, transformers_utils):
            if getattr(module, name, None) is None:
                setattr(module, name, value)
except Exception:
    pass

try:
    import transformers

    class _EndpointUnusedMT5Tokenizer:
        pass

    fallback_tokenizer = _EndpointUnusedMT5Tokenizer
    fallback_tokenizer_fast = _EndpointUnusedMT5Tokenizer
    try:
        fallback_tokenizer = getattr(transformers, "T5Tokenizer")
    except Exception:
        pass
    try:
        fallback_tokenizer_fast = getattr(transformers, "T5TokenizerFast")
    except Exception:
        fallback_tokenizer_fast = fallback_tokenizer
    transformers.MT5Tokenizer = fallback_tokenizer
    transformers.MT5TokenizerFast = fallback_tokenizer_fast
except Exception:
    pass
