from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer

from util.app_settings import EmbeddingSettings
from util.loader import load_config

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

tokenizer = None
model = None

@lru_cache(maxsize=1)
def settings() -> EmbeddingSettings:
    return load_config(EmbeddingSettings, section="embedding")

def load_embedding_model():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(
        settings().model_name,
        trust_remote_code=True,
        cache_dir=settings().cache_dir,
    )
    print("tokenizer initialized")
    quant_config = None
    if getattr(settings(), "use_4bit", False):
        if not _HAS_BNB:
            raise RuntimeError(
                "use_4bit=true but bitsandbytes isn't available. "
                "Install bitsandbytes + compatible CUDA build or set use_4bit=false."
            )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("quant_config initialized")

    model = AutoModel.from_pretrained(
        settings().model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        cache_dir=settings().cache_dir
    )
    print("model initialized")
    model.eval()