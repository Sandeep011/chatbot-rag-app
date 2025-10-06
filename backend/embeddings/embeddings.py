import os
from  typing import List, Dict, Any

from loguru import logger
from sentence_transformers import SentenceTransformer

_MODEL = None
_MODEL_DIM = None

def _to_float_list_vector(x) -> List[float]:
    """Coerce any 1-D vector-like (np array/list/tuple) to List[float]."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    raise TypeError(f"Expected 1-D vector-like, got {type(x)}")

def _to_float_list_matrix(x) -> List[List[float]]:
    """Coerce any 2-D matrix-like to List[List[float]]."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        # Could be list of lists OR already 1-D if N==1
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            return [[float(v) for v in row] for row in x]
        elif len(x) > 0 and isinstance(x[0], (float, int)):
            # Already a single vector (1-D) but caller expected 2-D
            return [[float(v) for v in x]]
    raise TypeError(f"Expected 2-D matrix-like, got {type(x)}")

def get_model():
    global _MODEL, _MODEL_DIM
    if not _MODEL:
        model_name = os.getenv("EMBED_MODEL","intfloat/e5-small-v2")
        logger.info(f"Loading embedidng model: {model_name}")
        _MODEL = SentenceTransformer(model_name)
        _MODEL_DIM = _MODEL.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. dimensions are {_MODEL_DIM}")
    return _MODEL


def embed_passage(passages: List[str]) -> List[List[float]]:
    model = get_model()
    out = model.encode([f"passage: {t}" for t in passages], normalize_embeddings=True)
    return _to_float_list_matrix(out)


def embed_query(text: str) -> List[float]:
    model = get_model()
    out = model.encode([f"query: {text}"], normalize_embeddings=True)
    # If model returns shape (1, dim), take row 0; if it returns (dim,), just coerce
    try:
        # Try treating as 2-D then collapse
        mat = _to_float_list_matrix(out)
        return mat[0]
    except TypeError:
        # Fallback: treat as 1-D
        return _to_float_list_vector(out)


def embedding_dim():
    get_model()
    return _MODEL_DIM
