from typing import List, Optional
import numpy as np


def get_openai_embeddings(model: Optional[str] = None):
    """Factory cho OpenAI embeddings — dùng key từ .env."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise ImportError("langchain-openai chưa cài: pip install langchain-openai") from e
    import os
    kwargs = {}
    if model:
        kwargs["model"] = model
    else:
        # config.OPENAI_EMBEDDING_MODEL sẽ được resolve ở nơi gọi
        pass
    # OpenAIEmbeddings tự đọc OPENAI_API_KEY từ env
    return OpenAIEmbeddings(**kwargs)


class EmbeddingAdapter:
    """
    Adapter responsibilities:
    - Truncate text according to the embedding model's limit
    - Batch embedding
    - NOT related to chunking
    """

    def __init__(self, embedder, tokenizer, max_len=512):
        """
        embedder: HuggingFaceEmbeddings | SentenceTransformer | OpenAIEmbeddings
        tokenizer: Corresponding HF tokenizer
        """
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _truncate(self, text: str) -> str:
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None
        )
        return self.tokenizer.decode(
            tokens["input_ids"],
            skip_special_tokens=True
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts safely
        """
        safe_texts = [self._truncate(t) for t in texts]

        return np.array(self.embedder.embed_documents(safe_texts))

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed single text (thin wrapper)
        """
        return self.embed_texts([text])[0]
