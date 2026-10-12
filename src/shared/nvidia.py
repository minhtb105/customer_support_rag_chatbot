"""
NVIDIA NIM provider (OpenAI-compatible) — chat + embeddings với xoay tua keys.

- Keys đọc từ .env theo đúng tên đã có: NVDIA_API_KEY, NVDIA_API_KEY2,
  NVDIA_API_KEY3 (chấp nhận cả alias chính tả NVIDIA_API_KEY*).
- Xoay tua: gặp 401 (key chết) hoặc 429 (rate limit) thì thử key tiếp theo.
  Lỗi khác (400/404 sai model...) fail-fast, không xoay (xoay cũng không cứu được).
- Embeddings dùng HTTP trực tiếp tới /v1/embeddings để gửi được `input_type`:
  model QA (llama-3.2-nv-embedqa-1b-v2) BẮT BUỘC đúng mode — `passage` khi index,
  `query` khi retrieve — sai mode rớt accuracy mạnh (theo docs NVIDIA).
- `NvidiaEmbeddings` bọc cho Chroma/langchain: embed_documents=passage,
  embed_query=query. Tương thích `EmbeddingAdapter.embed_documents`.
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

_KEY_NAMES = (
    "NVDIA_API_KEY", "NVDIA_API_KEY2", "NVDIA_API_KEY3",  # tên đang dùng trong .env
    "NVIDIA_API_KEY", "NVIDIA_API_KEY2", "NVIDIA_API_KEY3",  # alias chính tả đúng
)


def get_nvidia_keys() -> list[str]:
    """Trả về keys non-empty theo thứ tự ưu tiên, dedupe."""
    keys: list[str] = []
    for name in _KEY_NAMES:
        v = (os.getenv(name) or "").strip().strip("\"'")
        if v and v not in keys:
            keys.append(v)
    return keys


def _is_transient_error(exc: Exception | None) -> bool:
    """Single retryable-error predicate (5xx/401/429 + overload/capacity text).

    Covers both key-rotation and HARD->LIGHT fallback cases.
    """
    if exc is None:
        return False
    status = getattr(exc, "status_code", None)
    if status in (401, 429, 500, 502, 503, 529):
        return True
    msg = str(exc).lower()
    return ("rate limit" in msg or "ratelimit" in msg or "quota" in msg
            or "too many requests" in msg or "unauthorized" in msg
            or "invalid api key" in msg or "overload" in msg
            or "capacity" in msg or "exhausted" in msg
            or "temporarily" in msg or "bad gateway" in msg
            or "gateway timeout" in msg or "internal server error" in msg
            or "server error" in msg)


def _is_rotatable_error(exc: Exception) -> bool:
    return _is_transient_error(exc)


def _raise_if_no_keys() -> list[str]:
    keys = get_nvidia_keys()
    if not keys:
        raise RuntimeError(
            "Không tìm thấy NVIDIA key nào (NVDIA_API_KEY..3) trong .env")
    return keys


def embed_texts(texts: list[str], model: str,
                mode: str = "passage") -> list[list[float]]:
    """Embed 1 batch qua NVIDIA /v1/embeddings, xoay key khi 401/429."""
    import httpx

    keys = _raise_if_no_keys()
    last_err: Exception | None = None
    for i, key in enumerate(keys):
        try:
            with httpx.Client(timeout=120.0) as c:
                r = c.post(
                    f"{NVIDIA_BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "input": texts,
                          "input_type": mode,
                          "encoding_format": "float",
                          "truncate": "END"},
                )
                if r.status_code in (401, 429):
                    class _E(Exception):
                        pass
                    e = _E(f"NVIDIA embeddings {r.status_code}: "
                           f"{r.text[:200]}")
                    e.status_code = r.status_code  # type: ignore
                    raise e
                r.raise_for_status()
                return [d["embedding"] for d in r.json()["data"]]
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_rotatable_error(e) and i < len(keys) - 1:
                import time as _t
                _t.sleep(3)  # transient 5xx/429: nhường backend rồi thử key kế
                continue
            raise
    raise last_err  # type: ignore[misc]


class NvidiaEmbeddings:
    """Embedding function cho Chroma — passage khi index, query khi search."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        # Batch nhỏ: 1 request với hàng nghìn input bị API 400 (GINA 1875
        # chunks). Giữ thứ tự, xoay key từng batch như cũ.
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(embed_texts(texts[i:i + batch_size], self.model, mode="passage"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text], self.model, mode="query")[0]


class _CompletionsNS:
    def __init__(self, parent: "RotatingNvidiaClient"):
        self._parent = parent

    def _once(self, key: str, model: str, kwargs: dict):
        from openai import OpenAI

        client = OpenAI(api_key=key,
                        base_url=self._parent.base_url,
                        timeout=self._parent.timeout)
        kw = dict(kwargs)
        kw["model"] = model
        return client.chat.completions.create(**kw)

    def create(self, **kwargs):
        model = kwargs.get("model", "")
        keys = _raise_if_no_keys()
        last_err: Exception | None = None
        for i, key in enumerate(keys):
            try:
                return self._once(key, model, kwargs)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_rotatable_error(e) and i < len(keys) - 1:
                    continue
                raise
        # Hết keys vẫn lỗi quá tải + đang gọi model HARD -> fallback LIGHT 1 vòng
        if _is_overload(last_err) and model == hard_model():
            print(f"[nvidia] {model} overloaded, fallback {light_model()}")
            for key in keys:
                try:
                    return self._once(key, light_model(), kwargs)
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    continue
        raise last_err  # type: ignore[misc]


def _is_overload(exc: Exception | None) -> bool:
    if exc is None:
        return False
    status = getattr(exc, "status_code", None)
    if status in (503, 529):
        return True
    # Subset of the single transient predicate (overload-family text only).
    if not _is_transient_error(exc):
        return False
    msg = str(exc).lower()
    return "overload" in msg or "capacity" in msg or "exhausted" in msg


def hard_model() -> str:
    return (os.getenv("NVIDIA_CHAT_MODEL_HARD")
            or "nvidia/nemotron-3-ultra-550b-a55b")


def light_model() -> str:
    return (os.getenv("NVIDIA_CHAT_MODEL_LIGHT")
            or "meta/llama-3.2-11b-vision-instruct")


_HARD_SIGNALS = (
    "hbA1c", "hba1c", "mg/dl", "mmol/l", "mmol",
    "chẩn đoán", "chan doan", "phác đồ", "phac do",
    "tương tác", "tuong tac", "liều", "lieu", "insulin",
    "suy thận", "suy tim", "biến chứng", "bien chung",
    "tăng cường", "tang cuong", "phối hợp thuốc",
)


def looks_hard(question: str) -> bool:
    """Heuristic rẻ: câu có số liệu lab/thuốc/chỉ định thì đi model HARD."""
    q = (question or "").lower()
    if any(s.lower() in q for s in _HARD_SIGNALS):
        return True
    if any(ch.isdigit() for ch in q) and len(q) > 120:
        return True
    return False


def select_chat_model(question: str, override: str | None = None) -> str:
    """Chọn model theo độ khó. override (nếu có) luôn thắng."""
    if override:
        return override
    return hard_model() if looks_hard(question) else light_model()


class _ChatNS:
    def __init__(self, parent: "RotatingNvidiaClient"):
        self.completions = _CompletionsNS(parent)


class RotatingNvidiaClient:
    """Drop-in thay OpenAI client cho chat — cùng interface
    `client.chat.completions.create(...)`, tự xoay key khi 401/429."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = base_url or NVIDIA_BASE_URL
        self.timeout = timeout
        self.chat = _ChatNS(self)
