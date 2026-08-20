"""Cache-aware streaming encoder. Built and verified, deliberately not shipped
on the critical path -- see ENGINEERING.md §6."""

from .cache import EncoderCache, LayerCache
from .encoder import StreamingEncoder

__all__ = ["EncoderCache", "LayerCache", "StreamingEncoder"]
