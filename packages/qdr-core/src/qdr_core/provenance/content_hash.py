"""The single canonical content-addressing implementation (ADR-0007).

Nothing else in this repository may implement content hashing. Two independent hash
implementations silently produce two different addresses for the same bytes, which
defeats deduplication and makes provenance claims unfalsifiable. The control plane's
``core/storage/content_address.py`` and the engine's ``reproducibility/input_hashing.py``
are thin adapters over this module.

Canonicalisation rules for structured data, chosen so the same logical value hashes
identically on every platform and Python build:

  * object keys sorted; no insignificant whitespace
  * UTF-8, with non-ASCII emitted literally rather than ``\\uXXXX``-escaped
  * NaN and +/-Infinity rejected -- JSON cannot represent them and every encoder
    disagrees about the spelling
  * ``1`` and ``1.0`` hash *differently*: they are distinct JSON tokens, and silently
    collapsing them would let an int/float type change slip past a provenance check
"""

import hashlib
import json
from os import PathLike
from typing import Any

#: Algorithm prefix on every digest this module emits.
HASH_ALGORITHM = "sha256"

#: Read size for streaming file digests (1 MiB).
_CHUNK_SIZE = 1024 * 1024


def canonical_json(obj: Any) -> bytes:
    """Serialise ``obj`` to its canonical UTF-8 JSON encoding.

    Raises ``TypeError`` for anything not JSON-native. That strictness is deliberate:
    callers with domain types (NumPy arrays, dataclasses, enums) must convert
    explicitly, so the conversion is visible and reviewable rather than implicit in a
    fallback encoder that might change behaviour between library versions.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_bytes(data: bytes) -> str:
    """Return ``sha256:<hex>`` for raw bytes."""
    return f"{HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def content_hash(obj: Any) -> str:
    """Return ``sha256:<hex>`` for a JSON-native Python value."""
    return hash_bytes(canonical_json(obj))


def hash_file(path: str | PathLike[str]) -> str:
    """Return ``sha256:<hex>`` for a file's contents, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return f"{HASH_ALGORITHM}:{digest.hexdigest()}"


def is_content_hash(value: str) -> bool:
    """True when ``value`` is a well-formed digest produced by this module."""
    prefix, separator, hexdigest = value.partition(":")
    if separator != ":" or prefix != HASH_ALGORITHM or len(hexdigest) != 64:
        return False
    return all(c in "0123456789abcdef" for c in hexdigest)
