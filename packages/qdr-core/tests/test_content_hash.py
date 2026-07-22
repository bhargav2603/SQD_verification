"""Content addressing underpins every provenance and deduplication claim the platform
makes, so these tests pin the *exact* bytes -- not just internal self-consistency.

If a change makes a frozen vector below fail, the hash function has changed and every
artifact address already written to storage has been invalidated. That is a migration,
not a refactor."""

import math
from pathlib import Path

import pytest

from qdr_core.provenance.content_hash import (
    canonical_json,
    content_hash,
    hash_bytes,
    hash_file,
    is_content_hash,
)

# Frozen vectors. Regenerating these to make a test pass defeats their purpose.
FROZEN_VECTOR = {"beta": [1, 2, 3], "alpha": "x", "gamma": {"nested": True, "value": 1.5}}
FROZEN_DIGEST = "sha256:21e03ef8700e230496bb80fb453547a32895a192860de1f401ea725b1c1a4129"
EMPTY_OBJECT_DIGEST = "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
# The canonical SHA-256 of zero bytes, independently verifiable against any sha256 tool.
EMPTY_BYTES_DIGEST = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_frozen_vector_digest_is_stable() -> None:
    assert content_hash(FROZEN_VECTOR) == FROZEN_DIGEST


def test_empty_object_digest_is_stable() -> None:
    assert content_hash({}) == EMPTY_OBJECT_DIGEST


def test_empty_bytes_matches_canonical_sha256() -> None:
    """Cross-checks our framing against the outside world, not just against ourselves."""
    assert hash_bytes(b"") == EMPTY_BYTES_DIGEST


def test_key_order_does_not_affect_digest() -> None:
    a = {"one": 1, "two": 2, "three": 3}
    b = {"three": 3, "two": 2, "one": 1}
    assert content_hash(a) == content_hash(b)


def test_nested_key_order_does_not_affect_digest() -> None:
    a = {"outer": {"z": 1, "a": 2}}
    b = {"outer": {"a": 2, "z": 1}}
    assert content_hash(a) == content_hash(b)


def test_list_order_does_affect_digest() -> None:
    """Arrays are ordered; reordering one is a different value, not a formatting choice."""
    assert content_hash([1, 2, 3]) != content_hash([3, 2, 1])


def test_int_and_float_hash_differently() -> None:
    """1 and 1.0 are distinct JSON tokens. Collapsing them would let a type change
    slip past a provenance comparison unnoticed."""
    assert content_hash({"n": 1}) != content_hash({"n": 1.0})


def test_canonical_json_has_no_insignificant_whitespace() -> None:
    assert canonical_json({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_canonical_json_emits_utf8_literally() -> None:
    """Non-ASCII stays literal rather than \\uXXXX-escaped, so the digest is over the
    same bytes any other UTF-8 consumer would see."""
    assert canonical_json({"k": "café"}) == '{"k":"café"}'.encode()


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(bad_value: float) -> None:
    """JSON cannot represent these and encoders disagree on the spelling, so hashing
    them would produce an address that depends on the encoder version."""
    with pytest.raises(ValueError):
        content_hash({"n": bad_value})


def test_non_json_types_are_rejected() -> None:
    """Domain types must be converted explicitly by the caller, so the conversion is
    visible in review rather than hidden in a fallback encoder."""
    with pytest.raises(TypeError):
        content_hash({"when": object()})


def test_hash_file_matches_hash_bytes(tmp_path: Path) -> None:
    payload = b"molecular orbital integrals" * 1000
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    assert hash_file(path) == hash_bytes(payload)


def test_hash_file_streams_larger_than_one_chunk(tmp_path: Path) -> None:
    """Exercises the chunked read loop rather than a single-read fast path."""
    payload = b"x" * (3 * 1024 * 1024 + 17)
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    assert hash_file(path) == hash_bytes(payload)


@pytest.mark.parametrize(
    "value",
    [
        FROZEN_DIGEST,
        EMPTY_BYTES_DIGEST,
    ],
)
def test_is_content_hash_accepts_well_formed(value: str) -> None:
    assert is_content_hash(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "deadbeef",
        "md5:21e03ef8700e230496bb80fb453547a32895a192860de1f401ea725b1c1a4129",
        "sha256:tooshort",
        "sha256:" + "g" * 64,  # non-hex
        "sha256:" + "A" * 64,  # uppercase is not the form we emit
    ],
)
def test_is_content_hash_rejects_malformed(value: str) -> None:
    assert not is_content_hash(value)
