"""Canonical artifact reference type (ADR-0007)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from qdr_core.errors import ValidationError
from qdr_core.provenance.content_hash import is_content_hash


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """An immutable pointer to a stored artifact, addressed by content.

    ``uri`` says where a copy currently lives; ``content_hash`` says what it *is*. The
    hash is the identity -- a copy served from a different bucket with the same digest is
    the same artifact, and a digest mismatch on read is corruption, not a cache miss.
    """

    uri: str
    content_hash: str
    media_type: str
    size_bytes: int
    created_at: datetime

    def __post_init__(self) -> None:
        if not is_content_hash(self.content_hash):
            raise ValidationError(
                f"Malformed content hash: {self.content_hash!r}",
                content_hash=self.content_hash,
            )
        if self.size_bytes < 0:
            raise ValidationError(f"size_bytes must be non-negative, got {self.size_bytes}")
        if self.created_at.tzinfo is None:
            raise ValidationError("created_at must be timezone-aware")

    @classmethod
    def create(
        cls,
        uri: str,
        content_hash: str,
        media_type: str,
        size_bytes: int,
        created_at: datetime | None = None,
    ) -> "ArtifactRef":
        """Build a reference, defaulting ``created_at`` to now in UTC."""
        return cls(
            uri=uri,
            content_hash=content_hash,
            media_type=media_type,
            size_bytes=size_bytes,
            created_at=created_at or datetime.now(UTC),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-native form, matching contracts/json-schema/artifact-provenance.schema.json."""
        return {
            "uri": self.uri,
            "content_hash": self.content_hash,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
        }
