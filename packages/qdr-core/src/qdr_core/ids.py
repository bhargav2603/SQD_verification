"""Typed identifiers.

``NewType`` gives static separation at zero runtime cost: a ``RunId`` cannot be passed
where a ``JobId`` is expected, but at runtime both are ordinary ``str``.

IDs are UUIDv7 (RFC 9562): a 48-bit millisecond timestamp followed by random bits, so
lexicographic order matches creation order. That property matters because these become
primary keys -- time-ordered keys keep B-tree inserts local instead of scattering writes
across the index the way UUIDv4 does.
"""

import os
import time
from typing import NewType
from uuid import UUID

OrgId = NewType("OrgId", str)
ProjectId = NewType("ProjectId", str)
RunId = NewType("RunId", str)
JobId = NewType("JobId", str)
ArtifactId = NewType("ArtifactId", str)


def uuid7() -> UUID:
    """Generate a UUIDv7.

    Python's stdlib has no ``uuid.uuid7()`` as of 3.12, so we lay out the bits directly:
    48 bits of Unix time in milliseconds, 4 version bits, 12 random, 2 variant bits,
    62 random.
    """
    unix_ts_ms = int(time.time() * 1000) & 0xFFFF_FFFF_FFFF
    rand = int.from_bytes(os.urandom(10), "big")

    rand_a = (rand >> 62) & 0x0FFF  # 12 bits
    rand_b = rand & 0x3FFF_FFFF_FFFF_FFFF  # 62 bits

    value = unix_ts_ms << 80
    value |= 0x7 << 76  # version 7
    value |= rand_a << 64
    value |= 0b10 << 62  # RFC 9562 variant
    value |= rand_b
    return UUID(int=value)


def _new_id() -> str:
    return str(uuid7())


def new_org_id() -> OrgId:
    return OrgId(_new_id())


def new_project_id() -> ProjectId:
    return ProjectId(_new_id())


def new_run_id() -> RunId:
    return RunId(_new_id())


def new_job_id() -> JobId:
    return JobId(_new_id())


def new_artifact_id() -> ArtifactId:
    return ArtifactId(_new_id())
