# qdr-core

The shared Python kernel. Imported by the control plane, the compute plane and the
compute engine.

## Scope, and how to keep it

This package is a dependency of *everything*. Anything added here becomes a dependency
everywhere, and the blast radius of a change is the whole platform. So the bar is narrow
and deliberately awkward:

> Something belongs in `qdr-core` only if **more than one plane must agree on it exactly**.

Domain logic belongs to the plane that owns it. A helper used by one service is not shared
infrastructure. This package has **zero runtime dependencies** and that is a constraint to
defend, not an accident.

## What is here

| Module | Why it is shared |
|---|---|
| `limits.py` | The ≤40-qubit cap (ADR-0006) and chemical accuracy. Three enforcement points across three planes must use the same number. |
| `ids.py` | Typed identifiers (`RunId`, `JobId`, …) as UUIDv7 — time-ordered, so they stay well-behaved as database primary keys. |
| `errors.py` | The error taxonomy. Codes are a public contract: the API maps them to HTTP responses and the frontend switches on them. |
| `provenance/content_hash.py` | **The** content-addressing implementation (ADR-0007). |
| `provenance/artifact_ref.py` | The canonical artifact reference type. |

Still empty, awaiting the control-plane pass: `tenant.py`, `capabilities.py`, `events/`,
`messaging/`.

## Content addressing

There is exactly one hash implementation in this platform, and it is
`provenance/content_hash.py`. Two implementations produce two different addresses for the
same bytes, which defeats deduplication and makes every provenance claim unfalsifiable.
The control plane's `core/storage/content_address.py` and the engine's
`reproducibility/input_hashing.py` are thin adapters over this module.

Canonicalisation choices worth knowing:

- Keys sorted, no insignificant whitespace, UTF-8 emitted literally.
- NaN and ±Infinity are **rejected** — JSON cannot represent them and encoders disagree on
  the spelling, so hashing them would make the address depend on the encoder version.
- `1` and `1.0` hash **differently**. They are distinct JSON tokens, and collapsing them
  would let an int/float type change slip past a provenance comparison.
- Non-JSON types raise `TypeError`. Callers with domain types convert explicitly, so the
  conversion is visible in review rather than hidden in a fallback encoder.

`tests/test_content_hash.py` pins exact digests, including one cross-checked against the
canonical SHA-256 of the empty string. **If a frozen vector starts failing, the hash
function has changed and every artifact address already in storage is invalid — that is a
migration, not a refactor.**

## Tests

```bash
uv run pytest packages/qdr-core -q
```

No PySCF, no numpy, no network. Runs anywhere.
