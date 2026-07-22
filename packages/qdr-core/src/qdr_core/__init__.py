"""QDR shared kernel.

Scope is deliberately narrow -- this package is imported by every plane, so anything
added here becomes a dependency everywhere. It holds only what more than one plane must
agree on exactly: platform invariants, identifier types, the error taxonomy, and content
addressing. Domain logic belongs in the plane that owns it.
"""

__version__ = "0.1.0"
