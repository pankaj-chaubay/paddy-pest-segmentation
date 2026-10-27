"""Exception types raised by paddyseg.

Every error the package raises on purpose derives from :class:`PaddySegError`, so callers
can catch the whole family with one ``except`` clause and still tell the cases apart.
"""

from __future__ import annotations


class PaddySegError(Exception):
    """Base class for all errors raised deliberately by paddyseg."""


class AnnotationError(PaddySegError):
    """The annotation file is missing, malformed or internally inconsistent."""


class DatasetError(PaddySegError):
    """An image cannot be read or does not match its annotation record."""
