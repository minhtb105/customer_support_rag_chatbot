"""Shared helpers for tracker CRUD tests (PoC, conservative pass).

Provides tiny parametrize/classify helpers reused by the smallest
tracker test files. Additive only — existing tests keep their
explicit asserts; new helper-based tests duplicate a few cases
as proof-of-concept. Remaining files migrate in follow-up.
"""
import re

import pytest


def make_crud_case(args, expected):
    """Wrap (args, expected) as a pytest.param for classify tests."""
    if not isinstance(args, tuple):
        args = (args,)
    return pytest.param(args, expected)


def assert_classify_first(fn, args, expected):
    """Call classify fn with args tuple; assert first element == expected.

    Tracker classify fns return (cls, msg) tuples; handles scalar
    returns too.
    """
    if not isinstance(args, tuple):
        args = (args,)
    res = fn(*args)
    cls = res[0] if isinstance(res, (tuple, list)) else res
    assert cls == expected


def norm_ws(s: str) -> str:
    """Collapse whitespace for lossless-split comparisons."""
    return re.sub(r"\s+", " ", s or "").strip()


def assert_idempotent(r1: dict, r2: dict, *keys: str) -> None:
    """Assert seed reruns are idempotent on the given keys."""
    for k in keys:
        assert r1[k] == r2[k], f"{k}: {r1[k]!r} != {r2[k]!r}"


def assert_parse_ok(parse_fn, text: str, value, context: str) -> None:
    """Assert a voice-parse result is unambiguous with value/context."""
    r = parse_fn(text)
    assert r["ambiguous"] is False, f"{text!r} -> ambiguous ({r['reason']})"
    assert r["value"] == value, f"{text!r} -> {r['value']} != {value}"
    assert r["context"] == context, f"{text!r} -> {r['context']} != {context}"
    assert str(value) in r["confirm_text"]


def assert_parse_ambiguous(parse_fn, text: str) -> None:
    """Assert a voice-parse result is ambiguous (needs re-speak)."""
    r = parse_fn(text)
    assert r["ambiguous"] is True, f"{text!r} should be ambiguous, got {r}"
