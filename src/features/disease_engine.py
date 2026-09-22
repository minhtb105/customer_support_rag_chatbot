"""Generic Synthea-method-inspired disease progression engine (JSON mini state machine).

Subset v1 (D1): states Initial/Terminal/Guard/SetAttribute/ConditionOnset/
MedicationOrder/Observation/Symptom; transitions direct/distributed/conditional.
Unknown state/transition kinds FAIL loudly (ValueError). Seeded RNG everywhere.

A new disease = a new JSON module, no code change.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_STATE_KINDS = frozenset({
    "Initial", "Terminal", "Guard", "SetAttribute",
    "ConditionOnset", "MedicationOrder", "Observation", "Symptom",
})
SUPPORTED_TRANSITION_KINDS = frozenset({"direct", "distributed", "conditional"})
SUPPORTED_OPS = frozenset({"==", "!=", ">", ">=", "<", "<=", "in", "not_in"})


def load_module(path: str | Path) -> Dict[str, Any]:
    """Load + validate a disease JSON module (raises on unknown kinds)."""
    mod = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_module(mod)
    return mod


def validate_module(mod: Dict[str, Any]) -> None:
    """Fail loudly on unknown state/transition kinds, bad weights, bare probabilities."""
    states = mod.get("states") or {}
    if not mod.get("initial") or mod["initial"] not in states:
        raise ValueError("module needs 'initial' pointing at a defined state")
    for name, spec in states.items():
        kind = (spec or {}).get("kind")
        if kind not in SUPPORTED_STATE_KINDS:
            raise ValueError(f"unknown state kind {kind!r} at state {name!r}")
    seen_from: Dict[str, int] = {}
    for tr in mod.get("transitions") or []:
        kind = tr.get("kind")
        if kind not in SUPPORTED_TRANSITION_KINDS:
            raise ValueError(f"unknown transition kind {kind!r} from {tr.get('from')!r}")
        if tr.get("from") not in states:
            raise ValueError(f"transition from unknown state {tr.get('from')!r}")
        seen_from[tr["from"]] = seen_from.get(tr["from"], 0) + 1
        if kind == "direct" and tr.get("to") not in states:
            raise ValueError(f"direct transition to unknown state {tr.get('to')!r}")
        if kind == "distributed":
            targets = tr.get("targets") or []
            if not targets:
                raise ValueError(f"distributed transition from {tr['from']!r} needs targets")
            total = 0.0
            for t in targets:
                if t.get("to") not in states:
                    raise ValueError(f"distributed target unknown state {t.get('to')!r}")
                remarks = t.get("remarks") or {}
                if not ("source" in remarks or "assumption" in remarks):
                    raise ValueError(f"bare probability at {tr['from']!r}->{t.get('to')!r}: needs remarks.source|assumption")
                total += float(t.get("p", 0))
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"distributed weights from {tr['from']!r} sum to {total}, want ~1.0")
        if kind == "conditional":
            for br in tr.get("branches") or []:
                if br.get("to") not in states:
                    raise ValueError(f"conditional branch to unknown state {br.get('to')!r}")
                _check_condition(br.get("when") or {})
            if (tr.get("default") is not None) and tr["default"] not in states:
                raise ValueError(f"conditional default unknown state {tr.get('default')!r}")
    dupes = [k for k, v in seen_from.items() if v > 1]
    if dupes:
        raise ValueError(f"duplicate transition entries from states {dupes} (one entry per 'from')")


def _check_condition(cond: Dict[str, Any]) -> None:
    op = cond.get("op", "==")
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unknown condition op {op!r}")


def holds(cond: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    """Evaluate a Guard/branch condition against the patient ctx."""
    _check_condition(cond)
    attr, op, want = cond.get("attribute"), cond.get("op", "=="), cond.get("value")
    got = ctx.get(attr)
    if op == "==":
        return got == want
    if op == "!=":
        return got != want
    if op == "in":
        return got in (want or [])
    if op == "not_in":
        return got not in (want or [])
    try:
        if op == ">":
            return got > want
        if op == ">=":
            return got >= want
        if op == "<":
            return got < want
        if op == "<=":
            return got <= want
    except TypeError:
        return False
    return False  # pragma: no cover


class DiseaseEngine:
    """Weekly state-machine runner with seeded RNG."""

    def __init__(self, module: Dict[str, Any], seed: int = 0):
        validate_module(module)
        self.module = module
        self.rng = random.Random(seed)
        self._by_from = {tr["from"]: tr for tr in module.get("transitions") or []}

    def step(self, state: str, ctx: Dict[str, Any]) -> str:
        """One transition hop from `state` (Terminal stays put)."""
        kind = self.module["states"][state].get("kind")
        if kind == "Terminal":
            return state
        tr = self._by_from.get(state)
        if tr is None:
            return state  # no outgoing edge: hold
        tkind = tr["kind"]
        if tkind == "direct":
            return tr["to"]
        if tkind == "distributed":
            r = self.rng.random()
            acc = 0.0
            for t in tr["targets"]:
                acc += float(t["p"])
                if r < acc:
                    return t["to"]
            return tr["targets"][-1]["to"]
        # conditional
        for br in tr.get("branches") or []:
            if holds(br.get("when") or {}, ctx):
                return br["to"]
        return tr.get("default", state)

    def apply_state(self, state: str, ctx: Dict[str, Any], week: int) -> Dict[str, Any]:
        """Apply state effects to ctx; return the weekly record."""
        spec = self.module["states"][state]
        kind = spec.get("kind")
        rec: Dict[str, Any] = {"week": week, "state": state, "kind": kind}
        if kind == "SetAttribute":
            attr = spec["attribute"]
            if "value" in spec:
                ctx[attr] = spec["value"]
            elif "delta" in spec:
                ctx[attr] = ctx.get(attr, 0) + spec["delta"]
            elif "sample" in spec:
                lo, hi = spec["sample"]["min"], spec["sample"]["max"]
                ctx[attr] = self.rng.uniform(lo, hi)
            rec["attribute"] = attr
            rec["drift_delta"] = ctx.get("drift_delta", 0.0)
        elif kind == "ConditionOnset":
            ctx.setdefault("conditions", [])
            if spec["condition"] not in ctx["conditions"]:
                ctx["conditions"].append(spec["condition"])
            rec["condition"] = spec["condition"]
        elif kind == "MedicationOrder":
            ctx["medication"] = spec["medication"]
            rec["medication"] = spec["medication"]
        elif kind == "Observation":
            rec["observation"] = spec.get("code", state)
        elif kind == "Symptom":
            ctx.setdefault("symptoms", [])
            ctx["symptoms"].append(spec.get("symptom", state))
            rec["symptom"] = spec.get("symptom", state)
        elif kind == "Guard":
            rec["guard_holds"] = holds(spec.get("condition") or {}, ctx)
        return rec

    def run(self, ctx: Optional[Dict[str, Any]] = None, weeks: int = 12) -> Dict[str, Any]:
        """Weekly run from `initial`; halts early on Terminal."""
        live: Dict[str, Any] = dict(ctx or {})
        state = self.module["initial"]
        trajectory: List[Dict[str, Any]] = []
        for w in range(weeks):
            state = self.step(state, live)
            trajectory.append(self.apply_state(state, live, w))
            if self.module["states"][state].get("kind") == "Terminal":
                break
        return {"trajectory": trajectory, "ctx": live}
