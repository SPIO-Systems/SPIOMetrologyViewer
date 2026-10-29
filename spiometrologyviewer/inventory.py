"""Inventory: which wafers we have, which are released for stacking, which stacks were made, and
which pairs to stack next.

Kept in the workspace as two CSV files, read and written on every change (Nextcloud has no locking,
so whoever writes last wins; changes are small and rare):

    inventory/wafers.csv   wafer, plate, status, added, status_changed, changed_by, stack, notes
    inventory/stacks.csv   stack, item_a, item_b, status, stacked_on, created_by, parent,
                           good_devices, defective_devices, notes

A wafer is "in stock" when it is first found (sync), then "released" (for stacking) or "rejected"
by hand, and "stacked" once it is used in a stack. A stack is made of any two items: wafers or
other stacks (substacks), and is "available" until it is itself used in a stack.

Device quality: per wafer, the worst status (wafer.overall) over the measurement types of its plate
that have results for it, with each type's saved tolerances. Types on another position map than the
device map (P3 Alignment: 5 marks) are left out. A stack's device is the worst of its two items.
Devices are matched by number: device n of one item lies on device n of the other.
"""
import os
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

from . import measurements, wafer, workspace

FOLDER = workspace.PROJECT / "inventory"
WAFERS_CSV = FOLDER / "wafers.csv"
STACKS_CSV = FOLDER / "stacks.csv"
WAFER_COLUMNS = ["wafer", "plate", "status", "added", "status_changed", "changed_by", "stack", "notes"]
STACK_COLUMNS = ["stack", "item_a", "item_b", "status", "stacked_on", "created_by", "parent",
                 "good_devices", "defective_devices", "notes"]

IN_STOCK, RELEASED, REJECTED, STACKED, AVAILABLE = "in stock", "released", "rejected", "stacked", "available"
WAFER_STATUSES = [IN_STOCK, RELEASED, REJECTED, STACKED]
DEFECTIVE = {"fail"}   # default: warn counts as good


def now():
    return time.strftime("%Y-%m-%d %H:%M")


# ------------------------------------------------------------------ files

def _read(path, columns):
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df:
            df[c] = ""
    return df[columns + [c for c in df.columns if c not in columns]]


def _write(df, path):
    """Via a temporary file, so a crash or a Nextcloud sync never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except PermissionError:
        raise PermissionError(f"Can't write {path}: is it open in Excel? Close it and try again.") from None


def load():
    """(wafers, stacks) as DataFrames of strings."""
    return _read(WAFERS_CSV, WAFER_COLUMNS), _read(STACKS_CSV, STACK_COLUMNS)


def save(wafers=None, stacks=None):
    if wafers is not None:
        _write(wafers, WAFERS_CSV)
    if stacks is not None:
        _write(stacks, STACKS_CSV)


# ------------------------------------------------------------------ finding wafers

def _types(plates=None):
    plates = plates if plates is not None else measurements.load()
    return [m for ms in plates.values() for m in ms]


def sync(plates=None, progress=None):
    """Add every wafer that has images (legacy ones too) or results in any measurement type and isn't
    in the inventory yet, as "in stock". Never removes a wafer. Returns the names added."""
    found = {}
    for m in _types(plates):
        if progress:
            progress(f"Looking for wafers of {m.key}...")
        try:
            names = list(m.found()) + list(wafer.find_results(m.output))
        except Exception:   # unreachable image folder, unreadable config: the other types still count
            continue
        for name in names:
            if not name.endswith("(top level)"):
                found.setdefault(name, m.plate)
    wafers, _ = load()
    new = [n for n in found if n not in set(wafers["wafer"])]
    if new:
        added = now()
        rows = pd.DataFrame([{"wafer": n, "plate": found[n], "status": IN_STOCK, "added": added} for n in new])
        wafers = pd.concat([wafers, rows], ignore_index=True).fillna("")
        save(wafers=wafers)
    return new


# ------------------------------------------------------------------ device quality

_status_cache = {}    # wafer name -> {device: status}
_metrics_cache = {}   # measurement key -> tolerance metrics


def clear_cache():
    _status_cache.clear()
    _metrics_cache.clear()


def _metrics(m):
    if m.key not in _metrics_cache:
        from . import analyses
        _metrics_cache[m.key] = wafer.load_tolerances(m.tolerances, analyses.name_of(m.load_config()))
    return _metrics_cache[m.key]


def device_status(name, plate, plates=None):
    """{device: "pass" | "warn" | "fail"} of one wafer, worst over its plate's measurement types
    (on the device map) with results for it. Devices without results are left out."""
    if name in _status_cache:
        return _status_cache[name]
    statuses = {}
    for m in _types(plates):
        if m.plate != plate or m.positions.name != wafer.DEFAULT_POSITIONS.name:
            continue
        csv = m.output / name / "results.csv"
        if not csv.exists():
            continue
        try:
            res = pd.read_csv(csv)
            metrics = _metrics(m)
        except Exception:
            continue
        for row in res.to_dict("records"):
            dev = wafer.device_number(row.get("image"))
            s = wafer.overall(row, metrics)
            if dev is not None and s != "nodata" and \
                    wafer.SEVERITY[s] > wafer.SEVERITY[statuses.get(dev, "nodata")]:
                statuses[dev] = s
    _status_cache[name] = statuses
    return statuses


def item_status(item, wafers, stacks, plates=None):
    """{device: status} of a wafer, or of a stack (the worst of its two items, recursively)."""
    w = wafers[wafers["wafer"] == item]
    if len(w):
        return device_status(item, w.iloc[0]["plate"], plates)
    s = stacks[stacks["stack"] == item]
    if not len(s):
        return {}
    a = item_status(s.iloc[0]["item_a"], wafers, stacks, plates)
    b = item_status(s.iloc[0]["item_b"], wafers, stacks, plates)
    out = dict(a)
    for dev, st in b.items():
        if wafer.SEVERITY[st] > wafer.SEVERITY[out.get(dev, "nodata")]:
            out[dev] = st
    return out


def counts(statuses, devices=None):
    """{"pass": n, "warn": n, "fail": n, "nodata": n} over `devices` (default: those with a status)."""
    devices = statuses.keys() if devices is None else devices
    c = {s: 0 for s in ("pass", "warn", "fail", "nodata")}
    for d in devices:
        c[statuses.get(d, "nodata")] += 1
    return c


# ------------------------------------------------------------------ stacking two items

def compare(a, b, devices, defective=DEFECTIVE):
    """Per device of `devices`: "good" (defective in neither), "a" / "b" (defective in only that
    one: the stacked device is lost), "both" (overlapping defects: one bad stacked device instead
    of two) or "nodata" (no results in either). `a`, `b` = {device: status}.
    Returns (categories {device: category}, counts {category: n})."""
    cats = {}
    for d in devices:
        if d not in a and d not in b:
            cats[d] = "nodata"
            continue
        da, db = a.get(d) in defective, b.get(d) in defective
        cats[d] = "both" if da and db else "a" if da else "b" if db else "good"
    n = {k: 0 for k in ("good", "a", "b", "both", "nodata")}
    for c in cats.values():
        n[c] += 1
    return cats, n


def available(wafers, stacks):
    """Items that can go into a stack: released wafers and available stacks."""
    return (wafers.loc[wafers["status"] == RELEASED, "wafer"].tolist()
            + stacks.loc[stacks["status"] == AVAILABLE, "stack"].tolist())


def rank_pairs(items, wafers, stacks, devices, defective=DEFECTIVE, plates=None, partner_of=None):
    """Every pair of `items` (or every pair with `partner_of`), best first: most good stacked
    devices, then most overlapping defects. Returns [(a, b, counts), ...]."""
    st = {i: item_status(i, wafers, stacks, plates) for i in items}
    if partner_of is not None:
        st.setdefault(partner_of, item_status(partner_of, wafers, stacks, plates))
        pairs = [(partner_of, i) for i in items if i != partner_of]
    else:
        pairs = list(combinations(items, 2))
    ranked = [(a, b, compare(st[a], st[b], devices, defective)[1]) for a, b in pairs]
    ranked.sort(key=lambda p: (-p[2]["good"], -p[2]["both"], p[0], p[1]))
    return ranked


def best_pairing(ranked):
    """Greedy pairing from rank_pairs: take the best pair, drop its items, repeat."""
    used, out = set(), []
    for a, b, n in ranked:
        if a not in used and b not in used:
            out.append((a, b, n))
            used.update((a, b))
    return out


# ------------------------------------------------------------------ changes

def set_status(names, status, user=None):
    wafers, _ = load()
    sel = wafers["wafer"].isin(names)
    wafers.loc[sel, "status"] = status
    wafers.loc[sel, "status_changed"] = now()
    wafers.loc[sel, "changed_by"] = user or workspace.user()
    save(wafers=wafers)


def set_notes(kind, name, notes):
    wafers, stacks = load()
    df, col = (wafers, "wafer") if kind == "wafer" else (stacks, "stack")
    df.loc[df[col] == name, "notes"] = notes
    save(wafers=wafers if kind == "wafer" else None, stacks=stacks if kind == "stack" else None)


def next_stack_name(stacks=None):
    stacks = load()[1] if stacks is None else stacks
    nums = [int(s[1:]) for s in stacks["stack"] if s[:1] == "S" and s[1:].isdigit()]
    return f"S{max(nums, default=0) + 1:03d}"


def add_stack(name, a, b, devices, date=None, notes="", user=None, plates=None, defective=DEFECTIVE):
    """Record a stack made in production of items `a` and `b` (wafers or stacks): its wafers
    become "stacked" (with the stack's name), its substacks "stacked" (with this one as parent).
    The good / defective devices at this moment are saved with it."""
    clear_cache()   # the snapshot uses the latest results and tolerances
    wafers, stacks = load()
    name = name.strip()
    if not name:
        raise ValueError("The stack needs a name.")
    if name in set(stacks["stack"]) or name in set(wafers["wafer"]):
        raise ValueError(f"{name} exists already.")
    if a == b:
        raise ValueError("A stack needs two different items.")
    for item in (a, b):
        if item in set(wafers["wafer"]):
            if wafers.loc[wafers["wafer"] == item, "status"].iloc[0] in (STACKED, REJECTED):
                raise ValueError(f"Wafer {item} is {wafers.loc[wafers['wafer'] == item, 'status'].iloc[0]}.")
        elif item in set(stacks["stack"]):
            if stacks.loc[stacks["stack"] == item, "status"].iloc[0] != AVAILABLE:
                raise ValueError(f"Stack {item} is used in another stack already.")
        else:
            raise ValueError(f"{item} is not in the inventory.")
    cats, _ = compare(item_status(a, wafers, stacks, plates), item_status(b, wafers, stacks, plates),
                      devices, defective)
    good = ";".join(str(d) for d, c in sorted(cats.items()) if c == "good")
    bad = ";".join(str(d) for d, c in sorted(cats.items()) if c in ("a", "b", "both"))
    row = {"stack": name, "item_a": a, "item_b": b, "status": AVAILABLE, "stacked_on": date or now(),
           "created_by": user or workspace.user(), "parent": "", "good_devices": good,
           "defective_devices": bad, "notes": notes}
    stacks = pd.concat([stacks, pd.DataFrame([row])], ignore_index=True).fillna("")
    changed = now()
    for item in (a, b):
        sel = wafers["wafer"] == item
        if sel.any():
            wafers.loc[sel, ["status", "stack", "status_changed", "changed_by"]] = \
                [STACKED, name, changed, user or workspace.user()]
        else:
            stacks.loc[stacks["stack"] == item, ["status", "parent"]] = [STACKED, name]
    save(wafers, stacks)


def wafers_of(item, stacks):
    """The wafers in a stack (recursively), or [item] for a wafer."""
    s = stacks[stacks["stack"] == item]
    if not len(s):
        return [item]
    return wafers_of(s.iloc[0]["item_a"], stacks) + wafers_of(s.iloc[0]["item_b"], stacks)
