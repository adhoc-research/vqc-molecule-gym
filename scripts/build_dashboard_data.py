#!/usr/bin/env python3
"""Aggregate evaluation run logs into compact JSON for the project dashboard.

Reads ``runs/*.jsonl`` (real evaluator output: per-task energies, exact
references, reward components, circuit metrics) plus the RL curriculum tiers and
emits compact JSON files under ``assets/data/`` that power the self-contained,
interactive figures on the GitHub Pages project page.

Stdlib only -- no project imports, no heavy deps -- so it runs anywhere
(locally or in CI). Generated JSON is committed; the deploy workflow uploads the
repository root.

Usage:
    python3 scripts/build_dashboard_data.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Any

# --- Paths -------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_GLOB = os.path.join(ROOT, "runs", "*.jsonl")
OUT_DIR = os.path.join(ROOT, "assets", "data")

# Chemical accuracy: 1 kcal/mol ~= 1.594 mHa.
CHEM_ACC_MHA = 1.6

# --- Benchmark presentation metadata ----------------------------------------
# Display order is the curriculum-aligned narrative order (easy -> hard).
BENCH_META: dict[str, dict[str, str]] = {
    "h2_tiny": {"label": "H₂", "scan": "Bond length", "unit": "Å", "cas": "minimal"},
    "lih_bond_scan_v0": {"label": "LiH", "scan": "Bond length", "unit": "Å", "cas": "CAS(4e,4o)"},
    "c2h6_torsion_scan_v0": {"label": "C₂H₆", "scan": "Torsion angle", "unit": "°", "cas": "CAS(2e,2o)"},
    "h2o_angle_scan_v0": {"label": "H₂O", "scan": "H-O-H angle", "unit": "°", "cas": "CAS(4e,4o)"},
    "h2o_dimer_distance_scan_v0": {"label": "(H₂O)₂", "scan": "O···O distance", "unit": "Å", "cas": "CAS(4e,4o)"},
    "h4_small": {"label": "H₄", "scan": "Bond length", "unit": "Å", "cas": "CAS(4e,4o)"},
    "n2_bond_scan_v0": {"label": "N₂", "scan": "Bond length", "unit": "Å", "cas": "CAS(6e,6o)"},
}
BENCH_ORDER = list(BENCH_META.keys())

# Curriculum tiers (mirrors vqc_molecule_gym/curricula.py, canonicalized) with a
# short difficulty narrative for the overview cards.
CURRICULUM = [
    {
        "tier": "Easy",
        "key": "easy_curriculum",
        "benchmarks": ["h2_tiny", "lih_bond_scan_v0", "c2h6_torsion_scan_v0"],
        "note": "Chemical accuracy reachable or solved-by-base. The dense energy-improvement "
                "signal is reliable, making these strong candidates for early PPO reward learning.",
    },
    {
        "tier": "Medium",
        "key": "medium_curriculum",
        "benchmarks": ["h2o_angle_scan_v0", "h2o_dimer_distance_scan_v0"],
        "note": "Improve over base but not to chemical accuracy with current pools. "
                "Progress is graded, so these are downweighted for initial PPO.",
    },
    {
        "tier": "Hard",
        "key": "hard_curriculum",
        "benchmarks": ["h4_small", "n2_bond_scan_v0"],
        "note": "Strong base improvement, but error stalls above 10 mHa. Operator-pool or "
                "parameter work is needed, so these are best used as evaluation-only tasks.",
    },
]

# One-line interpretation per benchmark for the ceiling-check heatmap tooltip.
INTERPRETATION: dict[str, str] = {
    "c2h6_torsion_scan_v0": "Already/reliably chemically accurate.",
    "h2_tiny": "Chemical accuracy reachable.",
    "lih_bond_scan_v0": "Chemical accuracy reachable across the scan.",
    "h2o_angle_scan_v0": "Improves near threshold; useful graded RL progress.",
    "h2o_dimer_distance_scan_v0": "Close but not chemical accuracy; pool/grid limiting.",
    "h4_small": "Strong improvement over base; stalls at 10+ mHa.",
    "n2_bond_scan_v0": "Large improvement over base; far from chemical accuracy.",
}

AGENTS = ["random", "greedy", "beam_search"]


def parse_scan(task_id: str) -> tuple[float, str] | None:
    """Extract the scan variable (value, label) from a task id.

    Handles ``..._r1.40_...`` (distance), ``c2h6_phi090_...`` (torsion) and
    ``h2o_angle104.5_...`` (bond angle).
    """
    m = re.search(r"_r(\d+(?:\.\d+)?)", task_id)
    if m:
        return float(m.group(1)), m.group(1)
    m = re.search(r"phi(\d+(?:\.\d+)?)", task_id)
    if m:
        return float(m.group(1)), str(int(float(m.group(1))))
    m = re.search(r"angle(\d+(?:\.\d+)?)", task_id)
    if m:
        return float(m.group(1)), m.group(1)
    return None


def iter_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(glob.glob(RUNS_GLOB)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = rec.get("result") or {}
                if not result.get("valid"):
                    continue
                if rec.get("benchmark_id") not in BENCH_META:
                    continue
                records.append(rec)
    return records


def err_mha(result: dict[str, Any]) -> float | None:
    val = result.get("energy_error_mha")
    if val is None:
        ehart = result.get("energy_error_hartree")
        val = abs(ehart) * 1000.0 if ehart is not None else None
    return abs(val) if val is not None else None


def is_nonempty(rec: dict[str, Any]) -> bool:
    seq = (rec.get("action") or {}).get("operator_sequence") or []
    return len(seq) > 0


# --- Aggregation -------------------------------------------------------------

def build_pes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Best achieved energy vs exact reference, per task, per benchmark."""
    # best[(bench, task)] = (err, record)
    best: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for rec in records:
        if not is_nonempty(rec):
            continue
        result = rec["result"]
        e = err_mha(result)
        if e is None or result.get("reference_energy_hartree") is None:
            continue
        key = (rec["benchmark_id"], rec["task_id"])
        if key not in best or e < best[key][0]:
            best[key] = (e, rec)

    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (bench, task), (e, rec) in best.items():
        scan = parse_scan(task)
        if scan is None:
            continue
        result = rec["result"]
        by_bench[bench].append({
            "scan_value": scan[0],
            "scan_label": scan[1],
            "energy": round(result["energy_hartree"], 6),
            "reference": round(result["reference_energy_hartree"], 6),
            "error_mha": round(e, 4),
            "chemical_accuracy": bool(result.get("chemical_accuracy")),
            "num_operators": (result.get("circuit_metrics") or {}).get("num_operators"),
        })

    out = {"chem_acc_mha": CHEM_ACC_MHA, "benchmarks": {}}
    for bench in BENCH_ORDER:
        pts = sorted(by_bench.get(bench, []), key=lambda p: p["scan_value"])
        if not pts:
            continue
        meta = BENCH_META[bench]
        out["benchmarks"][bench] = {
            "label": meta["label"],
            "scan_axis": f"{meta['scan']} ({meta['unit']})",
            "cas": meta["cas"],
            "points": pts,
        }
    return out


def build_algo_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (benchmark, agent): best-per-task error stats + chem-acc hit rate."""
    # best error per (bench, agent, task)
    per_task: dict[tuple[str, str, str], float] = {}
    per_task_acc: dict[tuple[str, str, str], bool] = {}
    for rec in records:
        agent = rec.get("agent")
        if agent not in AGENTS:
            continue
        e = err_mha(rec["result"])
        if e is None:
            continue
        key = (rec["benchmark_id"], agent, rec["task_id"])
        if key not in per_task or e < per_task[key]:
            per_task[key] = e
            per_task_acc[key] = e < CHEM_ACC_MHA

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    acc: dict[tuple[str, str], int] = defaultdict(int)
    for (bench, agent, task), e in per_task.items():
        grouped[(bench, agent)].append(e)
        if per_task_acc[(bench, agent, task)]:
            acc[(bench, agent)] += 1

    out = {"agents": AGENTS, "benchmarks": []}
    for bench in BENCH_ORDER:
        if not any((bench, a) in grouped for a in AGENTS):
            continue
        entry = {"id": bench, "label": BENCH_META[bench]["label"], "by_agent": {}}
        for agent in AGENTS:
            errs = grouped.get((bench, agent))
            if not errs:
                entry["by_agent"][agent] = None
                continue
            entry["by_agent"][agent] = {
                "min_err": round(min(errs), 4),
                "median_err": round(statistics.median(errs), 4),
                "max_err": round(max(errs), 4),
                "n_tasks": len(errs),
                "chem_acc_hits": acc.get((bench, agent), 0),
            }
        out["benchmarks"].append(entry)
    return out


def build_reward_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One best point per (benchmark, task, agent): reward + circuit metrics."""
    best: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
    for rec in records:
        agent = rec.get("agent")
        if agent not in AGENTS or not is_nonempty(rec):
            continue
        result = rec["result"]
        e = err_mha(result)
        if e is None:
            continue
        key = (rec["benchmark_id"], rec["task_id"], agent)
        # keep the highest-reward record per cell
        reward = result.get("reward")
        if reward is None:
            continue
        if key not in best or reward > best[key][0]:
            best[key] = (reward, rec)

    points: list[dict[str, Any]] = []
    comp_keys = ["accuracy", "chemical_accuracy", "depth", "shots", "compactness"]
    for (bench, task, agent), (reward, rec) in best.items():
        result = rec["result"]
        cm = result.get("circuit_metrics") or {}
        comps = result.get("reward_components") or {}
        e = err_mha(result)
        points.append({
            "benchmark": bench,
            "label": BENCH_META[bench]["label"],
            "agent": agent,
            "error_mha": round(e, 4),
            "reward": round(reward, 4),
            "depth": cm.get("depth"),
            "gate_count": cm.get("gate_count"),
            "num_operators": cm.get("num_operators"),
            "components": {k: round(comps.get(k, 0.0), 4) for k in comp_keys},
        })
    points.sort(key=lambda p: (p["benchmark"], p["agent"]))
    return {"component_keys": comp_keys, "points": points}


def build_overview(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Ceiling-check summary (from beam_ceiling_check runs) + curriculum tiers."""
    # Restrict to the dedicated ceiling-check runs so numbers match the report.
    # Keep the best (lowest) error achieved per task, then reduce per benchmark.
    task_best: dict[tuple[str, str], float] = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "beam_ceiling_check_*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = rec.get("result") or {}
                bench = rec.get("benchmark_id")
                if not result.get("valid") or bench not in BENCH_META:
                    continue
                e = err_mha(result)
                if e is None:
                    continue
                key = (bench, rec.get("task_id"))
                if key not in task_best or e < task_best[key]:
                    task_best[key] = e

    per_bench: dict[str, list[float]] = defaultdict(list)
    for (bench, _task), e in task_best.items():
        per_bench[bench].append(e)

    rows = []
    for bench in BENCH_ORDER:
        errs = per_bench.get(bench)
        if not errs:
            continue
        hits = sum(1 for e in errs if e < CHEM_ACC_MHA)
        rows.append({
            "id": bench,
            "label": BENCH_META[bench]["label"],
            "n_tasks": len(errs),
            "chem_acc_hits": hits,
            "min_err": round(min(errs), 3),
            "median_err": round(statistics.median(errs), 3),
            "max_err": round(max(errs), 3),
            "interpretation": INTERPRETATION.get(bench, ""),
        })

    tiers = []
    for tier in CURRICULUM:
        tiers.append({
            "tier": tier["tier"],
            "note": tier["note"],
            "benchmarks": [
                {"id": b, "label": BENCH_META[b]["label"]}
                for b in tier["benchmarks"] if b in BENCH_META
            ],
        })

    return {"chem_acc_mha": CHEM_ACC_MHA, "ceiling": rows, "curriculum": tiers}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    records = iter_records()
    if not records:
        raise SystemExit("No valid run records found under runs/*.jsonl")

    artifacts = {
        "pes.json": build_pes(records),
        "algo_comparison.json": build_algo_comparison(records),
        "reward_metrics.json": build_reward_metrics(records),
        "overview.json": build_overview(records),
    }

    # Top-level summary stats for the hero stat row.
    benches = {r["benchmark_id"] for r in records}
    tasks = {(r["benchmark_id"], r["task_id"]) for r in records}
    total_chem_acc = sum(
        1 for r in records if (r["result"].get("chemical_accuracy"))
    )
    artifacts["summary.json"] = {
        "total_evaluations": len(records),
        "n_benchmarks": len(benches),
        "n_tasks": len(tasks),
        "n_agents": len(AGENTS),
        "chem_acc_evaluations": total_chem_acc,
    }

    for name, payload in artifacts.items():
        out_path = os.path.join(OUT_DIR, name)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        print(f"wrote {os.path.relpath(out_path, ROOT)}")

    # Also emit a JS bundle so the page works when opened directly from disk
    # (file:// blocks fetch of the JSON). dashboard.js prefers this global and
    # falls back to fetch when served over HTTP.
    bundle = {name[:-5]: payload for name, payload in artifacts.items()}
    js_path = os.path.join(OUT_DIR, "data.js")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write("window.DASH_DATA = ")
        json.dump(bundle, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"wrote {os.path.relpath(js_path, ROOT)}")

    # Console sanity summary
    pes = artifacts["pes.json"]["benchmarks"]
    print("\nPES point counts:")
    for bench in BENCH_ORDER:
        if bench in pes:
            print(f"  {bench:32s} {len(pes[bench]['points'])} points")


if __name__ == "__main__":
    main()
