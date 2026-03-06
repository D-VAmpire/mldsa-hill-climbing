#!/usr/bin/env python3
"""
Parse hill-climbing ML-DSA experiment log files.

Each experiment block starts with a '# python3 hillclimb_mldsa_noise.py ...' line
and ends after the '=== Summary ===' section. Blocks with no complete Summary
section are silently skipped (interrupted/partial runs).

Output: tab-separated rows to stdout.
  - All complete 5/5 blocks are emitted (Tier = "5/5").
  - For each (variant, j) group, the single 4/5 block with fewest inf-rels is
    also emitted (Tier = "4/5*"), regardless of whether a 5/5 block exists.
  - Rows are sorted by (variant, j, tier) so the two rows per configuration
    appear adjacent.
  - Incomplete/partial blocks (no Summary section) are silently skipped.

Usage:
    python3 parse_hillclimb_log.py <logfile>
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

RE_BLOCK_START   = re.compile(r'^# python3 hillclimb_mldsa(?:_noise)?\.py\b')
RE_PARAMS        = re.compile(r'--params\s+(\d+)')
RE_LEAKAGE       = re.compile(r'--leakage\s+(\d+)')
RE_INF_RELS_CMD  = re.compile(r'--inf-rels\s+(\d+)')
RE_BETA_EFF      = re.compile(r'beta_eff=(\d+)')
RE_INF_RELS_HDR  = re.compile(r'Informative relations:\s+(\d+)')
RE_LEAKAGE_HDR   = re.compile(r'Leakage index:\s+(\d+)')

RE_KEY_START     = re.compile(r'--- Key \d+/\d+ ---')
RE_REGRESSION    = re.compile(
    r'Regression:\s+(\d+)/256 correct \(([0-9.]+)%\),\s+D_0=(\d+)'
)
# First occurrence of "D=0 *" in the iteration log — this is the convergence point.
# The line looks like:  Iter 27: F=1108209, D=0 *  [...]
RE_FIRST_D0      = re.compile(r'Iter\s+(\d+):\s+F=\d+,\s+D=0\s+\*')

RE_SUCCESS       = re.compile(
    r'SUCCESS: key recovered, F=\d+, (\d+) iterations, (\d+) perturbation\(s\) \(([0-9.]+)s\)'
)
RE_FAILED        = re.compile(r'FAILED:')

RE_SUMMARY_START = re.compile(r'^=== Summary:')
RE_SUMMARY_RATE  = re.compile(r'=== Summary:\s+(\d+)/(\d+) keys recovered')
RE_AVG_ITERS     = re.compile(r'Avg iterations:\s+([0-9.]+)')
RE_AVG_PERTURBS  = re.compile(r'Avg perturbations:\s+([0-9.]+)')
RE_AVG_TIME      = re.compile(r'Avg hill-climb time:\s+([0-9.]+)s')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KeyResult:
    regression_acc: Optional[float] = None   # percentage, e.g. 70.7
    D0: Optional[int]   = None
    first_D0_iter: Optional[int] = None       # None if key was never recovered
    total_iters: Optional[int] = None
    perturbations: Optional[int] = None
    time_s: Optional[float] = None
    success: bool = False


@dataclass
class ExperimentBlock:
    variant: Optional[int] = None             # 44 / 65 / 87
    leakage_j: Optional[int] = None
    beta_eff: Optional[int] = None
    inf_rels: Optional[int] = None
    keys: list = field(default_factory=list)  # list[KeyResult]
    # summary fields (cross-check / fallback)
    summary_recovered: Optional[int] = None
    summary_total: Optional[int] = None
    summary_avg_iters: Optional[float] = None
    summary_avg_perturbs: Optional[float] = None
    summary_avg_time: Optional[float] = None
    complete: bool = False                    # True only if Summary was found


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def split_into_blocks(lines: list[str]) -> list[list[str]]:
    """Split log lines into per-experiment blocks at '# python3 ...' markers."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if RE_BLOCK_START.match(line):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def split_into_key_sections(block_lines: list[str]) -> list[list[str]]:
    """Within a block, split lines into per-key sections at '--- Key K/N ---'."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in block_lines:
        if RE_KEY_START.match(line.strip()):
            if current:
                sections.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        sections.append(current)
    return sections


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_config(block_lines: list[str], exp: ExperimentBlock) -> None:
    """Extract variant, leakage, inf_rels, beta_eff from command line or header."""
    for line in block_lines[:20]:   # config is always near the top
        if RE_BLOCK_START.match(line):
            m = RE_PARAMS.search(line)
            if m:
                exp.variant = int(m.group(1))
            m = RE_LEAKAGE.search(line)
            if m:
                exp.leakage_j = int(m.group(1))
            m = RE_INF_RELS_CMD.search(line)
            if m:
                exp.inf_rels = int(m.group(1))
        if exp.beta_eff is None:
            m = RE_BETA_EFF.search(line)
            if m:
                exp.beta_eff = int(m.group(1))
        if exp.inf_rels is None:
            m = RE_INF_RELS_HDR.search(line)
            if m:
                exp.inf_rels = int(m.group(1))
        if exp.leakage_j is None:
            m = RE_LEAKAGE_HDR.search(line)
            if m:
                exp.leakage_j = int(m.group(1))


def parse_key_section(key_lines: list[str]) -> KeyResult:
    kr = KeyResult()
    first_d0_found = False

    for line in key_lines:
        # Regression line
        if kr.regression_acc is None:
            m = RE_REGRESSION.search(line)
            if m:
                kr.regression_acc = float(m.group(2))
                kr.D0 = int(m.group(3))
                continue

        # First D=0 * — only record the very first occurrence
        if not first_d0_found:
            m = RE_FIRST_D0.search(line)
            if m:
                kr.first_D0_iter = int(m.group(1))
                first_d0_found = True
                continue

        # SUCCESS
        m = RE_SUCCESS.search(line)
        if m:
            kr.total_iters = int(m.group(1))
            kr.perturbations = int(m.group(2))
            kr.time_s = float(m.group(3))
            kr.success = True
            continue

        # FAILED
        if RE_FAILED.search(line):
            kr.success = False
            # total_iters / perturbations may still be on the --- Final --- line;
            # we leave them None since the key was not recovered.

    return kr


def parse_summary(block_lines: list[str], exp: ExperimentBlock) -> None:
    in_summary = False
    for line in block_lines:
        if RE_SUMMARY_START.match(line.strip()):
            in_summary = True
            m = RE_SUMMARY_RATE.search(line)
            if m:
                exp.summary_recovered = int(m.group(1))
                exp.summary_total = int(m.group(2))
            exp.complete = True
            continue
        if in_summary:
            m = RE_AVG_ITERS.search(line)
            if m:
                exp.summary_avg_iters = float(m.group(1))
            m = RE_AVG_PERTURBS.search(line)
            if m:
                exp.summary_avg_perturbs = float(m.group(1))
            m = RE_AVG_TIME.search(line)
            if m:
                exp.summary_avg_time = float(m.group(1))


def parse_block(block_lines: list[str]) -> Optional[ExperimentBlock]:
    exp = ExperimentBlock()
    parse_config(block_lines, exp)

    # Find where the summary begins so we don't feed it to key parsers
    summary_start_idx = None
    for i, line in enumerate(block_lines):
        if RE_SUMMARY_START.match(line.strip()):
            summary_start_idx = i
            break

    body_lines = block_lines[:summary_start_idx] if summary_start_idx is not None else block_lines

    # Parse individual key sections
    key_sections = split_into_key_sections(body_lines)
    for ks in key_sections:
        exp.keys.append(parse_key_section(ks))

    parse_summary(block_lines, exp)

    if not exp.complete:
        return None  # incomplete / interrupted run
    return exp


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def safe_avg(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def safe_max(values: list) -> Optional[int]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


HEADER = (
    "Tier\t"
    "Variant\t"
    "j\t"
    "beta_eff\t"
    "Inf_rels\t"
    "Success\t"
    "Avg_acc_%\t"
    "Avg_D0\t"
    "Avg_first_D0_iter\t"
    "Max_first_D0_iter\t"
    "Avg_total_iters\t"
    "Avg_perturbations\t"
    "Avg_time_s"
)


def fmt(val, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def experiment_to_row(exp: ExperimentBlock, tier: str) -> str:
    successful_keys = [k for k in exp.keys if k.success]
    n_recovered = len(successful_keys)
    n_total = len(exp.keys) if exp.keys else exp.summary_total

    avg_acc   = safe_avg([k.regression_acc for k in exp.keys])
    avg_D0    = safe_avg([k.D0 for k in exp.keys])

    # first-D0 stats: only over keys that actually reached D=0
    first_d0_iters = [k.first_D0_iter for k in successful_keys if k.first_D0_iter is not None]
    avg_first_d0 = safe_avg(first_d0_iters)
    max_first_d0 = safe_max(first_d0_iters)

    # total-iters, perturbations, time: use per-key data if available,
    # fall back to summary averages
    total_iters_list = [k.total_iters for k in successful_keys if k.total_iters is not None]
    perturbs_list    = [k.perturbations for k in successful_keys if k.perturbations is not None]
    times_list       = [k.time_s for k in successful_keys if k.time_s is not None]

    avg_total_iters  = safe_avg(total_iters_list) or exp.summary_avg_iters
    avg_perturbs     = safe_avg(perturbs_list)    or exp.summary_avg_perturbs
    avg_time         = safe_avg(times_list)       or exp.summary_avg_time

    success_str = f"{n_recovered}/{n_total}"

    return "\t".join([
        tier,
        fmt(exp.variant, 0),
        fmt(exp.leakage_j, 0),
        fmt(exp.beta_eff, 0),
        fmt(exp.inf_rels, 0),
        success_str,
        fmt(avg_acc, 1),
        fmt(avg_D0, 1),
        fmt(avg_first_d0, 1),
        fmt(max_first_d0, 0),
        fmt(avg_total_iters, 1),
        fmt(avg_perturbs, 1),
        fmt(avg_time, 1),
    ])


def n_recovered(exp: ExperimentBlock) -> int:
    """Number of successfully recovered keys, preferring per-key data over summary."""
    per_key = sum(1 for k in exp.keys if k.success)
    if per_key > 0 or exp.keys:
        return per_key
    return exp.summary_recovered or 0


def select_best_partial(group: list[ExperimentBlock]) -> Optional[ExperimentBlock]:
    """Return the 4/5 block with the fewest inf-rels from a (variant, j) group."""
    candidates = [
        exp for exp in group
        if n_recovered(exp) == 4
        and exp.inf_rels is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e.inf_rels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <logfile>", file=sys.stderr)
        sys.exit(1)

    logfile = sys.argv[1]
    try:
        with open(logfile, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]
    except OSError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    raw_blocks = split_into_blocks(lines)

    experiments: list[ExperimentBlock] = []
    skipped = 0
    for raw in raw_blocks:
        exp = parse_block(raw)
        if exp is None:
            skipped += 1
        else:
            experiments.append(exp)

    if skipped:
        print(f"# Skipped {skipped} incomplete/partial block(s).", file=sys.stderr)

    # Group by (variant, j)
    from collections import defaultdict
    groups: dict[tuple, list[ExperimentBlock]] = defaultdict(list)
    for exp in experiments:
        groups[(exp.variant, exp.leakage_j)].append(exp)

    # Build output rows: (sort_key, tier_label, exp)
    rows: list[tuple] = []

    for (variant, j), group in groups.items():
        # All 5/5 blocks
        for exp in group:
            if n_recovered(exp) == 5:
                sort_key = (variant or 0, j or 0, 0, exp.inf_rels or 0)
                rows.append((sort_key, "5/5", exp))

        # Best (min inf-rels) 4/5 block
        best_partial = select_best_partial(group)
        if best_partial is not None:
            sort_key = (variant or 0, j or 0, 1, best_partial.inf_rels or 0)
            rows.append((sort_key, "4/5*", best_partial))

    rows.sort(key=lambda r: r[0])

    print(HEADER)
    for _, tier, exp in rows:
        print(experiment_to_row(exp, tier))


if __name__ == "__main__":
    main()
