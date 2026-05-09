#!/usr/bin/env python3
"""
Aggregate and display all ML-DSA hill-climbing noise experiment results.

Discovers every noise-level-*-percent directory under the script's own
directory, parses all .txt log files within them, and prints results
grouped by:

    Noise level  →  ML-DSA variant  →  leakage index j

so the full experimental landscape is visible at a glance.

Usage:
    python3 overview_noise_experiments.py [--root <dir>]
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_hillclimb_noise_log import (
    split_into_blocks,
    parse_block,
    n_recovered,
    select_best_partial,
    experiment_to_row,
    HEADER,
)

_NOISE_DIR_RE = re.compile(r"noise-level-(\d+)-percent")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noise_pct(d: Path) -> int:
    m = _NOISE_DIR_RE.search(d.name)
    return int(m.group(1)) if m else 0


def _parse_file(path: Path) -> tuple[list, int]:
    """Return (rows, n_skipped).  rows = list of (tier, ExperimentBlock)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    raw_blocks = split_into_blocks(lines)

    experiments, skipped = [], 0
    for raw in raw_blocks:
        exp = parse_block(raw)
        if exp is None:
            skipped += 1
        else:
            experiments.append(exp)

    groups: dict = defaultdict(list)
    for exp in experiments:
        groups[(exp.variant, exp.leakage_j)].append(exp)

    rows = []
    for group in groups.values():
        for exp in group:
            if n_recovered(exp) == 5:
                rows.append(("5/5", exp))
        best = select_best_partial(group)
        if best is not None:
            rows.append(("4/5*", best))

    return rows, skipped


def _sort_key(tier_exp):
    tier, exp = tier_exp
    return (
        exp.variant or 0,
        exp.leakage_j or 0,
        0 if tier == "5/5" else 1,
        exp.inf_rels or 0,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

WIDE = 96

# Full column header, with Noise_% prepended
FULL_HEADER = "Noise_%\t" + HEADER


def _full_row(noise_pct: int, tier: str, exp) -> str:
    return f"{noise_pct}\t{experiment_to_row(exp, tier)}"


def _section_bar(char: str = "=") -> str:
    return char * WIDE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).parent,
                    help="Directory containing the noise-level-*-percent subdirectories "
                         "(default: directory of this script)")
    args = ap.parse_args()
    root: Path = args.root

    noise_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and _NOISE_DIR_RE.search(d.name)],
        key=_noise_pct,
    )
    if not noise_dirs:
        print(f"No noise-level-*-percent directories found under {root}.", file=sys.stderr)
        sys.exit(1)

    # Collect: noise_pct → sorted [(tier, ExperimentBlock)]
    data: dict[int, list] = {}
    total_skipped = 0

    for d in noise_dirs:
        pct = _noise_pct(d)
        all_rows: list = []
        for lf in sorted(d.glob("*.txt")):
            rows, skipped = _parse_file(lf)
            all_rows.extend(rows)
            total_skipped += skipped
        all_rows.sort(key=_sort_key)
        data[pct] = all_rows

    if total_skipped:
        print(
            f"# Note: skipped {total_skipped} incomplete/partial block(s) across all files.",
            file=sys.stderr,
        )

    # ---------------------------------------------------------------------------
    # Print overview
    # ---------------------------------------------------------------------------

    # ── Global summary table ────────────────────────────────────────────────
    print(_section_bar("="))
    print("  OVERVIEW  —  ML-DSA Hill-Climbing Noise Experiments")
    print(_section_bar("="))
    print()
    print("  Combinations covered  (noise% × variant × j):")
    print()

    # Collect all (noise, variant, j) with tier counts for the summary grid
    summary: dict[tuple, dict] = {}   # (noise, variant, j) → {5/5: n, 4/5*: n}
    for pct, rows in data.items():
        for tier, exp in rows:
            key = (pct, exp.variant, exp.leakage_j)
            if key not in summary:
                summary[key] = {"5/5": 0, "4/5*": 0}
            summary[key][tier] += 1

    # Print a compact grid: noise% as rows, (variant, j) as columns
    all_variants_j = sorted({(v, j) for (_, v, j) in summary}, key=lambda x: (x[0] or 0, x[1] or 0))
    col_w = 14
    header_cells = [f"ML-DSA-{v}/j={j}".center(col_w) for v, j in all_variants_j]
    print("  " + "Noise%".ljust(8) + "".join(header_cells))
    print("  " + "-" * (8 + col_w * len(all_variants_j)))
    for pct in sorted(data):
        cells = []
        for vj in all_variants_j:
            key = (pct, vj[0], vj[1])
            if key in summary:
                n5 = summary[key]["5/5"]
                n4 = summary[key]["4/5*"]
                if n5 and n4:
                    tag = f"{n5}×5/5+{n4}×4/5*"
                elif n5:
                    tag = f"{n5}×5/5"
                else:
                    tag = f"{n4}×4/5*"
            else:
                tag = "—"
            cells.append(tag.center(col_w))
        print("  " + f"{pct}%".ljust(8) + "".join(cells))
    print()

    # ── Detailed sections ────────────────────────────────────────────────────
    for pct, rows in sorted(data.items()):
        print()
        print(_section_bar("="))
        print(f"  NOISE LEVEL: {pct}%")
        print(_section_bar("="))

        if not rows:
            print("  (no complete experiments found)")
            continue

        # Sub-group by (variant, j)
        subgroups: dict[tuple, list] = defaultdict(list)
        for tier, exp in rows:
            subgroups[(exp.variant, exp.leakage_j)].append((tier, exp))

        for (variant, j), sub in sorted(
            subgroups.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0)
        ):
            beta = sub[0][1].beta_eff
            beta_str = f"beta_eff={beta}" if beta is not None else "beta_eff=N/A"
            n_five     = sum(1 for t, _ in sub if t == "5/5")
            n_partial  = sum(1 for t, _ in sub if t == "4/5*")
            count_str  = f"{n_five} × 5/5" + (f",  {n_partial} × 4/5*" if n_partial else "")

            print()
            print(f"  {'─' * (WIDE - 2)}")
            print(f"  ML-DSA-{variant}  |  j={j}  |  {beta_str}  |  {count_str}")
            print(f"  {'─' * (WIDE - 2)}")
            print(f"  {FULL_HEADER}")
            for tier, exp in sub:
                print(f"  {_full_row(pct, tier, exp)}")

    print()


if __name__ == "__main__":
    main()
