#!/usr/bin/env python3
"""select_subjects.py

Select subjects for pipeline processing using weekly PMA bins with
sex-balanced random capping (Wu et al. 2024; Pietsch et al. 2019).

Strategy:
  - Bin subjects by PMA at scan (rounded to nearest week) using first scan only
  - Use all available subjects per bin
  - Where a bin exceeds --max-per-bin (default 15), select a sex-balanced
    random subset: floor(max/2) of each sex, remainder filled randomly
  - Output a CSV tracking selected subjects with metadata

Usage:
    python select_subjects.py --dhcp-dir /path/to/dhcp --anat-source /path/to/anat_pipeline
    python select_subjects.py --dhcp-dir ... --anat-source ... --max-per-bin 15 --seed 42
    python select_subjects.py --dhcp-dir ... --anat-source ... --output selected_subjects.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select subjects for pipeline processing using weekly PMA bins.",
    )
    parser.add_argument(
        "--dhcp-dir", type=Path, required=True,
        help="Path to processed dHCP data directory (contains sub-*/ses-* dirs).",
    )
    parser.add_argument(
        "--anat-source", type=Path, required=True,
        help="Path to dHCP rel3 anat pipeline derivatives (contains participants.tsv).",
    )
    parser.add_argument(
        "--dropped-csv", type=Path, default=None,
        help="Path to dropped_subjects.csv (optional; if omitted, no exclusions applied).",
    )
    parser.add_argument(
        "--max-per-bin", type=int, default=15,
        help="Maximum subjects per weekly PMA bin (default: 15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: selected_subjects.csv in current directory)",
    )
    args = parser.parse_args(argv)

    dhcp_dir = args.dhcp_dir
    anat_dir = args.anat_source
    output_path = Path(args.output) if args.output else Path("selected_subjects.csv")

    if not dhcp_dir.is_dir():
        print(f"ERROR: dhcp directory not found: {dhcp_dir}", file=sys.stderr)
        return 1

    # Load dropped subjects
    dropped_set: set[tuple[str, str]] = set()
    if args.dropped_csv is not None:
        dropped_csv = args.dropped_csv
        if dropped_csv.is_file():
            with open(dropped_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dropped_set.add((row["SUBJECT"], row["SESSION"]))
            print(f"Loaded {len(dropped_set)} dropped subjects from {dropped_csv}")
        else:
            print(f"WARNING: dropped-csv not found: {dropped_csv}", file=sys.stderr)

    # Load participants metadata (birth_age, sex)
    part_tsv = anat_dir / "participants.tsv"
    if not part_tsv.is_file():
        print(f"ERROR: participants.tsv not found: {part_tsv}", file=sys.stderr)
        return 1

    birth_lookup: dict[str, float] = {}
    sex_lookup: dict[str, str] = {}
    with open(part_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["participant_id"]
            try:
                birth_lookup[pid] = float(row["birth_age"])
            except (ValueError, KeyError):
                pass
            sex_lookup[pid] = row.get("sex", "unknown")

    # Collect all sessions with scan_age from per-subject session TSVs
    records: list[dict] = []
    missing_scan_age = 0

    for sub_dir in sorted(dhcp_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                continue

            sub_id = sub_dir.name
            ses_id = ses_dir.name

            if (sub_id, ses_id) in dropped_set:
                continue

            ses_num = ses_id.replace("ses-", "")
            ses_tsv = anat_dir / sub_id / f"{sub_id}_sessions.tsv"

            scan_age = None
            scan_number = None
            if ses_tsv.is_file():
                with open(ses_tsv) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        if row["session_id"].strip() == ses_num:
                            try:
                                scan_age = float(row["scan_age"])
                            except (ValueError, KeyError):
                                pass
                            try:
                                scan_number = int(row["scan_number"])
                            except (ValueError, KeyError):
                                scan_number = 1
                            break

            if scan_age is None:
                missing_scan_age += 1
                continue

            cc_id = sub_id.replace("sub-", "")
            records.append({
                "subject": sub_id,
                "session": ses_id,
                "scan_age_pma": scan_age,
                "scan_number": scan_number or 1,
                "birth_age": birth_lookup.get(cc_id),
                "sex": sex_lookup.get(cc_id, "unknown"),
            })

    if missing_scan_age > 0:
        print(f"WARNING: {missing_scan_age} sessions missing scan_age, skipped")

    # Keep first scan only for binning
    records.sort(key=lambda r: (r["subject"], r["scan_number"]))
    seen_subjects: set[str] = set()
    first_scan_records: list[dict] = []
    second_scan_map: dict[str, dict] = {}

    for rec in records:
        if rec["subject"] not in seen_subjects:
            seen_subjects.add(rec["subject"])
            first_scan_records.append(rec)
        else:
            second_scan_map[rec["subject"]] = rec

    print(f"\nTotal sessions: {len(records)}")
    print(f"Unique subjects (first scan): {len(first_scan_records)}")
    print(f"Subjects with second scan: {len(second_scan_map)}")

    # Assign weekly PMA bins
    for rec in first_scan_records:
        rec["pma_bin"] = round(rec["scan_age_pma"])

    # Group by bin
    bins: dict[int, list[dict]] = {}
    for rec in first_scan_records:
        bins.setdefault(rec["pma_bin"], []).append(rec)

    # Select subjects per bin with sex-balanced random capping
    rng = random.Random(args.seed)
    max_per_bin = args.max_per_bin
    selected: list[dict] = []
    total_available = 0
    total_selected = 0

    print(f"\n{'Bin':>5} {'Avail':>6} {'M/F':>7} {'Sel':>4} {'Sel M/F':>8} {'Capped':>7}")
    print("-" * 50)

    for pma_bin in sorted(bins.keys()):
        subjects = bins[pma_bin]
        total_available += len(subjects)

        if len(subjects) <= max_per_bin:
            chosen = subjects
        else:
            males = [s for s in subjects if s["sex"] == "male"]
            females = [s for s in subjects if s["sex"] == "female"]
            rng.shuffle(males)
            rng.shuffle(females)

            n_each = max_per_bin // 2
            remainder = max_per_bin % 2

            sel_m = males[:min(n_each, len(males))]
            sel_f = females[:min(n_each, len(females))]

            # Fill shortfall from the other sex
            if len(sel_m) < n_each:
                extra_needed = n_each - len(sel_m)
                sel_f = females[:min(n_each + extra_needed, len(females))]
            elif len(sel_f) < n_each:
                extra_needed = n_each - len(sel_f)
                sel_m = males[:min(n_each + extra_needed, len(males))]

            chosen = sel_m + sel_f

            # Fill remainder (1 extra if max_per_bin is odd)
            if remainder > 0 and len(chosen) < max_per_bin:
                remaining = [s for s in subjects if s not in chosen]
                rng.shuffle(remaining)
                chosen.extend(remaining[: max_per_bin - len(chosen)])

        for s in chosen:
            s["selected"] = True
        selected.extend(chosen)
        total_selected += len(chosen)

        n_m = sum(1 for s in chosen if s["sex"] == "male")
        n_f = sum(1 for s in chosen if s["sex"] == "female")
        avail_m = sum(1 for s in subjects if s["sex"] == "male")
        avail_f = sum(1 for s in subjects if s["sex"] == "female")
        capped = "Yes" if len(subjects) > max_per_bin else "No"
        print(f"{pma_bin:>5}w {len(subjects):>6} {avail_m:>3}/{avail_f:<3} {len(chosen):>4} {n_m:>4}/{n_f:<3} {capped:>7}")

    print(f"\nTotal available: {total_available}")
    print(f"Total selected: {total_selected}")

    # Write output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject", "session", "pma_bin", "scan_age_pma",
        "birth_age", "sex", "has_second_scan", "second_session",
        "second_scan_age",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(selected, key=lambda r: (r["pma_bin"], r["subject"])):
            second = second_scan_map.get(rec["subject"])
            writer.writerow({
                "subject": rec["subject"],
                "session": rec["session"],
                "pma_bin": rec["pma_bin"],
                "scan_age_pma": f"{rec['scan_age_pma']:.2f}",
                "birth_age": f"{rec['birth_age']:.2f}" if rec["birth_age"] is not None else "",
                "sex": rec["sex"],
                "has_second_scan": "yes" if second else "no",
                "second_session": second["session"] if second else "",
                "second_scan_age": f"{second['scan_age_pma']:.2f}" if second else "",
            })

    print(f"\nSelected subjects written to: {output_path}")
    print(f"Random seed used: {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
