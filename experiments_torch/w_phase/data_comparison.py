#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_w_phase_datasets.py

Compare old W-phase dataset (flat values+bases) with new per-basis files.

Expected layout:

Old (same directory as this script):
  - w_phase_state.txt
  - w_phase_unique_bases.txt
  - w_phase_meas_values.txt
  - w_phase_meas_bases.txt

New:
  - state_vectors/w_phase_state.txt
      * header lines starting with: STATE / HEADER
      * one "Re Im" line per amplitude
  - measurements/w_phase_<BASIS>_<shots>.txt
      * header lines starting with: STATE / MEASUREMENT
      * one encoded shot per line (e.g. "XxZz")

This script:
  1) Checks if the state vectors match line-by-line after skipping
     any line that starts with HEADER, STATE, or MEASUREMENT.
  2) Uses w_phase_unique_bases.txt + old counts to infer samples_per_basis.
  3) For each basis in that order, loads the per-basis file,
     skips header lines, decodes uppercase/lowercase to bits,
     and reconstructs "new_values/new_bases".
  4) Compares old vs new line-by-line and prints a summary plus
     some example mismatches.
"""

from pathlib import Path
import sys

BASE_DIR = Path(".")
MEAS_DIR = BASE_DIR / "measurements"
STATE_DIR = BASE_DIR / "state_vectors"

# Lines starting with these tokens are treated as headers and skipped
HEADER_PREFIXES = ("HEADER", "STATE", "MEASUREMENT")


# =========================
# Helpers
# =========================

def is_header_line(s: str) -> bool:
    """True if the (already stripped) line starts with a known header prefix."""
    return s.startswith(HEADER_PREFIXES)


def read_state_file(path: Path):
    """
    Read a state file, returning only non-empty, non-header lines.

    Works for:
    - Old format: no header, only amplitudes.
    - New format: header lines at top; amplitudes follow.
    """
    if not path.is_file():
        return None
    lines = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or is_header_line(s):
                continue
            lines.append(s)
    return lines


def compare_states():
    old_path = BASE_DIR / "w_phase_state.txt"
    new_path = STATE_DIR / "w_phase_state.txt"

    old_state = read_state_file(old_path)
    new_state = read_state_file(new_path)

    if old_state is None or new_state is None:
        print("[state] Missing state file(s), skipping state comparison.")
        return

    if len(old_state) != len(new_state):
        print(f"[state] DIFF length: old={len(old_state)}, new={len(new_state)}")
        return

    mismatches = [i for i, (o, n) in enumerate(zip(old_state, new_state)) if o != n]

    if not mismatches:
        print("[state] OK: w_phase_state.txt matches state_vectors/w_phase_state.txt (amplitudes).")
    else:
        print(f"[state] {len(mismatches)} / {len(old_state)} amplitude lines differ.")
        print("        Example indices:", mismatches[:5])


def load_unique_bases(path: Path):
    bases = []
    with path.open("r") as f:
        for line in f:
            tokens = line.split()
            if tokens:
                bases.append(tokens)
    return bases


def load_old_measurements(values_path: Path, bases_path: Path):
    values = []
    bases = []

    with values_path.open("r") as fv:
        for line in fv:
            tokens = line.split()
            if tokens:
                values.append([int(t) for t in tokens])

    with bases_path.open("r") as fb:
        for line in fb:
            tokens = line.split()
            if tokens:
                bases.append(tokens)

    if len(values) != len(bases):
        print(f"[old] ERROR: values ({len(values)}) and bases ({len(bases)}) length mismatch.")
        sys.exit(1)

    return values, bases


def decode_per_basis_line(line: str, basis):
    """
    Given e.g. basis = ['X','X','Z','Z'] and line = 'XxZz',
    return bits [0,1,0,1]. Uppercase -> 0, lowercase -> 1.
    """
    line = line.strip()
    if len(line) != len(basis):
        raise ValueError(f"Line '{line}' does not match basis length {len(basis)}")

    bits = []
    for c in line:
        if c.isupper():
            bits.append(0)
        elif c.islower():
            bits.append(1)
        else:
            raise ValueError(f"Unexpected character '{c}' in line '{line}'")
    return bits


def reconstruct_new_from_per_basis(unique_bases, samples_per_basis):
    if not MEAS_DIR.is_dir():
        print(f"[new] ERROR: measurements/ directory not found.")
        sys.exit(1)

    new_values = []
    new_bases = []

    for basis in unique_bases:
        basis_code = "".join(basis)  # e.g. ['X','X','Z','Z'] -> 'XXZZ'
        per_file = MEAS_DIR / f"w_phase_{basis_code}_{samples_per_basis}.txt"

        if not per_file.is_file():
            print(f"[new] ERROR: Expected file not found: {per_file}")
            sys.exit(1)

        # Read lines, skipping headers and empties
        lines = []
        with per_file.open("r") as f:
            for ln in f:
                s = ln.strip()
                if not s or is_header_line(s):
                    continue
                lines.append(s)

        if len(lines) != samples_per_basis:
            print(f"[new] WARNING: {per_file} has {len(lines)} data lines, expected {samples_per_basis}")

        for line in lines:
            bits = decode_per_basis_line(line, basis)
            new_values.append(bits)
            new_bases.append(basis)

    return new_values, new_bases


def compare_measurements():
    ub_path = BASE_DIR / "w_phase_unique_bases.txt"
    vals_path = BASE_DIR / "w_phase_meas_values.txt"
    bas_path = BASE_DIR / "w_phase_meas_bases.txt"

    if not (ub_path.is_file() and vals_path.is_file() and bas_path.is_file()):
        print("[compare] ERROR: Old-format files missing; cannot compare.")
        sys.exit(1)

    unique_bases = load_unique_bases(ub_path)
    old_values, old_bases = load_old_measurements(vals_path, bas_path)

    num_bases = len(unique_bases)
    total_shots = len(old_values)

    if total_shots % num_bases != 0:
        print(f"[old] ERROR: total_shots={total_shots} not divisible by num_bases={num_bases}")
        sys.exit(1)

    samples_per_basis = total_shots // num_bases

    print(f"[info] Unique bases: {num_bases}")
    print(f"[info] Total shots (old): {total_shots}")
    print(f"[info] Samples per basis (inferred): {samples_per_basis}")

    # Optionally validate old ordering against unique_bases
    for i, basis in enumerate(unique_bases):
        start = i * samples_per_basis
        end = start + samples_per_basis
        if not all(old_bases[j] == basis for j in range(start, end)):
            print(f"[old] WARNING: Basis block {i} does not match unique_bases ordering.")
            break

    # Reconstruct new-format as old-style view
    new_values, new_bases = reconstruct_new_from_per_basis(unique_bases, samples_per_basis)

    if len(new_values) != len(old_values):
        print(f"[compare] LENGTH MISMATCH: old={len(old_values)}, new={len(new_values)}")

    n = min(len(old_values), len(new_values))
    if n == 0:
        print("[compare] No data to compare.")
        return

    mismatches = []
    for i in range(n):
        if old_bases[i] != new_bases[i] or old_values[i] != new_values[i]:
            mismatches.append(i)
            if len(mismatches) >= 20:  # cap printed examples
                break

    if not mismatches and len(old_values) == len(new_values):
        print("[compare] OK: measurements match line-by-line.")
    else:
        total_mismatch = sum(
            1
            for i in range(n)
            if old_bases[i] != new_bases[i] or old_values[i] != new_values[i]
        )
        print(f"[compare] {total_mismatch} / {n} lines differ (basis and/or bits).")
        if mismatches:
            print("[compare] First mismatches (index: old_basis old_bits -> new_basis new_bits):")
            for i in mismatches:
                print(
                    f"  {i}: "
                    f"{''.join(old_bases[i])} {old_values[i]}  ->  "
                    f"{''.join(new_bases[i])} {new_values[i]}"
                )


def main():
    compare_states()
    compare_measurements()


if __name__ == "__main__":
    main()
