"""
inspect_mat.py - One-shot inspector for the structure of a single subject .mat file.

Loads the first available .mat from MAT_DIR_S1, prints (and writes to mat_structure.txt)
the type, shape, and (if a struct) field names of every key relevant to the GraphMind
pipeline. Run once per significant change in the source data.
"""

import os
import sys
import numpy as np
import scipy.io as sio

# Add repo root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAT_DIR_S1

KEYS_OF_INTEREST = [
    "rest_jhu",
    "fa_jhu", "md_jhu",
    "vbm_gm_jhu", "vbm_wm_jhu", "palf_jhu",
]


def describe(value, prefix=""):
    """Return a string describing `value` (its type, shape, fields)."""
    lines = []
    t = type(value).__name__
    if isinstance(value, np.ndarray):
        lines.append(f"{prefix}type=ndarray  shape={value.shape}  dtype={value.dtype}")
    elif hasattr(value, "_fieldnames"):
        lines.append(f"{prefix}type=mat_struct  fields={list(value._fieldnames)}")
        for fname in value._fieldnames:
            sub = getattr(value, fname)
            lines.append(describe(sub, prefix=prefix + f"  .{fname}: "))
    else:
        lines.append(f"{prefix}type={t}  value={value!r}")
    return "\n".join(lines)


def main():
    try:
        mat_files = sorted(f for f in os.listdir(MAT_DIR_S1) if f.endswith(".mat"))
    except FileNotFoundError:
        raise SystemExit(f"MAT directory not found: {MAT_DIR_S1}")
    if not mat_files:
        raise SystemExit(f"No .mat files in {MAT_DIR_S1}")
    target = os.path.join(MAT_DIR_S1, mat_files[0])
    print(f"Inspecting: {target}")
    mat = sio.loadmat(target, squeeze_me=True, struct_as_record=False)

    out_lines = [f"# .mat structure inspection — {mat_files[0]}", ""]
    out_lines.append("## Top-level keys")
    keys = sorted(k for k in mat.keys() if not k.startswith("__"))
    for k in keys:
        out_lines.append(f"  {k}")
    out_lines.append("")

    out_lines.append("## Keys of interest")
    for k in KEYS_OF_INTEREST:
        out_lines.append(f"### {k}")
        if k not in mat:
            out_lines.append("  MISSING")
        else:
            out_lines.append(describe(mat[k], prefix="  "))
        out_lines.append("")

    text = "\n".join(out_lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "mat_structure.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
