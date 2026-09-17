"""
run_notebook.py
===============
Executes analysis.ipynb top to bottom using nbformat / nbconvert / exec
to confirm it executes cleanly with 0 errors.
"""
import sys
import os
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
notebook_path = ROOT / "analysis.ipynb"

print(f"Reading notebook: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")

# Environment setup for execution
global_env = {}
cell_idx = 0
failed = False

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        cell_idx += 1
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"\n--- Executing Code Cell #{cell_idx} ---")
        first_line = source.strip().split("\n")[0]
        print(f"Snippet: {first_line[:70]}...")
        try:
            exec(source, global_env)
            print(f"Cell #{cell_idx} PASSED OK")
        except Exception as exc:
            print(f"ERROR in Code Cell #{cell_idx}: {exc}")
            traceback.print_exc()
            failed = True
            break

if not failed:
    print("\n" + "=" * 60)
    print("ALL NOTEBOOK CELLS EXECUTED CLEANLY WITH NO ERRORS!")
    print("=" * 60)
    sys.exit(0)
else:
    print("\n" + "=" * 60)
    print("NOTEBOOK EXECUTION FAILED!")
    print("=" * 60)
    sys.exit(1)
