#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash apply_eval_custom_wall_time_patch.sh /path/to/dir/containing/eval_custom.py
#   bash apply_eval_custom_wall_time_patch.sh /path/to/eval_custom.py

TARGET="${1:-.}"
if [[ -d "$TARGET" ]]; then
  FILE="$TARGET/eval_custom.py"
else
  FILE="$TARGET"
fi

if [[ ! -f "$FILE" ]]; then
  echo "[ERROR] eval_custom.py not found: $FILE"
  exit 1
fi

python3 - "$FILE" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
original = text

if 'metrics["time"]["wall_time"]' in text:
    print(f"[INFO] Already patched: {path}")
    sys.exit(0)

# 1) Initialize wall_time once per instance loop.
pat1 = r"(\n\s*done = False\n)(\s*if config\.write_video:)"
rep1 = r"\1                wall_time = 0.0\n\2"
text, n1 = re.subn(pat1, rep1, text, count=1)

# 2) Accumulate per-step wall time.
pat2 = (
    r"(\n\s*time_start = time\.time\(\)\n"
    r"\s*terminated, truncated, info = evaluator\.step\(\)\n"
    r"\s*time_step = time\.time\(\) - time_start\n)"
)
rep2 = r"\1                    wall_time += time_step\n"
text, n2 = re.subn(pat2, rep2, text, count=1)

# 3) Persist wall_time in metrics["time"].
pat3 = (
    r"(\n\s*for metric in evaluator\.metrics:\n"
    r"\s*metrics\.update\(metric\.gather_results\(\)\)\n)"
)
rep3 = r'\1                metrics["time"]["wall_time"] = wall_time\n'
text, n3 = re.subn(pat3, rep3, text, count=1)

if not (n1 == 1 and n2 == 1 and n3 == 1):
    print("[ERROR] Patch failed: target code pattern not found (file may differ from expected).")
    print(f"[DEBUG] replacements: n1={n1}, n2={n2}, n3={n3}")
    sys.exit(2)

if text == original:
    print(f"[INFO] No changes made: {path}")
    sys.exit(0)

path.write_text(text)
print(f"[OK] Patched: {path}")
PY
