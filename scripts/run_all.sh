#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hazime Sendo
# Run both checks and write logs + environment info to results/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/results}"
mkdir -p "$OUT"
cd "$ROOT"
python3 -I src/dgut_elementary.py --seed 0 | tee "$OUT/elementary.log"
python3 -I src/dgut_ode_check.py  --seed 1 | tee "$OUT/ode_check.log"
python3 - <<'PY' > "$OUT/environment.txt"
import sys, platform, numpy, scipy
print("python", sys.version.split()[0]); print("platform", platform.platform())
print("numpy", numpy.__version__); print("scipy", scipy.__version__)
PY
( cd "$OUT" && sha256sum elementary.log ode_check.log > SHA256SUMS )
echo "logs written to $OUT"
