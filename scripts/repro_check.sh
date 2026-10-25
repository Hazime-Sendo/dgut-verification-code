#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hazime Sendo
# Run twice with the fixed seeds and compare the logs byte for byte.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T1="$(mktemp -d)"; T2="$(mktemp -d)"
"$ROOT/scripts/run_all.sh" "$T1" >/dev/null
"$ROOT/scripts/run_all.sh" "$T2" >/dev/null
for f in elementary.log ode_check.log; do
  if cmp -s "$T1/$f" "$T2/$f"; then echo "REPRODUCED $f ($(sha256sum "$T1/$f" | cut -c1-16)...)"; else echo "DIFFERS $f"; exit 1; fi
done
