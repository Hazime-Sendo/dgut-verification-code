# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hazime Sendo
"""Checks for the provisional linear ODE of DGUT 1.0 (Provisional Hypothesis P1):

    dZ/dt = a Z + b conj(Z) - gamma Z ,   a in iR, b in C, gamma > 0 .

Checks Remarks 1-3 of latex/dgut10.tex and Lambda(Phi_t) = ||exp(tM)||.
Only the provisional hypothesis is examined; Pi_Q, chi and the phase map are not.
"""
import argparse
import json
import os
import sys
import numpy as np
from scipy.linalg import expm

_all_ok = True
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "ode_examples.json")


def check(name, ok, info=""):
    global _all_ok
    _all_ok &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    return ok


def make_M(a, b, gamma):
    """Real 2x2 matrix of M: Z -> (a-gamma) Z + b conj(Z), with Z = x + i y."""
    alpha = complex(a) - gamma
    b = complex(b)
    A, B = alpha + b, alpha - b
    return np.array([[A.real, -B.imag], [A.imag, B.real]])


def f(Z, a, b, g):
    return a * Z + b * np.conj(Z) - g * Z


def lam(M, t=1.0):
    """Lambda(Phi_t) = operator 2-norm of exp(tM) (largest singular value)."""
    return np.linalg.svd(expm(M * t), compute_uv=False)[0]


def run(seed=1):
    global _all_ok
    _all_ok = True
    rng = np.random.default_rng(seed)
    with open(DATA, encoding="utf-8") as fh:
        ex = json.load(fh)
    c = lambda p: complex(p[0], p[1])

    # 0. make_M agrees with f
    bad = 0
    for _ in range(200):
        a = 1j * rng.normal(); b = complex(rng.normal(), rng.normal()); g = abs(rng.normal()) + 0.01
        Z = complex(rng.normal(), rng.normal())
        v = make_M(a, b, g) @ [Z.real, Z.imag]
        bad += abs(complex(*v) - f(Z, a, b, g)) > 1e-12
    check("make_M agrees with Z -> (a-gamma)Z + b conj(Z)", bad == 0)

    # 1. Remark 1
    bad = 0
    for _ in range(500):
        a2 = rng.normal(); b = complex(rng.normal(), rng.normal()); g = abs(rng.normal()) + 0.01
        ev = np.sort_complex(np.linalg.eigvals(make_M(1j * a2, b, g)))
        s = np.sqrt(complex(abs(b) ** 2 - a2 ** 2))
        bad += np.abs(ev - np.sort_complex(np.array([-g + s, -g - s]))).max() > 1e-8
    check("eigenvalues = -gamma +- sqrt(|b|^2 - a2^2) (real and complex-pair cases)", bad == 0)
    e1 = ex["remark1_growth_despite_gamma"]
    ev = np.sort(np.linalg.eigvals(make_M(c(e1["a"]), c(e1["b"]), e1["gamma"])).real)
    check("Remark 1 example: eigenvalue +1 although gamma>0", np.allclose(ev, np.sort(e1["expected_eigenvalues"])), f"(eigenvalues {ev})")
    bad = 0
    for _ in range(2000):
        a2 = rng.normal(); b = abs(rng.normal()) * 2; g = abs(rng.normal()) + 0.01
        grow = np.linalg.eigvals(make_M(1j * a2, b, g)).real.max() > 1e-9
        pred = (b ** 2 - a2 ** 2 > g ** 2) and (b >= abs(a2))
        bad += grow != pred
    check("growth iff |b|^2 - a2^2 > gamma^2 (real-root case)", bad == 0)

    # 2. Remark 2
    e2 = ex["remark2_nonnormal_transient"]
    M = make_M(c(e2["a"]), c(e2["b"]), e2["gamma"])
    n1 = lam(M)
    check("Remark 2 example: Re(eig) = -0.1 and ||e^M|| = 1.166 > 1",
          np.allclose(np.linalg.eigvals(M).real, e2["expected_real_part"]) and abs(n1 - e2["expected_norm_t1"]) < e2["norm_tolerance"],
          f"({n1:.4f})")
    ts = np.linspace(0, 40, 4001)
    ns = np.array([lam(M, t) for t in ts])
    tc = ts[np.argmax(ns < 1.0)] if (ns < 1).any() else float("nan")
    print(f"      max ||e^(tM)|| = {ns.max():.3f} at t={ts[ns.argmax()]:.2f}; first t with value < 1: {tc:.2f}")
    Zs, Ws = rng.normal(size=(2, 100000)) + 1j * rng.normal(size=(2, 100000))
    E = expm(M)
    img = lambda Z: E @ np.vstack([Z.real, Z.imag])
    r = np.hypot(*(img(Zs) - img(Ws))) / np.abs(Zs - Ws)
    check("sampled Lipschitz ratio <= ||e^M|| and close to it", r.max() <= n1 + 1e-9 and r.max() > 0.95 * n1, f"(max={r.max():.4f})")

    # 3. Remark 3
    def commute(a, b, g):
        Z = rng.normal(size=20) + 1j * rng.normal(size=20)
        return np.allclose(np.conj(f(Z, a, b, g)), f(np.conj(Z), a, b, g))

    def reverse(a, b, g):
        Z = rng.normal(size=20) + 1j * rng.normal(size=20)
        return np.allclose(np.conj(f(Z, a, b, g)), -f(np.conj(Z), a, b, g))
    badc = badr = 0
    for _ in range(3000):
        a2 = rng.choice([0.0, rng.normal()]); g = rng.choice([0.0, abs(rng.normal()) + 0.01])
        b = rng.choice([complex(rng.normal(), 0), complex(0, rng.normal()), complex(rng.normal(), rng.normal())])
        badc += commute(1j * a2, b, g) != ((a2 == 0) and (b.imag == 0))
        badr += reverse(1j * a2, b, g) != ((g == 0) and (b.real == 0))
    check("iota o f = f o iota  iff  a=0 and b in R", badc == 0)
    check("iota o f = -f o iota  iff  gamma=0 and b in iR  (never for gamma>0)", badr == 0)
    print("\nNote: linear-algebra properties of the provisional ODE only; not a verification of DGUT, Pi_Q or the phase map.")
    print("ALL", "PASS" if _all_ok else "FAIL")
    return 0 if _all_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    sys.exit(run(ap.parse_args().seed))
