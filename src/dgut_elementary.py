# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hazime Sendo
"""Numerical illustrations of the elementary statements of DGUT (Part 0).

Scope: only statements that are defined/proved in the manuscript (Chapter 0 of
latex/dgut10.tex):
  [1] sign decomposition  Z=phi+i*pi, iota(Z)=conj(Z), A=<grad F(phi), pi>
  [2] mapping pressure Lambda (least Lipschitz constant): estimate and classes
  [3] left-nested recursion bound  Lambda(f)*max(1,Lambda(f))**(n-1)
  [4] pointwise lambda_f vs global Lambda(f): the counterexample f=sign on R\\{0}
  [5] damped oscillator as an *example* of Sigma_-/Sigma_+ regions (not the DGUT dynamics)
These are illustrations, not proofs, and say nothing about Pi_Q, the state
assignment chi, or the identification of phases (all placeholders).
"""
import argparse
import sys
import numpy as np

_all_ok = True


def check(name, ok, info=""):
    global _all_ok
    _all_ok &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    return ok


def pressure(f, dist, sampler, n=200000):
    """Lower estimate of the least Lipschitz constant (max of sampled ratios)."""
    x, y = sampler(n), sampler(n)
    d = dist(x, y)
    m = d > 1e-9
    return (dist(f(x), f(y))[m] / d[m]).max()


def run(seed=0):
    global _all_ok
    _all_ok = True
    rng = np.random.default_rng(seed)

    # [1] sign decomposition
    gradF = lambda phi: phi                      # F(phi)=phi^2/2 (example)
    A = lambda Z: gradF(Z.real) * Z.imag
    iota = np.conj
    Z = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    check("iota o iota = id", np.allclose(iota(iota(Z)), Z))
    check("A(iota Z) = -A(Z)", np.allclose(A(iota(Z)), -A(Z)))
    sm, sp, s0 = (A(Z) < 0).sum(), (A(Z) > 0).sum(), (A(Z) == 0).sum()
    check("iota swaps Sigma- and Sigma+",
          (A(iota(Z)) < 0).sum() == sp and (A(iota(Z)) > 0).sum() == sm,
          f"(|S-|={sm}, |S+|={sp}, |S0|={s0})")

    # [2] mapping pressure
    cplx = lambda n: rng.normal(size=n) * 2 + 1j * rng.normal(size=n) * 2
    dC = lambda a, b: np.abs(a - b)
    maps = {
        "conj (Lambda=1)": (np.conj, 1.0),
        "rotation exp(0.7i) z (Lambda=1)": (lambda z: np.exp(0.7j) * z, 1.0),
        "0.5 z (Lambda=0.5)": (lambda z: 0.5 * z, 0.5),
        "0.9 exp(0.7i) z (Lambda=0.9)": (lambda z: 0.9 * np.exp(0.7j) * z, 0.9),
        "2 z (Lambda=2)": (lambda z: 2 * z, 2.0),
        "z + 0.3 sin(Re z) (Lambda=1.3)": (lambda z: z + 0.3 * np.sin(z.real) + 0j, 1.3),
    }
    print("\n[2] mapping pressure estimates (sample max; lower estimate of the true value)")
    for k, (f, exact) in maps.items():
        L = pressure(f, dC, cplx)
        cls = "<1 (contraction)" if L < 0.999 else ("=1 (non-expanding)" if L < 1.001 else ">1 (expanding)")
        check(k, abs(L - exact) < 0.02, f"estimate={L:.4f} class {cls}")

    # [3] left-nested recursion; dyad f(a,b)=c1*a+c2*b, sum metric, Lambda(f)=max(|c1|,|c2|)
    print("\n[3] left-nested recursion: Lambda(f^[n]) <= Lambda(f)*max(1,Lambda(f))^(n-1)")

    def fn(ms, c1, c2):
        v = ms[0]
        for m in ms[1:]:
            v = c1 * v + c2 * m
        return v
    print("  (a) coordinate-wise ratio = analytic coefficient (the map is linear, so this is a re-check of the exact value);")
    print("  (b) random perturbation of all coordinates: sampled ratio must not exceed the bound.")
    for (c1, c2) in [(0.5, 0.3), (1.0, 0.2), (1.2, 0.5), (0.9, 1.6)]:
        Lf = max(abs(c1), abs(c2))
        for n in (1, 3, 6, 10):
            N = 20000
            X = rng.normal(size=(N, n + 1)) + 1j * rng.normal(size=(N, n + 1))
            k = int(np.argmax([abs(c1 ** n)] + [abs(c2 * c1 ** (n - j)) for j in range(1, n + 1)]))
            Y = X.copy()
            Y[:, k] += 1.0
            est = (np.abs(fn(X.T, c1, c2) - fn(Y.T, c1, c2)) / np.abs(X[:, k] - Y[:, k])).max()
            D = rng.normal(size=(N, n + 1)) + 1j * rng.normal(size=(N, n + 1))
            rnd = (np.abs(fn(X.T, c1, c2) - fn((X + D).T, c1, c2)) / np.abs(D).sum(axis=1)).max()
            bound = Lf * max(1, Lf) ** (n - 1)
            check(f"c1={c1}, c2={c2}, n={n}", est <= bound + 1e-9 and rnd <= bound + 1e-9,
                  f"coordinate ratio={est:.4f}, random-direction max={rnd:.4f} <= bound={bound:.4f}")

    # [4] pointwise vs global
    print("\n[4] counterexample: X=R\\{0}, f=sign.  pointwise lambda_f=0, global Lambda=inf")
    x = np.array([-1e-6, 1e-6])
    Lglob = abs(np.sign(x[1]) - np.sign(x[0])) / abs(x[1] - x[0])
    h = 1e-4
    lam = abs(np.sign(0.5 + h) - np.sign(0.5)) / h
    check("lambda_f(0.5)=0 while ratio across 0 diverges", lam == 0 and Lglob > 1e5,
          f"(local ratio={lam}, ratio across 0={Lglob:.2e})")

    # [5] damped oscillator, example only
    print("\n[5] example: phi'' = -phi - gamma*phi', pi=phi', F=phi^2/2, A=phi*pi=dF/dt")

    def sim(gamma, T=40, dt=1e-3):
        phi, p = 1.0, 0.0
        out = []
        for _ in range(int(T / dt)):
            p += (-phi - gamma * p) * dt
            phi += p * dt
            out.append((phi, p))
        return np.array(out)
    for gamma in (0.0, 0.2, 5.0):
        tr = sim(gamma)
        a = A(tr[:, 0] + 1j * tr[:, 1])
        E = 0.5 * tr[:, 0] ** 2 + 0.5 * tr[:, 1] ** 2
        sw = (np.diff(np.sign(a)) != 0).sum()
        print(f"  gamma={gamma}: S-:{(a < 0).mean():.2f} S+:{(a > 0).mean():.2f}  sign changes of A: {sw}, E: {E[0]:.3f}->{E[-1]:.3f}")
        if gamma == 0.0:
            check("gamma=0: E approximately conserved", abs(E[-1] - E[0]) < 5e-3)
        else:
            check(f"gamma={gamma}: E non-increasing at every step", np.diff(E).max() <= 1e-12, f"(max step change {np.diff(E).max():.2e})")
        if gamma == 5.0:
            check("overdamped (gamma=5): A never positive, no sign change", (a > 1e-12).sum() == 0 and sw == 0,
                  "(consistent with the Remark on overdamped motion; a numerical example only)")
    print("\nNote: numerical illustrations of elementary statements; not a verification of DGUT claims, Pi_Q or the phase map.")
    print("ALL", "PASS" if _all_ok else "FAIL")
    return 0 if _all_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(run(ap.parse_args().seed))
