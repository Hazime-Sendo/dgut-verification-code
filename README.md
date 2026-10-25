# DGUT verification code

Numerical illustrations accompanying the preprint note
*Dynamic Gaussian Plane Unified Field Hypothesis (DGUT): A Mathematical Construction of the Two-Layer Structure φ, α …*
(author: Hazime Sendo, ORCID 0009-0007-3285-0308).

**Important.** The provisional ODE is *not* part of the manuscript `part0` (whose Hypothesis 4.1 states no equation of motion); it is a separate, clearly labelled hypothesis P1 of the note "DGUT 1.0". The main parts of `part0` (chain model, Π_Q, irreversibility hypothesis) have no code-based verification here.

**Scope.** The code checks only (i) the elementary statements of Chapter 0 and (ii) the linear-algebra properties of the *provisional* linear ODE
`dZ/dt = aZ + b·conj(Z) − γZ` (a ∈ iℝ, b ∈ ℂ, γ > 0) of "DGUT 1.0" (`latex/dgut10.tex`).
It does **not** verify DGUT as a physical theory, the interface Π_Q, the state assignment χ, or any identification of the three phases; these are placeholders in the manuscript.
A PASS means the stated elementary property was reproduced numerically (examples/estimates, not proofs).

## Layout

| Path | Content |
|---|---|
| `src/dgut_elementary.py` | sign decomposition, mapping pressure Λ, left-nested recursion bound, pointwise-vs-global counterexample, damped-oscillator example |
| `src/dgut_ode_check.py` | provisional ODE: eigenvalue formula, growth despite γ>0, non-normal transient (‖e^{tM}‖), Λ(Φ_t)=‖e^{tM}‖, ι-symmetry conditions |
| `data/ode_examples.json` | parameter sets quoted in the Remarks |
| `scripts/run_all.sh` | run both checks; write logs and environment info to `results/` |
| `scripts/repro_check.sh` | run twice with fixed seeds and compare logs byte for byte |
| `results/` | logs of the reference run (`elementary.log`, `ode_check.log`, `environment.txt`, `SHA256SUMS`) |
| `latex/dgut10.tex`, `latex/dgut10.pdf` | "DGUT 1.0: elementary structure and provisional dynamics" (XeLaTeX) |

## Run

```bash
python3 -m pip install -r requirements.txt     # numpy, scipy
bash scripts/run_all.sh                        # writes results/*.log
bash scripts/repro_check.sh                    # optional: reproducibility check
python3 src/dgut_elementary.py --seed 0        # or run a single check
python3 src/dgut_ode_check.py  --seed 1
```

Exit status is 0 if all checks pass, 1 otherwise. Seeds are fixed (`--seed`), so logs are reproducible with the same library versions.
Reference environment: see `results/environment.txt` (Python see file at the time of writing).

## Build the note

`xelatex latex/dgut10.tex` (twice). Fonts: Noto Serif CJK JP (free); change `\setmainfont` if unavailable.

## Dependencies

Python ≥ 3.9; `numpy ≥ 1.24`, `scipy ≥ 1.10` (see `requirements.txt`). LaTeX: XeLaTeX with `fontspec`, `amsmath`, `amssymb`, `hyperref`, `geometry`.

## References

- R. A. Horn, C. R. Johnson, *Matrix Analysis*, Cambridge University Press (1985) — spectral (operator) norm / singular values used for Λ(Φ_t)=‖e^{tM}‖.

## Contact / licence

Contact for editors is provided through the submission portal, not in this repository.
Licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`).
