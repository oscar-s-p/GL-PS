# CLAUDE.md — GIGA-Lens Point-Source Modeling Project

## Project Overview

This project implements the **flux ratio variant** of the strongly lensed point-source modeling
pipeline built on top of [GIGA-Lens](https://github.com/giga-lens/giga-lens) (Gu et al. 2022,
ApJ 935, 49). The broader point-source pipeline methodology is described in Baltasar,
Ratier-Werbin, Huang et al. 2026 (arXiv:2601.18787); **this specific flux ratio extension is
the work of Ratier-Werbin et al. (in prep.)**.

### Primary Source File
```
src/flux_ratios_pipeline_funcs.py
```
All core pipeline functions for the flux ratio modeling live here. This is the main file to
read, edit, and test when working on the pipeline.

### Scope of This Work
The flux ratio approach extends the base pipeline to handle **non-Ia SNe and quasars**, which
lack a standardizable candle and therefore cannot use absolute magnifications as constraints.
Instead, **flux ratios between images** are used, which are independent of the intrinsic source
brightness. This is the natural observable for quasars (stochastic light curves) and non-Ia SNe.

The pipeline runs on JAX with GPU acceleration (targeting 4× A100s, e.g. NERSC Perlmutter).

The key distinction from the original GIGA-Lens is that **no pixel-level image comparison is
used**. Instead, the loss function operates entirely on analytic observables extracted from the
lensed point source: image positions, flux ratios, and time delays.

---

## Physics & Model

### Mass Model
The lens mass is described by the **Elliptical Power Law (EPL)** profile plus **external shear**:

```
κ(x, y) = (3 - γ)/2 * (θ_E / sqrt(x² + y²/q²))^(γ-1)
```

- `θ_E`: Einstein radius (arcsec)
- `γ`: mass density slope (physically motivated range: 1.5–2.5; isothermal = 2.0)
- `q`: axis ratio, reparametrized as eccentricities `(ε₁, ε₂)` for sampling:
  `(ε₁, ε₂) = (1-q)/(1+q) * (cos(2φ), sin(2φ))`
- `(x_lens, y_lens)`: lens centroid coordinates
- `(γ_ext,1, γ_ext,2)`: external shear components

Total free parameters per system: **9**
`[θ_E, γ, ε₁, ε₂, x_lens, y_lens, γ_ext,1, γ_ext,2, A]`

`A = f_true / f_obs` is the SN Ia amplitude normalization, prior `N(1, 0.1)`, encoding the
~10% standardizability of SN Ia peak luminosity.

### Observables Used as Constraints
1. **Image positions** `(x_i, y_i)` — centroids from PSF fitting
2. **Flux ratios** `R_ij = F_i / F_j` — ratios between image fluxes (no standardizable candle needed)
3. **Time delays** `Δt_ij` — relative arrival times between images

> **Note:** The amplitude parameter `A` used in the SN Ia pipeline (Baltasar+ 2026) is **not
> a free parameter here** — flux ratios are intrinsic-flux-independent by construction. This
> reduces the model to **8 free parameters** for systems where `A` is dropped, unless an
> alternative normalization is needed.

---

## Loss Function

The total log-likelihood is a weighted sum of three terms:

```
log L = ω_D * log L_D + ω_F * log L_F + ω_TD * log L_TD
```

### Term 1 — Source-Plane Compactness (`L_D`)
Delenses each observed image position to the source plane and minimizes scatter:
```
log L_D = -Σ_i |β(x_i, Θ) - β̄(x, Θ)|²
```
Alone, this term is too weak and biases γ → 1. **Must be combined with flux term.**

### Term 2 — Flux Ratio (`L_F`) ← **This project's focus**
The base pipeline (Baltasar+ 2026) fits absolute magnifications using a standardizable SN Ia
candle. **This project replaces that with flux ratios**, which are independent of intrinsic
source brightness — making the pipeline applicable to quasars and non-Ia SNe.

Flux ratios are defined as `R_ij = F_i / F_j = μ_i / μ_j` (intrinsic flux cancels).
The loss term fits model-predicted magnification ratios against observed flux ratios:
```
log L_F = -Σ_{i≠ref} | (μ_i(Θ) / μ_ref(Θ)) - R_i^obs |²  / σ²_{R_i}
```
Key implementation file: `src/flux_ratios_pipeline_funcs.py`

The decomposition trick for the magnification singularity (fitting `det(A)` directly rather
than `1/det(A)`) still applies — see the base paper §2.2.2 for the mathematical motivation.
The squaring to remove parity dependence also applies to ratios.

### Term 3 — Time Delay (`L_TD`)
Two modes:
- **Fixed H₀** (default for lens model inference): minimizes Fermat potential differences
- **Fitting H₀**: minimizes time delay differences directly, treating H₀ as a free parameter

Time delays scale with θ_E; systems with θ_E ≲ 0.3″ have sub-day delays with large relative
uncertainties — useful for convergence and γ accuracy but not for H₀ constraints.

---

## Inference Pipeline

Three sequential stages (same as original GIGA-Lens):

1. **MAP** — multi-start gradient descent to find Maximum a Posteriori
2. **SVI** — Stochastic Variational Inference for a Gaussian surrogate posterior
3. **HMC** — Hamiltonian Monte Carlo for full posterior sampling

**AD note:** Forward-mode automatic differentiation is used for magnification computation
(nested AD: AD applied inside the magnification calculation, then again for optimization
gradients). ~25/20/40% speedup over finite differences in MAP/SVI/HMC respectively.

---

## Prior Distributions (Default for Archetypal Systems)

| Parameter | Prior |
|-----------|-------|
| `θ_E` | `U(0.5, 2.0)` — adjust per system |
| `γ` | `TN(2, 0.25; 1.5, 2.5)` — truncated Gaussian |
| `ε₁, ε₂` | `N(0, 0.1)` |
| `x_lens, y_lens` | `N(0, 0.1)` |
| `γ_ext,1, γ_ext,2` | `N(0, 0.1)` |
| `A` | `N(1, 0.1)` |

For compact systems (e.g. SN Zwicky θ_E ≈ 0.17″, SN iPTF16geu θ_E ≈ 0.29″):
use `θ_E ~ U(0, 0.5)`.

**Bijectors** map unconstrained ℝ^d → constrained physical space for all bounded parameters,
maintaining full differentiability. The log-posterior includes the Jacobian correction:
```
log p̃(Θ̃ | data) = log L(data | g(Θ̃)) + log p̃(g(Θ̃)) + log|J(Θ̃)|
```

---

## Weight Tuning (`ω_D`, `ω_F`, `ω_TD`)

**Critical:** Naive observational uncertainty weights are not sufficient. The flux weight `ω_F`
typically needs to be increased ~10× above the estimate from `ω_F = 1/(2σ²_{1/μ²})`.

Procedure:
1. Estimate starting weights from observational uncertainties:
   - `ω_F = 1 / (2 * σ²_{1/μ²})` where `σ²_{1/μ²} = (2σ_μ / μ³)²`
   - `ω_TD = 1 / (2 * σ²_TD)`
   - `ω_D` via magnification truncation: `μ_trunc^{-1} * σ_obs = (1/ω_D)^{1/2}`
2. Run modeling; check if γ is underestimated (sign of too-low `ω_F`)
3. Increase `ω_F` incrementally until accurate recovery + convergence

Signs of poor weighting:
- γ posterior converging toward 1.5 (lower bound) → increase `ω_F`
- Poor position/flux recovery → recheck all weights
- No convergence with flux only → distance term needed too

---

## Convergence Criteria

Using **Vehtari et al. 2021** (more stringent than Gelman & Rubin 1992):
- **R̂ < 1.01** for all parameters (target; R̂ < 1.1 is minimum acceptable)
- **ESS ≥ 100 per chain**

Always set `cross_chain_dims` correctly in TensorFlow Probability to avoid ESS overestimation.

Typical runtimes on 4× A100 GPUs (NERSC Perlmutter):
- Achieving R̂ < 1.1: seconds to ~1 min
- Achieving R̂ < 1.01: ~5–60 min depending on configuration
- Double (2-image) systems are harder to converge than quads

---

## Image Configurations

Five archetypal configurations, determined by source position relative to caustic:

| Config | Images | Notes |
|--------|--------|-------|
| Cross | 4 | Source inside caustic; standard Einstein cross |
| Long-cusp | 4 | Source near long-axis cusp; isolated image has negative parity |
| Short-cusp | 4 | Source near short-axis cusp; isolated image has positive parity |
| Fold | 4 | Two images merging across critical curve |
| Double | 2 | Source outside caustic; reduced constraining power |

**Parity rule:** The first two arriving images must have positive parity (Narayan & Bartelmann 1996).
This is a useful sanity check on any best-fit model.

---

## Known Gotchas & Edge Cases

- **Magnification singularity:** Images near the critical curve cause `det(A) → 0`. The
  decomposition-based flux loss handles this, but be aware that near-critical images have
  highly uncertain predicted magnifications (physically correct — large µ sensitivity).

- **Double systems:** Only 8 data points (4 positions + fluxes), same 9 parameters → poorly
  constrained. Position recovery is less accurate; time delays and fluxes are fine.

- **γ < 2:** Inner caustic and critical curve appear. At γ = 1.5 with q = 1.0, the inner
  critical curve and caustic precisely coincide (newly reported, not in prior literature).

- **High ellipticity + steep γ:** Can produce "knots" in the caustic (astroid-like structure)
  without needing angular Fourier perturbations. SN Zwicky (ε₁ ≈ 0.31, ε₂ ≈ 0.33, γ ≈ 2.11)
  is an example.

- **Mass Sheet Degeneracy (MSD):** Not currently modeled. For SN Ia, the standardizable
  luminosity already constrains mass normalization and partially mitigates MSD. Full MSD
  component is a planned extension.

- **H₀ constraints require θ_E ≳ 1″**: Small Einstein radii → sub-day time delays →
  uncertainties comparable to signal → no meaningful H₀ constraint.

---

## Real Systems Modeled

### SN iPTF16geu
- `z_s = 0.409`, `z_d = 0.2163`
- Einstein cross, 4 images ~0.3″ from lens center
- `θ_E ≈ 0.289″`, `γ ≈ 1.92` (consistent with Mörtsell+ 2020 within 1σ)
- Data: positions + magnifications from Dhawan et al. 2019 (HST + Keck AO)
- Flux anomalies present → microlensing likely; smooth model insufficient for fluxes
- Prior on θ_E: `U(0, 0.5)`

### SN Zwicky (SN 2022qmx)
- `z_s = 0.3544`, `z_d = 0.2262`
- Einstein cross, 4 images ~0.17″ from lens center
- `θ_E ≈ 0.173″`, `γ ≈ 2.11`, **high ellipticity** q ≈ 0.38 (ε₁ ≈ 0.31, ε₂ ≈ 0.33)
- Data: positions + magnifications + time delays from Larison et al. 2024
- Images A and C near critical curve → brighter; B and D farther → dimmer
- **Alternative model** to Goobar+ 2023 and Pierel+ 2023; arrival order A first (positive parity)
- Prior on θ_E: `U(0, 0.5)`

---

## Input Data Format

Each system requires:
```python
{
    "image_positions": [(x1, y1), (x2, y2), ...],  # arcsec, lens-centric
    "flux_ratios": [R_1ref, R_2ref, ...],            # F_i / F_ref, relative to reference image
    "flux_ratio_uncertainties": [sigma1, sigma2, ...],
    "time_delays": [dt1, dt2, ...],                  # days, relative to earliest image
    "td_uncertainties": [sigma_td1, ...],
    "redshifts": {"lens": z_d, "source": z_s},
    "cosmology": "LCDM"                              # assumed for D_Delta
}
```

> For the SN Ia absolute magnification pipeline (Baltasar+ 2026), replace `flux_ratios` with
> `magnifications` + `mag_uncertainties` and add `"A_prior": (1.0, 0.1)`.

---

## Key References

- **GIGA-Lens original:** Gu et al. 2022, ApJ 935, 49
- **Point-source pipeline (base):** Baltasar, Ratier-Werbin, Huang et al. 2026, arXiv:2601.18787
- **Flux ratio pipeline (this work):** Ratier-Werbin et al. (in prep.)
- **SN iPTF16geu discovery:** Goobar et al. 2017; modeling: Mörtsell et al. 2020
- **SN Zwicky discovery:** Goobar et al. 2023; HST: Pierel et al. 2023; time delays: Larison et al. 2024
- **MSD:** Schneider & Sluse 2013
- **Convergence diagnostics:** Vehtari et al. 2021 (R̂ < 1.01 target)
- **H₀ from lensed quasars:** Wong et al. 2019 (H0LiCOW)
