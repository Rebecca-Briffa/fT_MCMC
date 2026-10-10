# Constraining \(f(T)\) cosmology with RSD

Used in R. Briffa, C. Escamilla-Rivera, J. Levi Said, and J. Mifsud, *Growth of structures using redshift space distortion in \(f(T)\) cosmology*, Mon. Not. R. Astron. Soc. **528**, 2711 (2024), [arXiv:2310.09159](https://arxiv.org/abs/2310.09159).

Public copy of the late-time MCMC used to constrain teleparallel \(f(T)\) models, and ΛCDM, with cosmic chronometers, Pantheon+SH0ES, BAO, and redshift-space distortions (\(f\sigma_8\)).

Each model keeps its own solver, growth equation, and likelihood. \(\chi^2\) uses the same surveys. \(H(z)\) and \(r_s\) come from that model’s `solver.py` / `util.py`. ΛCDM has no extra \(f(T)\) parameter.

## Models

| Package | Original folder | Model | Run |
| --- | --- | --- | --- |
| `models/ft_plm/` | `PLM/` | power-law, \(E^{2p}\) | `python scripts/run_cc_pnp_bao_fs8.py` |
| `models/lm/` | `LM/` | exponential \(f(T)\) | `python scripts/run_lm_cc_pnp_bao_fs8.py` |
| `models/lm2/` | `LM2/` | second exponential form | `python scripts/run_lm2_cc_pnp_bao_fs8.py` |
| `models/logm/` | `LogM/` | logarithmic form | `python scripts/run_logm_cc_pnp_bao_fs8.py` |
| `models/lcdm/` | `LCDM/` | ΛCDM, no extra \(p\) | `python scripts/run_lcdm_cc_pnp_bao_fs8.py` |

Sampled parameters are \(H_0\), \(\Omega_m\), the extra \(f(T)\) parameter where the model has one, \(\sigma_8\), and the SN absolute magnitude \(M\).

## Setup

```bash
python -m pip install -r requirements.txt
```

Copy the survey files listed in `data/README.md` into `data/`. Pantheon+ is large and is not committed.

From the repository root, start with a short chain by lowering `niter` in the script you run.

## What goes into the posterior

Each `log_probability` builds \(H(z)\) on a redshift grid, integrates the comoving distance, solves the growth ODE, then adds:

- cosmic chronometers, \(H(z)\)
- Pantheon+SH0ES, distance moduli with Cepheid hosts anchoring \(M\)
- BAO (BOSS DR12, eBOSS, DR14 quasars)
- \(f\sigma_8\), including the WiggleZ covariance

ΛCDM uses the combined RSD+BAO vector (`RSD_BAO_data.txt` with `fs8_RSD_cov_tot.txt`) instead of the separate \(f\sigma_8\) table.

## Two checks on the original likelihood

**\(f\sigma_8\) / WiggleZ.** `fs8_cov_tot.txt` is in the same row order as `f_sigma8_updated.csv`, not in redshift order. The only large off-diagonal block sits on the three consecutive WiggleZ rows \(z = 0.44,\ 0.60,\ 0.73\).

The Drive code did `sort_values("z")` and then `self.fs8[index]` inside `enumerate`. On a pandas Series that lookup is by label, so `self.fs8[0]` is the original first row, not the lowest redshift. In this file that accident puts the residual back in CSV order, which is the covariance order, so the saved \(\chi^2\) is not the scrambled one. Taking `.values` after the sort would pair those WiggleZ rows with the wrong covariance entries and can drive \(f\sigma_8\) \(\chi^2\) too low. The likelihood now stores the three columns with `.to_numpy()` and does not sort.

**Pantheon+SH0ES calibrators.** Hubble-flow supernovae enter at \(z_{\mathrm{HD}} > 0.01\). Cepheid hosts (`IS_CALIBRATOR`) are kept at any redshift, and \(M\) is set by `CEPH_DIST` on those rows. The same mask is applied to the covariance. A cut \(z_{\mathrm{HD}} > 0.01\) on the whole catalog would drop most hosts, because they sit below \(0.01\), and \(M\) would no longer be anchored.

## Citation

If you use this code, please cite

R. Briffa, C. Escamilla-Rivera, J. Levi Said, and J. Mifsud, *Growth of structures using redshift space distortion in \(f(T)\) cosmology*, Mon. Not. R. Astron. Soc. **528**, 2711 (2024), [arXiv:2310.09159](https://arxiv.org/abs/2310.09159), [doi:10.1093/mnras/stae103](https://doi.org/10.1093/mnras/stae103).

The survey files remain the property of those collaborations.
