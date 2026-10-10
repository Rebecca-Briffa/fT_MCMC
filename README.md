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

## Citation

If you use this code, please cite

R. Briffa, C. Escamilla-Rivera, J. Levi Said, and J. Mifsud, *Growth of structures using redshift space distortion in \(f(T)\) cosmology*, Mon. Not. R. Astron. Soc. **528**, 2711 (2024), [arXiv:2310.09159](https://arxiv.org/abs/2310.09159), [doi:10.1093/mnras/stae103](https://doi.org/10.1093/mnras/stae103).

The survey files remain the property of those collaborations.
