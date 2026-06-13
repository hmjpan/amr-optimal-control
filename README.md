# AMR Optimal Control

**First-principles optimal control analysis of antimicrobial resistance using Pontryagin's Maximum Principle, calibrated to real ECDC surveillance data.**


---

## Overview

This repository contains the complete code, data pipeline, and analysis for the paper:

> *Optimal Antibiotic Use Policies for Carbapenem-Resistant Klebsiella pneumoniae: A Pontryagin Analysis of European Surveillance Data (2005-2024)*

**Core finding**: The resistant reproduction number R0_r defines a sharp bifurcation at R0_r = 1. Countries below this threshold can maintain higher antibiotic use; countries above must aggressively conserve. Italy, with R0_r = 1.13, sits at the bifurcation with genuine policy bimodality.

## Repository Structure

```
.
├── src/                          # Core library code
│   ├── core/amr_model.py         # Pontryagin solver, multi-start, S-I_s-I_r model
│   ├── inference/
│   │   ├── parameter_estimation.py  # MAP estimation (3 free parameters)
│   │   └── bootstrap.py          # Parametric bootstrap with 95% CI
│   ├── data/
│   │   ├── real_data_pipeline.py # Literature-calibrated data generation
│   │   └── data_pipeline.py      # ECDC download utilities
│   ├── validation/
│   │   ├── model_validation.py   # Cross-validation, counterfactual analysis
│   │   └── baseline_models.py    # Statistical baseline model comparison
│   └── viz/figures.py            # Publication-quality figure generation
│
├── experiments/                  # Reproducible analysis scripts
│   ├── final_analysis_v2.py      # MAIN: 8-country CRKP analysis (~3 min)
│   ├── real_data_analysis.py     # Real ECDC data pipeline
│   ├── bootstrap_scale.py        # 100-sample bootstrap for Italy (~18 min)
│   ├── sensitivity.py            # QALY weight sensitivity sweep (~3 min)
│   ├── verify_mathematics.py     # 23-point mathematical verification
│   ├── strict_audit.py           # Jacobian and adjoint correctness audit
│   ├── time_slice_validation.py  # Predictive validation (train 2006-2019, test 2023-2024)
│   ├── prediction_gap_analysis.py # Analysis of prediction gap causes
│   ├── diagnose_italy.py         # Italy bimodality diagnostic
│   ├── regenerate_figures.py     # Generate main publication figures
│   └── generate_word.py          # Generate MANUSCRIPT.docx with embedded figures
│
├── docs/math/                    # Mathematical derivations
│   ├── pkpd_derivation.py        # PK/PD derivation of alpha(u) from MSW hypothesis
│   └── derivations.py            # Symbolic mathematics (SymPy)
│
├── data/
│   ├── processed/ecdc_amr_real.csv  # Clean merged dataset (1953 records)
│   └── raw/ecdc/                    # Original ECDC CSV downloads
│       ├── K. pneumoniae_Carbapenems.csv
│       ├── Escherichia coli_Third-gen cephalosporins.csv
│       └── Staphylococcus aureus_ MRSA.csv
│
├── tests/test_core.py            # Unit tests (15 tests)
│
├── outputs/                      # Generated results
│   └── real_data/
│       ├── country_results.csv   # 8-country parameter estimates + optimal policies
│       ├── figures/              # Main figures (Fig1-Fig3) and supplementary figures (SupFig1-SupFig4)
│       ├── bootstrap/            # Bootstrap CI results (JSON)
│       └── sensitivity/          # Sensitivity analysis results
│
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License
```

## Data Sources

**Primary**: ECDC Surveillance Atlas (https://atlas.ecdc.europa.eu)
- K. pneumoniae carbapenem resistance: 547 observations, 30 countries, 2005-2024
- E. coli third-generation cephalosporin resistance: 690 observations
- S. aureus MRSA: 716 observations
- Downloaded: 11 June 2026

**Secondary** (consumption calibration): Klein EY et al. PNAS 2018; 115: E3463-E3470

The processed dataset is at `data/processed/ecdc_amr_real.csv`. Original ECDC CSV exports are preserved in `data/raw/ecdc/`.

## Installation

```bash
git clone https://github.com/hmjpan/amr-optimal-control.git
cd amr-optimal-control
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, NumPy 2.x, SciPy 1.x, Pandas, Matplotlib, scikit-learn, python-docx, requests

## Reproducing Results

All scripts are in the `experiments/` directory and should be run from the project root.

### Quick Start (core result, ~3 min)

```bash
python experiments/final_analysis_v2.py
```
Fits the model to 8 European countries, computes optimal policies, and compares against 4 alternative strategies (status quo, cycling, withdrawal, high-use). Outputs: `outputs/real_data/country_results.csv`.

### Full Reproducibility Pipeline

**1. Verify mathematical correctness (1 min)**
```bash
python experiments/verify_mathematics.py
```
23 tests checking: model equations, Pontryagin conditions, adjoint FD verification, cost convexity, numerical convergence. All must pass.

**2. Main analysis (3 min)**
```bash
python experiments/real_data_analysis.py
```
Loads real ECDC data, estimates parameters for 8 countries, computes optimal control, compares policies. Same as `final_analysis_v2.py` but with more diagnostic output.

**3. Bootstrap uncertainty quantification (~18 min)**
```bash
python experiments/bootstrap_scale.py
```
100 parametric bootstrap iterations for Italy CRKP. Outputs 95% confidence intervals for all parameters and policy outcomes. Expect ~72 valid samples.

**4. Sensitivity analysis (~3 min)**
```bash
python experiments/sensitivity.py
```
Sweeps the QALY weight ratio w_clinical/w_resistance from 0.05 to 0.50. Verifies that the R0_r bifurcation structure is preserved.

**5. Predictive validation (~2 min)**
```bash
python experiments/time_slice_validation.py
```
Trains on Italy 2006-2019, excludes 2020-2022, and predicts 2023-2024. Compares the mechanistic model against logistic, linear, persistence, and historical-mean baselines.

**6. Prediction gap analysis (~2 min)**
```bash
python experiments/prediction_gap_analysis.py
```
Quantifies how much of the post-2019 prediction gap is attributable to pandemic-period and post-pandemic Italian policy changes.

**7. Generate figures**
```bash
python experiments/regenerate_figures.py
```
Produces main figures `Fig1-Fig3` and supplementary figures `SupFig1-SupFig4` in `outputs/real_data/figures/` (300 DPI).


**8. Run tests**
```bash
python -m pytest tests/test_core.py -v
```
15 unit tests covering the core model, Pontryagin solver, and epidemiological quantities.

## Key Results (Table 3 from manuscript)

| Country  | R0_r  | u*     | Strategy     |
|----------|-------|--------|--------------|
| Greece   | 2.62  | 0.008  | CONSERVE     |
| Romania  | 1.62  | 0.014  | CONSERVE     |
| Spain    | 1.21  | 0.306  | CONSERVE     |
| Italy    | 1.13  | 0.836* | CONSERVE     |
| Germany  | 1.00  | 0.955  | TREAT        |
| France   | 0.97  | 0.957  | TREAT        |
| Sweden   | 0.92  | 0.974  | TREAT        |

*Italy: bootstrap median u*=0.041, 95% CI [0.003, 0.871] (bimodal at bifurcation)

See MANUSCRIPT.md for the full paper with all tables and references.

## Mathematical Methods (Summary)

The model uses Pontryagin's Maximum Principle to solve a two-point boundary value problem for the optimal time-varying antibiotic use policy u*(t) minimizing:

```
J(u) = w_T*I_r(T) + integral [w_r*I_r + w_s*I_s*(1-u) - w_c*u*I_s + w_u*u^2] dt
```

subject to the S-I_s-I_r compartment model with antibiotic-induced resistance acquisition alpha(u) = phi + sigma*u^2/(1+u^2). Full derivations are in SUPPLEMENTARY_APPENDIX.

