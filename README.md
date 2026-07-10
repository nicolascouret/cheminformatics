# Cheminformatics & ML

Personal self-taught project exploring Machine Learning applied to chemistry,
built alongside my Master's degree in Chemistry (ENS Paris-Saclay).

## Objective

Explore how ML can be combined with chemical data and molecular descriptors
to predict physicochemical properties of molecules from their structure.

## Projects

### 01 — Aqueous Solubility Prediction (ESOL)
`notebooks/01_esol_solubility.ipynb`

Prediction of logS (aqueous solubility) for 1128 molecules using molecular descriptors
computed with RDKit. Comparison of Linear Regression, Random Forest, and XGBoost.

**Results:**
| Model | R² | RMSE |
|---|---|---|
| Linear Regression | 0.765 | 1.054 |
| Random Forest | 0.859 | 0.816 |
| RF + Morgan Fingerprints | 0.866 | 0.795 |
| XGBoost + Morgan Fingerprints | 0.879 | 0.757 |

### 02 — Organic Solvent Solubility Prediction
`notebooks/02_organic_solvent_solubility.ipynb`

Prediction of logS in three organic solvents (ethanol, benzene, acetone) using
a combination of RDKit descriptors, DFT-derived descriptors (Boobier et al., 2020)
and Morgan fingerprints.

**Results:**
| Approach | Ethanol R² | Benzene R² | Acetone R² |
|---|---|---|---|
| RDKit + Fingerprints | 0.491 | 0.465 | 0.296 |
| DFT descriptors (Boobier) | 0.547 | 0.620 | 0.476 |
| DFT + Fingerprints | 0.597 | 0.677 | 0.531 |

## Tech Stack

- **RDKit** — molecular manipulation and descriptor calculation from SMILES
- **scikit-learn** — ML models (Linear Regression, Random Forest)
- **XGBoost** — gradient boosting
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization

## Solubility Predictor App

Interactive app to predict aqueous and organic solvent solubility from a SMILES string.

```bash
conda activate chemml
cd path/to/cheminformatics
python -m streamlit run app.py
```

## References

Boobier et al. (2020). *Machine learning with physicochemical relationships: solubility prediction in organic solvents and water.* Nature Communications, 11, 5753.\\
Sorkun, M.C., Khetan, A. & Er, S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Sci Data 6, 143 (2019). https://doi.org/10.1038/s41597-019-0151-1

## Reproduce the environment

```bash
conda create -n chemml python=3.10
conda activate chemml
conda install -c conda-forge rdkit
pip install scikit-learn pandas numpy matplotlib seaborn jupyter xgboost
```