# Cheminformatics & ML

Personal self-taught project exploring Machine Learning applied to chemistry,
built alongside my Master's degree in Chemistry (ENS Paris-Saclay).

## Objective

Explore how ML can be combined with chemical data and molecular descriptors
to predict physicochemical properties and biological activity of molecules from
their structure. The project moves progressively from physicochemical endpoints
to structure-activity relationships and a deployed prediction app. All models are
trained on publicly available datasets and evaluated with rigorous protocols
(cross-dataset validation, scaffold splits, ablation studies, applicability-domain
checks), with explicit reporting of the ceilings imposed by the data.

## Projects

### 01 — Aqueous Solubility Prediction
`notebooks/01_esol_solubility.ipynb`

Prediction of aqueous solubility (logS, mol/L) from molecular structure using
RDKit descriptors and Morgan fingerprints on the ESOL benchmark dataset [1].

**Training protocol:**
All models were trained using an 80/20 random train/test split (random_state=42
for reproducibility). No validation set was used for hyperparameter tuning in
this initial benchmark — default hyperparameters were applied throughout.
The 6 molecular features used are: LogP, molecular weight, number of H-bond
donors and acceptors, topological polar surface area (TPSA), and number of
aromatic rings. Morgan fingerprints (radius=2, 2048 bits) were added in a
second step and concatenated with the 6 descriptors to form a 2054-dimensional
feature vector.

**Internal benchmark (ESOL test set, n=226):**
| Model | R² | RMSE (log mol/L) |
|---|---|---|
| Linear Regression | 0.765 | 1.054 |
| Random Forest | 0.859 | 0.816 |
| RF + Morgan Fingerprints (r=2, 2048 bits) | 0.866 | 0.795 |
| XGBoost + Morgan Fingerprints | 0.879 | 0.757 |

**Cross-dataset validation (trained on AqSolDB\ESOL, tested on ESOL):**
| Training set | n (train) | Test set | n (test) | R² | RMSE |
|---|---|---|---|---|---|
| ESOL | 902 | ESOL | 226 | 0.879 | 0.757 |
| AqSolDB\ESOL | 9378 | ESOL | 226 | **0.906** | **0.668** |

**Discussion:**
The XGBoost model achieves R²=0.879 and RMSE=0.757 log mol/L on the ESOL
test set, consistent with published results on this benchmark: Delaney [1]
reported RMSE=0.89 with a linear model, and more recent deep learning
approaches achieve RMSE≈0.58 [5]. Our result sits between these two benchmarks,
which is expected given the limited feature set (6 physicochemical descriptors
+ 2D fingerprints) and absence of hyperparameter optimization.

The most predictive individual feature is LogP (lipophilicity), consistent
with the known negative correlation between hydrophobicity and aqueous
solubility. TPSA and H-bond donor/acceptor counts capture polar interactions
that promote solvation. The addition of Morgan fingerprints provides marginal
improvement (+0.7% R²), suggesting that the 6 physicochemical descriptors
already capture most of the relevant variance for this dataset.

Cross-dataset validation reveals that a model trained on the larger AqSolDB
dataset [2] (9378 molecules after removal of ESOL overlap) generalizes better
to external data (R²=0.906, RMSE=0.668), despite lower internal metrics.
This highlights a key principle: internal R² is insufficient to assess
true predictive power — external validation on leak-free datasets is essential.

Performance could be further improved by: (i) incorporating solvation energy
from DFT calculations (shown by Boobier et al. [3] to be the most predictive
descriptor for aqueous solubility), (ii) adding experimental melting point as
a proxy for lattice energy, and (iii) using graph neural networks (GNNs) to
learn structural features directly from molecular graphs rather than
hand-crafted descriptors.

---

### 02 — Organic Solvent Solubility Prediction
`notebooks/02_organic_solvent_solubility.ipynb`

Prediction of logS in three organic solvents (ethanol, benzene, acetone) using
the dataset from Boobier et al. [3]. Three feature sets compared: RDKit
descriptors + Morgan fingerprints, DFT-derived descriptors (14 descriptors
selected from B3LYP/6-31+G(d) calculations), and their combination.

**Training protocol:**
Each solvent was modelled independently. An 80/20 train/test split was applied
per solvent (random_state=42). The 14 DFT descriptors used are those retained
by Boobier et al. after correlation analysis: MW, MP (melting point), molar
volume, solvation free energy (ΔG_solv), dipole moment in solution (solv_dip),
orbital interaction energies (LsoluHsolv, LsolvHsolu), solvent-accessible
surface area (SASA), and partial charge descriptors (O_charges, C_charges,
Het_charges, Most_neg, Most_pos). DFT values were pre-computed at the
B3LYP/6-31+G(d) level with IEFPCM solvation (Gaussian 09 [4]) by Boobier
et al. and used directly from their published dataset.

**XGBoost results by solvent and feature set:**
| Feature set | Ethanol R² | Benzene R² | Acetone R² |
|---|---|---|---|
| RDKit + Morgan Fingerprints | 0.491 | 0.465 | 0.296 |
| DFT descriptors (Boobier et al.) | 0.547 | 0.620 | 0.476 |
| DFT + Morgan Fingerprints | 0.597 | 0.677 | 0.531 |

**Discussion:**
R² values for organic solvents (0.30–0.68) are substantially lower than for
aqueous solubility. Three factors explain this:

1. **Dataset size**: each solvent contains only 370–553 training molecules,
compared to >9000 for the aqueous model. Small datasets increase variance and
limit generalization.

2. **Experimental noise**: Boobier et al. [3] report that ethanol and acetone
data are particularly noisy due to water contamination and solvent volatility,
making R² and RMSE less reliable metrics in these cases. Their own ET models
achieve R²=0.50 (ethanol) and R²=0.42 (acetone), consistent with our results.

3. **Missing descriptors**: melting point is identified by Boobier et al. as
the single most important descriptor for organic solvent solubility, as it
reflects the lattice energy of the solid. It is included in the DFT feature
set here but was not available for the RDKit-only model. Its high importance
explains why the DFT model outperforms the RDKit model particularly in benzene
(+15% R²), where solute-solute interactions dominate.

The benzene model performs best (R²=0.677), consistent with [3], likely
because benzene interactions are dominated by well-captured van der Waals
forces. Performance could be improved by expanding the training set to
additional solvents and incorporating conformational averaging of DFT
descriptors.

---

### 03 — Hydration Free Energy & Lipophilicity Benchmarks
`notebooks/03_freesolv_lipophilicity.ipynb`

Benchmarking the exact same XGBoost + ECFP4 pipeline (unchanged, not re-optimized)
on two further MoleculeNet properties, to test how well the approach generalizes
across different molecular endpoints: hydration free energy (FreeSolv [6], n=642,
ΔG_hyd in kcal/mol) and the octanol/water distribution coefficient (Lipophilicity,
logD at pH 7.4, n=4200).

**Training protocol:**
Identical featurization and models to notebook 01 (2054-dimensional vectors;
Linear Regression, Random Forest, XGBoost with default hyperparameters).
Evaluation uses both RepeatedKFold (5x5, mean ± std) and a Bemis-Murcko scaffold
split. Parameters are deliberately not re-optimized so that any difference
reflects the property and data, not tuning.

**RepeatedKFold results:**
| Dataset | Model | RMSE | R² |
|---|---|---|---|
| FreeSolv (kcal/mol) | Linear Regression | 1.37 | 0.868 |
| | Random Forest | 1.24 | 0.894 |
| | **XGBoost** | **1.12** | **0.913** |
| Lipophilicity (logD) | Linear Regression | 1.20 | ~0.00 |
| | Random Forest | 0.69 | 0.666 |
| | **XGBoost** | **0.705** | **0.656** |

**Scaffold split vs published GNN benchmarks:**
| Dataset | This work (XGBoost, scaffold) | MPNN [7] | D-MPNN [8] | AttentiveFP [9] |
|---|---|---|---|---|
| FreeSolv | RMSE 3.22 | 1.40 | 1.37 | 0.736 |
| Lipophilicity | RMSE 0.815 | 0.672 | 0.555 | 0.578 |

**Discussion:**
Protocol matters. Under random cross-validation, XGBoost on FreeSolv (RMSE 1.12)
already beats the MPNN and D-MPNN benchmarks; but those benchmarks use scaffold
splitting, so the fair comparison is the scaffold number (3.22), which trails all
GNNs. Comparing across protocols would be misleading. On Lipophilicity the pipeline
is genuinely competitive with graph networks.

FreeSolv is the hardest endpoint despite describing a physically simpler process
(gas-to-water, no crystal lattice), because the dataset is small and its
experimental values carry a noise floor of ~0.6 kcal/mol; the best GNN
(AttentiveFP, 0.736) already operates near that floor. A revealing contrast:
linear regression is strong on FreeSolv (R²=0.868, ΔG_hyd nearly additive in the
descriptors) but collapses on logD (R²≈0), which is governed by non-linear,
substructure-specific effects — the same pipeline ranks its models differently
depending on the property, which is the central argument for benchmarking across
several endpoints.

A transfer experiment (Option B) added a logD value predicted by the Lipophilicity
model as an extra feature to the ESOL solubility model. It produced no improvement
(R² 0.898 to 0.899), because the predicted logD is largely redundant (r=0.876) with
the MolLogP descriptor already present — more features help only when they add
non-redundant information.

---

### 04 — From Fixed Fingerprints to Learned Representations (MLP & GNN)
`notebooks/04_neural_network.ipynb`

Notebook 03 showed that on a fixed fingerprint representation, the choice of model
is not the limiting factor. This notebook tests the alternative — learning the
representation from the molecular graph — on the Lipophilicity dataset, building an
MLP and a graph neural network from first principles.

**Training protocol:**
The MLP (PyTorch) is trained on the standardized 2054-dimensional features with
Adam, dropout and weight decay, and early stopping on a validation set. The GNN
(PyTorch Geometric) consumes the molecular graph (one-hot-encoded atom features)
through message-passing layers, global pooling and a linear head, trained with
masked MSE and early stopping. The final ablation is replicated over 5 random
seeds and reported as mean ± std.

**Key results (Lipophilicity, validation RMSE):**
| Model | RMSE |
|---|---|
| MLP on fingerprints (regularized) | ~0.76 |
| GNN, minimal (GCN, 6 raw atom features) | 1.14 |
| **GNN, improved (GraphConv, rich features)** | **0.589 ± 0.024** |

**Ablation study (2x2, 5 seeds, RMSE):**
| Model | Old features (6) | Rich features (29) |
|---|---|---|
| GCN (minimal) | 1.156 ± 0.035 | 0.885 ± 0.056 |
| GraphConv (improved) | 0.876 ± 0.044 | 0.589 ± 0.024 |

**Discussion:**
On the fixed fingerprint, the regularized MLP only matches XGBoost (0.705),
confirming that the representation, not the model, sets the ceiling. A naive GNN
underfits (uniformly high train and validation error), but enriching the atom
features and the architecture brings it to 0.589 ± 0.024, competitive with
published graph networks (D-MPNN 0.555, AttentiveFP 0.578). The replicated ablation
shows that richer features and a stronger architecture each lower the error by
about 0.28 RMSE, roughly equally and additively; an apparent interaction seen on a
single split did not survive replication, a reminder to weigh effects against
noise. A useful diagnostic recurs throughout: a large train/validation gap
indicates overfitting (regularize), while uniformly high error indicates
underfitting (enrich the model).

---

### 05 — Single-Target QSAR (BACE1 & hERG)
`notebooks/05_qsar_single_target.ipynb`

Moving from physicochemical properties to biological activity (pIC50), which is
assay-dependent and noisier. Two therapeutically relevant targets: BACE1
(Alzheimer's disease; MoleculeNet, n=1513) [10] and hERG (cardiotoxicity; curated
from ChEMBL, n=8348) [11].

**Training protocol:**
Same pipeline as before with one added descriptor (number of rotatable bonds,
2055-dimensional vector), motivated by the role of conformational flexibility in
binding. Models: Linear Regression, Random Forest, XGBoost. Evaluation: RepeatedKFold
and Bemis-Murcko scaffold split; hERG is additionally framed as a binary
blocker/non-blocker classification (pIC50 > 5) evaluated by ROC-AUC. hERG data were
curated from ChEMBL (IC50 only, exact relation, biochemical assay type, nanomolar
units, median deduplication per molecule).

**Regression results:**
| Dataset | Model | Protocol | RMSE | R² |
|---|---|---|---|---|
| BACE1 | XGBoost | RepeatedKFold | 0.707 | 0.72 |
| | XGBoost | Scaffold split | 0.847 | 0.40 |
| | RF benchmark [8] | scaffold | 1.07 | - |
| | D-MPNN [8] | scaffold | 0.791 | - |
| hERG | Random Forest | RepeatedKFold | 0.578 | 0.58 |
| | Random Forest | Scaffold split | 0.683 | 0.32 |

**hERG classification:** ROC-AUC = 0.70 (threshold pIC50 > 5).

**Discussion:**
On the matched (scaffold) protocol, XGBoost on BACE1 (RMSE 0.847) beats the
published random-forest benchmark (1.07) and approaches the D-MPNN graph network
(0.791) — a strong result for a non-optimized classical pipeline. Linear regression
collapses on BACE1 (pocket-specific, non-linear SAR) but not on hERG, whose blockade
has a partly linear dependence on lipophilicity. hERG regression sits close to the
~0.5 log inter-laboratory noise floor, so its modest R² reflects data quality rather
than model inadequacy, and reframing it as classification does not create signal that
is not there.

An activity-cliff analysis found that 4.2% of structurally similar BACE1 pairs
(Tanimoto > 0.8) differ by more than 2 pIC50 units. Each such cliff is an error no
structure-based model can avoid, imposing a hard ceiling. Consistently,
leakage-aware hyperparameter tuning (search on a training split, evaluation on an
untouched test set) improved BACE1 RMSE only marginally (0.697 to 0.675), confirming
that the data, not the model, is the limiting factor.

---

### 06 — Multi-Target Activity Profiling
`notebooks/06_multitarget_qsar.ipynb`

Predicting a compound's activity profile across nine protein targets simultaneously,
for selectivity and off-target (polypharmacology) assessment. Data for nine targets
(hERG, BACE1, EGFR, CDK2, JAK2, PARP1, BRD4, Aurora A, PIK3CA) were curated from
ChEMBL [12] (~58,000 molecule-target records).

**Training protocol:**
Each target's IC50 data were curated and converted to pIC50 as in notebook 05.
A key data-integrity step: every target ChEMBL identifier was verified against its
preferred name before modelling — several identifiers initially used resolved to the
wrong protein and were corrected. Two strategies were compared: Approach A trains one
XGBoost per target; Approach B trains a single XGBoost on all molecule-target pairs
with a one-hot target encoding. Both are evaluated on held-out data (stratified by
target for B).

**Approach A (per-target) vs Approach B (unified), held-out R²:**
| Target | A (per-target) | B (unified one-hot) |
|---|---|---|
| BRD4 | 0.758 | 0.633 |
| PARP1 | 0.722 | 0.498 |
| PIK3CA | 0.664 | 0.445 |
| JAK2 | 0.659 | 0.516 |
| BACE1 | 0.654 | 0.513 |
| CDK2 | 0.653 | 0.380 |
| EGFR | 0.625 | 0.489 |
| Aurora A | 0.553 | 0.316 |
| hERG | 0.528 | 0.392 |

**Discussion:**
The per-target ensemble outperforms the unified one-hot model on all nine targets.
A one-hot flag is too weak a mechanism to let a single tree model serve proteins with
genuinely different structure-activity relationships: it settles on a compromise that
fits no target as well as its dedicated model. Multi-target learning is not free — it
requires related targets or a mechanism that learns a shared representation with
target-specific outputs (a multi-task graph neural network), identified here as the
principled next step and left as future work.

A qualitative validation on known drugs behaved sensibly: gefitinib (an EGFR
inhibitor) ranked EGFR as its top predicted target, and aspirin scored low
throughout (a negative control). It also exposed a calibration bias — the Aurora A
model predicts high activity for almost any molecule — showing that cross-target
ranking is only meaningful when per-target models are comparably calibrated, a
further argument for a jointly trained model.

The per-target ensemble was deployed in the interactive app (see below).

---

### Streamlit App — Solubility & Target Activity Predictor
`app.py`

Interactive web application with two tabs, predicting from a SMILES string:

- **Solubility**: aqueous logS and solubility in three organic solvents. Water model
  trained on AqSolDB\ESOL (n=9378, XGBoost + Morgan fingerprints).
- **Target Activity**: a biological activity profile (pIC50) across the nine protein
  targets of notebook 06, colour-coded by activity level, with a dedicated hERG
  cardiotoxicity flag (high predicted hERG activity is a safety risk, the opposite of
  an efficacy target).

Both tabs include a Tanimoto-based applicability-domain indicator (Morgan
fingerprints, radius=2) to flag predictions outside the training domain, and a
disclaimer that predictions are for research purposes only, not clinical advice.

```bash
conda activate chemml
cd path/to/cheminformatics
python -m streamlit run app.py
```

---

## Tech Stack

- **RDKit** (2023.09) — molecular manipulation, descriptor calculation, fingerprints
- **scikit-learn** — Linear Regression, Random Forest, model selection, metrics
- **XGBoost** — gradient boosting regressor
- **PyTorch / PyTorch Geometric** — MLP and graph neural network (notebook 04)
- **chembl_webresource_client** — bioactivity data extraction (notebooks 05-06)
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **Streamlit** — interactive web application

## References

[1] Delaney, J.S. ESOL: Estimating aqueous solubility directly from molecular
structure. *J. Chem. Inf. Comput. Sci.* **44**, 1000–1005 (2004).
https://doi.org/10.1021/ci034243x

[2] Sorkun, M.C., Khetan, A. & Er, S. AqSolDB, a curated reference set of
aqueous solubility and 2D descriptors for a diverse set of compounds.
*Sci. Data* **6**, 143 (2019). https://doi.org/10.1038/s41597-019-0151-1

[3] Boobier, S., Hose, D.R.J., Blacker, A.J. & Nguyen, B.N. Machine learning
with physicochemical relationships: solubility prediction in organic solvents
and water. *Nat. Commun.* **11**, 5753 (2020).
https://doi.org/10.1038/s41467-020-19594-z

[4] Frisch, M.J. et al. Gaussian 09, Revision D.03. Gaussian, Inc.,
Wallingford CT (2016).

[5] Lusci, A., Pollastri, G. & Baldi, P. Deep architectures and deep learning
in chemoinformatics: the prediction of aqueous solubility for drug-like
molecules. *J. Chem. Inf. Model.* **53**, 1563–1575 (2013).
https://doi.org/10.1021/ci400187y

[6] Mobley, D.L. & Guthrie, J.P. FreeSolv: a database of experimental and
calculated hydration free energies, with input files. *J. Comput.-Aided Mol. Des.*
**28**, 711–720 (2014). https://doi.org/10.1007/s10822-014-9747-x

[7] Wu, Z. et al. MoleculeNet: a benchmark for molecular machine learning.
*Chem. Sci.* **9**, 513–530 (2018). https://doi.org/10.1039/C7SC02664A

[8] Yang, K. et al. Analyzing learned molecular representations for property
prediction. *J. Chem. Inf. Model.* **59**, 3370–3388 (2019).
https://doi.org/10.1021/acs.jcim.9b00237

[9] Xiong, Z. et al. Pushing the boundaries of molecular representation for drug
discovery with the graph attention mechanism. *J. Med. Chem.* **63**, 8749–8760
(2020). https://doi.org/10.1021/acs.jmedchem.9b00959

[10] Subramanian, G. et al. Computational modeling of beta-secretase 1 (BACE-1)
inhibitors using ligand-based approaches. *J. Chem. Inf. Model.* **56**, 1936–1949
(2016). https://doi.org/10.1021/acs.jcim.6b00290

[11] Sanguinetti, M.C. & Tristani-Firouzi, M. hERG potassium channels and cardiac
arrhythmia. *Nature* **440**, 463–469 (2006). https://doi.org/10.1038/nature04710

[12] Zdrazil, B. et al. The ChEMBL Database in 2023: a drug discovery platform
spanning multiple bioactivity data types and time periods. *Nucleic Acids Res.*
**52**, D1180–D1192 (2024). https://doi.org/10.1093/nar/gkad1004

## Reproduce the environment

```bash
conda create -n chemml python=3.10
conda activate chemml
conda install -c conda-forge rdkit
pip install scikit-learn pandas numpy matplotlib seaborn jupyter xgboost streamlit chembl_webresource_client torch torch_geometric
```

## Author

**Nicolas Couret** — M2 Ingénierie Moléculaire du Vivant (IMoV), Sorbonne Université,
as part of my Chemistry curriculum at ENS Paris-Saclay.
A self-taught cheminformatics portfolio (developed with AI assistance) exploring
machine learning for drug discovery.

Contact: [nicolas.couret@ens-paris-saclay.fr](mailto:nicolas.couret@ens-paris-saclay.fr) · [LinkedIn](https://www.linkedin.com/in/nicolas-couret-97b78x/)