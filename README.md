# Cheminformatics & ML

Projet personnel d'apprentissage du Machine Learning appliqué à la chimie.
Démarche entièrement autodidacte, construite en parallèle de mon Master de chimie (ENS Paris-Saclay).

## Objectif

Explorer comment le ML peut être couplé à des données chimiques et des descripteurs moléculaires
pour prédire des propriétés physicochimiques de molécules à partir de leur structure.

## Projets

### 01 — Prédiction de solubilité aqueuse (ESOL)
`notebooks/01_esol_solubility.ipynb`

Prédiction du logS (solubilité aqueuse) de 1128 molécules à partir de descripteurs moléculaires
calculés avec RDKit. Comparaison d'une régression linéaire et d'un Random Forest.

**Résultats :**
| Modèle | R² | RMSE |
|---|---|---|
| Régression linéaire | 0.765 | 1.054 |
| Random Forest | 0.859 | 0.816 |
| RF + Morgan fingerprints | 0.866 | 0.795 |

## Stack technique

- **RDKit** — manipulation de molécules et calcul de descripteurs depuis SMILES
- **scikit-learn** — modèles ML (régression linéaire, Random Forest)
- **pandas / numpy** — manipulation de données
- **matplotlib / seaborn** — visualisation

[📓 Voir le notebook](https://tub-aspique.github.io/cheminformatics/notebooks/01_esol_solubility.html)

## Reproduire l'environnement

```bash
conda create -n chemml python=3.10
conda activate chemml
conda install -c conda-forge rdkit
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```
