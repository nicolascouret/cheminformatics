import os, sys, random
import numpy as np, joblib
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED, RDConfig, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
RDLogger.DisableLog("rdApp.*")
 
REPO = "/mnt/c/Users/coure/Projects/cheminformatics"
ensemble = joblib.load(f"{REPO}/models/egfr_ensemble.joblib")
xtb_clf  = joblib.load(f"{REPO}/models/xtb_stability_clf.joblib")
LIPO     = joblib.load(f"{REPO}/models/lipophilicity.joblib")
ref = joblib.load(f"{REPO}/models/train_smiles.joblib")
ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
           for s in random.Random(42).sample(list(ref), 2000)
           if (m := Chem.MolFromSmiles(s)) is not None]
 
_p = FilterCatalogParams()
_p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
CATALOG = FilterCatalog.FilterCatalog(_p)
LAM = 2.0

##Docking surrogate
sys.path.append(REPO)                       # to import scale_surrogate
LAM_DOCK = 1.0                              # pessimism strength (mu - LAM_DOCK*sigma)
DOCK_S0, DOCK_W = 8.0, 1.0                  # sigmoid desirability center/width on -affinity scale
_SURR = None
def _get_surrogate():
    global _SURR
    if _SURR is None:
        from scale_surrogate import load_surrogate
        _SURR = load_surrogate()            # rebuilds sigma-GP Cholesky once (~2-3 min)
    return _SURR
def _dock_desir(s):
    return 1.0 / (1.0 + np.exp(-(s - DOCK_S0) / DOCK_W))
 
## Morgan fingerprints and physical descriptors
def compute_fp(mol, radius=2, nbits=2048):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
 
def feat(mol):
    if mol is None:
        return None
    fpb = compute_fp(mol)
    desc = [Descriptors.MolLogP(mol), Descriptors.MolWt(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol), Descriptors.NumRotatableBonds(mol)]
    return np.hstack([np.array(fpb), desc]).reshape(1, -1).astype(np.float32)
 
## Guardrails PAINS/Brenk
ALLOWED = {"C", "N", "O", "F", "S", "Cl", "Br"}
 
def guards(mol):
    if mol is None:
        return False
    elif any(a.GetSymbol() not in ALLOWED for a in mol.GetAtoms()):
        return False
    elif Descriptors.MolWt(mol) < 150 or Descriptors.MolWt(mol) > 600:
        return False
    elif abs(Chem.GetFormalCharge(mol)) > 1:
        return False
    elif CATALOG.HasMatch(mol):
        return False
    return True
 
## Activity desirability (pessimistic, mu - lambda*sigma)
def activity_desirability(mol):
    x = feat(mol)
    if x is None:
        return np.nan
    preds = np.array([m.predict(x)[0] for m in ensemble])
    mu, sigma = preds.mean(), preds.std()
    lcb = mu - LAM * sigma
    return float(np.clip(lcb / 10.0, 0.0, 1.0))
 
## Applicability domain (max Tanimoto to the training reference)
def ad(sim):
    if sim <= 0.3:
        return sim / 0.3
    elif sim >= 0.7:
        return max(0.0, (1 - sim) / 0.3)
    else:
        return 1.0
 
def ad_desirability(mol):
    fpb = compute_fp(mol)
    sim = max(DataStructs.BulkTanimotoSimilarity(fpb, ref_fps))
    return ad(sim)
 
## Synthetic accessibility
def sa_desirability(mol):
    SA = sascorer.calculateScore(mol)
    return (10 - SA) / 9
 
## Quantum viability (xTB stability surrogate)
def xtb_viability(mol):
    x = np.array(compute_fp(mol)).reshape(1, -1)
    p_unstable = xtb_clf.predict_proba(x)[0, 1]
    return 1.0 - float(p_unstable)
 
## Lipophilicity window (logD desirability)
def logd_window(v, lo=1.0, hi=3.0, margin=2.0):
    if v < lo:
        return max(0.0, 1 - (lo - v) / margin)
    if v > hi:
        return max(0.0, 1 - (v - hi) / margin)
    return 1.0
 
def logd_desirability(mol):
    pred = float(LIPO.predict(feat(mol))[0])
    return logd_window(pred)
 
## Docking desirability (single-molecule path, used by reward())
def dock_desirability(mol):
    mu, sg = _get_surrogate()([Chem.MolToSmiles(mol)])
    return float(np.nan_to_num(_dock_desir(mu[0] - LAM_DOCK * sg[0]), nan=0.0))

def dock_naive_desirability(mol):
    mu, _ = _get_surrogate()([Chem.MolToSmiles(mol)])
    return float(np.nan_to_num(_dock_desir(mu[0]), nan=0.0))

## Choosable reward
TERMS = {
    "activity":   activity_desirability,
    "qed":        lambda mol: QED.qed(mol),
    "ad":         ad_desirability,
    "sa":         sa_desirability,
    "xtb":        xtb_viability,
    "logd":       logd_desirability,
    "dock":       dock_desirability,
    "dock_naive": dock_naive_desirability,
}

def reward(smi, terms=("activity", "qed", "ad", "sa"), combine="geometric"):
    """Single-molecule objective (readable path). Does NOT touch the budget."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.nan
    if not guards(mol):
        return 0.0
    vals = np.array([TERMS[t](mol) for t in terms], dtype=float)
    if combine == "geometric":
        return float(np.prod(vals) ** (1.0 / len(vals)))
    elif combine == "product":
        return float(np.prod(vals))
    else:
        raise ValueError(f"combine inconnu : {combine}")

## Budget-aware batch API (what optimizers call) -- surrogate batched ONCE/batch
_CACHE = {}
_BUDGET = {"calls": 0}

def _score_batch(smiles, terms, combine, invalid_value):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    out = np.full(len(smiles), invalid_value, dtype=float)
    keep = [i for i, m in enumerate(mols) if m is not None and guards(m)]
    kset = set(keep)
    for i, m in enumerate(mols):                 # valid but blocked -> 0.0
        if m is not None and i not in kset:
            out[i] = 0.0
    if not keep:
        return out
    K = [mols[i] for i in keep]
    per_term = {}
    if "dock" in terms or "dock_naive" in terms:  # ONE surrogate call for the batch
        mu, sg = _get_surrogate()([Chem.MolToSmiles(m) for m in K])
        mu = np.nan_to_num(mu, nan=0.0); sg = np.nan_to_num(sg, nan=1.0)
        if "dock" in terms:
            per_term["dock"] = np.nan_to_num(_dock_desir(mu - LAM_DOCK * sg), nan=0.0)
        if "dock_naive" in terms:
            per_term["dock_naive"] = np.nan_to_num(_dock_desir(mu), nan=0.0)
    for t in terms:                               # light terms stay per-molecule
        if t not in per_term:
            per_term[t] = np.array([TERMS[t](m) for m in K], dtype=float)
    M = np.vstack([per_term[t] for t in terms])
    prod = np.prod(M, axis=0)
    vals = prod ** (1.0 / len(terms)) if combine == "geometric" else prod
    out[np.array(keep)] = vals
    return out

def score(smiles, terms=("activity", "qed", "ad", "sa", "xtb"),
          combine="geometric", invalid_value=float("nan")):
    if isinstance(smiles, str):
        smiles = [smiles]
    kt = tuple(terms)
    miss = [i for i, s in enumerate(smiles) if (s, kt, combine) not in _CACHE]
    if miss:
        vals = _score_batch([smiles[i] for i in miss], terms, combine, invalid_value)
        for j, i in enumerate(miss):
            _CACHE[(smiles[i], kt, combine)] = vals[j]
            _BUDGET["calls"] += 1
    return np.array([_CACHE[(s, kt, combine)] for s in smiles], dtype=float)

def n_oracle_calls():
    return _BUDGET["calls"]

def reset_budget():
    _BUDGET["calls"] = 0
    _CACHE.clear()