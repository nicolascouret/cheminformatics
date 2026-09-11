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
 
## Choosable reward
TERMS = {
    "activity": activity_desirability,
    "qed":      lambda mol: QED.qed(mol),
    "ad":       ad_desirability,
    "sa":       sa_desirability,
    "xtb":      xtb_viability,
    "logd":     logd_desirability,
}
 
def reward(smi, terms=("activity", "qed", "ad", "sa"), combine="geometric"):
    """Pure single-molecule objective. Does NOT touch the oracle budget."""
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
 
## Budget-aware batch API (what optimizers call)
# The "oracle" = one full scoring of a molecule. Repeats are cached and free,
# so n_oracle_calls() reflects unique molecules evaluated = the true budget.
_CACHE = {}
_BUDGET = {"calls": 0}
 
def score(smiles, terms=("activity", "qed", "ad", "sa", "xtb"),
          combine="geometric", invalid_value=float("nan")):
    if isinstance(smiles, str):
        smiles = [smiles]
    key_terms = tuple(terms)
    out = np.empty(len(smiles), dtype=float)
    for i, smi in enumerate(smiles):
        key = (smi, key_terms, combine)
        if key in _CACHE:
            out[i] = _CACHE[key]
            continue
        val = reward(smi, terms=terms, combine=combine)
        if isinstance(val, float) and np.isnan(val):
            val = invalid_value
        _CACHE[key] = val
        _BUDGET["calls"] += 1         
        out[i] = val
    return out
 
def n_oracle_calls():
    return _BUDGET["calls"]
 
def reset_budget():
    _BUDGET["calls"] = 0
    _CACHE.clear()