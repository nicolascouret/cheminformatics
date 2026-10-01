"""Scale-up of the DECOUPLED docking surrogate to the full ~9600-molecule set.

Design (decided on the 1500-molecule subset):
  mu(smiles)    <- combined-kernel GP  (FCFP-Tanimoto + RBF phys + RBF 3D)   -> accurate mean
  sigma(smiles) <- mono-Tanimoto GP    (FCFP)                                -> calibrated uncertainty
  pessimistic reward downstream: R = mu - lambda * sigma
"""
import os, time, json
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

REPO   = "/mnt/c/Users/coure/Projects/cheminformatics"
CSV    = f"{REPO}/models/docking_seed.csv"
X3DCACHE = f"{REPO}/models/X3d_9600.npy"
BUNDLE = f"{REPO}/models/surrogate_9600.joblib"

MU_HP    = dict(a=0.26, b=37.95, c=4.54, lp=18.09, l3=5.54, sn=0.26)
SIGMA_HP = dict(sf=1.0, sn=0.5)
SEED = 0

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def fcfp_bits(mols):
    B = np.zeros((len(mols), 2048), dtype=np.uint8)
    for i, m in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useFeatures=True)
        B[i, list(fp.GetOnBits())] = 1
    return B

def phys_block(mols):
    def f(m):
        return [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                Descriptors.NumRotatableBonds(m),
                rdMolDescriptors.CalcNumAromaticRings(m),
                rdMolDescriptors.CalcFractionCSP3(m)]
    return np.array([f(m) for m in mols], float)

def shape_block(mols):
    out = np.full((len(mols), 6), np.nan)
    for i, m in enumerate(mols):
        mh = Chem.AddHs(m)
        if AllChem.EmbedMolecule(mh, randomSeed=SEED) != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mh)
            out[i] = [Descriptors3D.Asphericity(mh), Descriptors3D.Eccentricity(mh),
                      Descriptors3D.NPR1(mh), Descriptors3D.NPR2(mh),
                      Descriptors3D.RadiusOfGyration(mh), Descriptors3D.SpherocityIndex(mh)]
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            log(f"  3D embedding {i+1}/{len(mols)}")
    return out

def tanimoto(Ba, Bb, asum=None, bsum=None):
    inter = Ba @ Bb.T
    a = Ba.sum(1) if asum is None else asum
    b = Bb.sum(1) if bsum is None else bsum
    union = a[:, None] + b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float64)

def rbf(Xa, Xb, l):
    return np.exp(-cdist(Xa, Xb, "sqeuclidean") / (2.0 * l**2))

def combined_cross(Ba, Bb, Pa, Pb, Ta, Tb, hp):
    K = hp["a"] * tanimoto(Ba, Bb)
    K += hp["b"] * rbf(Pa, Pb, hp["lp"])
    K += hp["c"] * rbf(Ta, Tb, hp["l3"])
    return K

def fit_mu(Btr, Ptr, Ttr, ytr, hp):
    mu_y = ytr.mean()
    n = len(ytr)
    K = combined_cross(Btr, Btr, Ptr, Ptr, Ttr, Ttr, hp)
    K[np.diag_indices(n)] += hp["sn"]**2 + 1e-6
    L = cho_factor(K, lower=True, overwrite_a=True)
    alpha = cho_solve(L, ytr - mu_y)
    return alpha, mu_y

def fit_sigma_L(Btr, hp):
    n = Btr.shape[0]
    K = hp["sf"] * tanimoto(Btr.astype(np.float32), Btr.astype(np.float32))
    K[np.diag_indices(n)] = hp["sf"] + hp["sn"]**2 + 1e-6
    return cho_factor(K, lower=True, overwrite_a=True)

def predict_mu(Bnew, Pnew, Tnew, Btr, Ptr, Ttr, alpha, mu_y, hp):
    Ks = combined_cross(Bnew, Btr, Pnew, Ptr, Tnew, Ttr, hp)
    return Ks @ alpha + mu_y

def predict_sigma(Bnew, Btr, L, hp):
    Ks = hp["sf"] * tanimoto(Bnew.astype(np.float32), Btr.astype(np.float32))
    v = cho_solve(L, Ks.T)
    var = hp["sf"] - np.einsum("ij,ji->i", Ks, v)
    return np.sqrt(np.clip(var, 1e-9, None))

def main():
    t0 = time.time()
    import pandas as pd
    df = pd.read_csv(CSV)
    smiles = df["smiles"].tolist()
    y = -df["affinity"].to_numpy(float)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in keep]; y = y[keep]; smiles = [smiles[i] for i in keep]
    n = len(mols)
    log(f"loaded {n} valid molecules")

    log("FCFP + physchem ...")
    B = fcfp_bits(mols)
    Xphys = phys_block(mols)

    if os.path.exists(X3DCACHE):
        X3d = np.load(X3DCACHE)
        log(f"3D loaded from cache ({X3d.shape})")
    else:
        log("3D embedding (the long pole) ...")
        X3d = shape_block(mols)
        np.save(X3DCACHE, X3d)
        log(f"3D done, cached ({np.isnan(X3d).any(1).sum()} failures imputed)")

    col_mean = np.nanmean(X3d, axis=0)
    bad = np.isnan(X3d).any(1)
    X3d[bad] = np.where(np.isnan(X3d[bad]), col_mean, X3d[bad])

    log("evaluation split 80/20 ...")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n); ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]

    sp = StandardScaler().fit(Xphys[tr]); Pt = sp.transform(Xphys[tr]); Pe = sp.transform(Xphys[te])
    s3 = StandardScaler().fit(X3d[tr]);   Tt = s3.transform(X3d[tr]);   Te = s3.transform(X3d[te])
    Bt32, Be32 = B[tr].astype(np.float32), B[te].astype(np.float32)

    alpha, mu_y = fit_mu(Bt32, Pt, Tt, y[tr], MU_HP)
    mu = predict_mu(Be32, Pe, Te, Bt32, Pt, Tt, alpha, mu_y, MU_HP)
    L = fit_sigma_L(B[tr], SIGMA_HP)
    sg = predict_sigma(B[te], B[tr], L, SIGMA_HP)
    err = np.abs(mu - y[te])
    r2 = r2_score(y[te], mu); sp_c = spearmanr(sg, err).correlation
    log(f"SCALE EVAL | R2(mu)={r2:.3f} | Spearman(sigma,err)={sp_c:.3f} | sigma spread {sg.min():.2f}-{sg.max():.2f}")
    del L

    log("full fit on all molecules ...")
    spF = StandardScaler().fit(Xphys); PF = spF.transform(Xphys)
    s3F = StandardScaler().fit(X3d);   TF = s3F.transform(X3d)
    B32 = B.astype(np.float32)
    alphaF, mu_yF = fit_mu(B32, PF, TF, y, MU_HP)

    bundle = dict(
        mu_hp=MU_HP, sigma_hp=SIGMA_HP, mu_y=float(mu_yF),
        train_fcfp_bits=B, train_phys_std=PF, train_shape_std=TF,
        phys_scaler=spF, shape_scaler=s3F, alpha_mu=alphaF,
        eval=dict(r2=float(r2), spearman=float(sp_c), n=n),
    )
    joblib.dump(bundle, BUNDLE, compress=3)
    log(f"saved -> {BUNDLE}")
    log(f"TOTAL {(time.time()-t0)/60:.1f} min")
    print(json.dumps(bundle["eval"], indent=2))

def load_surrogate(path=BUNDLE):
    b = joblib.load(path)
    L = fit_sigma_L(b["train_fcfp_bits"], b["sigma_hp"])
    Btr = b["train_fcfp_bits"].astype(np.float32)
    Ptr, Ttr = b["train_phys_std"], b["train_shape_std"]
    def predict(smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        ok = [i for i, m in enumerate(mols) if m is not None]
        mm = [mols[i] for i in ok]
        Bn = fcfp_bits(mm).astype(np.float32)
        Pn = b["phys_scaler"].transform(phys_block(mm))
        X3n = shape_block(mm)
        X3n[np.isnan(X3n).any(1)] = 0.0
        Tn = b["shape_scaler"].transform(np.nan_to_num(X3n))
        mu = predict_mu(Bn, Pn, Tn, Btr, Ptr, Ttr, b["alpha_mu"], b["mu_y"], b["mu_hp"])
        sg = predict_sigma(Bn, b["train_fcfp_bits"], L, b["sigma_hp"])
        MU = np.full(len(smiles_list), np.nan); SG = np.full(len(smiles_list), np.nan)
        MU[ok] = mu; SG[ok] = sg
        return MU, SG
    return predict

if __name__ == "__main__":
    main()
