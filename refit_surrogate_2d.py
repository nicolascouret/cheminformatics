"""Refit the docking-mu surrogate WITHOUT the 3D shape block.

Motivation: the full surrogate (surrogate_9600.joblib) uses a combined kernel
    a*Tanimoto(FCFP) + b*RBF(phys) + c*RBF(shape3D). Predicting mu therefore needs
    a conformer per molecule (EmbedMolecule + MMFF), which is far too slow to serve
    as a GFlowNet reward called on every generated molecule, every iteration.

This builds a 2D-only mu:  a*Tanimoto(FCFP) + b*RBF(phys)  -> no conformers, fast.
    We reuse the hyperparameters already tuned for the 3D model (a, b, lp, sn) and
    simply re-solve alpha on the 2D kernel. sigma is unchanged (it was already a
    mono-Tanimoto / 2D GP). The 80/20 eval is reported on the SAME split/seed as
    scale_surrogate.py so R2 is directly comparable to the 0.87 of the 3D model.

If R2 drops a lot here, re-optimize (a, b, lp, sn) on the 2D kernel before using it.
"""
import os, time, json
import numpy as np
import joblib
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# reuse the exact featurizers / kernels / sigma-GP from the 3D script
from scale_surrogate import (fcfp_bits, phys_block, tanimoto, rbf,
                             fit_sigma_L, predict_sigma, MU_HP, SIGMA_HP, SEED)

REPO   = "/mnt/c/Users/coure/Projects/cheminformatics"
CSV    = f"{REPO}/models/docking_seed.csv"
BUNDLE = f"{REPO}/models/surrogate_2d.joblib"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 2D kernel: drop the c*RBF(shape3D) term -------------------------------
def cross_2d(Ba, Bb, Pa, Pb, hp):
    return hp["a"] * tanimoto(Ba, Bb) + hp["b"] * rbf(Pa, Pb, hp["lp"])


def fit_mu_2d(Btr, Ptr, ytr, hp):
    mu_y = ytr.mean()
    n = len(ytr)
    K = cross_2d(Btr, Btr, Ptr, Ptr, hp)
    K[np.diag_indices(n)] += hp["sn"] ** 2 + 1e-6
    L = cho_factor(K, lower=True, overwrite_a=True)
    alpha = cho_solve(L, ytr - mu_y)
    return alpha, mu_y


def predict_mu_2d(Bnew, Pnew, Btr, Ptr, alpha, mu_y, hp):
    Ks = cross_2d(Bnew, Btr, Pnew, Ptr, hp)
    return Ks @ alpha + mu_y


def main():
    t0 = time.time()
    import pandas as pd
    df = pd.read_csv(CSV)
    smiles = df["smiles"].tolist()
    y = -df["affinity"].to_numpy(float)                 # higher = better binding
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in keep]; y = y[keep]
    n = len(mols)
    log(f"loaded {n} valid molecules (no 3D -> fast)")

    log("FCFP + physchem ...")
    B = fcfp_bits(mols)
    Xphys = phys_block(mols)

    # same 80/20 split/seed as scale_surrogate.py -> comparable R2
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n); ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]

    sp = StandardScaler().fit(Xphys[tr])
    Pt, Pe = sp.transform(Xphys[tr]), sp.transform(Xphys[te])
    Bt32, Be32 = B[tr].astype(np.float32), B[te].astype(np.float32)

    alpha, mu_y = fit_mu_2d(Bt32, Pt, y[tr], MU_HP)
    mu = predict_mu_2d(Be32, Pe, Bt32, Pt, alpha, mu_y, MU_HP)
    L = fit_sigma_L(B[tr], SIGMA_HP)
    sg = predict_sigma(B[te], B[tr], L, SIGMA_HP)
    err = np.abs(mu - y[te])
    r2 = r2_score(y[te], mu)
    sp_c = spearmanr(sg, err).correlation
    log(f"2D EVAL | R2(mu)={r2:.3f} | Spearman(sigma,err)={sp_c:.3f} "
        f"| sigma {sg.min():.2f}-{sg.max():.2f}   (3D ref: R2=0.87)")
    del L

    log("full fit on all molecules ...")
    spF = StandardScaler().fit(Xphys); PF = spF.transform(Xphys)
    alphaF, mu_yF = fit_mu_2d(B.astype(np.float32), PF, y, MU_HP)

    bundle = dict(
        mu_hp=MU_HP, sigma_hp=SIGMA_HP, mu_y=float(mu_yF),
        train_fcfp_bits=B, train_phys_std=PF,
        phys_scaler=spF, alpha_mu=alphaF,
        eval=dict(r2=float(r2), spearman=float(sp_c), n=n), mode="2d",
    )
    joblib.dump(bundle, BUNDLE, compress=3)
    log(f"saved -> {BUNDLE}")
    log(f"TOTAL {(time.time()-t0)/60:.1f} min")
    print(json.dumps(bundle["eval"], indent=2))


def load_surrogate_2d(path=BUNDLE):
    """Fast predictor: (MU, SG) for a list of SMILES, NO conformer generation."""
    b = joblib.load(path)
    L = fit_sigma_L(b["train_fcfp_bits"], b["sigma_hp"])
    Btr = b["train_fcfp_bits"].astype(np.float32)
    Ptr = b["train_phys_std"]

    def predict(smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        ok = [i for i, m in enumerate(mols) if m is not None]
        mm = [mols[i] for i in ok]
        MU = np.full(len(smiles_list), np.nan)
        SG = np.full(len(smiles_list), np.nan)
        if mm:
            Bn = fcfp_bits(mm).astype(np.float32)
            Pn = b["phys_scaler"].transform(phys_block(mm))
            MU[ok] = predict_mu_2d(Bn, Pn, Btr, Ptr, b["alpha_mu"], b["mu_y"], b["mu_hp"])
            SG[ok] = predict_sigma(Bn, b["train_fcfp_bits"], L, b["sigma_hp"])
        return MU, SG

    return predict


if __name__ == "__main__":
    main()
