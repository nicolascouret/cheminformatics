"""HER2 anti-target docking surrogate (2D) — same recipe as refit_surrogate_2d,
trained on the HER2 docking labels (models/her2_dock_9600.csv).

Reuses the EGFR-tuned hyperparameters (a, b, lp, sn): same featurization and a
very similar docking-score distribution, so we transfer the HPs and just re-solve
alpha on the HER2 kernel. The 80/20 R2 is reported so you can sanity-check it vs
the EGFR 2D surrogate (R2=0.862). The saved bundle is format-compatible with
load_surrogate_2d(), so scoring reuses that loader with path=this bundle.
"""
import time, numpy as np, joblib
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from scale_surrogate import (fcfp_bits, phys_block, fit_sigma_L, predict_sigma,
                             MU_HP, SIGMA_HP, SEED)
from refit_surrogate_2d import fit_mu_2d, predict_mu_2d

REPO   = "/mnt/c/Users/coure/Projects/cheminformatics"
CSV    = f"{REPO}/models/her2_dock_9600.csv"
BUNDLE = f"{REPO}/models/surrogate_her2_2d.joblib"


def main():
    import pandas as pd
    t0 = time.time()
    df = pd.read_csv(CSV)
    smiles = df["smiles"].tolist()
    y = -df["affinity"].to_numpy(float)                 # higher = stronger HER2 binding
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in keep]; y = y[keep]
    n = len(mols)
    print(f"loaded {n} HER2-docked molecules")

    B = fcfp_bits(mols)
    Xphys = phys_block(mols)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n); ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]

    sp = StandardScaler().fit(Xphys[tr])
    Pt, Pe = sp.transform(Xphys[tr]), sp.transform(Xphys[te])
    Bt, Be = B[tr].astype(np.float32), B[te].astype(np.float32)

    alpha, mu_y = fit_mu_2d(Bt, Pt, y[tr], MU_HP)
    mu = predict_mu_2d(Be, Pe, Bt, Pt, alpha, mu_y, MU_HP)
    L = fit_sigma_L(B[tr], SIGMA_HP)
    sg = predict_sigma(B[te], B[tr], L, SIGMA_HP)
    print(f"HER2 2D | R2(mu)={r2_score(y[te], mu):.3f} | "
          f"Spearman(sigma,err)={spearmanr(sg, np.abs(mu - y[te])).correlation:.3f} "
          f"(EGFR 2D ref: 0.862)")
    del L

    spF = StandardScaler().fit(Xphys); PF = spF.transform(Xphys)
    alphaF, mu_yF = fit_mu_2d(B.astype(np.float32), PF, y, MU_HP)
    joblib.dump(dict(mu_hp=MU_HP, sigma_hp=SIGMA_HP, mu_y=float(mu_yF),
                     train_fcfp_bits=B, train_phys_std=PF,
                     phys_scaler=spF, alpha_mu=alphaF, mode="2d_her2"),
                BUNDLE, compress=3)
    print(f"saved -> {BUNDLE}   ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
