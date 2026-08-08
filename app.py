import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Solubility Predictor", page_icon="flask", layout="centered")
st.title("Molecular Solubility and Target Activity Predictor")
st.markdown("Enter a SMILES string to predict aqueous/organic solubility or a biological target activity profile.")


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
def calc_descriptors_6(smiles):
    """Six descriptors used by the solubility models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "LogP": Descriptors.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def calc_descriptors_7(smiles):
    """Seven descriptors used by the target-activity models (adds rotatable bonds)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = calc_descriptors_6(smiles)
    d["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
    return d


def calc_fingerprint(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits))


def featurize_solubility(smiles):
    fp = calc_fingerprint(smiles)
    desc = calc_descriptors_6(smiles)
    if fp is None or desc is None:
        return None
    return np.hstack([fp, list(desc.values())]).reshape(1, -1).astype(np.float32)


def featurize_activity(smiles):
    fp = calc_fingerprint(smiles)
    desc = calc_descriptors_7(smiles)
    if fp is None or desc is None:
        return None
    return np.hstack([fp, list(desc.values())]).reshape(1, -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading (cached so it runs only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_solubility_models():
    df_aqsol = pd.read_csv("data/curated-solubility-dataset.csv")
    df_aqsol = df_aqsol[["SMILES", "Solubility"]].rename(columns={"SMILES": "smiles", "Solubility": "logS"})

    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    smiles_esol = set(pd.read_csv(url)["smiles"].tolist())
    df_aqsol = df_aqsol[~df_aqsol["smiles"].isin(smiles_esol)]

    desc_list = df_aqsol["smiles"].apply(calc_descriptors_6).tolist()
    mask = [d is not None for d in desc_list]
    df_valid = df_aqsol[mask].reset_index(drop=True)
    desc_valid = pd.DataFrame([d for d in desc_list if d is not None])
    fp_valid = np.array([calc_fingerprint(s) for s in df_valid["smiles"]])

    X_water = np.hstack([fp_valid, desc_valid.values])
    y_water = df_valid["logS"].values
    model_water = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_water.fit(X_water, y_water)

    solvent_models = {}
    for solvent, fname in zip(
        ["Ethanol", "Benzene", "Acetone"],
        ["ethanol_solubility_data.csv", "benzene_solubility_data.csv", "acetone_solubility_data.csv"],
    ):
        df_s = pd.read_csv("data/Solubility Data/" + fname)
        desc_s = pd.DataFrame(df_s["SMILES"].apply(calc_descriptors_6).tolist())
        fp_s = np.array([calc_fingerprint(s) for s in df_s["SMILES"]])
        X_s = np.hstack([fp_s, desc_s.values])
        y_s = df_s["LogS"].values
        X_tr, _, y_tr, _ = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
        m = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        m.fit(X_tr, y_tr)
        solvent_models[solvent] = m

    return model_water, solvent_models, df_valid["smiles"].tolist()


@st.cache_resource
def load_target_models():
    models = joblib.load("models/target_models.joblib")
    ref_smiles = joblib.load("models/train_smiles.joblib")
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
               for s in ref_smiles if (m := Chem.MolFromSmiles(s)) is not None]
    return models, ref_fps


# ---------------------------------------------------------------------------
# Applicability domain (max Tanimoto to a reference set)
# ---------------------------------------------------------------------------
def tanimoto_max_sampled(smiles, ref_smiles, n=1000):
    q = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048)
    sample = random.Random(42).sample(list(ref_smiles), min(n, len(ref_smiles)))
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
           for s in sample if (m := Chem.MolFromSmiles(s)) is not None]
    sims = DataStructs.BulkTanimotoSimilarity(q, fps)
    return max(sims) if sims else 0.0


def tanimoto_max_fps(smiles, ref_fps):
    q = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048)
    sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
    return max(sims) if sims else 0.0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Solubility", "Target Activity"])

with tab1:
    model_water, solvent_models, sol_ref_smiles = load_solubility_models()
    smiles_sol = st.text_input("SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)", key="sol_smiles")

    if st.button("Predict Solubility", key="btn_sol"):
        x = featurize_solubility(smiles_sol) if smiles_sol else None
        if x is None:
            st.error("Invalid or empty SMILES.")
        else:
            sim = tanimoto_max_sampled(smiles_sol, sol_ref_smiles)
            if sim > 0.7:
                st.success("Applicability domain: max Tanimoto = " + str(round(sim, 2)) + " (reliable)")
            elif sim > 0.4:
                st.warning("Applicability domain: max Tanimoto = " + str(round(sim, 2)) + " (use with caution)")
            else:
                st.error("Applicability domain: max Tanimoto = " + str(round(sim, 2)) + " (outside training domain)")

            st.subheader("Results")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Water logS", round(float(model_water.predict(x)[0]), 3))
            with c2:
                st.markdown("**Descriptors**")
                st.dataframe(pd.DataFrame([calc_descriptors_6(smiles_sol)]).T.rename(columns={0: "Value"}))

            st.subheader("Organic solvents")
            cols = st.columns(3)
            for col, (solvent, m) in zip(cols, solvent_models.items()):
                with col:
                    st.metric(solvent + " logS", round(float(m.predict(x)[0]), 3))

with tab2:
    st.header("Target Activity Profile")
    st.markdown(
        "Predict biological activity (pIC50) against 9 protein targets. "
        "Disclaimer: predictions come from QSAR models trained on ChEMBL IC50 data. "
        "For research purposes only, not clinical advice."
    )
    smiles_bio = st.text_input("SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)", key="bio_smiles")

    if st.button("Predict Activity Profile", key="btn_bio"):
        x = featurize_activity(smiles_bio) if smiles_bio else None
        if x is None:
            st.error("Invalid or empty SMILES.")
        else:
            models, ref_fps = load_target_models()
            names = list(models.keys())
            preds = [float(models[n].predict(x)[0]) for n in names]

            sim = tanimoto_max_fps(smiles_bio, ref_fps)
            if sim < 0.3:
                st.warning("Out of applicability domain (max Tanimoto = " + str(round(sim, 2)) + "). Prediction unreliable.")

            ACTIVE, MODERATE = 6.0, 5.0

            def tier_color(name, p):
                if name == "hERG":
                    return "red" if p >= ACTIVE else "lightgray"   # hERG haut = danger
                if p >= ACTIVE:
                    return "seagreen"                              # actif probable (efficacite)
                if p >= MODERATE:
                    return "orange"                                # modere
                return "lightgray"                                 # inactif probable

            order = np.argsort(preds)
            names_s = [names[i] for i in order]
            preds_s = [preds[i] for i in order]
            colors = [tier_color(n, p) for n, p in zip(names_s, preds_s)]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(names_s, preds_s, color=colors, edgecolor="black")
            ax.axvline(ACTIVE, color="black", linestyle="--", linewidth=1)
            for b, p in zip(bars, preds_s):
                ax.text(p + 0.05, b.get_y() + b.get_height() / 2, str(round(p, 1)), va="center", fontsize=8)
            ax.set_xlabel("Predicted pIC50")
            ax.set_title("Predicted target activity profile")
            st.pyplot(fig)

            # verdict en texte
            likely = [n for n, p in zip(names, preds) if n != "hERG" and p >= ACTIVE]
            st.markdown("**Likely active (pIC50 >= 6):** " + (", ".join(likely) if likely else "none"))
            herg_p = preds[names.index("hERG")]
            if herg_p >= ACTIVE:
                st.error("hERG alert: predicted pIC50 = " + str(round(herg_p, 2)) + " (potential cardiotoxicity)")
            else:
                st.success("hERG: predicted pIC50 = " + str(round(herg_p, 2)) + " (low blockade risk)")
            st.caption("Green = likely active, orange = moderate, gray = likely inactive. hERG in red = toxicity risk.")