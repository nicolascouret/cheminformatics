import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
import subprocess, tempfile
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem

DOCK_DIR = os.environ.get("DOCK_DIR", os.path.expanduser("~/egfr_dock"))
RECEPTOR = f"{DOCK_DIR}/receptor.pdbqt"   # local default; cluster: DOCK_DIR=~/dock_run
REF_LIG  = f"{DOCK_DIR}/ligand.pdbqt"

def dock(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return (smi, None)
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, randomSeed=0) != 0: return (smi, None)
        AllChem.MMFFOptimizeMolecule(m)
        with tempfile.TemporaryDirectory() as d:
            sdf, pq, out = f"{d}/l.sdf", f"{d}/l.pdbqt", f"{d}/o.pdbqt"
            Chem.MolToMolFile(m, sdf)
            subprocess.run(["obabel", sdf, "-O", pq, "-p", "7.4",
                            "--partialcharge", "gasteiger"], capture_output=True)
            r = subprocess.run(["smina", "-r", RECEPTOR, "-l", pq,
                 "--autobox_ligand", REF_LIG, "--autobox_add", "4",
                 "--seed", "0", "--exhaustiveness", "8", "--cpu", "1",
                 "--num_modes", "1", "-o", out], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                p = line.split()
                if len(p) >= 2 and p[0] == "1":
                    return (smi, float(p[1]))
        return (smi, None)
    except Exception:
        return (smi, None)

if __name__ == "__main__":
    smis = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with Pool(int(os.environ.get("NCORES", os.cpu_count()))) as pool:
        rows = pool.map(dock, smis)
    out = sys.argv[1].replace(".smi", "_out.csv")
    with open(out, "w") as f:
        f.write("smiles,affinity\n")
        for smi, aff in rows:
            f.write(f"{smi},{'' if aff is None else aff}\n")
