from pathlib import Path

ALL_TERMS = ["activity", "qed", "ad", "sa", "xtb", "logd"]
REPLICATES = [0, 1, 2, 3, 4]          
OUT = Path.home() / "REINVENT4" / "ablation_runs"
OUT.mkdir(exist_ok=True)

conditions = {"full": ALL_TERMS}
for t in ALL_TERMS:
    conditions[f"drop_{t}"] = [x for x in ALL_TERMS if x != t]

conditions["activity_only"] = ["activity"]

TEMPLATE = '''run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_logs"

[parameters]
summary_csv_prefix = "ablation_runs/{name}"
use_checkpoint = false
purge_memories = false
prior_file = "priors/reinvent.prior"
agent_file = "priors/reinvent.prior"
batch_size = 64
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[[stage]]
chkpt_file = "ablation_runs/{name}.chkpt"
termination = "simple"
max_score = 1.0
min_steps = 190
max_steps = 190

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.Scorer]
[[stage.scoring.component.Scorer.endpoint]]
name = "{name}"
weight = 1
[stage.scoring.component.Scorer.endpoint.params]
terms = "{terms_csv}"
combine = "geometric"
'''

n = 0
for cond, terms in conditions.items():
    for r in REPLICATES:
        name = f"abl_{cond}_s{r}"
        (OUT / f"{name}.toml").write_text(
            TEMPLATE.format(name=name, terms_csv=",".join(terms)))
        n += 1
print(f"{n} TOMLs générés dans {OUT}")   