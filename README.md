# NCA World — Kaggle runner (initial connectivity test)

This repository contains the **restricted** Kaggle-side worker for our JEPA + Neural Cellular Automata research. The initial job is only a PyTorch CUDA smoke test. **No training has been started yet.**

## Setup in Kaggle

1. Create a Kaggle Notebook; in **Settings** select **GPU T4 x2** (or another GPU) and enable **Internet**.
2. In GitHub, create a **fine-grained personal access token** restricted to this repository only:
   - Repository contents: **Read-only**
   - Issues: **Read and write**
   - Metadata: Read (automatic)
   Do not use a classic full-access token.
3. In Kaggle, open **Add-ons → Secrets** (the interface may vary); create a secret named `GITHUB_TOKEN`, paste the token there and enable notebook access. **Do not put the token in a cell, code, or chat.**
4. Paste the cell below and run it:

```python
from kaggle_secrets import UserSecretsClient
import base64
import os
import requests

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo = "crystalcali/nca-world-kaggle"
url = f"https://api.github.com/repos/{repo}/contents/agent.py"
response = requests.get(
    url,
    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    timeout=30,
)
response.raise_for_status()
agent_source = base64.b64decode(response.json()["content"]).decode("utf-8")
os.environ["GITHUB_TOKEN"] = token
exec(compile(agent_source, "agent.py", "exec"), {"__name__": "__main__"})
```

The agent polls GitHub for up to 10 minutes, executes **one** approved job, publishes results as a comment on that job's GitHub issue, and exits. Kaggle may reclaim idle sessions; start the cell again for new jobs. The agent does not run arbitrary commands received from GitHub Issues.

## Safety & scope

- The agent only accepts issues created by `crystalcali` with a title starting `[COLAB-JOB]`, body exactly `{"task": "gpu_smoke"}`, and without a prior claim/result comment.
- It does not run shell scripts or arbitrary issue commands. Extend the allowlist only after reviewing the implementation.
- A token scoped to this one repository limits damage if credentials are compromised. Revoke it when no longer needed.
- **Currently this repository is public.** Change it to private if you do not want source code or results to be visible to others.
- Colab/Kaggle GPU sessions are time-limited. This is not an always-on autonomous service.


## Experiment v1: synthetic GUI intervention

The checked-in `train_v1.py` implements three models with identical random-data generation, loss definition and held-out evaluation seeds:

| Model | Dynamic mechanism |
|---|---|
| `local` | NCA, click signal only at the control panel |
| `world` | NCA + latent encoder/GRU conditioned by actions + EMA JEPA-like latent prediction |
| `world_tiles` | As above, with nonlocal tile self-attention |

Input: two observed frames, one stepwise action signal. Targets: future video frames, including remote effects of toggling a control on the opposite side of the interface.

Metrics: overall frame MAE, panel MAE, foreground MAE, object position error, intervention-only object position error; short and long open-loop horizons.

**These models are not parameter-matched** (local approx 8k; others approx 44-47k). The pilot uses only 30 training iterations per model to validate end-to-end execution; do not interpret it as evidence that one architecture is better. Longer runs with more seeds and capacity-matched controls are required.

**To run:** after creating a new issue with exact title prefix `[COLAB-JOB]` and JSON body `{"task":"nca_pilot"}`, rerun the Kaggle notebook cell. The agent runs a fixed allowlisted training script and posts a JSON metrics report to that issue. Checkpoints and full loss history remain in the Kaggle filesystem under `/kaggle/working/nca_world/pilot_output`. The agent executes one job per notebook run.

Security: do not give the GitHub token `Contents: write` permission or post it in a notebook cell/chat. GitHub Issues do **not** grant arbitrary code execution; only the two fixed job names are handled.


## Experiment v2: distinguish dynamics from copying

train_v2.py strengthens the synthetic benchmark. Every episode has at most one distant button event, which reverses an object's motion. Evaluation compares paired futures with identical first frames but different future actions.

Additional metrics:
- copy_last_position_error_px — baseline that freezes the last observed object position.
- predicted_counterfactual_separation_px — model's response to a button action.
- true_counterfactual_separation_px — difference between ground-truth futures.

V2 pilot: 60 training iterations per model. V2 full run: 900 iterations per model. The model sizes are not matched. No claims of better generalization until longer runs and matched-capacity baselines.

Allowed GitHub issue bodies: {"task":"nca_v2_pilot"} or {"task":"nca_v2_train"}. The unchanged Kaggle notebook launcher processes one new task per run. Outputs and checkpoints stay in /kaggle/working/nca_world/v2_pilot or v2_train.

## Experiment v3 — anti-copying training

The v2 pilot found an almost frozen-frame solution. V3 preserves the same visual task and counterfactual testing, but adds supervised foreground segmentation and differentiable ball-position loss. Both losses use ground-truth object masks available from our synthetic simulator.

All models receive identical seeded training episodes; v2 did not fully control this. This is a more informative architecture test, not a claim of learned physics. JEPA is a small EMA latent predictor, not a pretrained universal JEPA.

V3 pilot: 160 steps per model and 14-frame long-horizon test. Full train: 1100 steps/model. Still not capacity matched. Agent accepted tasks: {"task":"nca_v3_pilot"}, {"task":"nca_v3_train"}. Run the original Kaggle launcher again to pick the next pending issue. Checkpoints are stored in /kaggle/working/nca_world/v3_pilot or v3_train.
