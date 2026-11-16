# Developmental Motivation: Kaggle T4 experiment

A **falsifiable, functional** experiment on whether increasing the number of independent motivational signals, learning to regulate their priorities, and gradually introducing competing goals improves adaptation in a new environment. It does **not** test whether neural networks experience emotions, whether these variables are animal emotions, or whether emotion is necessary for general intelligence.

The code uses **only PyTorch, NumPy, pandas, and Matplotlib** (already available in Kaggle). No model downloads, API keys, datasets, Gym, transformers or network access are required. Kaggle notebook: **[`kaggle_emotional_development_t4.ipynb`](kaggle_emotional_development_t4.ipynb)**. The notebook contains its own complete code, so it can run offline after opening the notebook in Kaggle.

## One-minute launch (Kaggle)

1. Open `kaggle_emotional_development_t4.ipynb` in Kaggle (upload the notebook from this branch).
2. Notebook settings → Accelerator → **GPU T4 x2** (or T4). Internet may remain **Off**. The experiment uses **one** GPU (`cuda:0`) reliably; it does not claim to parallelize two cards.
3. Run all cells. The default `PRESET = "pilot"` runs all **five** comparison arms with one seed; it checks the workflow but cannot establish statistical significance.
4. Switch to `PRESET = "full"` for two or more seeds and longer training. Export figures, CSV logs, config and trained checkpoints from `/kaggle/working/emotional_development/<preset>/` (available as Kaggle outputs after saving a notebook version).

A Kaggle notebook needs **GPU enabled**, otherwise the first cell fails with instructions. For a local CPU smoke test set `REQUIRE_GPU=False` and `PRESET="smoke"` in the first cell. Do **not** infer research conclusions from the pilot.

## What is tested

| Arm | Reward decomposition | Learned motivational gating | Sequential curriculum |
| --- | --- | --- | --- |
| `scalar` | one scalar value critic | no | no |
| `multi` | four value critics (one per motive) | no | no |
| `regulated` | four value critics | **yes** | no |
| `developmental` | four value critics | **yes** | **yes** |
| `curriculum_control` | four value critics | no | **yes** |

All arms share the same environment, actor/encoder/GRU size, training interaction budget, random initializations (per seed), action space and held-out evaluation episodes. This is **not an exact trainable-parameter-match**: the learned regulator is used only in `regulated`/`developmental`; `scalar` has a one-output critic instead of four. The code deliberately reports no guaranteed superiority. The curriculum variants are compared against the `curriculum_control` so its effects are not confused with gating effects.

### Environment

A batched 9×9 grid with randomized food, shelter, social contact and a moving hazard. Each episode lasts up to 64 steps. Maintaining energy competes with avoiding hazards, meeting a social-contact need, and exploring unseen cells. All policies see the same positions and need measurements. The four separately measurable reward components are:

- **energy:** positive changes in energy, negative depletion and exhaustion death;
- **safety:** health changes, risk exposure and death;
- **social:** changes in social-resource level, decays without contact;
- **exploration:** reward for first visits, diminishing returns for repeat visits.

The global reward is a fixed weighted combination: `0.34 * energy + 0.32 * safety + 0.18 * social + 0.16 * exploration` (the staged curricula temporarily disable selected rewards and renormalize the rest). There are no claims of actual subjective fear, happiness or hunger.

`Agent = encoder → GRUCell → motivation gate → actor + critic`. For regulated arms the gate has four learned state-dependent priorities: `0.4 + 2.4·softmax(controller([GRU state, current needs]))`. Its components sum to 4. The same small motivation adapter is present in all arms and receives neutral gates for controls. The shared RNN state maintains a history beyond the current observation; it is *not* a JEPA or large language model. This pilot isolates the motivational hypothesis before adding a JEPA world model or language encoder.

For each episode, the environment returns `[r_energy, r_safety, r_social, r_exploration]`. Multi-critic variants predict discounted returns for each of these signals. The policy gradient uses the weighted sum of the four per-drive advantages. The controller receives task-reward gradients **through its effect on the policy**. The scalar arm predicts the combined target with one critic. Unrolled A2C with entropy regularization is used, rather than PPO; this is a compact baseline, not a SOTA agent.

### Developmental curriculum

- First ~30% of updates: energy + safety (`0.52, 0.48, 0, 0`).
- Next ~30%: energy + safety + social (`0.41, 0.39, 0.20, 0`).
- Last ~40%: all four drives (`0.34, 0.32, 0.18, 0.16`).

These phases are training conditions, not a claim that human children develop in those exact stages. At evaluation, every arm faces the same full environment and the same metrics.

## Predeclared measurements

- **Primary:** held-out shifted-layout score = `0.22 * energy + 0.22 * health + 0.13 * social + 0.18 * explored_fraction + 0.25 * survival` (all terms bounded 0–1). Same held-out seeds across arms.
- **Secondary:** standard-layout score, shifted-layout reward, survival, energy/health/social level, explored fraction; individual reward channels.
- **Diagnostics:** train return, critic/actor loss, action entropy, average regulator gates, trajectories and gate traces.
- **Comparisons:** `multi−scalar` (value decomposition), `regulated−multi` (dynamic regulation), `developmental−regulated` (staging), `developmental−curriculum_control` (regulator given the same curriculum), and `curriculum_control−multi` (curriculum alone).

With the full preset, report per-seed scores and mean ± sample standard deviation. Two seeds are still far too few for strong conclusions; aim for **5–10 independent seeds**, confidence intervals and multiple independently generated maps. Results under the default pilot are **feasibility checks only**. Use fixed seeds and identical evaluation datasets; no per-model hyperparameter cherry-picking.

## Run locally and test

```bash
python -m pip install torch numpy pandas matplotlib
python -m unittest discover -s experiments/emotional_development/tests -v
```

A local programmatic run is also possible:

```python
import sys
sys.path.insert(0, 'experiments/emotional_development')
from emotion_dev import Config, run, plot_results
cfg = Config(device='cpu', updates=3, batch_size=8, horizon=12,
             hidden=24, eval_every=1, eval_episodes=8, seeds=(17,),
             output_dir='./emotional_development_smoke')
train, ev, final, models = run(cfg)
summary = plot_results(train, ev, final, models, cfg)
print(summary)
```

## Research caveats

1. A gridworld with handcrafted needs is not evidence that emotional complexity *causes* biological intelligence or subjective experience.
2. More reward channels or a richer model can change sample efficiency for reasons unrelated to emotion. Stronger controls would match **active trainable parameters**, compute and reward scaling, and introduce randomly partitioned reward channels.
3. The held-out layout tests **zero-shot generalization**, not online test-time adaptation: no weights are updated at evaluation.
4. Sequential curriculum may help or hurt by withholding the full objective early in training; it should not be assumed to be beneficial.
5. A successful experiment warrants follow-ups with procedurally generated tasks, ablations, genuine world-model prediction and social multi-agent interactions.

## Files

- `emotion_dev.py` — complete simulator, PyTorch recurrent actor-critic, experiments, metrics, plots.
- `kaggle_emotional_development_t4.ipynb` — self-contained Kaggle notebook; contains a synced copy of the module definitions.
- `tests/test_emotion_dev.py` — CPU shape, determinism, differentiable gating, curriculum and all-arm smoke tests.