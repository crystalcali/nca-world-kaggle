"""Motivation and developmental learning: fully vectorized synthetic experiment.

Pure PyTorch, no Gym dependencies, no external datasets or pretrained checkpoints.
Functional drive variables are NOT claims about subjective emotions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

DRIVES = ("energy", "safety", "social", "exploration")
VARIANTS = ("scalar", "multi", "regulated", "developmental", "curriculum_control")
BASE_WEIGHTS = (0.34, 0.32, 0.18, 0.16)


@dataclass
class Config:
    updates: int = 100
    batch_size: int = 128
    horizon: int = 64
    hidden: int = 96
    learning_rate: float = 7e-4
    gamma: float = 0.98
    entropy_coef: float = 0.025
    value_coef: float = 0.40
    eval_every: int = 10
    eval_episodes: int = 128
    seeds: tuple[int, ...] = (17, 29)
    variants: tuple[str, ...] = VARIANTS
    device: str = "auto"
    output_dir: str = "/kaggle/working/emotional_development"
    grid_size: int = 9

    def __post_init__(self):
        assert self.grid_size >= 7
        assert self.batch_size >= 1 and self.horizon >= 2 and self.updates >= 1
        assert self.eval_every >= 1
        assert all(v in VARIANTS for v in self.variants)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select GPU T4 x2 in Kaggle settings")
    return torch.device(device)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NeedsGrid:
    """Batched partially goal-directed grid environment with 4 reward channels.

    All versions of the agent receive the same observations and fixed-size network.
    No external dependency or dataset. Reward channels are measurable and auditable.
    """

    def __init__(self, batch: int, horizon: int, size: int, device: torch.device,
                 seed: int, shifted: bool = False):
        self.B, self.H, self.S, self.device = batch, horizon, size, device
        self.shifted = shifted
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)
        self.ids = torch.arange(batch, device=device)
        self.moves = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
                                  device=device, dtype=torch.long)
        self.reset()

    def _randpos(self, anchor):
        jitter = torch.randint(-1, 2, (self.B, 2), device=self.device, generator=self.g)
        return (torch.tensor(anchor, device=self.device) + jitter).clamp(1, self.S - 2)

    def reset(self):
        s = self.S
        self.pos = torch.randint(1, s - 1, (self.B, 2), device=self.device, generator=self.g)
        # Held-out test changes the spatial arrangement of resources.
        self.food = self._randpos((1, s - 2) if self.shifted else (s - 2, 1))
        self.shelter = self._randpos((s - 2, 1) if self.shifted else (1, 1))
        self.ally = self._randpos((1, 1) if self.shifted else (s - 2, s - 2))
        self.hazard = self._randpos((s // 2, s // 2))
        self.energy = torch.full((self.B,), 0.72, device=self.device)
        self.health = torch.full((self.B,), 0.95, device=self.device)
        self.social = torch.full((self.B,), 0.63, device=self.device)
        self.alive = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self.visited = torch.zeros((self.B, s, s), dtype=torch.float32, device=self.device)
        self.visited[self.ids, self.pos[:, 0], self.pos[:, 1]] = 1.
        self.t = 0
        self.total_reward = torch.zeros((self.B, 4), device=self.device)
        return self.observe()

    @staticmethod
    def _distance(a, b):
        return (a - b).abs().sum(dim=-1)

    def drives(self):
        d = self._distance(self.pos, self.hazard)
        near = (d <= 2).float()
        coverage = (self.visited > 0).float().mean(dim=(1, 2))
        return torch.stack((1 - self.energy, (1 - self.health) * .8 + near * .6,
                            1 - self.social, 1 - coverage), dim=-1).clamp(0, 1.5)

    def observe(self):
        s = float(self.S - 1)
        coverage = (self.visited > 0).float().mean(dim=(1, 2))
        d = self._distance(self.pos, self.hazard).float()
        parts = [q.float() / s for q in (self.pos, self.food, self.shelter, self.ally, self.hazard)]
        parts += [self.energy[:, None], self.health[:, None], self.social[:, None],
                  torch.full((self.B, 1), self.t / self.H, device=self.device),
                  (d / (self.S * 2))[:, None], coverage[:, None], self.drives(),
                  self.alive.float()[:, None]]
        return torch.cat(parts, dim=-1)

    def step(self, actions: torch.Tensor):
        assert actions.shape == (self.B,)
        old_alive = self.alive.clone()
        old_energy, old_health, old_social = self.energy, self.health, self.social
        delta = self.moves[actions.long()]
        self.pos = torch.where(old_alive[:, None],
                               (self.pos + delta).clamp(0, self.S - 1), self.pos)
        if self.t % 6 == 5:
            hazard_move = self.moves[torch.randint(0, 5, (self.B,),
                generator=self.g, device=self.device)]
            self.hazard = (self.hazard + hazard_move).clamp(0, self.S - 1)
        before_visits = self.visited[self.ids, self.pos[:, 0], self.pos[:, 1]].clone()
        self.visited[self.ids, self.pos[:, 0], self.pos[:, 1]] += old_alive.float()
        at_food = (self._distance(self.pos, self.food) == 0).float()
        at_shelter = (self._distance(self.pos, self.shelter) == 0).float()
        at_ally = (self._distance(self.pos, self.ally) == 0).float()
        dist_h = self._distance(self.pos, self.hazard)
        collision = (dist_h == 0).float()
        danger = (dist_h <= 1).float()
        moving = (actions != 0).float()
        self.energy = (old_energy - .012 - .003 * moving + .72 * at_food).clamp(0, 1)
        self.health = (old_health - .20 * collision - .025 * danger
                       + .045 * at_shelter).clamp(0, 1)
        self.social = (old_social - .010 + .70 * at_ally).clamp(0, 1)
        self.energy = torch.where(old_alive, self.energy, old_energy)
        self.health = torch.where(old_alive, self.health, old_health)
        self.social = torch.where(old_alive, self.social, old_social)
        died = old_alive & ((self.energy <= 1e-6) | (self.health <= 1e-6))
        self.alive = old_alive & ~died
        components = torch.stack((
            1.6 * (self.energy - old_energy) - died.float() * (self.energy <= 1e-6).float(),
            1.8 * (self.health - old_health) - .015 * danger -
                died.float() * (self.health <= 1e-6).float(),
            1.4 * (self.social - old_social),
            .11 * (before_visits == 0).float() + .016 / torch.sqrt(before_visits + 1)
                - .012,
        ), dim=-1)
        components = components * old_alive.float()[:, None]
        self.total_reward += components
        self.t += 1
        return self.observe(), components

    def summary(self):
        coverage = (self.visited > 0).float().mean(dim=(1, 2))
        survival = self.alive.float()
        # Separate, pre-registered external evaluation metric, NOT training reward.
        score = (.22 * self.energy + .22 * self.health + .13 * self.social +
                 .18 * coverage + .25 * survival)
        return {
            "score": score.mean().item(),
            "survival": survival.mean().item(),
            "energy": self.energy.mean().item(),
            "health": self.health.mean().item(),
            "social": self.social.mean().item(),
            "coverage": coverage.mean().item(),
            **{f"r_{name}": self.total_reward[:, i].mean().item()
               for i, name in enumerate(DRIVES)},
        }


class Agent(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, variant: str):
        super().__init__()
        self.variant = variant
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.recurrent = nn.GRUCell(hidden, hidden)
        self.regulator = nn.Sequential(nn.Linear(hidden + 4, hidden // 2),
                                       nn.Tanh(), nn.Linear(hidden // 2, 4))
        self.motivation_adapter = nn.Linear(4, hidden, bias=False)
        self.actor = nn.Linear(hidden, 5)
        self.multi_critic = nn.Linear(hidden, 4)
        self.scalar_critic = nn.Linear(hidden, 1)

    def forward(self, obs, state):
        core = self.recurrent(self.encoder(obs), state)
        needs = obs[:, -5:-1]
        if self.variant in ("regulated", "developmental"):
            # Bounded, differentiable gate; 1.0 means neutral motive priority.
            gates = .4 + 2.4 * F.softmax(self.regulator(torch.cat((core, needs), -1)), -1)
        else:
            gates = torch.ones_like(needs)
        features = torch.tanh(core + .25 * self.motivation_adapter(needs * gates))
        logits = self.actor(features)
        value = self.scalar_critic(features) if self.variant == "scalar" else self.multi_critic(features)
        return logits, value, core, gates


def weights_at(variant: str, update_idx: int, updates: int, device):
    if variant in ("developmental", "curriculum_control"):
        p = (update_idx + 1) / updates
        if p < .30:
            vals = (.52, .48, 0., 0.)
        elif p < .60:
            vals = (.41, .39, .20, 0.)
        else:
            vals = BASE_WEIGHTS
    else:
        vals = BASE_WEIGHTS
    return torch.tensor(vals, dtype=torch.float32, device=device)


def returns_from_rewards(reward, gamma):
    out = []
    running = torch.zeros_like(reward[0])
    for r in reversed(reward.unbind(0)):
        running = r + gamma * running
        out.append(running)
    return torch.stack(out[::-1])


def train_one(agent: Agent, config: Config, variant: str, seed: int, device,
              verbose: bool = True):
    set_seed(seed)
    agent.to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    hist = []
    evals = []
    for upd in range(config.updates):
        agent.train()
        env = NeedsGrid(config.batch_size, config.horizon, config.grid_size,
                        device, seed=seed * 100003 + upd)
        obs = env.observe()
        hidden = torch.zeros(config.batch_size, config.hidden, device=device)
        log_probs, entropies, values, rewards, gates_list = [], [], [], [], []
        for t in range(config.horizon):
            logits, v, hidden, gates = agent(obs, hidden)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
            logp_all = F.log_softmax(logits, dim=-1)
            log_probs.append(logp_all.gather(1, action[:, None]).squeeze(1))
            entropies.append(-(probs * logp_all).sum(-1))
            values.append(v)
            gates_list.append(gates.detach())
            obs, r = env.step(action)
            rewards.append(r)
        V = torch.stack(values)
        R = torch.stack(rewards)
        lp = torch.stack(log_probs)
        ent = torch.stack(entropies).mean()
        returns = returns_from_rewards(R, config.gamma)
        weights = weights_at(variant, upd, config.updates, device)
        if variant == "scalar":
            target = (returns * weights).sum(-1)
            advantage = target - V.squeeze(-1)
            value_loss = F.mse_loss(V.squeeze(-1), target)
        else:
            advantage = ((returns - V) * weights).sum(-1)
            value_loss = F.mse_loss(V, returns)
        # Detach and normalize only for the actor (critic uses unnormalized returns).
        A = advantage.detach()
        A = (A - A.mean()) / (A.std(unbiased=False) + 1e-6)
        pg_loss = -(lp * A).mean()
        loss = pg_loss + config.value_coef * value_loss - config.entropy_coef * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        stats = env.summary()
        stats.update(variant=variant, seed=seed, update=upd + 1,
                     loss=float(loss.detach()), actor_loss=float(pg_loss.detach()),
                     critic_loss=float(value_loss.detach()), entropy=float(ent.detach()),
                     weighted_return=float((R * weights).sum(-1).sum(0).mean()),
                     stage_weights=weights.detach().cpu().tolist())
        gate_means = torch.stack(gates_list).mean((0, 1)).cpu().numpy()
        stats.update({f"gate_{d}": float(g) for d, g in zip(DRIVES, gate_means)})
        hist.append(stats)
        if (upd + 1) % config.eval_every == 0 or upd == 0 or upd + 1 == config.updates:
            for shifted in (False, True):
                ev = evaluate(agent, config, seed=314159, shifted=shifted, device=device)
                ev.update(variant=variant, seed=seed, update=upd + 1,
                          eval_set="shifted" if shifted else "standard")
                evals.append(ev)
            if verbose:
                s = evals[-2]["score"]
                sh = evals[-1]["score"]
                print(f"{variant:>18} seed={seed} update={upd+1:3d}/{config.updates} "
                      f"train-score={stats['score']:.3f} eval={s:.3f} shifted={sh:.3f} "
                      f"entropy={float(ent.detach()):.2f}", flush=True)
    return agent, hist, evals


@torch.no_grad()
def evaluate(agent: Agent, config: Config, seed: int, shifted: bool, device):
    agent.eval()
    env = NeedsGrid(config.eval_episodes, config.horizon, config.grid_size,
                    device, seed=seed, shifted=shifted)
    obs = env.observe()
    hidden = torch.zeros(config.eval_episodes, config.hidden, device=device)
    weights = torch.tensor(BASE_WEIGHTS, device=device)
    gate_means = torch.zeros(4, device=device)
    for _ in range(config.horizon):
        logits, _, hidden, gates = agent(obs, hidden)
        gate_means += gates.mean(0)
        actions = logits.argmax(-1)
        obs, _ = env.step(actions)
    result = env.summary()
    result["weighted_return"] = float((env.total_reward * weights).sum(-1).mean())
    result.update({f"gate_{d}": float(x) for d, x in
                   zip(DRIVES, (gate_means / config.horizon).tolist())})
    return result


@torch.no_grad()
def trajectory(agent: Agent, config: Config, device, shifted=False, seed=404):
    """One deterministic roll-out for qualitative visualization, never for model selection."""
    agent.eval()
    env = NeedsGrid(1, config.horizon, config.grid_size, device, seed=seed, shifted=shifted)
    obs = env.observe()
    hidden = torch.zeros(1, config.hidden, device=device)
    positions = [env.pos[0].cpu().tolist()]
    gates = []
    for _ in range(config.horizon):
        logits, _, hidden, gate = agent(obs, hidden)
        obs, _ = env.step(logits.argmax(-1))
        positions.append(env.pos[0].cpu().tolist())
        gates.append(gate[0].cpu().tolist())
    return {"positions": np.asarray(positions), "gates": np.asarray(gates),
            "food": env.food[0].cpu().tolist(), "shelter": env.shelter[0].cpu().tolist(),
            "ally": env.ally[0].cpu().tolist(), "hazard_final": env.hazard[0].cpu().tolist(),
            "summary": env.summary()}


def run(config: Config):
    device = resolve_device(config.device)
    out = Path(config.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    print("device:", device, "GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("variants:", config.variants, "seeds:", config.seeds)
    histories, evaluations, models = [], [], {}
    started = time.time()
    for variant in config.variants:
        for seed in config.seeds:
            # Same initialization across variants at a given seed, except the output heads
            # and the regulator gradients selected by the experiment design.
            set_seed(seed)
            model = Agent(obs_dim=21, hidden=config.hidden, variant=variant)
            model, hist, ev = train_one(model, config, variant, seed, device)
            histories.extend(hist)
            evaluations.extend(ev)
            models[(variant, seed)] = model
            pd.DataFrame(histories).to_csv(out / "train_history.csv", index=False)
            pd.DataFrame(evaluations).to_csv(out / "evaluation_history.csv", index=False)
            torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                        "variant": variant, "seed": seed}, out / f"{variant}_seed{seed}.pt")
    history = pd.DataFrame(histories)
    evaluation = pd.DataFrame(evaluations)
    final = evaluation[evaluation["update"] == config.updates].copy()
    final.to_csv(out / "final_metrics.csv", index=False)
    print("Completed in", round((time.time() - started) / 60, 2), "minutes")
    return history, evaluation, final, models


def plot_results(history, evaluation, final, models, config: Config):
    import matplotlib.pyplot as plt
    out = Path(config.output_dir)
    plt.figure(figsize=(10, 5))
    for variant, g in evaluation[evaluation.eval_set == "shifted"].groupby("variant"):
        av = g.groupby("update").score.agg(["mean", "std"])
        x, y = av.index.to_numpy(), av["mean"].to_numpy()
        sd = av["std"].fillna(0).to_numpy()
        plt.plot(x, y, label=variant)
        plt.fill_between(x, y - sd, y + sd, alpha=.12)
    plt.xlabel("Training update")
    plt.ylabel("Held-out shifted-layout score ↑")
    plt.title("Out-of-distribution generalization (same eval seeds)")
    plt.legend(fontsize=8)
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(out / "generalization_curve.png", dpi=170)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    score = final.groupby(["variant", "eval_set"]).score.agg(["mean", "std"]).reset_index()
    labels = list(config.variants)
    for ax, eval_set in zip(axes, ["standard", "shifted"]):
        means = [float(score[(score.variant == v) & (score.eval_set == eval_set)]["mean"].iloc[0])
                 for v in labels]
        stds = [float(score[(score.variant == v) & (score.eval_set == eval_set)]["std"].fillna(0).iloc[0])
                for v in labels]
        ax.bar(range(len(labels)), means, yerr=stds, capsize=4)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_title(eval_set)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Behavioral score ↑")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Final evaluation: mean ± seed std (not a confidence interval)")
    fig.tight_layout()
    fig.savefig(out / "final_scores.png", dpi=170)
    plt.show()

    plt.figure(figsize=(10, 4))
    for variant, g in history.groupby("variant"):
        s = g.groupby("update").weighted_return.mean()
        plt.plot(s.index, s.values, label=variant)
    plt.xlabel("Training update")
    plt.ylabel("On-policy weighted return")
    plt.title("Training objective: compare only within the same curriculum stage")
    plt.legend(fontsize=8)
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(out / "training_returns.png", dpi=170)
    plt.show()

    # Actual learned motive gates, and a single concrete trajectory.
    for variant in ("regulated", "developmental"):
        key = next((k for k in models if k[0] == variant), None)
        if key is None:
            continue
        tr = trajectory(models[key], config, resolve_device(config.device), shifted=True)
        pos = tr["positions"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.plot(pos[:, 1], pos[:, 0], "-o", markersize=2, alpha=.65)
        for name, marker in (("food", "s"), ("shelter", "^"), ("ally", "D"), ("hazard_final", "x")):
            xy = tr[name]
            ax.scatter(xy[1], xy[0], s=100, marker=marker, label=name)
        ax.set_xlim(-.5, config.grid_size - .5)
        ax.set_ylim(config.grid_size - .5, -.5)
        ax.set_aspect("equal")
        ax.set_title(f"{variant}: example trajectory")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        for i, name in enumerate(DRIVES):
            axes[1].plot(tr["gates"][:, i], label=name)
        axes[1].set_ylim(.3, 2.9)
        axes[1].set_xlabel("Environment step")
        axes[1].set_ylabel("Regulator gate")
        axes[1].set_title("State-dependent motivational priorities")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{variant}_trajectory_gates.png", dpi=170)
        plt.show()
    return final.groupby(["variant", "eval_set"])[["score", "survival", "energy", "health",
        "social", "coverage", "weighted_return"]].agg(["mean", "std"]).round(3)