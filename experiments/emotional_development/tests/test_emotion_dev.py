"""Fast CPU tests (no external services, no datasets)."""
import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from emotion_dev import (Agent, Config, NeedsGrid, VARIANTS, DRIVES,
                         BASE_WEIGHTS, evaluate, returns_from_rewards,
                         set_seed, train_one, weights_at)


class EmotionDevelopmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device("cpu")
        self.cfg = Config(updates=3, batch_size=8, horizon=12, hidden=20,
                          eval_every=3, eval_episodes=9, seeds=(11,), device="cpu")

    def test_env_shapes_bounds_and_finiteness(self):
        env = NeedsGrid(7, 14, 9, self.device, seed=42)
        self.assertEqual(tuple(env.observe().shape), (7, 21))
        self.assertEqual(tuple(env.drives().shape), (7, 4))
        for _ in range(14):
            obs, r = env.step(torch.randint(0, 5, (7,)))
            self.assertEqual(tuple(r.shape), (7, 4))
            self.assertTrue(torch.isfinite(obs).all())
            self.assertTrue(torch.isfinite(r).all())
            self.assertTrue(((env.pos >= 0) & (env.pos < 9)).all())
            for name in ("energy", "health", "social"):
                self.assertTrue(((getattr(env, name) >= 0) & (getattr(env, name) <= 1)).all())
        self.assertTrue(0 <= env.summary()["score"] <= 1)

    def test_seeded_rollouts_are_deterministic(self):
        one = NeedsGrid(11, 24, 9, self.device, seed=64)
        two = NeedsGrid(11, 24, 9, self.device, seed=64)
        for t in range(24):
            a = torch.full((11,), t % 5)
            obs1, r1 = one.step(a)
            obs2, r2 = two.step(a)
            self.assertTrue(torch.equal(obs1, obs2))
            self.assertTrue(torch.equal(r1, r2))

    def test_stage_curriculum(self):
        e = self.device
        self.assertTrue(torch.allclose(weights_at("scalar", 0, 100, e),
                                       torch.tensor(BASE_WEIGHTS)))
        self.assertEqual(weights_at("developmental", 0, 100, e)[2:].sum().item(), 0.)
        self.assertGreater(weights_at("developmental", 35, 100, e)[2].item(), 0.)
        self.assertEqual(weights_at("developmental", 35, 100, e)[3].item(), 0.)
        self.assertGreater(weights_at("developmental", 70, 100, e)[3].item(), 0.)
        self.assertTrue(torch.equal(weights_at("developmental", 10, 100, e),
                                    weights_at("curriculum_control", 10, 100, e)))

    def test_returns(self):
        r = torch.ones((4, 3, 4))
        out = returns_from_rewards(r, .5)
        self.assertAlmostEqual(out[0, 1, 2].item(), 1.875)
        self.assertAlmostEqual(out[-1, 1, 2].item(), 1.)

    def test_regulator_is_state_dependent_and_backpropagates(self):
        set_seed(44)
        agent = Agent(21, 20, "regulated")
        x = torch.randn(8, 21)
        l, v, h, gates = agent(x, torch.zeros(8, 20))
        self.assertEqual(tuple(v.shape), (8, 4))
        self.assertTrue(torch.allclose(gates.sum(-1), torch.full((8,), 4.), atol=1e-5))
        self.assertTrue(((gates >= .4) & (gates <= 2.8)).all())
        self.assertGreater(gates.var(0).sum().item(), 1e-7)
        (l.square().sum() + v.square().sum()).backward()
        self.assertIsNotNone(agent.regulator[0].weight.grad)
        self.assertGreater(agent.regulator[0].weight.grad.abs().sum().item(), 0.)

    def test_all_variants_complete_training_and_evaluation(self):
        for variant in VARIANTS:
            set_seed(77)
            model = Agent(21, self.cfg.hidden, variant)
            model, hist, val = train_one(model, self.cfg, variant, 77, self.device,
                                         verbose=False)
            self.assertEqual(len(hist), self.cfg.updates)
            self.assertEqual(len(val), 4)  # update 1 and update 3, each two sets
            for row in val:
                self.assertTrue(0 <= row["score"] <= 1)
                self.assertTrue(all(torch.isfinite(torch.tensor(row[f"gate_{d}"])) for d in DRIVES))


if __name__ == "__main__":
    unittest.main()