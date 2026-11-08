#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "scipy>=1.17", "arviz>=0.23"]
# ///
"""Independent small-system checks of posterior validation; no research outputs."""

import unittest

import numpy as np

from eta_validation import (
    ExactPosterior, convergence_diagnostics, fixed_coordinate_scale,
    invariant_traces, run_replica_hmc, swap_log_ratio,
)
from rbm_exact import kl, log_prob_visible, overlaps, spin_states


class ValidationChecks(unittest.TestCase):
    def test_complete_posterior_gradient(self):
        rng = np.random.default_rng(43)
        target = ExactPosterior(spin_states(4)[[1, 2, 2, 7, 13]], 2, 0.3)
        W = rng.normal(size=(4, 2)) * 0.4
        _, grad = target(W)
        numerical = np.zeros_like(W)
        for j in range(W.size):
            plus, minus = W.copy(), W.copy()
            plus.ravel()[j] += 1e-6
            minus.ravel()[j] -= 1e-6
            numerical.ravel()[j] = (target(plus)[0] - target(minus)[0]) / 2e-6
        np.testing.assert_allclose(grad, numerical, atol=1e-7, rtol=1e-7)

    def test_exchange_ratio_includes_prior(self):
        target = ExactPosterior(spin_states(4)[[1, 3, 11]], 2, 1.7)
        rng = np.random.default_rng(8)
        left, right = rng.normal(size=(2, 4, 2))
        ll, lr = float(target(left)[0]), float(target(right)[0])
        a, b = 0.2, 5.0
        direct = (a * lr + b * ll) - (a * ll + b * lr)
        self.assertAlmostEqual(swap_log_ratio(a, b, ll, lr), direct, places=12)

    def test_hidden_prediction_and_exact_symmetry(self):
        rng = np.random.default_rng(56)
        target = ExactPosterior(spin_states(4)[[0, 2, 6]], 2, 0.7)
        V = spin_states(4)
        xis = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=float)
        W = rng.normal(size=(4, 2))
        transformed = W[:, [1, 0]] * np.array([-1, 1])
        samples = np.stack([W, transformed]).reshape(1, 2, 1, 4, 2)
        traces = invariant_traces(samples, target, xis, V[:3])
        lp = log_prob_visible(W, np.zeros(4), np.zeros(2), 1.0, V)
        q = overlaps(lp, V @ xis / 4)
        for i in range(2):
            self.assertAlmostEqual(traces[f"q{i + 1}"][0, 0, 0], q[i], places=12)
        for i in range(3):
            self.assertAlmostEqual(traces[f"logp_probe{i}"][0, 0, 0], lp[i], places=12)
        for array in traces.values():
            np.testing.assert_allclose(array[:, 0], array[:, 1], atol=1e-12)

    def test_rank_diagnostics_detect_stuck_chains(self):
        rng = np.random.default_rng(9)
        good = rng.normal(size=(1000, 4, 1))
        bad = good + np.array([0, 0, 5, 5])[None, :, None]
        diagnostics = convergence_diagnostics({"good": good, "bad": bad}, [1])
        self.assertLess(diagnostics["1.0"]["good"]["rhat_rank"], 1.02)
        self.assertGreater(diagnostics["1.0"]["bad"]["rhat_rank"], 1.3)

    def test_replica_sampler_gaussian_moments(self):
        def normal(W):
            return -0.5 * (W ** 2).sum((-2, -1)), -W

        rng = np.random.default_rng(1701)
        etas = np.array([0.5, 1.0, 2.0])
        result = run_replica_hmc(
            normal, rng.normal(size=(8, 3, 2, 1)) * 3, etas,
            np.array([0.5, 1.5])[None, :, None] / np.sqrt(etas[:, None, None]), rng,
            warmup=500, sample=2000, thin=2, progress_every=0,
        )
        x = result["samples"]
        mean = x.mean((0, 1, 3, 4))
        variance = (x ** 2).mean((0, 1, 3, 4))
        np.testing.assert_allclose(mean, 0, atol=0.05)
        np.testing.assert_allclose(variance, 1 / etas, rtol=0.05)
        self.assertTrue(np.all(result["swap_counts"].sum(0) > 100))
        self.assertEqual(int(result["divergences"].sum()), 0)

    def test_rbm_against_two_dimensional_quadrature(self):
        # This posterior is integrated independently on an explicit 2-D grid.
        # It checks eta on BOTH likelihood and prior, plus predictive averaging.
        data = np.array([[1, 1], [-1, -1], [1, -1]], dtype=float)
        target = ExactPosterior(data, 1, 1.4)
        V = spin_states(2)
        teacher_W = np.array([[0.7], [0.4]])
        lp_teacher = log_prob_visible(teacher_W, np.zeros(2), np.zeros(1), 1.0, V)
        grid = np.linspace(-7, 7, 281)
        gx, gy = np.meshgrid(grid, grid, indexing="ij")
        Wgrid = np.stack([gx.ravel(), gy.ravel()], axis=1)[..., None]
        base = target(Wgrid)[0]
        lpgrid = log_prob_visible(Wgrid, np.zeros(2), np.zeros(1), 1.0, V, chunk=256)
        etas = np.array([0.5, 1.0, 2.0])
        rng = np.random.default_rng(1846)
        scale, _ = fixed_coordinate_scale(target, np.zeros((2, 1)), etas, "unit")
        result = run_replica_hmc(
            target, rng.normal(size=(8, 3, 2, 1)) * 2, etas, scale, rng,
            warmup=600, sample=2400, thin=3, progress_every=0,
        )
        sample = result["samples"]
        for ei, eta in enumerate(etas):
            weight = np.exp(eta * (base - base.max()))
            weight /= weight.sum()
            edge = ((np.abs(Wgrid[:, :, 0]) >= 7 - 1e-10).any(1))
            self.assertLess(weight[edge].sum(), 1e-6)
            exact_norm2 = weight @ (Wgrid ** 2).sum((-2, -1))
            chain_norm2 = (sample[:, :, ei] ** 2).sum((-2, -1)).mean(0)
            sem = chain_norm2.std(ddof=1) / np.sqrt(len(chain_norm2))
            self.assertLess(abs(chain_norm2.mean() - exact_norm2), max(6 * sem, 0.02))
            exact_pp = weight @ np.exp(lpgrid)
            Wsample = sample[:, :, ei].reshape(-1, 2, 1)
            model_lp = log_prob_visible(Wsample, np.zeros(2), np.zeros(1), 1.0, V, chunk=256)
            sampled_pp = np.exp(model_lp).mean(0)
            exact_kl = float(kl(np.log(exact_pp), lp_teacher) / 2)
            sampled_kl = float(kl(np.log(sampled_pp), lp_teacher) / 2)
            self.assertLess(abs(sampled_kl - exact_kl), 0.005)
            np.testing.assert_allclose(sampled_pp, exact_pp, atol=0.01)
            print(f"quadrature eta={eta:g}: norm² {chain_norm2.mean():.5f}/{exact_norm2:.5f}; "
                  f"predictive KL {sampled_kl:.6f}/{exact_kl:.6f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
