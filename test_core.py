"""Small checks for the model, allocation and saved paper schedules."""
import csv
import unittest
import numpy as np
from ear.model import make_model
from ear.integrate import integrate, objective_gradient
from ear.optimize import project_schedule
from ear.hold import emotional_bound, positive_weight, select_interval, contraction_margin
from experiments import ROOT, load_case, evaluate


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model(np.array([[.2, 1.], [.5, .1]]),
                                np.array([[.1, .7], [.4, .2]]))
        self.z = np.array([.06, .08, .25, .20])
        self.grid = np.array([0., .5, 1.])
        self.U = np.tile([.2, .3, .04, .03], (2, 1))

    def test_gradient(self):
        _, gradient, _ = objective_gradient(self.model, self.z, self.U, self.grid)
        direction = np.random.default_rng(3).normal(size=self.U.shape)
        epsilon = 1e-6
        plus = integrate(self.model, self.z, self.U + epsilon * direction, self.grid).cost
        minus = integrate(self.model, self.z, self.U - epsilon * direction, self.grid).cost
        self.assertAlmostEqual((plus - minus) / (2 * epsilon),
                               float(np.sum(gradient * direction)), places=7)

    def test_projection(self):
        U = project_schedule(np.full_like(self.U, 2.), self.model)
        self.assertTrue(np.all(U >= 0))
        self.assertTrue(np.all(U <= self.model.upper_controls()))
        self.assertTrue(np.all(U @ self.model.resource_vector() <= self.model.budget))
        np.testing.assert_array_equal(project_schedule(U, self.model), U)

    def test_hold(self):
        q = positive_weight(self.model, np.r_[self.model.u_max, np.zeros(2)])
        decision = select_interval(self.model, self.z[2:], self.U[0], np.array([1., 2., 4.]), q, .01)
        self.assertIsNotNone(decision.action)
        _, ceiling = emotional_bound(self.model, self.z[2:], decision.action, decision.duration)
        self.assertTrue(np.all(contraction_margin(self.model, ceiling, decision.action, q, .01) <= 1e-12))

    def test_saved_reference(self):
        model, z, _ = load_case("rt_mt")
        with np.load(ROOT / "data/schedules.npz") as data:
            U, grid = data["rt_mt_joint_held_controls"], data["rt_mt_joint_held_grid"]
        row, _, _ = evaluate(model, z, U, grid)
        with (ROOT / "data/results.csv").open() as f:
            expected = next(float(r["value"]) for r in csv.DictReader(f)
                            if r["experiment"] == "allocation" and r["network"] == "rt_mt"
                            and r["scenario"] == "initial_y=0.25" and r["policy"] == "joint_held"
                            and r["metric"] == "objective")
        self.assertAlmostEqual(row["objective"], expected, places=10)


if __name__ == "__main__":
    unittest.main()
