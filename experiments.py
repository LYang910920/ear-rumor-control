"""Allocation, review and withdrawal experiments for EAR."""
from dataclasses import fields, replace
from pathlib import Path
import csv
import json
import numpy as np
from ear.model import Model, make_model
from ear.integrate import integrate, objective_gradient
from ear.optimize import optimize, project_schedule, Solution
from ear.hold import positive_weight, select_interval
from ear.aggregation import quotient_from_sparse
from networks import contacts, synthetic_network

ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / "parameters.json").read_text())
TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
POLICIES = ("no_intervention", "joint_held", "optimized_constant",
            "downranking_only", "emotion_only", "peer_free_design")


def load_case(network, overrides=None):
    with np.load(ROOT / "data/inputs.npz") as data:
        A, B, pi, membership = (data[f"{network}_{k}"]
                                for k in ("A", "B", "pi", "membership"))
    parameters = dict(SETTINGS["common"])
    parameters.update({k: SETTINGS["networks"][network][k] for k in ("beta", "eta")})
    initial = dict(SETTINGS["initial"])
    for key, value in (overrides or {}).items():
        if key in ("initial_x", "initial_y"):
            initial[key[-1]] = value
        else:
            parameters[key] = value
    model = make_model(A, B, pi, **parameters)
    z = np.r_[np.full(model.K, initial["x"]), np.full(model.K, initial["y"])]
    return model, z, membership


def grid_for(end, hold):
    return np.r_[np.arange(0., end, hold), end]


def starts_for(model, grid):
    shape = (len(grid) - 1, 2 * model.K)
    return [np.zeros(shape), project_schedule(np.tile(.5 * model.upper_controls(),
                                                     (shape[0], 1)), model)]


def policy_on_grid(U, grid, new_grid):
    result = np.zeros((len(new_grid) - 1, U.shape[1]))
    for j, (left, right) in enumerate(zip(new_grid[:-1], new_grid[1:])):
        overlap = np.maximum(0., np.minimum(right, grid[1:]) - np.maximum(left, grid[:-1]))
        tail = max(0., right - max(left, grid[-1]))
        result[j] = (overlap @ U + tail * U[-1]) / (right - left)
    return result


def spectral_solve(model, z, grid, U, max_iterations, tolerance, max_step):
    """Spectral projected updates for the common-tolerance group comparison."""
    U = U.copy(); theta = model.K / float(np.mean(np.diff(grid)))
    fixed_scale = theta; previous = None; status = "iteration_limit"
    for iteration in range(max_iterations + 1):
        J, G, trajectory = objective_gradient(model, z, U, grid, max_step)
        residual = float(np.max(np.abs(project_schedule(U - fixed_scale * G, model) - U)
                                / model.upper_controls()))
        if previous is not None:
            old_U, old_G = previous; s = U - old_U; y = G - old_G
            sy = float(np.sum(s * y))
            if sy > 1e-20:
                theta = float(np.clip(np.sum(s * s) / sy, 1e-6, 1e6))
        if residual <= tolerance:
            status = "stationary_tolerance"; break
        if iteration == max_iterations:
            break
        direction = project_schedule(U - theta * G, model) - U
        slope = float(np.sum(G * direction))
        if slope >= 0:
            status = "non_descent_numerical_stop"; break
        alpha = 1.; accepted = False
        for _ in range(40):
            trial = U + alpha * direction
            if integrate(model, z, trial, grid, max_step).cost <= J + 1e-4 * alpha * slope:
                previous = (U.copy(), G.copy()); U = trial; accepted = True; break
            alpha *= .5
        if not accepted:
            status = "line_search_limit"; break
    return Solution(U, trajectory, iteration, residual, status)


def best_solution(model, z, grid, iterations, refined=False, previous=None):
    starts = starts_for(model, grid)
    if previous is not None:
        starts[0] = project_schedule(policy_on_grid(*previous, grid), model)
    solver = spectral_solve if refined else optimize
    runs = [solver(model, z, grid, U, max_iterations=iterations,
                   tolerance=SETTINGS["optimizer"]["tolerance"],
                   max_step=SETTINGS["time"]["max_step"]) for U in starts]
    selected = min(range(len(runs)), key=lambda j: runs[j].trajectory.cost)
    return runs[selected], selected, runs


def lift_model(group, A, B, membership):
    parameters = {}
    for field in fields(Model):
        name = field.name
        if name not in ("A", "B", "pi"):
            value = getattr(group, name)
            parameters[name] = value if name == "budget" else value[membership]
    return Model(A, B, np.full(len(membership), 1 / len(membership)), **parameters)


def population_run(model, z, U, grid, step):
    times, sharing, emotion = [], [], []
    def observe(t, state):
        times.append(t)
        sharing.append(float(model.pi @ state[:model.K]))
        emotion.append(float(model.pi @ state[model.K:]))
    trajectory = integrate(model, z, U, grid, step, store_history=False, observer=observe)
    return trajectory, np.asarray(times), np.asarray(sharing), np.asarray(emotion)


def evaluate(model, z, U, grid, followup=12.):
    step = SETTINGS["time"]["max_step"]
    tr, t, x, y = population_run(model, z, U, grid, step)
    row = dict(objective=tr.cost, sharing_h_per_user=float(TRAP(x, t)),
               emotion_h_per_user=float(TRAP(y, t)),
               resource_h_per_user=float(np.diff(grid) @ (U @ model.resource_vector())),
               release_sharing=float(x[-1]), release_emotion=float(y[-1]))
    arrays = dict(times=t, sharing=x, emotion=y, controls=U, grid=grid)
    release = tr.states[-1]
    if followup > 0:
        zero = np.zeros((1, 2 * model.K))
        ft, tt, fx, fy = population_run(model, release, zero,
                                      np.array([grid[-1], grid[-1] + followup]), step)
        slope = model.rhs(ft.states[-1], zero[0], *model.zero_forcing(tt[-1]))
        row.update(followup_h=followup, followup_sharing_h_per_user=float(TRAP(fx, tt)),
                   followup_emotion_h_per_user=float(TRAP(fy, tt)),
                   rebound_fraction=float(max(0., fx.max() - x[-1])),
                   endpoint_sharing=float(fx[-1]),
                   endpoint_sharing_rate_per_h=float(model.pi @ slope[:model.K]))
        arrays.update(followup_times=tt, followup_sharing=fx, followup_emotion=fy)
    return row, arrays, release


def write_results(output, rows, arrays=None, starts=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("metrics.csv", rows), ("solutions.csv", starts)):
        if values:
            columns = list(dict.fromkeys(key for row in values for key in row))
            with (output / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader(); writer.writerows(values)
    if arrays:
        np.savez_compressed(output / "trajectories.npz", **arrays)
    print(f"Results: {output / 'metrics.csv'}")


def reproduce(network, output, level="group", contact_dir=None, policies=None, followup=12.):
    model, z, membership = load_case(network)
    if level == "node":
        model = lift_model(model, *contacts(contact_dir), membership)
        z = np.r_[z[:len(z)//2][membership], z[len(z)//2:][membership]]
    rows, arrays = [], {}
    with np.load(ROOT / "data/schedules.npz") as saved:
        available = [k[len(network)+1:-len("_controls")] for k in saved.files
                     if k.startswith(network + "_") and k.endswith("_controls")]
        if policies is None:
            policies = [p for p in POLICIES if p in available]
        for policy in policies:
            if level == "node" and policy not in POLICIES:
                raise ValueError("Node comparisons use the original fixed-grid schedules.")
            U = saved[f"{network}_{policy}_controls"]
            grid = saved[f"{network}_{policy}_grid"]
            if level == "node":
                k = U.shape[1] // 2
                U = np.c_[U[:, :k][:, membership], U[:, k:][:, membership]]
            row, series, _ = evaluate(model, z, U, grid, followup)
            rows.append(dict(network=network, model=level, policy=policy, **row))
            arrays.update({policy + "_" + key: value for key, value in series.items()
                           if level == "group" or key not in ("controls",)})
            print(policy, f"J={row['objective']:.6g}", flush=True)
    write_results(output, rows, arrays)


def solve(network, output, overrides=None, refined=False, policies=None):
    model, z, _ = load_case(network, overrides)
    T, hold = SETTINGS["time"]["horizon"], SETTINGS["time"]["hold"]
    designs = {"joint_held": model, "optimized_constant": model,
               "downranking_only": replace(model, v_max=np.zeros(model.K)),
               "emotion_only": replace(model, u_max=np.zeros(model.K)),
               "peer_free_design": replace(model, eta=np.zeros(model.K))}
    if policies is None:
        policies = ["joint_held", "optimized_constant", "peer_free_design"]
    cap = SETTINGS["optimizer"]["refined_max_iterations" if refined else "max_iterations"]
    rows, arrays, stops = [], {}, []
    for policy in policies:
        grid = np.array([0., T]) if policy == "optimized_constant" else grid_for(T, hold)
        if policy == "no_intervention":
            U = np.zeros((len(grid) - 1, 2 * model.K))
        else:
            sol, selected, runs = best_solution(designs[policy], z, grid, cap, refined)
            U = sol.controls
            for j, candidate in enumerate(runs):
                stops.append(dict(policy=policy, start=j, selected=j == selected,
                                  objective=candidate.trajectory.cost, status=candidate.status,
                                  updates=candidate.iterations, projected_step=candidate.projected_step))
        row, series, _ = evaluate(model, z, U, grid)
        rows.append(dict(network=network, model="group", policy=policy, **row))
        arrays.update({policy + "_" + key: value for key, value in series.items()})
        print(policy, f"J={row['objective']:.6g}", flush=True)
    write_results(output, rows, arrays, stops)


def review(network, interval, output):
    model, z0, _ = load_case(network)
    cfg = SETTINGS["review"]; T = SETTINGS["time"]["horizon"]
    adaptive = interval == "adaptive"
    hold = SETTINGS["time"]["hold"] if adaptive else float(interval)
    q = positive_weight(model, np.r_[model.u_max, np.zeros(model.K)]) if adaptive else None
    t = 0.; z = z0.copy(); previous = None; actions = []; times = [0.]; decisions = []
    while t < T - 1e-10:
        candidate = np.unique(np.minimum(cfg["menu"], T - t))
        nominal = float(candidate.max()) if adaptive else min(hold, T - t)
        end = min(T, t + max(cfg["prediction_horizon"], nominal))
        future = np.r_[t, t + nominal]
        if future[-1] < end - 1e-10:
            future = np.r_[future, grid_for(end - future[-1], hold)[1:] + future[-1]]
        sol, _, _ = best_solution(model, z, future,
                                  SETTINGS["optimizer"]["review_max_iterations"], previous=previous)
        action, duration, mode = sol.controls[0], nominal, "fixed"
        if adaptive:
            decision = select_interval(model, z[model.K:], action, candidate, q, cfg["decline_rate"])
            if decision.action is None:
                duration, mode = float(candidate.min()), "earliest_review"
            else:
                action, duration, mode = decision.action, decision.duration, decision.mode
        tr = integrate(model, z, action[None, :], np.array([t, t + duration]),
                       SETTINGS["time"]["max_step"])
        decisions.append(dict(time=t, duration=duration, action_choice=mode))
        z = tr.states[-1]; t += duration; times.append(t); actions.append(action)
        previous = (sol.controls.copy(), future.copy())
        print(f"Review at {times[-2]:g} h: hold {duration:g} h ({mode})", flush=True)
    row, arrays, _ = evaluate(model, z0, np.asarray(actions), np.asarray(times))
    row.update(network=network, model="group", policy="review_" + interval, reviews=len(actions))
    write_results(output, [row], arrays, decisions)


def withdrawal(contact_dir, output):
    """Compare emotion feedback and grouping from the same node release state."""
    group, initial, membership = load_case("rt_mt")
    A, B = contacts(contact_dir); node = lift_model(group, A, B, membership)
    with np.load(ROOT / "data/schedules.npz") as saved:
        U, grid = saved["rt_mt_joint_held_controls"], saved["rt_mt_joint_held_grid"]
    V = np.c_[U[:, :group.K][:, membership], U[:, group.K:][:, membership]]
    z = np.r_[initial[:group.K][membership], initial[group.K:][membership]]
    release = population_run(node, z, V, grid, SETTINGS["time"]["max_step"])[0].states[-1]
    rows, arrays = [], {}
    cases = [("original_nodes", node, release),
             ("without_amplification", replace(node, alpha=np.zeros(node.K)), release)]
    for k in (16, 32, 64, 128):
        q = quotient_from_sparse(A, B, max_groups=k, compute_residual=False)
        counts = np.bincount(q.membership)
        x = np.bincount(q.membership, weights=release[:node.K]) / counts
        y = np.bincount(q.membership, weights=release[node.K:]) / counts
        params = {f.name: float(getattr(group, f.name)[0]) for f in fields(Model)
                  if f.name not in ("A", "B", "pi", "budget")}
        params["budget"] = group.budget
        cases.append((f"groups_{k}", make_model(q.A, q.B, q.pi, **params), np.r_[x, y]))
    for label, model, start in cases:
        row, series, _ = evaluate(model, start, np.zeros((1, 2 * model.K)),
                                  np.array([24., 36.]), followup=0.)
        rows.append(dict(network="rt_mt", policy=label, **row))
        arrays.update({label + "_" + key: value for key, value in series.items()
                       if key != "controls"})
    write_results(output, rows, arrays)


def timing(network, contact_dir, output):
    """Apply the original joint schedule and its time average to the same contacts."""
    group, initial, membership = load_case(network)
    node = lift_model(group, *contacts(contact_dir), membership)
    z = np.r_[initial[:group.K][membership], initial[group.K:][membership]]
    with np.load(ROOT / "data/schedules.npz") as saved:
        U = saved[f"{network}_joint_held_controls"]
        grid = saved[f"{network}_joint_held_grid"]
    coarse = np.array([grid[0], grid[-1]])
    average = policy_on_grid(U, grid, coarse)
    rows, arrays = [], {}
    for label, actions, times in (("joint_held", U, grid), ("time_averaged", average, coarse)):
        V = np.c_[actions[:, :group.K][:, membership], actions[:, group.K:][:, membership]]
        row, series, _ = evaluate(node, z, V, times)
        rows.append(dict(network=network, model="node", policy=label, **row))
        arrays.update({label + "_" + key: value for key, value in series.items() if key != "controls"})
    write_results(output, rows, arrays)


def demo(output):
    A, B = synthetic_network(100, 20260907)
    q = quotient_from_sparse(A / 13., B / 11., max_groups=4, compute_residual=False)
    model = make_model(q.A, q.B, q.pi)
    z = np.r_[np.full(model.K, .06), np.full(model.K, .25)]
    grid = np.array([0., 1., 2.]); initial = starts_for(model, grid)[1]
    sol = optimize(model, z, grid, initial, max_iterations=10)
    row, arrays, _ = evaluate(model, z, sol.controls, grid, followup=1.)
    write_results(output, [dict(network="example", model="group", policy="joint_held", **row)], arrays)
