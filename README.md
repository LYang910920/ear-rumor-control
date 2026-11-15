# EAR: Emotion-aware Allocation and Review

Code for *Rumor Rebound and Emotional Persistence: Update-Constrained Mitigation
on Multilayer Networks*, by Lu-Xing Yang, Qingyi Zhu, Chenquan Gan and Gang Li.

EAR combines piecewise-constant allocation of delivery reduction and communication,
selection of the next review, and evaluation of propagation after withdrawal.

## Start here

Python 3.10 or later; NumPy and SciPy; CPU execution.

```sh
python -m pip install -r requirements.txt
python run.py demo
python run.py reproduce --network rt_mt
```

The example takes a few seconds. `reproduce` evaluates the saved paper schedules
and writes population trajectories and metrics under `outputs/`. `python test_core.py`
checks the gradient, constraints, holding condition and a saved reference result.

## Paper experiments

| Command | Calculation |
|---|---|
| `reproduce --network rt_mt` | Original group schedules, intervention and withdrawal |
| `reproduce --network follower_mt` | Original follower-based group schedules |
| `reproduce --network rt_mt --policies refined_joint refined_constant refined_peer_free` | Common-tolerance group schedules |
| `reproduce --network rt_mt --policies review_fixed_1h review_fixed_2h review_fixed_4h review_adaptive` | Saved review policies |
| `solve --network rt_mt` | Recompute joint, constant and peer-free allocation from two starts |
| `solve --network rt_mt --scenario budget_low` | Recompute a prescribed sensitivity setting |
| `solve --network rt_mt --refined` | Recompute the common-tolerance group comparison |
| `review --network rt_mt --interval adaptive` | Recompute first-hold selection and action adjustment |

Prefix commands with `python run.py`. `--help` lists options. Optimization uses
2,000 updates per start (400 at each review) and tolerance `1e-3`. Both final
solutions are retained in `solutions.csv`. Original and `refined_*` schedules
are distinct; the paper's node comparisons use the original schedules.

For complete node contacts:

```sh
python run.py prepare --network rt_mt
python run.py reproduce --network rt_mt --model node
python run.py reproduce --network rt_mt --model node --policies joint_held optimized_constant --followup-hours 24
python run.py timing --network rt_mt
python run.py withdrawal
```

`prepare` downloads the retweet and mention files from the
[SNAP Higgs dataset](https://snap.stanford.edu/data/higgs-twitter.html).
Use `--network follower_mt` to prepare and evaluate follower–mention contacts;
`--raw-dir PATH` reuses downloaded files. `prepare --network synthetic` builds the
10,000-user four-community example. `timing` compares the joint schedule with its
time average at identical cumulative inputs. `withdrawal` compares emotion
amplification and 16/32/64/128-group predictions from the same RT–MT node release state.

All endpoint users are retained. Delivery matrices use `[retweeter, author]` or
`[follower, followed]`; mention matrices use `[mentioned, mentioning]`.
Contacts are binary, nonself, and normalized by the positive incoming-degree median
with a per-user cap of five. Behavioral rates in `parameters.json` are fixed when
moving from groups to nodes or varying the emotional exit rate. Higgs provides
contacts; activity states and intervention effects are model scenarios.

## Files

| File | Role |
|---|---|
| `run.py` | Command entry point |
| `experiments.py` | Allocation, review, timing and withdrawal comparisons |
| `networks.py` | Contact preparation and synthetic network |
| `ear/model.py` | Coupled activities, controls and objective |
| `ear/integrate.py` | SSPRK3 integration and discrete adjoint |
| `ear/optimize.py` | Whole-interval projected allocation (Algorithm 2) |
| `ear/hold.py` | Emotional bound and first-hold selection (Theorem 2, Algorithm 3) |
| `ear/aggregation.py`, `ear/bounds.py` | Shared-action grouping and transfer bounds |
| `parameters.json` | Reference values and prescribed sensitivity ranges |
| `data/inputs.npz`, `data/schedules.npz` | Three group models and 28 final schedules |
| `data/results.csv` | Final paper results with model, policy, metric, unit and time window |

The reported full-network experiments used an Apple M4 Pro with 24 GiB memory
and one requested computational thread. Full optimization takes longer than
evaluation of saved schedules. Output directories are excluded from version control.
