# EAR: Emotion-aware Allocation and Review

Code for *Rumor Rebound and Emotional Persistence: Update-Constrained Mitigation
on Multilayer Networks*, by Lu-Xing Yang, Qingyi Zhu, Chenquan Gan and Gang Li.

EAR combines piecewise-constant allocation of delivery reduction and communication,
selection of the next review, and evaluation of propagation after withdrawal.

## Start here

Python 3.10 or later.

```sh
python -m pip install -r requirements.txt
python run.py demo
python run.py reproduce --network rt_mt
```

`reproduce` evaluates the saved paper schedules and writes population trajectories
and metrics under `outputs/`. `python test_core.py` runs the core tests.

## Paper experiments

| Command | Calculation |
|---|---|
| `reproduce --network rt_mt` | Original group schedules, intervention and withdrawal |
| `reproduce --network follower_mt` | Original follower-based group schedules |
| `reproduce --network rt_mt --policies refined_joint refined_constant refined_peer_free` | Group policies optimized to the same stopping tolerance |
| `reproduce --network rt_mt --policies review_fixed_1h review_fixed_2h review_fixed_4h review_adaptive` | Saved review policies |
| `solve --network rt_mt` | Optimize joint and constant actions, and actions planned without emotion spread |
| `solve --network rt_mt --scenario budget_low` | Recompute a prescribed sensitivity setting |
| `solve --network rt_mt --refined` | Optimize the group policies to the same stopping tolerance |
| `review --network rt_mt --interval adaptive` | Recompute first-hold selection and action adjustment |

Prefix commands with `python run.py`; `--help` lists options. The paper
evaluates the original schedules on individual contacts and the `refined_*`
schedules on group models at a common stopping tolerance.

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
with a per-user cap of five. Group and node evaluations use the same behavioral rates. Emotional-exit
comparisons vary `mu` while keeping the other rates fixed. Higgs provides
contacts; activity states and intervention effects are model scenarios.

## Paper-to-code map

| File / function | Paper location | Role |
|---|---|---|
| `run.py` | Section V | Single command entry point |
| `ear/model.py` | Equations (2), (3), (6) | Coupled activities, controls and objective |
| `ear/aggregation.py` | Section II-D, Algorithm 1 | Contact-profile groups and shared actions |
| `ear/optimize.py` | Theorem 3, Algorithm 2 | Joint allocation across all holding intervals |
| `ear/integrate.py` | Algorithm 2, Appendix D | SSPRK3 integration and discrete adjoint |
| `ear/hold.py` | Theorem 2, Algorithm 3, Appendix E | Emotional upper bound and longest eligible first hold |
| `ear/bounds.py` | Lemma F.1 | Group-to-node state and cumulative-activity bounds |
| `networks.py` | Section V-A, Appendix G-A | Complete contacts and synthetic network |
| `experiments.py`: `solve`, `reproduce`, `timing` | Sections V-B–V-C | Allocation and timing at equal cumulative inputs |
| `experiments.py`: `withdrawal` | Section V-D | Emotional amplification and group predictions after release |
| `experiments.py`: `review`, `solve` | Sections V-E–V-F | Review decisions and parameter sensitivity |
| `parameters.json` | Table G.I | Model parameters and sensitivity ranges |
| `data/inputs.npz`, `data/schedules.npz` | Section V, Appendix G | Three group models and 28 final schedules |
| `data/results.csv` | Section V, Appendix G | Final metrics with model, policy, unit and time window |
