"""Run EAR examples and paper comparisons."""
import argparse
from pathlib import Path

# Set thread counts before importing the numerical libraries.
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from experiments import SETTINGS, POLICIES, demo, reproduce, solve, review, withdrawal, timing
from networks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    for name in ("demo", "prepare", "reproduce", "solve", "review", "withdrawal", "timing"):
        command = tasks.add_parser(name)
        command.add_argument("--output", type=Path)
        if name not in ("demo", "withdrawal"):
            command.add_argument("--network", choices=list(SETTINGS["networks"]), default="rt_mt")
        if name in ("reproduce", "withdrawal", "timing"):
            command.add_argument("--contacts", type=Path)
        if name == "prepare":
            command.add_argument("--raw-dir", default="data/raw")
        if name == "reproduce":
            command.add_argument("--model", choices=["group", "node"], default="group")
            command.add_argument("--policies", nargs="+")
            command.add_argument("--followup-hours", type=float, default=12.)
        if name == "solve":
            command.add_argument("--refined", action="store_true")
            command.add_argument("--policies", nargs="+", choices=POLICIES)
            scenarios = ["main", "mu_low", "mu_high", "emotion_low", "emotion_high"]
            scenarios += [f"{key}_{side}" for key in SETTINGS["sensitivity"] for side in ("low", "high")]
            command.add_argument("--scenario", choices=scenarios, default="main")
        if name == "review":
            command.add_argument("--interval", choices=["adaptive", "1", "2", "4"], default="adaptive")
    args = parser.parse_args()
    network = getattr(args, "network", "rt_mt")
    output = args.output or Path("outputs") / args.task / network
    if args.task == "demo":
        demo(output)
    elif args.task == "prepare":
        prepare(network, args.output or Path("data/contacts") / network, args.raw_dir)
    elif args.task == "reproduce":
        folder = args.contacts or Path("data/contacts") / network
        reproduce(network, output, args.model, folder, args.policies, args.followup_hours)
    elif args.task == "solve":
        overrides = {}
        if args.refined and args.policies and any(p not in ("joint_held", "optimized_constant", "peer_free_design") for p in args.policies):
            parser.error("The common-tolerance comparison uses joint, constant and peer-free policies.")
        if args.scenario != "main":
            parameter, side = args.scenario.rsplit("_", 1)
            if network != "rt_mt" and not (parameter == "emotion" and network == "synthetic"):
                parser.error("This parameter comparison uses RT–MT; initial-emotion comparisons also use synthetic contacts.")
            values = SETTINGS["sensitivity"].get(parameter, {}).get("values")
            if parameter == "mu":
                values = SETTINGS["exit_rates"]
            elif parameter == "emotion":
                parameter = "initial_y"; values = SETTINGS["initial_emotion"]
            overrides[parameter] = values[0 if side == "low" else -1]
        solve(network, output, overrides, args.refined, args.policies)
    elif args.task == "review":
        review(network, args.interval, output)
    else:
        folder = args.contacts or Path("data/contacts") / network
        if args.task == "withdrawal":
            withdrawal(folder, output)
        else:
            timing(network, folder, output)


if __name__ == "__main__":
    main()
