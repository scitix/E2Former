# -*- coding: utf-8 -*-
"""
Do molecular dynamics simulation with a trained model from fairchem.
Modified from: https://github.com/kyonofx/MDsim/blob/main/simulate.py
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import mdsim.md.integrator as md_integrator
import numpy as np
import torch
import yaml
from ase import units
from ase.io import Trajectory
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.utils import load_config
from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from mdsim.md.ase_utils import Simulator, data_to_atoms


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runcmd(cmd_list):
    return subprocess.run(
        cmd_list,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def eval_and_init(config):
    # load model.
    model_dir = config["model_dir"]

    model_ckpt = str(Path(model_dir) / "best_checkpoint.pt")
    model_config = str(config["model_config_yml"])

    if "test_dataset_src" not in config:
        config["test_dataset_src"] = config["dataset_src"]
    calculator = OCPCalculator(
        config_yml=model_config, checkpoint_path=model_ckpt, cpu=False
    )
    test_metrics = {}
    test_metrics["num_params"] = sum(
        p.numel() for p in calculator.trainer.model.parameters()
    )
    return calculator, test_metrics


def simulate(config, calculator, test_metrics):
    (Path(config["model_dir"]) / config["save_name"]).mkdir(parents=True, exist_ok=True)
    trajectory_path = Path(config["model_dir"]) / config["save_name"] / "atoms.traj"
    thermo_log_path = Path(config["model_dir"]) / config["save_name"] / "thermo.log"

    RESTART = False
    if trajectory_path.exists():
        if not thermo_log_path.exists():
            raise ValueError("trajectory exists but thermo.log does not exist.")
        history = Trajectory(trajectory_path)
        if len(history) > 0 and not config["purge"]:
            atoms = history[-1]
            with open(thermo_log_path, "r") as f:
                last_line = f.read().splitlines()[-1]
            simulated_time = [float(x) for x in last_line.split(" ") if x][0]
            simulated_step = int(
                simulated_time / config["integrator_config"]["timestep"] * 1000
            )
            RESTART = True
            print(f"Found existing simulation. Simulated time: {simulated_time} ps")
        else:
            os.remove(trajectory_path)
            os.remove(thermo_log_path)

    if not RESTART:
        test_dataset = LmdbDataset({"src": config["dataset_src"]})
        if "init_idx" in config:
            init_idx = config["init_idx"]
        else:
            init_idx = random.randint(0, len(test_dataset))

        init_data = test_dataset[init_idx]
        atoms = data_to_atoms(init_data)
        simulated_time = 0
        simulated_step = 0
        print("Start simulation from scratch.")

    if simulated_step > config["steps"]:
        print(
            f'Simulated step {simulated_step} > {config["steps"]}. Simulation already complete.'
        )
        return

    # set calculator.
    atoms.set_calculator(calculator)

    # adjust units.
    config["integrator_config"]["timestep"] *= units.fs
    if config["integrator"] in ["NoseHoover", "NoseHooverChain"]:
        config["integrator_config"]["temperature"] *= units.kB

    # set up simulator.
    integrator = getattr(md_integrator, config["integrator"])(
        atoms, **config["integrator_config"]
    )
    simulator = Simulator(
        atoms,
        integrator,
        config["T_init"],
        restart=RESTART,
        start_time=simulated_time,
        save_dir=Path(config["model_dir"]) / config["save_name"],
        save_frequency=config["save_freq"],
    )

    # run simulation.
    start_time = time.time()
    early_stop, step = simulator.run(config["steps"] - simulated_step)
    elapsed = time.time() - start_time
    test_metrics["running_time"] = elapsed
    test_metrics["early_stop"] = early_stop
    test_metrics["simulated_frames"] = step

    with open(
        Path(config["model_dir"]) / config["save_name"] / "test_metric.json", "w"
    ) as f:
        json.dump(test_metrics, f)


def main(config):
    seed_everywhere(config["seed"])
    save_name = "md"
    if config["identifier"] is not None:
        save_name = "md_" + config["identifier"] + "_" + str(config["seed"])
    if "init_idx" in config:
        save_name = save_name + "_init_" + str(config["init_idx"])
    config["save_name"] = save_name
    os.makedirs(Path(config["model_dir"]) / save_name, exist_ok=True)
    with open(
        os.path.join(Path(config["model_dir"]) / save_name, "config.yml"), "w"
    ) as yf:
        yaml.dump(config, yf, default_flow_style=False)

    calculator, test_metrics = eval_and_init(config)
    simulate(config, calculator, test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation_config_yml", required=True, type=Path)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_config_yml", type=str)
    parser.add_argument("--identifier", type=str)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--init_idx",
        type=int,
        help="the index of the initial state selected from the init dataset.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="if <True>, remove the previous run if exists.",
    )

    args, override_args = parser.parse_known_args()
    config, _, _ = load_config(args.simulation_config_yml)
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(overrides)
    main(config)
