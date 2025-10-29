# -*- coding: utf-8 -*-
import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yml", required=True)
    parser.add_argument("--cvd", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-dir", default="./")
    parser.add_argument("--timestamp-id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--identifier", default=None)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    # Get the config name without file type (.yaml) and without the 'configs':
    config_name = Path(args.config_yml).stem

    logdir = Path("./logs")
    logdir.mkdir(exist_ok=True)
    timestr = datetime.now().isoformat()

    # Make list of all files in logdir that has 'config_name'
    if args.timestamp_id is not None:
        indentifier = args.timestamp_id
    elif args.identifier is not None:
        indentifier = args.identifier
    else:
        indentifier = config_name
    log_files = [f for f in logdir.glob(f"*{indentifier}*")]
    # Count the number, add one and add that to the log file name
    counter = len(log_files) + 1

    tmux_session_name = f"{args.cvd}_v{counter}_{indentifier}"
    cmd = f"CUDA_VISIBLE_DEVICES={args.cvd}"
    cmd += " python main.py"
    cmd += f" --config-yml {args.config_yml}"
    cmd += f" --mode {args.mode}"
    cmd += f" --run-dir {args.run_dir}"
    if args.timestamp_id is not None:
        cmd += f" --timestamp-id {args.timestamp_id}"
    if args.checkpoint is not None:
        cmd += f" --checkpoint {args.checkpoint}"
    if args.identifier is not None:
        cmd += f" --identifier {args.identifier}"
    if args.amp:
        cmd += " --amp"

    log_path = logdir / f"{tmux_session_name}_{timestr}.txt"
    subprocess_run_cmd = (
        f"tmux new-session -d -s {tmux_session_name} '{cmd} > {log_path} 2>&1'"
    )

    print(f"Running command: {subprocess_run_cmd}")
    subprocess.run(subprocess_run_cmd, shell=True)

    print("tmux session stated with name:", tmux_session_name)
    print("Low priority kill string:")
    print(f"sudo -u {os.getlogin()} kill {os.getpid()}")
