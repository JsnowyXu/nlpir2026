#!/usr/bin/env python
"""Persistent, bounded job queue; never terminates processes outside this queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from _bootstrap import configure

configure()
from patentflow_main.io import ROOT, PACKAGE_ROOT, atomic_json


def main(args):
    plan = json.loads(Path(args.plan).read_text())
    state_path = ROOT / "state" / (plan["name"] + ".json")
    lock = state_path.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = plan["jobs"]
    records = (
        json.loads(state_path.read_text()).get("jobs", {})
        if state_path.exists()
        else {}
    )
    running = {}
    env = dict(
        os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4"
    )
    while True:
        for key, (process, handle, gpu) in list(running.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                job = next(j for j in jobs if j["id"] == key)
                complete = (ROOT / job["output"]).exists()
                records[key].update(
                    status="complete" if code == 0 and complete else "failed",
                    returncode=code,
                    finished=time.time(),
                )
                print(key, records[key]["status"], flush=True)
                del running[key]
        used = {value[2] for value in running.values()}
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            available = {
                int(row.split(",")[0]): int(row.split(",")[1])
                for row in output.strip().splitlines()
            }
        except (OSError, subprocess.SubprocessError):
            available = {}
        for job in jobs:
            key = job["id"]
            if key in running:
                continue
            if (ROOT / job["output"]).exists():
                records.setdefault(key, {}).update(
                    status="complete", output=job["output"]
                )
                continue
            if records.get(key, {}).get("status") == "failed":
                continue
            missing = [p for p in job.get("requires", []) if not (ROOT / p).exists()]
            if missing:
                records[key] = {"status": "waiting_for_inputs", "missing": missing}
                continue
            if job.get("device") == "cpu":
                choices = [
                    "cpu:%d" % i
                    for i in range(args.cpu_slots)
                    if "cpu:%d" % i not in used
                ]
            else:
                choices = [
                    g
                    for g in args.gpus
                    if g not in used
                    and available.get(g, 0) >= job.get("minimum_free_mb", 1800)
                ]
            if not choices:
                records[key] = {
                    "status": (
                        "waiting_for_cpu"
                        if job.get("device") == "cpu"
                        else "waiting_for_gpu"
                    )
                }
                continue
            gpu = (
                choices[0]
                if job.get("device") == "cpu"
                else max(choices, key=lambda g: available[g])
            )
            device = "cpu" if job.get("device") == "cpu" else "cuda:%d" % gpu
            command = [sys.executable, str(PACKAGE_ROOT / "train.py")] + [
                str(v).replace("{device}", device) for v in job["argv"][1:]
            ]
            log = ROOT / "logs" / (plan["name"] + "_" + key + ".log")
            handle = log.open("a")
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            records[key] = {
                "status": "running",
                "pid": process.pid,
                "gpu": gpu,
                "started": time.time(),
                "command": command,
                "log": str(log),
            }
            running[key] = (process, handle, gpu)
            used.add(gpu)
            print("START", key, process.pid, gpu, flush=True)
        atomic_json(
            state_path,
            {
                "updated_unix": time.time(),
                "supervisor_pid": os.getpid(),
                "jobs": records,
            },
        )
        if not running and all(
            records.get(j["id"], {}).get("status") in ("complete", "failed")
            for j in jobs
        ):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--cpu-slots", type=int, default=0)
    main(parser.parse_args())
