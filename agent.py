"""Kaggle-side agent for an explicitly allowlisted GitHub job queue.

The agent never executes shell commands or arbitrary source from an issue.
Only issues opened by the repository owner with exact, supported JSON jobs
are accepted. Credentials are read from Kaggle Secrets, never committed.
"""
from __future__ import annotations

import json
import base64
import subprocess
from pathlib import Path
import os
import platform
import sys
import time
import traceback

import requests

REPO = "crystalcali/nca-world-kaggle"
OWNER = "crystalcali"
BASE = f"https://api.github.com/repos/{REPO}"
TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
POLL_SECONDS = 15
MAX_WAIT_SECONDS = 600
PREFIX = "[COLAB-JOB]"
CLAIM = "[colab-agent-claim]"
RESULT = "[colab-agent-result]"

if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN is missing; set it using Kaggle Secrets.")

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})


def github(method: str, path: str, **kwargs):
    response = SESSION.request(method, BASE + path, timeout=30, **kwargs)
    if not response.ok:
        # Never print request headers or tokens.
        raise RuntimeError(f"GitHub {method} {path}: HTTP {response.status_code} {response.text[:300]}")
    return response.json() if response.content else None


def smoke_test() -> dict:
    import torch

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if not torch.cuda.is_available():
        report["status"] = "no_gpu"
        return report

    for index in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(index)
        report["gpus"].append({
            "name": p.name,
            "memory_gib": round(p.total_memory / (1024 ** 3), 2),
        })

    device = torch.device("cuda:0")
    with torch.no_grad():
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(8):
            c = a @ b
        torch.cuda.synchronize()
        report["matmul_8x1024_ms"] = round(1000 * (time.perf_counter() - start), 2)
        report["output_mean"] = round(c.float().mean().item(), 6)
    report["status"] = "ok"
    return report


def experiment_pilot() -> dict:
    # Only this checked-in source and fixed arguments are allowed.
    # Issues do not contain executable instructions.
    raw = github("GET", "/contents/train_v1.py")
    content = base64.b64decode(raw["content"])
    folder = Path("/kaggle/working/nca_world")
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "train_v1.py"
    script.write_bytes(content)
    command = [sys.executable, "-u", str(script), "--pilot",
               "--out", str(folder / "pilot_output")]
    env = os.environ.copy()
    env.pop("GITHUB_TOKEN", None)
    print("Starting allowlisted pilot", flush=True)
    process = subprocess.Popen(command, cwd=str(folder), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    final = None
    last_lines = []
    for line in process.stdout:
        line = line.rstrip()
        print(line, flush=True)
        last_lines.append(line)
        last_lines = last_lines[-20:]
        if line.startswith("FINAL_RESULTS_JSON="):
            final = json.loads(line.partition("=")[2])
    status = process.wait()
    if status != 0:
        raise RuntimeError("Experiment failed with exit code " + str(status) +
                           ". Last output: " + "\\n".join(last_lines)[-1600:])
    if final is None:
        raise RuntimeError("Experiment ended without FINAL_RESULTS_JSON")
    return final


def experiment_v2(pilot: bool) -> dict:
    # Fixed repository path and fixed arguments, not user-provided commands.
    raw = github("GET", "/contents/train_v2.py")
    content = base64.b64decode(raw["content"])
    folder = Path("/kaggle/working/nca_world")
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "train_v2.py"
    script.write_bytes(content)
    output = folder / ("v2_pilot" if pilot else "v2_train")
    command = [sys.executable, "-u", str(script), "--out", str(output)]
    if pilot:
        command += ["--pilot"]
    env = os.environ.copy()
    env.pop("GITHUB_TOKEN", None)
    print("Running fixed NCA v2", "pilot" if pilot else "training", flush=True)
    process = subprocess.Popen(command, cwd=str(folder), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    final = None
    last_lines = []
    for line in process.stdout:
        line = line.rstrip()
        print(line, flush=True)
        last_lines.append(line)
        last_lines = last_lines[-30:]
        if line.startswith("FINAL_RESULTS_JSON="):
            final = json.loads(line.partition("=")[2])
    status = process.wait()
    if status != 0:
        raise RuntimeError("V2 run failed with exit code " + str(status) +
                           ". Last output: " + "\n".join(last_lines)[-2000:])
    if final is None:
        raise RuntimeError("V2 run ended without FINAL_RESULTS_JSON")
    return final


def experiment_v3(pilot: bool) -> dict:
    # Fixed repository path and fixed arguments, not user-provided commands.
    raw = github("GET", "/contents/train_v3.py")
    content = base64.b64decode(raw["content"])
    folder = Path("/kaggle/working/nca_world")
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "train_v3.py"
    script.write_bytes(content)
    output = folder / ("v3_pilot" if pilot else "v3_train")
    command = [sys.executable, "-u", str(script), "--out", str(output)]
    if pilot:
        command += ["--pilot"]
    env = os.environ.copy()
    env.pop("GITHUB_TOKEN", None)
    print("Running fixed NCA v3", "pilot" if pilot else "training", flush=True)
    process = subprocess.Popen(command, cwd=str(folder), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    final = None
    last_lines = []
    for line in process.stdout:
        line = line.rstrip()
        print(line, flush=True)
        last_lines.append(line)
        last_lines = last_lines[-30:]
        if line.startswith("FINAL_RESULTS_JSON="):
            final = json.loads(line.partition("=")[2])
    status = process.wait()
    if status != 0:
        raise RuntimeError("V3 run failed with exit code " + str(status) +
                           ". Last output: " + "\n".join(last_lines)[-2000:])
    if final is None:
        raise RuntimeError("V3 run ended without FINAL_RESULTS_JSON")
    return final


# Fixed, audited task names. No arbitrary code from GitHub issues.
ALLOWED_TASKS = {"gpu_smoke": smoke_test, "nca_pilot": experiment_pilot,
                 "nca_v2_pilot": lambda: experiment_v2(True),
                 "nca_v2_train": lambda: experiment_v2(False),
                 "nca_v3_pilot": lambda: experiment_v3(True),
                 "nca_v3_train": lambda: experiment_v3(False)}


def is_processed(issue_number: int) -> bool:
    comments = github("GET", f"/issues/{issue_number}/comments?per_page=100")
    return any(
        CLAIM in (item.get("body") or "") or RESULT in (item.get("body") or "")
        for item in comments
    )


def report_comment(issue_number: int, marker: str, payload: str) -> None:
    github("POST", f"/issues/{issue_number}/comments", json={"body": marker + "\n\n" + payload})


def process_one() -> bool:
    issues = github("GET", "/issues?state=open&per_page=100&sort=created&direction=asc")
    for issue in issues:
        if not str(issue.get("title", "")).startswith(PREFIX):
            continue
        if issue.get("user", {}).get("login", "").lower() != OWNER:
            continue
        number = issue["number"]
        try:
            job = json.loads(issue.get("body") or "")
        except (TypeError, ValueError):
            continue
        if not isinstance(job, dict) or set(job) != {"task"}:
            continue
        task_name = job["task"]
        if task_name not in ALLOWED_TASKS:
            continue
        if is_processed(number):
            continue
        report_comment(number, CLAIM, "Kaggle agent accepted: " + task_name)
        print(f"Executing {task_name} (GitHub issue #{number})", flush=True)
        try:
            data = ALLOWED_TASKS[task_name]()
            output = {"task": task_name, "result": data}
        except Exception as exc:
            output = {"task": task_name, "error": str(exc), "traceback": traceback.format_exc()[-1800:]}
        report_comment(number, RESULT, "```json\n" + json.dumps(output, indent=2, ensure_ascii=False) + "\n```")
        print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
        return True
    return False


def main() -> None:
    print(f"Agent started for {REPO}; polling for up to {MAX_WAIT_SECONDS // 60} minutes.", flush=True)
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            if process_one():
                return
        except requests.RequestException as exc:
            print(f"Network error: {type(exc).__name__}. Retrying...", flush=True)
        except RuntimeError as exc:
            print(f"Agent error: {exc}", flush=True)
        time.sleep(POLL_SECONDS)
    print("No unprocessed approved job was found before timeout.", flush=True)


if __name__ == "__main__":
    main()
