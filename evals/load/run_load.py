"""Drive the cache-ON / cache-OFF load comparison end to end.

    PYTHONPATH=. .venv/bin/python -m evals.load.run_load --minutes 3 --users 3

For each mode it starts a dedicated uvicorn on its own port with
`CACHE_ENABLED` set accordingly, waits for `/health` to go 200 (warmup loads
the embedding model, the cross-encoder and the Colang parse -- 11-24s), snapshots
`/metrics`, runs Locust headless, snapshots `/metrics` again, and stops the
server.

Two details that decide whether the comparison means anything:

**A fresh server per mode.** Restarting is not incidental tidiness. `/metrics`
is a process-lifetime counter, so a mode-2 run against a warm mode-1 process
would report mode 1's latencies inside its own percentiles. Two processes, two
clean counters.

**Warmup is excluded on purpose.** Every run begins after `/health` is green,
so no cold-start cost lands in the percentiles. Phase 7's cold-6.3s /
warm-3.5s figure is the cold number; this report is steady-state and says so.

The cache-ON run goes SECOND and is not pre-warmed. Its first request for each
repeated question is a genuine miss, so the reported hit rate is what a freshly
deployed service would actually see over the window -- not the ceiling you get
by seeding the cache first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"


def wait_ready(base: str, timeout: float = 180.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"{base}/health", timeout=5).status_code == 200:
                return round(time.time() - t0, 1)
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(1)
    raise RuntimeError(f"{base} never became ready")


def metrics(base: str) -> dict:
    return httpx.get(f"{base}/metrics", timeout=10).json()


def run_mode(mode: str, *, port: int, minutes: int, users: int) -> dict:
    base = f"http://localhost:{port}"
    env = {**os.environ, "PYTHONPATH": str(REPO),
           "CACHE_ENABLED": "true" if mode == "cache_on" else "false"}
    log = RESULTS / f"{mode}.server.log"
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {mode}: starting uvicorn on :{port} ===", flush=True)
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [str(REPO / ".venv/bin/uvicorn"), "app.api.main:app",
             "--port", str(port), "--log-level", "warning"],
            cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True)
    try:
        warmup_s = wait_ready(base)
        print(f"    ready in {warmup_s}s", flush=True)
        before = metrics(base)

        csv_prefix = RESULTS / mode
        cmd = [str(REPO / ".venv/bin/locust"),
               "-f", str(HERE / "locustfile.py"), "--headless",
               "-u", str(users), "-r", "1", "-t", f"{minutes}m",
               "--host", base, "--csv", str(csv_prefix),
               "--only-summary"]
        lenv = {**os.environ, "PYTHONPATH": str(REPO),
                "APP_STATS_OUT": str(RESULTS / f"{mode}.app_stats.json")}
        subprocess.run(cmd, cwd=REPO, env=lenv, check=True)

        after = metrics(base)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    app_stats = json.loads((RESULTS / f"{mode}.app_stats.json").read_text())
    return {"mode": mode, "port": port, "minutes": minutes, "users": users,
            "warmup_s": warmup_s, "metrics_before": before,
            "metrics_after": after, "app_stats": app_stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=3)
    ap.add_argument("--users", type=int, default=3)
    ap.add_argument("--port-off", type=int, default=8091)
    ap.add_argument("--port-on", type=int, default=8092)
    args = ap.parse_args()

    if shutil.which(str(REPO / ".venv/bin/locust")) is None:
        print("locust not installed in .venv", file=sys.stderr)
        return 1

    out = {"started_at": time.time(), "runs": []}
    # cache OFF first: it is the baseline, and running it second against a
    # warm provider connection pool would flatter it.
    for mode, port in (("cache_off", args.port_off), ("cache_on", args.port_on)):
        out["runs"].append(run_mode(mode, port=port, minutes=args.minutes,
                                    users=args.users))

    dest = RESULTS / "runs.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
