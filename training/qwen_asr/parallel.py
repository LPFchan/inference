#!/usr/bin/env python3
"""Admit and supervise at most four concurrent Qwen3-ASR model workers within shared memory budgets.

Slot 1 is the serial drivers' existing `.gpu-loop.lock`; slots 2-4 live under
`ROOT/slots/`. Each job owns a directory under `ROOT/parallel/` claimed by
`.parallel.lock`. A worker inherits its slot and job locks, so both stay busy
until the worker itself exits, even if this launcher dies. Every live
`train.py` or `batching.py` interpreter counts, registered or not. Budgets are
reservations checked against measured physical footprints (which include Metal
buffers); MLX `--memory-gb` is a soft guideline, and nothing here is kernel
isolation. A launcher only ever stops workers it started. Standard library only.
"""
import argparse
import contextlib
import ctypes
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

GB = 1_000_000_000
WORKER_BYTES = 5 * GB  # largest budget for one new worker; unregistered model processes reserve at least this
AGGREGATE_BYTES = 20 * GB
SYSTEM_ALLOWANCE_BYTES = 4 * GB  # set aside for macOS and apps, not measured free memory
SLOTS = 4
NORMAL, CRITICAL = 1, 4  # kern.memorystatus_vm_pressure_level
DENIED = 75  # EX_TEMPFAIL: retry later
SCHEDULE = ("ko", "ja", "zh") * 3 + ("en",)
SCRIPT = re.compile(r"qwen_asr/(?:train|batching)\.py$")
MODEL = re.compile(r"^\S*python[\d.]*\s+(?:-\S+\s+)*\S*qwen_asr/(?:train|batching)\.py(?:\s|$)", re.I)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slot_lock(root, slot):
    return root / ".gpu-loop.lock" if slot == 1 else root / "slots" / f"{slot}.lock"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- macOS measurements (no sudo, bounded) -------------------------------------

class _Usage(ctypes.Structure):  # rusage_info_v4: uuid then 35 uint64 fields
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]


@functools.lru_cache(maxsize=None)
def _libproc():
    return ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint(pid):
    """Physical footprint now and lifetime maximum, including Metal buffers; None if gone."""
    usage = _Usage()
    if _libproc().proc_pid_rusage(pid, 4, ctypes.byref(usage)):
        return None
    return {"bytes": usage.values[7], "max_bytes": usage.values[28]}


def _run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=10).stdout


def system():
    """Activity Monitor-style used memory (app + wired + compressed), swap and kernel pressure level."""
    page = int(_run("sysctl", "-n", "hw.pagesize"))
    stats = {k: int(v) for k, v in re.findall(r'^"?([^":\n]+)"?:\s+(\d+)\.', _run("vm_stat"), re.M)}
    used = (stats["Anonymous pages"] - stats["Pages purgeable"] + stats["Pages wired down"]
            + stats["Pages occupied by compressor"]) * page
    swap = float(re.search(r"used = ([\d.]+)M", _run("sysctl", "-n", "vm.swapusage")).group(1))
    return {"total_bytes": int(_run("sysctl", "-n", "hw.memsize")), "used_bytes": used,
            "swap_used_bytes": int(swap * (1 << 20)), "swapouts": stats["Swapouts"],
            "pageouts": stats["Pageouts"],
            "pressure_level": int(_run("sysctl", "-n", "kern.memorystatus_vm_pressure_level"))}


def models():
    """PIDs whose interpreter runs qwen_asr/train.py or batching.py directly (not shells or launchers)."""
    found = set()
    for line in _run("ps", "-axo", "pid=,args=").splitlines():
        pid, _, args = line.strip().partition(" ")
        if pid.isdigit() and MODEL.match(args.strip()):
            found.add(int(pid))
    return found


def records(root):
    """Launcher records whose worker process is still alive."""
    found = {}
    for path in (root / "slots").glob("[1-4].json"):
        record = json.loads(path.read_text())
        if record.get("status") == "running" and alive(record["worker_pid"]):
            found[record["slot"]] = record
    return found


def measure(root):
    """Footprint and budget of every live model process. Registered workers keep their own budget."""
    registered = {r["worker_pid"]: r for r in records(root).values()}
    return {pid: {**(footprint(pid) or {"bytes": 0, "max_bytes": 0}),
                  "budget_bytes": registered[pid]["budget_bytes"] if pid in registered else WORKER_BYTES,
                  "slot": registered[pid]["slot"] if pid in registered else None}
            for pid in models() | set(registered)}


# --- admission ----------------------------------------------------------------

def gate(root):
    """Serialize admission decisions; held until the admitted worker is registered."""
    (root / "slots").mkdir(parents=True, exist_ok=True)
    handle = (root / "slots/admission.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def census(root, claim=False, aggregate=AGGREGATE_BYTES, allowance=SYSTEM_ALLOWANCE_BYTES):
    """Busy slots, live workers and reservations (caller holds the gate). With claim, keep a free slot locked."""
    busy, mine = [], None
    for slot in (2, 3, 4, 1):  # touch the serial drivers' lock last
        path = slot_lock(root, slot)
        if not claim and not path.exists():
            continue
        handle = path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            busy.append(slot)
            continue
        if claim and mine is None:
            mine = (slot, handle)
        else:
            handle.close()
    units = measure(root)
    legacy = 1 in busy and all(u["slot"] != 1 for u in units.values())
    # A serial driver holding slot 1 between trials has no model process yet: reserve one worker for it.
    idle_legacy = legacy and all(u["slot"] for u in units.values())
    state = system()
    models_bytes = sum(u["bytes"] for u in units.values())
    facts = {"busy_slots": sorted(busy), "legacy_slot": legacy,
             "workers": {str(pid): u for pid, u in sorted(units.items())},
             "worker_count": len(units) + idle_legacy,
             "reserved_bytes": sum(max(u["budget_bytes"], u["max_bytes"]) for u in units.values())
             + WORKER_BYTES * idle_legacy,
             "limit_bytes": min(aggregate, state["total_bytes"] - allowance),
             "models_bytes": models_bytes, "outside_models_estimate_bytes": state["used_bytes"] - models_bytes,
             "system": state}
    return facts, mine


def claim_job(job):
    """Lock the job directory, or None if another live process holds it."""
    job.mkdir(parents=True, exist_ok=True)
    handle = (job / ".parallel.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def admit(root, job, budget, idle=False, **limits):
    """Claim a slot and the job (caller holds the gate). Returns ((slot, slot_lock, job_lock) or None, facts)."""
    facts, mine = census(root, claim=True, **limits)
    lock = claim_job(job)
    reasons = [text for failed, text in (
        ((root / "STOP").exists(), "ROOT/STOP present"),
        (mine is None, "no free slot lock"),
        (facts["worker_count"] + 1 > SLOTS, "four model workers already live"),
        (facts["reserved_bytes"] + budget > facts["limit_bytes"], "memory reservation unavailable"),
        (facts["system"]["pressure_level"] != NORMAL, "memory pressure not normal"),
        (idle and (facts["busy_slots"] or facts["workers"]), "benchmark requires an idle machine"),
        (lock is None, "job directory claimed by another process"),
    ) if failed]
    facts.update(budget_bytes=budget, denied=reasons)
    if reasons:
        for handle in (mine and mine[1], lock):
            if handle:
                handle.close()
        return None, facts
    return (mine[0], mine[1], lock), facts


def script_of(command):
    return next((Path(arg) for arg in command if SCRIPT.search(arg)), None)


def validate(root, command, budget, job=None):
    """Return (job directory, problems) for a train.py or batching.py command line."""
    problems = []
    script = script_of(command)
    if script is None or (root / "code") not in script.resolve().parents:
        problems.append("command must run train.py or batching.py from a frozen ROOT/code/ source")
    probe = script is not None and script.name == "batching.py"

    def value(flag):
        return command[command.index(flag) + 1] if flag in command[:-1] else None
    output = Path(value("--output") or root / "missing").resolve()
    job = Path(job).resolve() if job else None if probe else output
    if job is None or (root / "parallel") not in job.parents:
        problems.append("job directory must be inside ROOT/parallel/ (probes need --job-dir)")
    elif probe and (job not in output.parents or output.exists()):
        problems.append("probe --output must be a new report inside its --job-dir")
    elif not probe and output != job:
        problems.append("trainer --job-dir must be its --output directory")
    memory = value("--memory-gb")
    if memory is None or float(memory) * (1 << 30) > budget:
        problems.append(f"--memory-gb must be at most {budget / (1 << 30):.3f} GiB (budget {budget} bytes)")
    if not 0 < budget <= WORKER_BYTES:
        problems.append(f"budget must be at most {WORKER_BYTES} bytes")
    if not probe and (output / "run.json").exists() and "--resume" not in command:
        problems.append("existing run requires explicit --resume")
    return job, problems


# --- supervision --------------------------------------------------------------

def spawn(root, admitted, command, job, budget, facts):
    slot, slot_handle, job_handle = admitted
    stream = (job / "process.log").open("a")
    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                             pass_fds=(slot_handle.fileno(), job_handle.fileno()),
                             env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    stream.close()
    record = {"slot": slot, "status": "running", "launcher_pid": os.getpid(), "worker_pid": child.pid,
              "job": str(job), "command": command, "budget_bytes": budget, "admitted": time.time(),
              "supervisor_sha256": sha256(__file__), "script_sha256": sha256(script_of(command)),
              "admission": facts}
    path = root / "slots" / f"{slot}.json"
    if path.exists():
        with (root / "slots/history.jsonl").open("a") as history:
            history.write(json.dumps(json.loads(path.read_text()), ensure_ascii=False) + "\n")
    atomic_json(path, record)
    atomic_json(job / "admission.json", record)
    return {"slot": slot, "handles": (slot_handle, job_handle), "child": child, "record": record,
            "path": path, "job": job}


def newest(root, workers):
    """This launcher's running worker if it is the newest one whose launcher still supervises it."""
    latest = max((r for r in records(root).values() if alive(r["launcher_pid"])),
                 key=lambda r: r["admitted"], default=None)
    return next((w for w in workers if latest and w["child"].pid == latest["worker_pid"]
                 and w["child"].poll() is None), None)


def supervise(root, workers, interval=10.0, strikes=3, aggregate=AGGREGATE_BYTES):
    """Sample memory; after `strikes` consecutive bad samples stop an owned worker to restore bounds.

    A worker above its own budget is stopped. Aggregate or critical-pressure overage stops only the
    newest supervised worker, and only by its own launcher. STOP, signals and sampler failures stop
    every owned worker; legacy and unrelated processes are never touched.
    """
    stopping = []
    previous = {sig: signal.signal(sig, lambda *_: stopping.append(True)) for sig in (signal.SIGTERM, signal.SIGINT)}
    pressure = total = 0
    try:
        while any(w["child"].poll() is None for w in workers):
            try:
                state, units = system(), measure(root)
            except Exception as error:  # never leave owned workers running unsupervised
                for w in workers:
                    _terminate(w, "sampler-failed", repr(error))
                break
            live = sum(max(u["budget_bytes"], u["max_bytes"]) for u in units.values())
            pressure = pressure + 1 if state["pressure_level"] >= CRITICAL else 0
            total = total + 1 if live > aggregate else 0
            for w in workers:
                if w["child"].poll() is not None:
                    continue
                unit = units.get(w["child"].pid, {"bytes": 0, "max_bytes": 0})
                over = [text for failed, text in ((unit["bytes"] > w["record"]["budget_bytes"], "worker"),
                                                  (live > aggregate, "aggregate"),
                                                  (state["pressure_level"] >= CRITICAL, "critical")) if failed]
                w["strikes"] = w.get("strikes", 0) + 1 if "worker" in over else 0
                w["peak"] = max(w.get("peak", 0), unit["max_bytes"])
                sample = {"time": round(time.time(), 1), "bytes": unit["bytes"], "max_bytes": unit["max_bytes"],
                          "live_reserved_bytes": live, "used_bytes": state["used_bytes"],
                          "swap_used_bytes": state["swap_used_bytes"], "pressure_level": state["pressure_level"],
                          "over": over}
                with (w["job"] / "memory.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                if w["strikes"] >= strikes:
                    _terminate(w, "evicted-worker-budget")
            if stopping or (root / "STOP").exists():
                for w in workers:
                    _terminate(w, "stopped")
            elif pressure >= strikes or total >= strikes:
                victim = newest(root, workers)
                if victim:
                    _terminate(victim, "evicted-memory-pressure" if pressure >= strikes else "evicted-aggregate")
                pressure = total = 0
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline and any(w["child"].poll() is None for w in workers):
                time.sleep(min(0.2, interval))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for w in workers:
            _terminate(w, "launcher-error")  # no-op unless the loop raised
            record = {**w["record"], "status": w.get("status", "finished"), "error": w.get("error"),
                      "returncode": w["child"].returncode, "finished": time.time(),
                      "peak_footprint_bytes": w.get("peak", 0)}
            atomic_json(w["job"] / "admission.json", record)
            atomic_json(w["path"], record)
            for handle in w["handles"]:
                handle.close()
    return workers


def _terminate(worker, status, error=None):
    """Stop one owned worker with SIGTERM, then SIGKILL after 30 s. Logs and checkpoints stay."""
    if worker["child"].poll() is not None:
        return
    worker.update(status=status, error=error)
    worker["child"].terminate()
    try:
        worker["child"].wait(timeout=30)
    except subprocess.TimeoutExpired:
        worker["child"].kill()
        worker["child"].wait()


def run(a):
    root = a.root.resolve()
    job, problems = validate(root, a.command, a.budget_bytes, a.job_dir)
    if problems:
        print(json.dumps({"admitted": False, "denied": problems}), flush=True)
        return 2
    while True:
        with gate(root):
            admitted, facts = admit(root, job, a.budget_bytes)
            if admitted:
                worker = spawn(root, admitted, a.command, job, a.budget_bytes, facts)
        print(json.dumps({"admitted": bool(admitted), "slot": admitted and admitted[0], "job": str(job),
                          "denied": facts["denied"], "reserved_bytes": facts["reserved_bytes"],
                          "limit_bytes": facts["limit_bytes"],
                          "pressure_level": facts["system"]["pressure_level"]}), flush=True)
        if admitted:
            break
        if not a.wait or (root / "STOP").exists():
            return DENIED
        time.sleep(a.wait)
    [worker] = supervise(root, [worker])
    status = worker.get("status", "finished")
    print(json.dumps({"job": str(job), "status": status, "returncode": worker["child"].returncode,
                      "peak_footprint_bytes": worker.get("peak", 0)}), flush=True)
    if status.startswith("evicted"):
        return DENIED
    return worker["child"].returncode if status == "finished" else 1


# --- throughput benchmark -----------------------------------------------------

def sampled_audio_seconds(data, seed, examples):
    """Audio seconds train.py's SFT sampler visits for these indices."""
    rows = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines() if line.strip()]
    pools = {lang: [r for r in rows if r["language"] == lang] for lang in set(SCHEDULE)}
    return sum(random.Random(seed + i).choice(pools[SCHEDULE[i % len(SCHEDULE)]])["seconds"]
               for i in range(examples))


def bench(a):
    root, source = a.root.resolve(), a.source.resolve()
    folder = root / "parallel" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    examples = a.steps * a.accumulate
    audio = sampled_audio_seconds(a.data, a.seed, examples)
    summary = {"settings": {k: str(v) for k, v in vars(a).items() if k != "func"}, "examples_per_worker": examples,
               "audio_seconds_per_worker": audio, "phases": []}
    for count in a.workers:
        workers = []
        start_system = system()
        with gate(root):
            for index in range(count):
                output = folder / f"w{count}-{index}"
                command = [str(a.runtime), "-u", str(source / "training/qwen_asr/train.py"), "sft",
                           "--model", str(a.model), "--data", str(a.data), "--protected-test", str(a.protected_test),
                           "--output", str(output), "--steps", str(a.steps), "--accumulate", str(a.accumulate),
                           "--save-every", str(a.steps), "--skip-initial-eval", "--eval-clips", "1",
                           "--seed", str(a.seed), "--learning-rate", "3e-5", "--memory-gb", str(a.memory_gb)]
                job, problems = validate(root, command, a.budget_bytes)
                admitted, facts = (None, {"denied": []}) if problems else \
                    admit(root, job, a.budget_bytes, idle=index == 0)
                if not admitted:
                    for w in workers:
                        _terminate(w, "aborted")
                    supervise(root, workers, interval=1)
                    summary["aborted"] = {"workers": count, "denied": problems + facts["denied"]}
                    atomic_json(folder / "bench.json", summary)
                    print(json.dumps(summary["aborted"]), flush=True)
                    return DENIED
                workers.append(spawn(root, admitted, command, job, a.budget_bytes, facts))
        began = time.perf_counter()
        supervise(root, workers, interval=2)
        wall = time.perf_counter() - began
        end_system = system()
        rates, peaks = [], []
        for w in workers:
            steps = [json.loads(x) for x in (w["job"] / "metrics.jsonl").read_text().splitlines()] \
                if (w["job"] / "metrics.jsonl").exists() else []
            if w["child"].returncode or len(steps) != a.steps:
                summary["aborted"] = {"workers": count, "returncode": w["child"].returncode, "job": str(w["job"])}
                atomic_json(folder / "bench.json", summary)
                return 1
            rates.append(examples / sum(s["seconds"] for s in steps))
            peaks.append({"mlx_bytes": max(s["peak_memory_bytes"] for s in steps), "footprint_bytes": w.get("peak", 0)})
        phase = {"workers": count, "wall_seconds": wall,
                 "end_to_end_examples_per_second": count * examples / wall,
                 "end_to_end_audio_seconds_per_second": count * audio / wall,
                 "steady_examples_per_second": sum(rates),
                 "steady_audio_seconds_per_second": sum(rates) * audio / examples,
                 "per_worker_examples_per_second": rates, "peaks": peaks,
                 "swap_used_delta_bytes": end_system["swap_used_bytes"] - start_system["swap_used_bytes"],
                 "swapouts_delta": end_system["swapouts"] - start_system["swapouts"],
                 "pageouts_delta": end_system["pageouts"] - start_system["pageouts"],
                 "max_pressure_level": max((json.loads(x)["pressure_level"] for w in workers
                                            if (w["job"] / "memory.jsonl").exists()
                                            for x in (w["job"] / "memory.jsonl").read_text().splitlines()),
                                           default=end_system["pressure_level"])}
        summary["phases"].append(phase)
        atomic_json(folder / "bench.json", summary)
        print(json.dumps(phase), flush=True)
        if (root / "STOP").exists():
            return 1
    base = summary["phases"][0]["steady_examples_per_second"]
    for phase in summary["phases"]:
        phase["steady_speedup_vs_first"] = phase["steady_examples_per_second"] / base
    atomic_json(folder / "bench.json", summary)
    print(json.dumps({"bench": str(folder / "bench.json")}), flush=True)
    return 0


def status(a):
    """Read-only snapshot: never creates slot files and never claims a slot."""
    root = a.root.resolve()
    with gate(root) if (root / "slots").exists() else contextlib.nullcontext():
        facts, _ = census(root)
    facts["running"] = {slot: {**r, "launcher_alive": alive(r["launcher_pid"])} for slot, r in records(root).items()}
    print(json.dumps(facts, indent=2))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if "--" in argv:  # everything after the first `--` is the child command, passed unchanged
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1:]
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    for name, func in (("run", run), ("bench", bench), ("status", status)):
        s = sub.add_parser(name)
        s.set_defaults(func=func)
        s.add_argument("root", type=Path)
        if name != "status":
            s.add_argument("--budget-bytes", type=int, default=WORKER_BYTES,
                           help=f"physical-footprint budget per worker, at most {WORKER_BYTES}")
        if name == "run":
            s.add_argument("--wait", type=float, default=0, help="retry admission every N seconds")
            s.add_argument("--job-dir", type=Path, help="required for batching.py probes; trainers use --output")
        if name == "bench":
            for flag in ("--runtime", "--source", "--model", "--data", "--protected-test"):
                s.add_argument(flag, type=Path, required=True)
            s.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
            s.add_argument("--steps", type=int, default=60)
            s.add_argument("--accumulate", type=int, default=4)
            s.add_argument("--seed", type=int, default=21774173)
            s.add_argument("--memory-gb", type=float, default=4.5)
    a = p.parse_args(argv)
    if (a.func is run) != bool(command):
        p.error("run needs `-- COMMAND ...`; other subcommands take no command")
    a.command = command
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
