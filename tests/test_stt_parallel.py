"""Qwen3-ASR admission and supervision: CLI, budgets, probes, locks and eviction (CPU only, no models)."""
import fcntl
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/qwen_asr"))
import parallel  # noqa: E402

GB = parallel.GB
FAKE = """import json, os, sys, time
a = sys.argv
out = a[a.index('--output') + 1]
os.makedirs(out if a[0].endswith('train.py') else os.path.dirname(out), exist_ok=True)
if '--steps' in a:
    with open(out + '/metrics.jsonl', 'w') as f:
        for s in range(int(a[a.index('--steps') + 1])):
            f.write(json.dumps({'step': s + 1, 'seconds': 0.5, 'peak_memory_bytes': 4_000_000_000}) + '\\n')
time.sleep(float(a[a.index('--sleep') + 1]) if '--sleep' in a else 0)
if a[0].endswith('batching.py'):
    open(out, 'x').write('{}')
"""


@pytest.fixture
def host(tmp_path, monkeypatch):
    """Fake Mac: real flock, processes and `ps` scan (scoped to this root); mocked footprints and VM stats."""
    root = tmp_path.resolve() / "root"
    (root / "parallel").mkdir(parents=True)
    source = root / "code/x/training/qwen_asr"
    source.mkdir(parents=True)
    for name in ("train.py", "batching.py"):
        (source / name).write_text(FAKE)
    state = {"root": root, "source": source, "now": {}, "peaks": {}, "pressure": 1, "children": []}
    monkeypatch.setattr(parallel, "MODEL", re.compile(
        r"^\S*python[\d.]*\s+(?:-\S+\s+)*" + re.escape(str(root)) + r"/code/\S*qwen_asr/(?:train|batching)\.py", re.I))
    monkeypatch.setattr(parallel, "footprint", lambda pid: {
        "bytes": state["now"].get(pid, 0), "max_bytes": state["peaks"].get(pid, state["now"].get(pid, 0))})
    monkeypatch.setattr(parallel, "system", lambda: {
        "total_bytes": 25_769_803_776, "used_bytes": 10 * GB, "swap_used_bytes": 0,
        "swapouts": 0, "pageouts": 0, "pressure_level": state["pressure"]})
    yield state
    for child in state["children"]:
        if child.poll() is None:
            child.kill()
            child.wait()


def command(host, output, script="train.py", memory="4.5", sleep=60):
    return [sys.executable, "-u", str(host["source"] / script), "sft", "--output", str(output),
            "--memory-gb", memory, "--sleep", str(sleep)]


def visible(pid):
    """Wait until `ps` shows the interpreter running the script (after exec)."""
    for _ in range(50):
        if pid in parallel.models():
            return pid
        time.sleep(0.05)
    raise AssertionError(f"{pid} never looked like a model process")


def legacy(host, now=4_900_000_000, peak=5_169_499_328):
    """An unregistered trainer like the batch-32 run (its driver holds slot 1 separately)."""
    child = subprocess.Popen(command(host, host["root"] / "trials/legacy"))
    host["children"].append(child)
    host["now"][child.pid], host["peaks"][child.pid] = now, peak
    visible(child.pid)
    return child


def launch(host, name, budget=parallel.WORKER_BYTES, sleep=60):
    """Admit and spawn one trainer exactly like `parallel.py run`."""
    root = host["root"]
    job = root / "parallel" / name
    with parallel.gate(root):
        admitted, facts = parallel.admit(root, job, budget)
        if not admitted:
            return None, facts
        worker = parallel.spawn(root, admitted, command(host, job, sleep=sleep), job, budget, facts)
    host["children"].append(worker["child"])
    visible(worker["child"].pid)
    return worker, facts


def hold(path):
    handle = Path(path).open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def held(path):
    try:
        hold(path).close()
    except BlockingIOError:
        return True
    return False


# --- CLI ----------------------------------------------------------------------

def test_options_before_separator_parse_and_child_arguments_pass_unchanged(host, monkeypatch):
    seen = []
    monkeypatch.setattr(parallel, "run", lambda a: seen.append(a) or 0)
    child = ["python", "-c", "pass", "--wait", "1", "--", "--job-dir"]
    assert parallel.main(["run", "/tmp/root", "--wait", "60", "--job-dir", "/tmp/j", "--", *child]) == 0
    assert seen[0].wait == 60 and seen[0].job_dir == Path("/tmp/j") and seen[0].command == child
    with pytest.raises(SystemExit):
        parallel.main(["run", "/tmp/root", "--wait", "60"])  # no child command


def test_main_runs_an_admitted_probe_end_to_end(host):
    root = host["root"]
    job = root / "parallel/batching-acceptance/m2"
    child = [sys.executable, "-u", str(host["source"] / "batching.py"), "--microbatch", "2",
             "--output", str(job / "parity.json"), "--memory-gb", "4.5", "--sleep", "0.3"]
    assert parallel.main(["run", str(root), "--wait", "5", "--job-dir", str(job), "--", *child]) == 0
    record = json.loads((job / "admission.json").read_text())
    assert record["status"] == "finished" and record["returncode"] == 0 and record["command"] == child
    assert record["job"] == str(job) and record["slot"] == 2 and record["budget_bytes"] == 5 * GB
    assert record["script_sha256"] == parallel.sha256(host["source"] / "batching.py")
    assert (job / "parity.json").is_file() and (job / "memory.jsonl").read_text().strip()
    assert json.loads((root / "slots/2.json").read_text())["status"] == "finished"
    assert parallel.main(["run", str(root), "--job-dir", str(job), "--", *child]) == 2  # report exists


def test_status_is_read_only_before_any_launch(host, capsys):
    assert parallel.main(["status", str(host["root"])]) == 0
    assert json.loads(capsys.readouterr().out)["busy_slots"] == []
    assert not (host["root"] / "slots").exists() and not (host["root"] / ".gpu-loop.lock").exists()


# --- validation and job claims -------------------------------------------------

def test_validation_separates_probe_reports_from_job_directories(host):
    root, source = host["root"], host["source"]
    probe = [sys.executable, str(source / "batching.py"), "--output"]
    job = root / "parallel/batching-acceptance/p"

    def problems(cmd, budget=5 * GB, job_dir=None):
        return parallel.validate(root, cmd, budget, job_dir)[1]
    assert problems(probe + [str(job / "parity.json"), "--memory-gb", "4.5"], job_dir=job) == []
    assert parallel.validate(root, probe + [str(job / "r.json"), "--memory-gb", "4.5"], 5 * GB, job)[0] == job
    assert problems(probe + [str(job / "parity.json")], job_dir=job)  # batching.py defaults to 5 GiB
    assert problems(probe + [str(job / "parity.json"), "--memory-gb", "4.5"])  # probes need --job-dir
    assert problems(probe + [str(root / "parallel/elsewhere.json"), "--memory-gb", "4.5"], job_dir=job)
    job.mkdir(parents=True)
    (job / "parity.json").write_text("{}")
    assert "probe --output must be a new report inside its --job-dir" in problems(
        probe + [str(job / "parity.json"), "--memory-gb", "4.5"], job_dir=job)

    trainer = [sys.executable, str(source / "train.py"), "sft", "--output", str(root / "parallel/t1")]
    assert problems(trainer + ["--memory-gb", "4.5"]) == []
    assert "4.657 GiB" in problems(trainer + ["--memory-gb", "4.7"])[0]
    assert problems(trainer)  # train.py defaults to 18 GiB
    assert problems(trainer + ["--memory-gb", "4.5"], job_dir=root / "parallel/other")
    assert problems(trainer + ["--memory-gb", "4.5"], budget=6 * GB) == [f"budget must be at most {5 * GB} bytes"]
    legacy_output = [sys.executable, str(source / "train.py"), "--output", str(root / "trials/x"), "--memory-gb", "4"]
    assert problems(legacy_output)
    assert problems(["python", "/repo/training/qwen_asr/train.py", *trainer[3:], "--memory-gb", "4"])
    (root / "parallel/t1").mkdir()
    (root / "parallel/t1/run.json").write_text("{}")
    assert problems(trainer + ["--memory-gb", "4"]) == ["existing run requires explicit --resume"]
    assert problems(trainer + ["--memory-gb", "4", "--resume"]) == []


def test_a_claimed_job_directory_refuses_second_probe_or_trainer(host):
    root = host["root"]
    job = root / "parallel/batching-acceptance/p"
    first = command(host, job / "a.json", script="batching.py")
    with parallel.gate(root):
        admitted, facts = parallel.admit(root, job, 5 * GB)
        worker = parallel.spawn(root, admitted, first, job, 5 * GB, facts)
    host["children"].append(worker["child"])
    worker["handles"][1].close()  # the probe itself keeps the inherited job lock
    with parallel.gate(root):
        denied, facts = parallel.admit(root, job, 5 * GB)
    assert denied is None and facts["denied"] == ["job directory claimed by another process"]


# --- reservations --------------------------------------------------------------

def test_legacy_peak_and_registered_overage_are_reserved_once(host):
    root = host["root"]
    driver = hold(root / ".gpu-loop.lock")
    legacy(host)
    first, facts = launch(host, "a")
    assert first["slot"] == 2 and facts["legacy_slot"] and facts["worker_count"] == 1
    assert facts["reserved_bytes"] == 5_169_499_328
    second, _ = launch(host, "b")
    third, facts = launch(host, "c")
    assert third is None and facts["denied"] == ["memory reservation unavailable"]
    assert facts["reserved_bytes"] == 15_169_499_328 and facts["limit_bytes"] == 20 * GB
    host["peaks"][second["child"].pid] = 5_200_000_000  # a registered worker above its budget counts its peak
    third, facts = launch(host, "c", budget=4_640_000_000)
    assert third is None and facts["reserved_bytes"] == 15_369_499_328
    third, _ = launch(host, "c", budget=4_630_000_000 - 1)
    assert third["slot"] == 4
    fifth, facts = launch(host, "d", budget=1)
    assert fifth is None and facts["denied"] == ["no free slot lock", "four model workers already live"]
    driver.close()


def test_idle_serial_driver_reserves_one_worker(host):
    driver = hold(host["root"] / ".gpu-loop.lock")
    worker, facts = launch(host, "a")
    assert facts["worker_count"] == 1 and facts["reserved_bytes"] == 5 * GB
    driver.close()


def test_four_new_workers_fill_every_slot_and_block_a_serial_driver(host):
    slots = [launch(host, name)[0]["slot"] for name in "abcd"]
    assert slots == [2, 3, 4, 1]
    fifth, facts = launch(host, "e", budget=1)
    assert fifth is None and {"no free slot lock", "four model workers already live"} <= set(facts["denied"])
    with pytest.raises(BlockingIOError):  # what cycle.py / the batch-32 driver do first
        hold(host["root"] / ".gpu-loop.lock")


def test_orphan_model_process_without_a_slot_still_occupies_capacity(host):
    legacy(host, now=1, peak=1)
    for name in "abc":
        assert launch(host, name)[0]
    denied, facts = launch(host, "d")
    assert denied is None and facts["worker_count"] == 4 and "four model workers already live" in facts["denied"]


@pytest.mark.parametrize("change,reason", [
    (lambda host: host.update(pressure=2), "memory pressure not normal"),
    (lambda host: (host["root"] / "STOP").touch(), "ROOT/STOP present"),
])
def test_pressure_or_stop_denies_admission(host, change, reason):
    change(host)
    worker, facts = launch(host, "a")
    assert worker is None and reason in facts["denied"]
    assert not (host["root"] / "slots/2.json").exists()
    host.update(pressure=1)
    (host["root"] / "STOP").unlink(missing_ok=True)
    assert launch(host, "a")[0]["slot"] == 2


# --- lock lifetime ---------------------------------------------------------------

def test_locks_survive_launcher_death_and_free_when_the_worker_exits(host):
    root = host["root"]
    job = root / "parallel/t"
    pid = os.fork()
    if pid == 0:  # the launcher: real main(), killed below without any cleanup
        code = 1
        try:
            code = parallel.main(["run", str(root), "--", *command(host, job)])
        finally:
            os._exit(code)
    for _ in range(100):
        if (job / "admission.json").exists():
            break
        time.sleep(0.05)
    worker = json.loads((job / "admission.json").read_text())["worker_pid"]
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    assert parallel.alive(worker) and held(root / "slots/2.lock") and held(job / ".parallel.lock")
    facts, _ = parallel.census(root)
    assert facts["busy_slots"] == [2] and facts["workers"][str(worker)]["slot"] == 2
    assert launch(host, "t")[0] is None  # job still claimed
    assert launch(host, "u")[0]["slot"] == 3
    os.kill(worker, signal.SIGKILL)
    for _ in range(100):
        if not held(root / "slots/2.lock"):
            break
        time.sleep(0.05)
    assert not held(root / "slots/2.lock") and not held(job / ".parallel.lock")
    recovered, _ = launch(host, "t")
    assert recovered["slot"] == 2
    history = [json.loads(x) for x in (root / "slots/history.jsonl").read_text().splitlines()]
    assert history[0]["worker_pid"] == worker


# --- supervision -----------------------------------------------------------------

def test_worker_over_its_own_budget_is_stopped_and_others_continue(host):
    other = legacy(host)
    hog, _ = launch(host, "hog")
    calm, _ = launch(host, "calm", sleep=1.5)
    host["now"][hog["child"].pid] = 5_100_000_000
    parallel.supervise(host["root"], [hog, calm], interval=0.1, strikes=2)
    assert hog["status"] == "evicted-worker-budget" and "status" not in calm
    assert calm["child"].returncode == 0 and other.poll() is None
    samples = [json.loads(x) for x in (host["root"] / "parallel/hog/memory.jsonl").read_text().splitlines()]
    assert samples[-1]["over"] == ["worker"]
    record = json.loads((host["root"] / "parallel/hog/admission.json").read_text())
    assert record["status"] == "evicted-worker-budget" and (host["root"] / "parallel/hog/process.log").exists()


@pytest.mark.parametrize("overload,status", [
    ("aggregate", "evicted-aggregate"),
    ("pressure", "evicted-memory-pressure"),
])
def test_aggregate_or_critical_overage_evicts_only_the_newest_owned_worker(host, monkeypatch, overload, status):
    other = legacy(host)
    older, _ = launch(host, "old", sleep=1.5)
    newer, _ = launch(host, "new")
    if overload == "aggregate":  # the legacy job grows; only stopping the newest owned job restores 20 GB
        host["peaks"][other.pid] = 11 * GB  # current footprint already fell; its reservation keeps the peak
    else:
        fake = parallel.system
        monkeypatch.setattr(parallel, "system",
                            lambda: {**fake(), "pressure_level": 4 if newer["child"].poll() is None else 1})
    parallel.supervise(host["root"], [older, newer], interval=0.1, strikes=2)
    assert newer["status"] == status and "status" not in older and older["child"].returncode == 0
    assert other.poll() is None
    assert json.loads((host["root"] / "parallel/old/admission.json").read_text())["status"] == "finished"


@pytest.mark.parametrize("critical", [False, True])
def test_worker_eviction_does_not_also_evict_a_healthy_worker(host, monkeypatch, critical):
    workers = [launch(host, name, sleep=1.5)[0] for name in ("hog", "healthy-a", "healthy-b", "newest")]
    hog, *healthy = workers
    host["now"][hog["child"].pid] = 5_100_000_000
    host["peaks"][hog["child"].pid] = 5_100_000_000
    if critical:
        fake = parallel.system
        monkeypatch.setattr(parallel, "system", lambda: {
            **fake(), "pressure_level": 4 if hog["child"].poll() is None else 1})
    parallel.supervise(host["root"], workers, interval=0.1, strikes=2)
    assert hog["status"] == "evicted-worker-budget"
    assert all(w["child"].returncode == 0 and "status" not in w for w in healthy)


def test_sampler_failure_stops_owned_workers_and_records_why(host, monkeypatch):
    other = legacy(host)
    worker, _ = launch(host, "a")
    monkeypatch.setattr(parallel, "system", lambda: (_ for _ in ()).throw(subprocess.TimeoutExpired("vm_stat", 10)))
    parallel.supervise(host["root"], [worker], interval=0.1)
    record = json.loads((host["root"] / "parallel/a/admission.json").read_text())
    assert record["status"] == "sampler-failed" and "TimeoutExpired" in record["error"]
    assert worker["child"].returncode is not None and other.poll() is None
    assert not held(host["root"] / "slots/2.lock")


def test_model_scan_counts_interpreters_not_launchers_or_shells():
    match = parallel.MODEL.match
    assert match("/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/"
                 "Resources/Python.app/Contents/MacOS/Python -u /r/code/655b245/training/qwen_asr/train.py sft")
    assert match("/h/.venv/bin/python3.12 /r/code/fee870c/training/qwen_asr/batching.py --output /r/p.json")
    assert not match("/h/.venv/bin/python /r/code/x/training/qwen_asr/parallel.py run /r -- "
                     "/h/.venv/bin/python -u /r/code/x/training/qwen_asr/train.py sft")
    assert not match("sh -c /h/.venv/bin/python -u /r/code/x/training/qwen_asr/train.py sft")
    assert not match("/usr/bin/python3 -u /tmp/stt-qwen-batch32-full-20261029.py")


def test_model_census_keeps_long_commands_at_narrow_terminal_width(host, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    worker = legacy(host)
    assert worker.pid in parallel.models()


# --- benchmark ---------------------------------------------------------------------

def test_benchmark_audio_mirrors_the_trainer_sampler(tmp_path):
    rows = [{"id": f"{lang}{i}", "language": lang, "seconds": i + 1.0}
            for lang in ("ko", "ja", "zh", "en") for i in range(5)]
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    pools = {lang: [r for r in rows if r["language"] == lang] for lang in ("ko", "ja", "zh", "en")}
    expected = sum(random.Random(7 + i).choice(pools[parallel.SCHEDULE[i % 10]])["seconds"] for i in range(40))
    assert parallel.sampled_audio_seconds(tmp_path, 7, 40) == expected


def test_benchmark_runs_identical_phases_and_reports_combined_rates(host):
    root = host["root"]
    data = root / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("\n".join(json.dumps({"id": lang, "language": lang, "seconds": 2.0})
                                                for lang in ("ko", "ja", "zh", "en")))
    code = parallel.main(["bench", str(root), "--runtime", sys.executable, "--source", str(root / "code/x"),
                          "--model", "m", "--data", str(data), "--protected-test", "t",
                          "--workers", "1", "2", "--steps", "3"])
    assert code == 0
    [summary] = [json.loads(p.read_text()) for p in (root / "parallel").glob("bench-*/bench.json")]
    one, two = summary["phases"]
    assert summary["examples_per_worker"] == 12 and summary["audio_seconds_per_worker"] == 24.0
    assert one["steady_examples_per_second"] == pytest.approx(8.0)
    assert two["steady_examples_per_second"] == pytest.approx(16.0)
    assert two["steady_speedup_vs_first"] == pytest.approx(2.0)
    assert {p["mlx_bytes"] for p in two["peaks"]} == {4_000_000_000}
    assert not parallel.records(root)  # every bench slot was released and recorded finished


def test_benchmark_records_a_pre_admission_validation_denial(host):
    root = host["root"]
    data = root / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("\n".join(
        json.dumps({"id": lang, "language": lang, "seconds": 2.0})
        for lang in ("ko", "ja", "zh", "en")))
    result = parallel.main(["bench", str(root), "--runtime", sys.executable,
                            "--source", str(root / "code/x"), "--model", "m",
                            "--data", str(data), "--protected-test", "t",
                            "--workers", "1", "--steps", "3", "--memory-gb", "5"])
    assert result == parallel.DENIED
    [path] = (root / "parallel").glob("bench-*/bench.json")
    summary = json.loads(path.read_text())
    assert summary["phases"] == [] and summary["aborted"]["workers"] == 1
    assert "--memory-gb" in summary["aborted"]["denied"][0]
    assert not parallel.records(root)
