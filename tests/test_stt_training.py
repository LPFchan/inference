"""Leakage, supervision alignment and GRPO update-direction regression tests."""
import importlib.util
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "training/qwen_asr"
sys.path.insert(0, str(DIRECTORY))
from common import advantages, reward, training_targets, validate_splits


def row(identifier, text, speaker, digest):
    return dict(id=identifier, language="ko", text=text, speaker=speaker, sha256=digest)


def test_first_transcript_token_and_eos_are_supervised():
    inputs, targets, mask = training_targets([11, 12, 13], [21, 22, 99])
    assert inputs == [11, 12, 13, 21, 22]
    assert [t for t, use in zip(targets, mask) if use] == [21, 22, 99]
    assert mask == [False, False, True, True, True]


def test_reward_uses_characters_for_cjk_and_words_for_english():
    assert reward("한국어", "한국어", "ko") == 1
    assert reward("日本語", "日本", "ja") == pytest.approx(2 / 3)
    assert reward("中文", "中", "zh") == 0.5
    assert reward("two words", "two", "en") == 0.5
    assert reward("a", "abcdef", "en") == 0


def test_identical_rollouts_have_no_fabricated_advantage():
    assert advantages([0.8, 0.8]) == [0, 0]
    low, high = advantages([0.5, 0.9])
    assert low < 0 < high
    assert low + high == pytest.approx(0)


def test_fixed_advantage_scale_divides_centered_rewards():
    near_tie = [0.5625, 0.7291666666666667, 0.7291666666666667, 0.5625]
    assert advantages(near_tie, None) == advantages(near_tie)
    assert advantages(near_tie) == pytest.approx([-1, 1, 1, -1], rel=1e-4)  # std + 1e-6 default
    spread = 1 / 12 / 0.03  # each sample sits 1/12 from the group mean: 2.7778
    assert advantages(near_tie, 0.03) == pytest.approx([-spread, spread, spread, -spread], rel=1e-12)
    assert advantages([0.8, 0.8], 0.03) == [0, 0]
    assert advantages([-1.0] * 4, 0.03) == [0] * 4
    for invalid in (0, -0.03, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="advantage scale"):
            advantages(near_tie, invalid)


@pytest.mark.parametrize("field,value", [("sha256", "a"), ("speaker", "s1"),
                                         ("text", "안녕하세요!")])
def test_train_dev_overlap_is_rejected(field, value):
    train = row("1", "안녕하세요", "s1", "a")
    dev = row("2", "반갑습니다", "s2", "b")
    dev[field] = value
    with pytest.raises(ValueError, match="overlap"):
        validate_splits([train], [dev], [])


@pytest.mark.parametrize("protected", [dict(reference="안녕하세요!", sha256="x"),
                                       dict(reference="다른 문장", sha256="a")])
def test_frozen_test_text_and_audio_are_excluded(protected):
    with pytest.raises(ValueError, match="test .*overlap"):
        validate_splits([row("1", "안녕하세요", "s1", "a")], [], [protected])


def test_duplicate_sample_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_splits([row("1", "가", "s1", "a"), row("1", "나", "s2", "b")], [], [])


def test_disjoint_inputs_pass():
    validate_splits([row("1", "가", "s1", "a")], [row("2", "나", "s2", "b")], [])


def parse_training_args(monkeypatch, stage, extra):
    spec = importlib.util.spec_from_file_location("qwen_train_args", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(sys, "argv", ["train.py", stage, "--model", "model", "--data", "data",
                                     "--protected-test", "test.jsonl", "--output", "run", *extra])
    return trainer, trainer.arguments()


@pytest.mark.parametrize("stage,extra,base_reference", [
    ("grpo", ["--base-reference"], True),
    ("grpo", ["--init-adapter", "sft-checkpoint"], False),
    ("sft", [], False),
    ("evaluate", ["--init-adapter", "sft-checkpoint"], False),
])
def test_reference_modes_are_explicit_and_bound_to_resume_settings(monkeypatch, stage, extra, base_reference):
    trainer, args = parse_training_args(monkeypatch, stage, extra)
    assert args.base_reference == base_reference
    assert trainer.training_settings(args)["base_reference"] == base_reference
    if base_reference:
        assert args.init_adapter is None
    if "--init-adapter" in extra:
        assert args.init_adapter == Path("sft-checkpoint")


@pytest.mark.parametrize("stage,extra", [
    ("grpo", []),
    ("grpo", ["--base-reference", "--init-adapter", "sft-checkpoint"]),
    ("sft", ["--base-reference"]),
    ("evaluate", ["--base-reference"]),
])
def test_missing_or_conflicting_reference_modes_are_rejected(monkeypatch, stage, extra):
    with pytest.raises(SystemExit) as error:
        parse_training_args(monkeypatch, stage, extra)
    assert error.value.code == 2


@pytest.mark.parametrize("extra,scale", [([], None), (["--advantage-scale", "0.03"], 0.03)])
def test_advantage_scale_is_bound_to_grpo_resume_settings(monkeypatch, extra, scale):
    trainer, args = parse_training_args(monkeypatch, "grpo", ["--base-reference", *extra])
    assert args.scale == 1.0  # LoRA multiplier is a separate setting
    assert trainer.training_settings(args)["advantage_scale"] == scale


@pytest.mark.parametrize("stage,value", [
    ("grpo", "0"), ("grpo", "-0.03"), ("grpo", "nan"), ("grpo", "inf"),
    ("sft", "0.03"), ("evaluate", "0.03"),
])
def test_invalid_advantage_scale_is_rejected(monkeypatch, stage, value):
    extra = ["--base-reference"] if stage == "grpo" else []
    with pytest.raises(SystemExit) as error:
        parse_training_args(monkeypatch, stage, [*extra, "--advantage-scale", value])
    assert error.value.code == 2


def test_grpo_gradient_increases_rewarded_token_probability():
    mx = pytest.importorskip("mlx.core")
    spec = importlib.util.spec_from_file_location("qwen_train", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    old = mx.array([-0.5, -0.4])
    for advantage in (1.0, -1.0):
        grad = mx.grad(lambda lp: trainer.grpo_loss(lp, old, old, advantage, 0.02))(old)
        mx.eval(grad)
        assert all((v < 0 if advantage > 0 else v > 0) for v in grad.tolist())
    loss = trainer.grpo_loss(old, old, old, 0.0, 0.02)
    assert float(loss.item()) == pytest.approx(0)


def test_korean_spacing_can_receive_word_error_reward():
    reference, hypothesis = "정말 좋은 하루", "정말좋은 하루"
    assert reward(reference, hypothesis, "ko") == 1
    assert reward(reference, hypothesis, "ko", korean_wer_weight=0.5) < 1


def evaluation_rows(extra=True):
    refs = {"en": "some english words", "ko": "정말 좋은 하루", "ja": "日本語音声", "zh": "中文语音"}
    return {f"{lang}-{i}": {"id": f"{lang}-{i}", "language": lang, "reference": ref,
            "hypothesis": ref + (" wrong" if lang in ("en", "ko") else "錯") if extra else ref,
            "ended": True} for lang, ref in refs.items() for i in range(12)}


def test_selection_accepts_real_gain_and_rejects_ties():
    from selection import assess
    anchor = evaluation_rows()
    better = evaluation_rows(False)
    assert assess(better, anchor, anchor, repetitions=40)["passes"]
    assert not assess(anchor, anchor, anchor, repetitions=40)["passes"]


def test_selection_retains_english_guard():
    from selection import assess
    anchor, candidate = evaluation_rows(), evaluation_rows(False)
    for r in candidate.values():
        if r["language"] == "en":
            r["hypothesis"] += " many additional incorrect words"
    result = assess(candidate, anchor, anchor, repetitions=40)
    assert not result["passes"]
    assert not result["gates"]["english_limit"]


def test_selection_refuses_changed_evaluation_references():
    from selection import assess
    anchor, candidate = evaluation_rows(), evaluation_rows(False)
    candidate["ko-0"]["reference"] = "다른 문장"
    with pytest.raises(ValueError, match="references differ"):
        assess(candidate, anchor, anchor, repetitions=40)


def test_aishell_ingestion_excludes_official_dev_and_test():
    from expand_clean import aishell_training_member
    assert aishell_training_member("train/S0002/BAC009S0002W0122.wav")
    assert not aishell_training_member("dev/S0724/BAC009S0724W0122.wav")
    assert not aishell_training_member("test/S0901/BAC009S0901W0122.wav")


def test_distinct_training_recordings_may_repeat_a_sentence():
    validate_splits([row("1", "같은 문장", "speaker-a", "a"),
                     row("2", "같은 문장", "speaker-b", "b")],
                    [row("3", "다른 문장", "speaker-c", "c")], [])


@pytest.mark.parametrize("free", [100 * (1 << 30) - 1, 100 * (1 << 30)])
def test_clean_expansion_preserves_holdouts_and_distinct_recordings(tmp_path, monkeypatch, free):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("scipy.signal")
    import pyarrow.parquet as pq
    import io
    import json
    import tarfile
    from types import SimpleNamespace
    from common import atomic_json, load_manifests, sha256
    import expand_clean
    from expand_clean import build, source_holdout
    from prefetch import SOURCES
    # The intake's 100 GiB free-space guard, independent of this host's disk.
    monkeypatch.setattr(expand_clean.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))

    root, core, output = (tmp_path / name for name in ("root", "core", "output"))
    for folder in (root, core / "wav", output):
        folder.mkdir(parents=True)
    rng = np.random.default_rng(42)

    def audio_bytes():
        stream = io.BytesIO()
        sf.write(stream, rng.uniform(-0.1, 0.1, 32000), 16000, format="WAV", subtype="PCM_16")
        return stream.getvalue()

    manifests = {}
    for split in ("train", "dev"):
        rows = []
        for lang in ("en", "ko", "ja", "zh"):
            identifier = f"{split}-{lang}"
            wav = core / "wav" / f"{identifier}.wav"
            wav.write_bytes(audio_bytes())
            rows.append(dict(id=identifier, language=lang, text=identifier, speaker=identifier,
                             audio=f"wav/{identifier}.wav", sha256=sha256(wav), seconds=2,
                             source="fixture", source_revision="fixture", license="fixture"))
        manifest = core / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifests[f"{split}_sha256"] = sha256(manifest)
    protected = root / "test.jsonl"
    protected.write_text(json.dumps(dict(reference="sealed test", sha256="sealed")) + "\n")
    atomic_json(core / "data.json", {**manifests, "protected_manifest_sha256": sha256(protected)})
    completed = {}

    def cached(key, content):
        path = root / str(len(completed))
        path.write_bytes(content)
        source = SOURCES[key.split("/")[0]]
        completed[key] = dict(path=str(path), sha256=sha256(path), source=source[0],
                              revision=source[1], license=source[2])

    speaker = next(i for i in range(100) if not source_holdout(f"zeroth/{i}"))
    table = pa.Table.from_pylist([
        dict(id=str(i), speaker_id=speaker, text=text, audio=dict(bytes=audio_bytes()))
        for i, text in enumerate(("같은 추가 문장", "같은 추가 문장", "sealed test", "dev-ko"))])
    stream = io.BytesIO()
    pq.write_table(table, stream)
    cached("zeroth/data/train-00000.parquet", stream.getvalue())

    cached("cv17/transcript/ja/train.tsv",
           b"client_id\tpath\tsentence\nallowed\ta.mp3\tnew Japanese\nheld\tb.mp3\tshould exclude\n")
    cached("cv17/transcript/ja/dev.tsv", b"client_id\tpath\tsentence\nheld\tc.mp3\tsource holdout\n")

    def archive_content(members, mode="w"):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode=mode) as archive:
            for name in members:
                data = audio_bytes()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return stream.getvalue()

    cached("cv17/audio/ja/train/00000.tar", archive_content(["a.mp3", "b.mp3"]))
    cached("aishell/data_aishell/transcript/aishell_transcript_v0.8.txt",
           "train1 中 文 音 频\ndev1 不该训练\ntest1 也不该训练\n".encode())
    cached("aishell/data_aishell/wav/S0002.tar.gz", archive_content([
        "train/S0002/train1.wav", "dev/S0724/dev1.wav", "test/S0901/test1.wav"], "w:gz"))
    atomic_json(root / "prefetch.json", dict(status="complete", completed=completed))
    settings = SimpleNamespace(root=root, core=core, output=output, protected_test=protected,
                               aishell_hours=40, seed=42)
    if free < 100 * (1 << 30):
        with pytest.raises(RuntimeError, match="100 GiB free"):
            build(settings)
        return
    build(settings)
    train, dev, metadata = load_manifests(output, protected)
    assert len(train) == 8
    assert sum(r["text"] == "같은 추가 문장" for r in train) == 2
    assert sum(r["source"] == SOURCES["aishell"][0] for r in train) == 1
    assert (output / "dev.jsonl").read_bytes() == (core / "dev.jsonl").read_bytes()
    assert metadata["dev_clips"] == len(dev) == 4


@pytest.mark.parametrize("checkpoint_step", [0, 2])
def test_resume_preserves_unsaved_work_and_replays_from_published_checkpoint(tmp_path, checkpoint_step):
    import json
    from common import recover_uncommitted_updates
    original = "".join(json.dumps(dict(step=i)) + "\n" for i in range(1, 4)) + '{"step":'
    (tmp_path / "metrics.jsonl").write_text(original)
    (tmp_path / "step-000003").mkdir()
    (tmp_path / "step-000003/adapters.safetensors").write_bytes(b"incomplete")
    recover_uncommitted_updates(tmp_path, checkpoint_step)
    active = [json.loads(line)["step"] for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert active == list(range(1, checkpoint_step + 1))
    recovery, = (tmp_path / "interrupted").iterdir()
    assert (recovery / "metrics.jsonl").read_text() == original
    assert (recovery / "step-000003/adapters.safetensors").read_bytes() == b"incomplete"
    assert not (tmp_path / "step-000003").exists()
    recover_uncommitted_updates(tmp_path, checkpoint_step)
    assert list((tmp_path / "interrupted").iterdir()) == [recovery]


def test_controller_verdict_identifies_the_selected_confirmation(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import cycle
    from common import atomic_json
    for name in ("base", "sft200", "grpo50"):
        folder = tmp_path / "calibration" / name
        folder.mkdir(parents=True)
        atomic_json(folder / "run.json", {"status": "complete"})
        atomic_json(folder / "dev-initial.json", {"rows": list(evaluation_rows().values())})
    for name in ("data-raw5", "data-v001-16h"):
        folder = tmp_path / name
        folder.mkdir()
        atomic_json(folder / "data.json", {})
    # Keep the base as parent; both new candidates improve, with the repeated
    # seed scoring better. Exercise the actual controller's persisted evidence.
    monkeypatch.setattr(cycle, "normalized_score", lambda *_: 1.0)
    scores = iter((0.8, 0.7))
    monkeypatch.setattr(cycle, "assess", lambda *_: {"passes": True, "candidate_score": next(scores)})

    class CompletedProcess:
        returncode = 0
        pid = 123

        def __init__(self, args, **kwargs):
            out = Path(args[args.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / "run.json", {"status": "complete"})
            atomic_json(out / "latest.json", {"directory": "step-005000"})
            atomic_json(out / "dev-step-005000.json", {"rows": list(evaluation_rows(False).values())})
            if out.name.endswith("-confirm"):
                (tmp_path / "STOP").touch()

        def poll(self):
            return 0

    monkeypatch.setattr(cycle.subprocess, "Popen", CompletedProcess)
    monkeypatch.setattr(cycle.signal, "signal", lambda *_: None)
    monkeypatch.setattr(cycle.shutil, "disk_usage", lambda *_: SimpleNamespace(free=200 * (1 << 30)))
    cycle.loop(SimpleNamespace(root=tmp_path, runtime=Path("python"),
                              initial_root=tmp_path / "initial", protected_test=tmp_path / "test"))
    state = json.loads((tmp_path / "loop-state.json").read_text())
    verdict, = (tmp_path / "trials").glob("*/verdict.json")
    accepted = json.loads(verdict.read_text())
    assert accepted["status"] == "accepted"
    assert "-confirm/" in accepted["checkpoint"]
    assert accepted["checkpoint"] == state["incumbent"]["checkpoint"]
    assert accepted["evaluation"] == state["incumbent"]["evaluation"]


def sentence_pools():
    """Two recordings of every source sentence per language, plus one ID-less sentence read twice."""
    pools = {}
    for lang in ("en", "ko", "ja", "zh"):
        rows = [dict(id=f"{lang}/{s}-{r}", language=lang, text=f"{lang} sentence {s}",
                     source_sentence_id=s) for s in range(12) for r in range(2)]
        rows += [dict(id=f"{lang}/plain-{r}", language=lang, text=text)
                 for r, text in enumerate([f"{lang} Plain words.", f"{lang} plain  WORDS"])]
        pools[lang] = rows
    return pools


def test_sentence_epoch_draws_one_seeded_recording_per_sentence():
    from common import sentence_epoch_order, sentence_key
    pools = sentence_pools()
    order = sentence_epoch_order(pools, 7)
    for lang, rows in order.items():
        keys = [sentence_key(row) for row in rows]
        assert len(keys) == len(set(keys)) == 13  # ID 0 is a sentence; the two plain readings are one
        assert ("id", 0) in keys and sum(key[0] == "text" for key in keys) == 1
        assert all(row in pools[lang] for row in rows)
    assert sentence_epoch_order(sentence_pools(), 7) == order
    other = sentence_epoch_order(pools, 8)
    assert [r["id"] for r in other["ko"]] != [r["id"] for r in order["ko"]]
    shuffled = {lang: [r["id"] for r in rows] for lang, rows in order.items()}
    assert len(set(map(tuple, shuffled.values()))) == 4  # each language draws from its own generator


def test_sentence_epoch_draw_keeps_schedule_and_refuses_exhaustion():
    from common import sentence_epoch_draw, sentence_epoch_order, sentence_key
    trainer = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location("qwen_train_schedule", DIRECTORY / "train.py"))
    spec.loader.exec_module(trainer)
    schedule = trainer.SCHEDULE
    order = sentence_epoch_order(sentence_pools(), 7)
    # 13 sentences per language; KO's 14th draw (microstep 43) would repeat one.
    drawn = [sentence_epoch_draw(order, index, schedule) for index in range(43)]
    assert [row["language"] for row in drawn] == [schedule[i % 10] for i in range(43)]
    for lang in ("ko", "ja", "zh", "en"):
        assert [row for row in drawn if row["language"] == lang] == order[lang][:len(
            [row for row in drawn if row["language"] == lang])]
    assert len({(r["language"], sentence_key(r)) for r in drawn}) == 43
    with pytest.raises(ValueError, match="exhausted for ko at microstep 43"):
        sentence_epoch_draw(order, 43, schedule)
    resumed = sentence_epoch_order(sentence_pools(), 7)  # a resumed process rebuilds the order
    assert [sentence_epoch_draw(resumed, i, schedule) for i in range(24, 43)] == drawn[24:]


def test_default_sft_draws_are_unchanged_and_sentence_epoch_is_opt_in(monkeypatch):
    import random
    from common import sentence_epoch_draw, sentence_epoch_order
    trainer, args = parse_training_args(monkeypatch, "sft", [])
    assert args.sampler == "replacement" and trainer.training_settings(args)["sampler"] == "replacement"
    pools, schedule = sentence_pools(), trainer.SCHEDULE
    assert [trainer.sft_row(pools, None, 11, i) for i in range(40)] == [
        random.Random(11 + i).choice(pools[schedule[i % 10]]) for i in range(40)]
    order = sentence_epoch_order(pools, 11)
    assert [trainer.sft_row(pools, order, 11, i) for i in range(40)] == [
        sentence_epoch_draw(order, i, schedule) for i in range(40)]
    trainer, args = parse_training_args(monkeypatch, "sft", ["--sampler", "sentence-epoch"])
    assert trainer.training_settings(args)["sampler"] == "sentence-epoch"


@pytest.mark.parametrize("stage,extra", [
    ("grpo", ["--base-reference"]), ("evaluate", []), ("evaluate", ["--external", "probe.jsonl"])])
def test_sentence_epoch_is_rejected_outside_sft(monkeypatch, stage, extra):
    with pytest.raises(SystemExit) as error:
        parse_training_args(monkeypatch, stage, [*extra, "--sampler", "sentence-epoch"])
    assert error.value.code == 2


@pytest.mark.parametrize("steps,refused", [(3, False), (4, True)])
def test_sentence_epoch_budget_is_refused_before_model_load(tmp_path, monkeypatch, steps, refused):
    import json
    from types import ModuleType, SimpleNamespace
    from common import atomic_json, sha256
    data = tmp_path / "data"
    data.mkdir()
    rows = {"train": [], "dev": []}
    for split in rows:
        for lang in ("en", "ko", "ja", "zh"):
            audio = data / f"{split}-{lang}.wav"
            audio.write_bytes(f"{split} {lang}".encode())
            rows[split].append(dict(id=f"{split}/{lang}", language=lang, text=f"{split} {lang} words",
                                    speaker=f"{split}-{lang}", audio=audio.name, sha256=sha256(audio),
                                    source_sentence_id=0))
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[split]))
    protected = tmp_path / "test.jsonl"
    protected.write_text("")
    atomic_json(data / "data.json", {"train_sha256": sha256(data / "train.jsonl"),
                                     "dev_sha256": sha256(data / "dev.jsonl"),
                                     "protected_manifest_sha256": sha256(protected)})
    loads = []
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx.optimizers", "mlx.utils", "mlx_audio", "mlx_audio.stt",
                 "mlx_audio.lm", "mlx_audio.lm.generate", "mlx_lm", "mlx_lm.sample_utils",
                 "mlx_lm.tuner", "mlx_lm.tuner.lora"):
        module = ModuleType(name)
        module.__dict__.update(tree_flatten=None, tree_map=None, tree_unflatten=None, generate_step=None,
                               make_sampler=None, LoRALinear=None)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", SimpleNamespace(load_model=loads.append))
    trainer, args = parse_training_args(monkeypatch, "sft", ["--sampler", "sentence-epoch",
                                                             "--steps", str(steps), "--accumulate", "1"])
    args.data, args.protected_test, args.output = data, protected, tmp_path / "run"
    args.model = tmp_path / "missing-model"
    args.output.mkdir()
    # One sentence per language: KO, JA and ZH fill the first three microsteps; the fourth repeats KO.
    with pytest.raises(ValueError, match="missing model weights" if not refused else
                       "sentence epoch exhausted for ko at microstep 3"):
        trainer.run(args)
    assert not loads and not (args.output / "run.json").exists()
