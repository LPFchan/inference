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


def test_one_seed_lists_match_single_evaluations():
    from selection import assess
    anchor, candidate = evaluation_rows(), evaluation_rows(False)
    single = assess(candidate, anchor, anchor, repetitions=40)
    assert assess([candidate], [anchor], anchor, repetitions=40) == single
    assert single["seeds"] == 1


def test_two_seeds_pool_errors_per_clip():
    from selection import assess, error_counts
    anchor, clean = evaluation_rows(), evaluation_rows(False)
    result = assess([clean, anchor], [anchor, anchor], anchor, repetitions=40)
    ko = [error_counts(r) for r in anchor.values() if r["language"] == "ko"]
    assert result["candidate_errors"]["ko"] == sum(e for e, _ in ko) / (2 * sum(n for _, n in ko))
    assert result["seeds"] == 2 and result["passes"]
    with pytest.raises(ValueError, match="same number of seeds"):
        assess([clean, clean], anchor, anchor, repetitions=40)


def test_two_seed_gates_count_empty_outputs_over_both_runs():
    from selection import assess
    anchor, clean, empty = evaluation_rows(), evaluation_rows(False), evaluation_rows(False)
    empty["zh-0"]["hypothesis"] = ""
    assert not assess([clean, empty], [anchor, anchor], anchor, repetitions=40)["gates"]["empty_output_limit"]


def test_selection_makes_korean_spacing_optional():
    from selection import error_counts
    ko = lambda reference, hypothesis, glosses=(): {"language": "ko", "reference": reference,
                                                    "hypothesis": hypothesis, "glosses": list(glosses)}
    reference = "슈퍼 대회전에서 13위로"
    assert error_counts(ko(reference, "슈퍼대회전 에서 13위로")) == (0, 3)
    assert error_counts(ko(reference, "슈퍼 대회전에서 14위로")) == (1, 3)
    assert error_counts(ko(reference, "슈퍼 대회전에서 아주 13위로")) == (1, 3)
    assert error_counts(ko(reference, "")) == (3, 3)
    assert error_counts(ko("슈퍼 대회전super g에서", "슈퍼대회전에서", ["super g"])) == (0, 2)
    english = {"language": "en", "reference": "some english words", "hypothesis": "someenglish words"}
    assert error_counts(english) == (2, 3)


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


@pytest.mark.parametrize("pool_size", [1, 2, 7, 1000])
def test_grpo_attempt_zero_keeps_the_legacy_row(pool_size):
    import random
    from common import grpo_row
    pool = [f"row-{i}" for i in range(pool_size)]
    for seed in (0, 21325231, 21326232):
        for step in (1, 2, 10, 1000):
            assert grpo_row(pool, seed, step, 0) == random.Random(seed + step).choice(pool)


def test_grpo_redraws_follow_the_string_seed_law_without_touching_global_random():
    import random
    from common import grpo_row
    pool = list(range(1000))
    random.seed(99)
    state = random.getstate()
    draws = [grpo_row(pool, 21325231, step, attempt) for step, attempt in ((1, 1), (1, 7), (2, 1), (1000, 7))]
    assert draws == [520, 63, 322, 13]  # fixed across processes and PYTHONHASHSEED
    for seed, step, attempt in ((0, 1, 1), (21325231, 3, 2), (21326232, 999, 7)):
        expected = random.Random(f"grpo-redraw:{seed}:{step}:{attempt}").choice(pool)
        assert grpo_row(pool, seed, step, attempt) == grpo_row(pool, seed, step, attempt) == expected
    assert random.getstate() == state
    assert len({grpo_row(pool, 7, 3, attempt) for attempt in range(1, 8)}) > 1  # attempt changes the row
    # Redraws are with replacement: a two-row pool must repeat across eight attempts, a single row always.
    assert len({grpo_row(["a", "b"], 7, 3, attempt) for attempt in range(8)}) <= 2
    assert {grpo_row(["only"], 7, 3, attempt) for attempt in range(8)} == {"only"}


def test_informative_groups_need_raw_reward_spread():
    from common import informative
    assert not informative([0.0] * 4)
    assert not informative([0.8] * 4)
    assert not informative([-1.0] * 4)  # all truncated
    assert informative([0.8, 0.8, 0.8, 0.7])
    assert informative([-1.0, -1.0, -1.0, 0.0])


@pytest.mark.parametrize("stage,extra", [
    ("grpo", ["--base-reference"]), ("sft", []), ("evaluate", [])])
def test_max_draws_defaults_to_one_and_is_bound_to_resume_settings(monkeypatch, stage, extra):
    trainer, args = parse_training_args(monkeypatch, stage, extra)
    assert args.max_draws == 1 and trainer.training_settings(args)["max_draws"] == 1
    if stage == "grpo":
        trainer, args = parse_training_args(monkeypatch, stage, [*extra, "--max-draws", "8"])
        assert trainer.training_settings(args)["max_draws"] == 8


@pytest.mark.parametrize("stage,value", [
    ("grpo", "0"), ("grpo", "-1"), ("grpo", "1.5"), ("sft", "8"), ("evaluate", "2"), ("sft", "0")])
def test_invalid_max_draws_is_rejected(monkeypatch, stage, value):
    extra = ["--base-reference"] if stage == "grpo" else []
    with pytest.raises(SystemExit) as error:
        parse_training_args(monkeypatch, stage, [*extra, "--max-draws", value])
    assert error.value.code == 2


GRPO_TEXT = {1: "abcd", 2: "abc", 3: "zzzz"}  # rewards 1.0, 0.75 and 0.0 against "abcd"
GRPO_EOS = 9
GRPO_REFERENCE = [4, 5]  # encoding of the TRAIN transcript "abcd"; never sampled


def fake_grpo_runtime(monkeypatch, tmp_path, script):
    """Real MLX on CPU with a tiny decoder; rollouts replay `script` completions in call order."""
    import json
    from types import ModuleType, SimpleNamespace
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    pytest.importorskip("jiwer")
    from common import atomic_json, sha256

    class LoRALinear(nn.Module):
        @staticmethod
        def from_base(linear, r, scale):
            layer = LoRALinear()
            out_dims, in_dims = linear.weight.shape
            layer.linear, layer.scale = linear, scale
            layer.lora_a = mx.random.uniform(-0.1, 0.1, (in_dims, r))
            layer.lora_b = mx.zeros((r, out_dims))
            return layer

        def __call__(self, x):
            return self.linear(x) + self.scale * (x @ self.lora_a) @ self.lora_b

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(4, 4, bias=False) for _ in range(4))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 4)
            self.layers = [Block() for _ in range(28)]

        def __call__(self, inputs_embeds):
            h = inputs_embeds
            for block in self.layers:
                h = h + 0.1 * block.o_proj(block.q_proj(h) + block.k_proj(h) + block.v_proj(h))
            return h

    encoded = []

    class Tokenizer:
        eos_token_id = GRPO_EOS

        def encode(self, text, add_special_tokens=True):
            encoded.append((text, add_special_tokens))
            return list(GRPO_REFERENCE) if text == "abcd" else [1]

        def decode(self, tokens, skip_special_tokens=True):
            return "".join(GRPO_TEXT[t] for t in tokens)

    contexts = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.lm_head = Decoder(), nn.Linear(4, 10, bias=False)
            self._tokenizer = Tokenizer()

        def _eos_token_ids(self):
            return [GRPO_EOS]

        def _preprocess_audio(self, path):
            marker = int(Path(path).read_text())  # each clip's audio is its own marker
            contexts.append(marker)
            return None, None, marker

        def get_audio_features(self, features, mask):
            return mx.zeros((1, 1, 4))

        def _build_prompt(self, count, language):
            return mx.array([[count, 0, 0]])

        def _build_inputs_embeds(self, prompt, audio):
            return mx.full((1, prompt.shape[1], 4), float(prompt[0, 0].item() % 7) / 7)

    remaining = list(script)

    def generate_step(prompt, model, input_embeddings, max_tokens, sampler, prefill_step_size):
        completion = remaining.pop(0) if remaining and int(prompt[0].item()) < 1000 else [1, GRPO_EOS]
        for token in completion[:max_tokens]:
            yield token, None

    for name in ("mlx_audio", "mlx_audio.stt", "mlx_audio.lm", "mlx_lm", "mlx_lm.tuner"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", SimpleNamespace(load_model=lambda path: Model()))
    monkeypatch.setitem(sys.modules, "mlx_audio.lm.generate", SimpleNamespace(generate_step=generate_step))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=lambda temp: None))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.lora", SimpleNamespace(LoRALinear=LoRALinear))
    monkeypatch.setattr(mx, "set_memory_limit", lambda limit: None)
    monkeypatch.setattr(mx, "set_cache_limit", lambda limit: None)
    seeds, updates, rates = [], [], []
    real_seed = mx.random.seed
    monkeypatch.setattr(mx.random, "seed", lambda value: (seeds.append(value), real_seed(value))[1])

    class RecordingAdamW(optim.AdamW):
        def update(self, model, gradients):
            updates.append(len(targets))
            rates.append(float(self.learning_rate.item()))  # the rate this update actually uses
            return super().update(model, gradients)

    monkeypatch.setattr(optim, "AdamW", RecordingAdamW)

    model_dir, data = tmp_path / "model", tmp_path / "data"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")
    atomic_json(model_dir / "training-source.json", {"id": "mlx-community/Qwen3-ASR-1.7B-8bit",
                                                     "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"})
    data.mkdir()
    rows = {"train": [], "dev": []}
    for split, base in (("train", 0), ("dev", 1000)):
        for index, lang in enumerate(lang for lang in ("en", "ko", "ja", "zh") for _ in range(3 if split == "train" else 1)):
            marker = base + index
            audio = data / f"{marker}.wav"
            audio.write_text(str(marker))
            rows[split].append(dict(id=f"{split}/{lang}/{marker}", language=lang,
                                    text="abcd" if split == "train" else f"dev {lang}",
                                    speaker=f"speaker-{marker}", audio=audio.name, sha256=sha256(audio)))
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[split]))
    protected = tmp_path / "test.jsonl"
    protected.write_text("")
    atomic_json(data / "data.json", {"train_sha256": sha256(data / "train.jsonl"),
                                     "dev_sha256": sha256(data / "dev.jsonl"),
                                     "protected_manifest_sha256": sha256(protected)})
    trainer = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location("qwen_train_redraw", DIRECTORY / "train.py"))
    spec.loader.exec_module(trainer)
    targets = []
    real_targets = trainer.training_targets

    def recording_targets(prompt, completion):
        targets.append((prompt[0], tuple(completion)))  # which context and completion feed a likelihood
        return real_targets(prompt, completion)

    monkeypatch.setattr(trainer, "training_targets", recording_targets)

    def configure(extra, stage="grpo", output="run"):
        (tmp_path / output).mkdir(exist_ok=True)
        monkeypatch.setattr(sys, "argv", ["train.py", stage, "--model", str(model_dir), "--data", str(data),
                                          "--protected-test", str(protected), "--output", str(tmp_path / output),
                                          *(["--base-reference"] if stage == "grpo" else []),
                                          "--skip-initial-eval", "--eval-clips", "1",
                                          "--max-tokens", "2", "--seed", "11", *extra])
        return trainer.arguments()

    (tmp_path / "run").mkdir()
    pools = {lang: [r for r in rows["train"] if r["language"] == lang] for lang in ("ko", "ja", "zh")}
    return SimpleNamespace(trainer=trainer, configure=configure, pools=pools, contexts=contexts,
                           targets=targets, seeds=seeds, updates=updates, rates=rates, remaining=remaining,
                           encoded=encoded, data=data, protected=protected,
                           output=tmp_path / "run", root=tmp_path, cpu=lambda: mx.stream(mx.cpu))


EQUAL, TRUNCATED, MIXED = [[1, GRPO_EOS]] * 4, [[3, 3, 3]] * 4, [[1, GRPO_EOS], [2, GRPO_EOS]] * 2


@pytest.mark.parametrize("max_draws,steps", [
    # KO wins on the last draw after an all-perfect and an all-truncated group; JA wins at once;
    # ZH exhausts the cap on equal groups and still takes its normal update.
    (3, [[EQUAL, TRUNCATED, MIXED], [MIXED], [EQUAL, TRUNCATED, EQUAL]]),
    (1, [[MIXED], [EQUAL], [TRUNCATED]]),
])
def test_grpo_redraw_updates_once_per_step_on_the_stopping_group(monkeypatch, tmp_path, max_draws, steps):
    import json
    from common import grpo_row
    script = [completion for groups in steps for group in groups for completion in group]
    env = fake_grpo_runtime(monkeypatch, tmp_path, script)
    args = env.configure(["--steps", "3", "--max-draws", str(max_draws)])
    with env.cpu():  # keep the fake decoder off the GPU
        env.trainer.run(args)
    assert not env.remaining
    rewards_of = {id(EQUAL): [1.0] * 4, id(TRUNCATED): [-1.0] * 4, id(MIXED): [1.0, 0.75] * 2}
    texts_of = {id(EQUAL): ["abcd"] * 4, id(TRUNCATED): ["zzzz" * 2] * 4, id(MIXED): ["abcd", "abc"] * 2}
    events = [json.loads(line) for line in (env.output / "metrics.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events] == [1, 2, 3]
    assert env.seeds == [11, 12, 13, 14]  # once at load, then once per logical step, never per draw
    assert len(env.updates) == 3  # exactly one optimizer update per logical step
    drawn, start = [], 0
    for step, (event, groups, lang) in enumerate(zip(events, steps, ("ko", "ja", "zh")), 1):
        rows = [grpo_row(env.pools[lang], 11, step, attempt) for attempt in range(len(groups))]
        final, final_row = groups[-1], rows[-1]
        drawn += [int(r["audio"][:-4]) for r in rows]
        assert event["draws"] == len(groups)
        assert event["exhausted"] == (final is not MIXED)
        assert event["skipped"] == [{"id": r["id"], "rewards": rewards_of[id(g)]}
                                    for r, g in zip(rows[:-1], groups[:-1])]
        assert (event["id"], event["rewards"]) == (final_row["id"], rewards_of[id(final)])
        assert event["completions"] == texts_of[id(final)]
        assert event["zero_advantage"] == (final is not MIXED)
        # Old, reference and policy likelihoods all use the stopping context and completions only.
        used = env.targets[start:env.updates[step - 1]]
        completions = [tuple(c[:2]) for c in final]
        assert used == [(int(final_row["audio"][:-4]), c) for c in completions] * 3
        start = env.updates[step - 1]
    assert env.contexts[:len(drawn)] == drawn  # discarded draws were transcribed, then dropped
    assert env.contexts[len(drawn):] == [1000, 1001, 1002, 1003]  # final development evaluation
    assert json.loads((env.output / "run.json").read_text())["contract"]["settings"]["max_draws"] == max_draws
    changed = env.configure(["--steps", "4", "--max-draws", str(max_draws + 1), "--resume"])
    with pytest.raises(ValueError, match="identical inputs/settings/code"):
        env.trainer.run(changed)


def test_grpo_resume_marks_run_active_before_model_load(monkeypatch, tmp_path):
    import json
    from common import sha256
    env = fake_grpo_runtime(monkeypatch, tmp_path, [*EQUAL, *MIXED])
    with env.cpu():
        env.trainer.run(env.configure(["--steps", "1"]))
    before = json.loads((env.output / "run.json").read_text())
    assert before["status"] == "complete" and "finished" in before and "peak_memory_bytes" in before
    checkpoint_hash = sha256(env.output / "step-000001/adapters.safetensors")
    loader = sys.modules["mlx_audio.stt.utils"].load_model
    seen = []

    def inspect_loading(path):
        state = json.loads((env.output / "run.json").read_text())
        seen.append(state)
        assert state["status"] == "running"
        assert "finished" not in state and "peak_memory_bytes" not in state
        assert state["contract"] == before["contract"] and state["started"] == before["started"]
        return loader(path)

    monkeypatch.setattr(sys.modules["mlx_audio.stt.utils"], "load_model", inspect_loading)
    with env.cpu():
        env.trainer.run(env.configure(["--steps", "2", "--resume"]))
    assert len(seen) == 1
    assert json.loads((env.output / "run.json").read_text())["status"] == "complete"
    assert json.loads((env.output / "latest.json").read_text())["step"] == 2
    assert sha256(env.output / "step-000001/adapters.safetensors") == checkpoint_hash


def test_learning_rate_decay_keeps_the_prefix_then_falls_to_a_positive_last_update():
    from common import decayed_learning_rate
    schedule, peak = {"decay_start": 500, "decay_end": 944}, 3e-5
    assert decayed_learning_rate(peak, 500, None) is None
    assert [decayed_learning_rate(peak, t, schedule) for t in (1, 499, 500)] == [None] * 3
    assert decayed_learning_rate(peak, 501, schedule) == peak * 444 / 445
    assert decayed_learning_rate(peak, 944, schedule) == peak / 445
    tail = [decayed_learning_rate(peak, t, schedule) for t in range(501, 945)]
    assert len(tail) == 444 and all(a > b > 0 for a, b in zip(tail, tail[1:]))
    with pytest.raises(ValueError, match="past the learning-rate decay end"):
        decayed_learning_rate(peak, 945, schedule)


@pytest.mark.parametrize("extra,schedule", [
    ([], None),
    (["--steps", "944", "--lr-decay-start", "500"], {"decay_start": 500, "decay_end": 944}),
    (["--steps", "500", "--lr-decay-start", "500", "--lr-decay-end", "944"], {"decay_start": 500, "decay_end": 944}),
    (["--steps", "100", "--lr-decay-start", "500", "--lr-decay-end", "944"], {"decay_start": 500, "decay_end": 944}),
    (["--steps", "1", "--lr-decay-start", "0", "--lr-decay-end", "1"], {"decay_start": 0, "decay_end": 1}),
])
def test_learning_rate_schedule_is_resolved_and_bound_to_resume_settings(monkeypatch, extra, schedule):
    trainer, args = parse_training_args(monkeypatch, "sft", extra)
    assert trainer.training_settings(args)["lr_schedule"] == schedule
    for stage, reference in (("grpo", ["--base-reference"]), ("evaluate", [])):
        trainer, args = parse_training_args(monkeypatch, stage, reference)
        assert trainer.training_settings(args)["lr_schedule"] is None


@pytest.mark.parametrize("stage,extra", [
    ("sft", ["--lr-decay-end", "944"]),
    ("grpo", ["--base-reference", "--lr-decay-start", "1"]),
    ("evaluate", ["--lr-decay-start", "1"]),
    ("sft", ["--steps", "500", "--lr-decay-start", "500"]),
    ("sft", ["--lr-decay-start", "500", "--lr-decay-end", "500"]),
    ("sft", ["--lr-decay-start", "-1", "--lr-decay-end", "944"]),
    ("sft", ["--steps", "945", "--lr-decay-start", "500", "--lr-decay-end", "944"]),
    ("sft", ["--lr-decay-start", "1.5", "--lr-decay-end", "944"]),
])
def test_invalid_learning_rate_schedules_are_rejected(monkeypatch, stage, extra):
    with pytest.raises(SystemExit) as error:
        parse_training_args(monkeypatch, stage, extra)
    assert error.value.code == 2


def sft_events(output):
    import json
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def named_tensors(path):
    import mlx.core as mx
    return {name: (value.dtype, value.shape, memoryview(value).tobytes())
            for name, value in mx.load(str(path)).items()}


@pytest.mark.parametrize("pause", [3, 4])  # at the decay start, and inside the decay
def test_sft_decay_keeps_the_constant_prefix_and_resumes_exactly(monkeypatch, tmp_path, pause):
    import json
    env = fake_grpo_runtime(monkeypatch, tmp_path, [])
    peak, decay = 0.05, ["--lr-decay-start", "3", "--lr-decay-end", "6"]
    common = ["--learning-rate", str(peak), "--accumulate", "2", "--save-every", "1"]
    with env.cpu():
        env.trainer.run(env.configure([*common, "--steps", "3"], "sft", "constant"))
        constant_rates = list(env.rates)
        env.rates.clear()
        env.trainer.run(env.configure([*common, "--steps", "6", *decay], "sft", "scheduled"))
        scheduled_rates = list(env.rates)
        env.rates.clear()
        env.trainer.run(env.configure([*common, "--steps", str(pause), *decay], "sft", "paused"))
        env.trainer.run(env.configure([*common, "--steps", "6", *decay, "--resume"], "sft", "paused"))
    constant, scheduled, paused = (env.root / name for name in ("constant", "scheduled", "paused"))
    import mlx.core as mx
    f32 = lambda x: float(mx.array(x).item())  # the optimizer holds its rate as float32
    # The prefix never writes the rate: adapter and optimizer tensors match a run without the flags.
    for kind in ("adapters", "optimizer"):
        assert named_tensors(constant / f"step-000003/{kind}.safetensors") == named_tensors(
            scheduled / f"step-000003/{kind}.safetensors")
    assert constant_rates == scheduled_rates[:3] == [f32(peak)] * 3
    assert scheduled_rates[3:] == [f32(peak * 3 / 4), f32(peak * 2 / 4), f32(peak * 1 / 4)]
    keys = ("step", "loss", "gradient_norm")
    assert [[e[k] for k in keys] for e in sft_events(constant)] == [
        [e[k] for k in keys] for e in sft_events(scheduled)][:3]
    assert all("learning_rate" not in e for e in sft_events(constant))
    assert [e["learning_rate"] for e in sft_events(scheduled)] == scheduled_rates
    # A paused run resumed to the same endpoint reproduces uninterrupted scheduled training.
    keys += ("learning_rate",)
    assert [[e[k] for k in keys] for e in sft_events(paused)] == [
        [e[k] for k in keys] for e in sft_events(scheduled)]
    assert env.rates[pause:] == scheduled_rates[pause:]
    for kind in ("adapters", "optimizer"):
        assert named_tensors(paused / f"step-000006/{kind}.safetensors") == named_tensors(
            scheduled / f"step-000006/{kind}.safetensors")
    contract = json.loads((paused / "run.json").read_text())["contract"]
    assert contract == json.loads((scheduled / "run.json").read_text())["contract"]
    assert contract["settings"]["lr_schedule"] == {"decay_start": 3, "decay_end": 6}


def test_changed_decay_endpoint_is_refused_before_model_load(monkeypatch, tmp_path):
    env = fake_grpo_runtime(monkeypatch, tmp_path, [])
    common = ["--learning-rate", "0.05", "--accumulate", "1", "--lr-decay-start", "2"]
    with env.cpu():
        env.trainer.run(env.configure([*common, "--steps", "2", "--lr-decay-end", "4"], "sft"))
    loads = []
    monkeypatch.setattr(sys.modules["mlx_audio.stt.utils"], "load_model", loads.append)
    for changed in (["--lr-decay-end", "5"], ["--lr-decay-end", "3"]):
        with pytest.raises(ValueError, match="identical inputs/settings/code"):
            env.trainer.run(env.configure([*common, "--steps", "3", *changed, "--resume"], "sft"))
    with pytest.raises(ValueError, match="identical inputs/settings/code"):
        env.trainer.run(env.configure(["--learning-rate", "0.05", "--accumulate", "1", "--steps", "3",
                                       "--resume"], "sft"))
    with pytest.raises(SystemExit):
        env.configure([*common, "--steps", "5", "--lr-decay-end", "4", "--resume"], "sft")
    assert not loads


@pytest.mark.parametrize("stage,extra", [("grpo", ["--base-reference"]), ("sft", []), ("evaluate", [])])
def test_reference_member_defaults_off_and_is_bound_to_resume_settings(monkeypatch, stage, extra):
    trainer, args = parse_training_args(monkeypatch, stage, extra)
    assert args.reference_member is False and trainer.training_settings(args)["reference_member"] is False
    if stage == "grpo":
        for flags in (["--reference-member"], ["--reference-member", "--max-draws", "1"]):
            trainer, args = parse_training_args(monkeypatch, stage, [*extra, *flags])
            assert trainer.training_settings(args)["reference_member"] is True


@pytest.mark.parametrize("stage,extra", [
    ("sft", []), ("evaluate", []), ("evaluate", ["--external", "probe.jsonl"]),
    ("grpo", ["--base-reference", "--max-draws", "2"]), ("grpo", ["--base-reference", "--max-draws", "8"])])
def test_reference_member_is_rejected_outside_single_draw_grpo(monkeypatch, stage, extra):
    with pytest.raises(SystemExit) as error:
        parse_training_args(monkeypatch, stage, [*extra, "--reference-member"])
    assert error.value.code == 2


@pytest.mark.parametrize("language,text", [
    ("en", "Two words."), ("ko", "정말 좋은 하루"), ("ja", "日本語の音声"), ("zh", "中文语音")])
def test_reference_transcript_scores_one_under_the_unchanged_reward(language, text):
    for weight in (0.0, 0.5):
        assert reward(text, text, language, weight) == 1.0


def test_reference_member_advantages_push_the_label_and_spare_perfect_groups():
    for sampled in ([0.75] * 4, [-1.0] * 4, [0.0] * 4):  # equal but imperfect: no signal without the reference
        assert advantages(sampled) == [0] * 4
        *samples, ref = advantages(sampled + [1.0])
        assert ref > 0 and all(value < 0 for value in samples)
    assert advantages([1.0] * 5) == [0] * 5  # all perfect stays zero-advantage
    for sampled in ([1.0, 0.75, 1.0, 0.75], [1.0, -1.0, 0.5, 0.9], [0.1, 0.2, 0.3, 0.999999]):
        *samples, ref = advantages(sampled + [1.0])
        assert ref >= 0 and ref == max(samples + [ref])


def test_reference_member_policy_gradient_is_probability_weighted_and_unclipped():
    mx = pytest.importorskip("mlx.core")
    spec = importlib.util.spec_from_file_location("qwen_train_reference", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    probs = [0.05, 0.5, 0.79, 0.85, 0.999]  # below, inside and above the 0.8 clip edge
    lp = mx.log(mx.array(probs))
    behaviour = mx.zeros_like(lp)  # behaviour probability one
    for advantage in (0.0, 0.3, 1.7):
        grad = mx.grad(lambda x: trainer.grpo_loss(x, behaviour, lp, advantage, 0.0))(lp)
        assert grad.tolist() == pytest.approx([-advantage * p / len(probs) for p in probs], rel=1e-5, abs=1e-12)
        loss = trainer.grpo_loss(lp, behaviour, lp, advantage, 0.0)
        assert float(loss.item()) == pytest.approx(-advantage * sum(probs) / len(probs), rel=1e-5, abs=1e-12)
    # The equivalence needs a nonnegative advantage: a negative one would be clipped below 0.8.
    grad = mx.grad(lambda x: trainer.grpo_loss(x, behaviour, lp, -1.0, 0.0))(lp)
    assert grad.tolist()[0] == 0


IMPERFECT = [[2, GRPO_EOS]] * 4


def test_grpo_reference_member_joins_each_group_as_a_fifth_member(monkeypatch, tmp_path):
    import json
    import math
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten
    from common import grpo_row, informative
    steps = [MIXED, EQUAL, IMPERFECT, TRUNCATED]  # KO, JA, ZH, KO
    env = fake_grpo_runtime(monkeypatch, tmp_path, [c for group in steps for c in group])
    losses, member_grads = [], []
    real_value_and_grad = nn.value_and_grad

    def recording_value_and_grad(model, fn):
        def run(*args):
            loss, grad = real_value_and_grad(model, fn)(*args)
            mx.eval(loss, grad)
            losses.append(float(loss.item()))
            member_grads.append(dict(tree_flatten(grad)))
            return loss, grad
        return run

    monkeypatch.setattr(nn, "value_and_grad", recording_value_and_grad)
    calls = []
    real_loss = env.trainer.grpo_loss

    def recording_loss(log_probs, old, ref, advantage, beta):
        calls.append((old.tolist(), ref.tolist(), advantage))
        return real_loss(log_probs, old, ref, advantage, beta)

    monkeypatch.setattr(env.trainer, "grpo_loss", recording_loss)
    args = env.configure(["--steps", "4", "--reference-member", "--learning-rate", "0.005"])
    with env.cpu():
        env.trainer.run(args)
    assert not env.remaining
    assert env.seeds == [11, 12, 13, 14, 15]  # sampling seeds unchanged: load, then once per step
    assert len(env.updates) == 4 and len(calls) == len(losses) == 20
    reference = (*GRPO_REFERENCE, GRPO_EOS)
    # The TRAIN transcript is encoded once per step without special tokens; dev text only at the end.
    assert env.encoded == [("abcd", False)] * 4 + [(f"dev {lang}", False) for lang in ("en", "ko", "ja", "zh")]
    events = [json.loads(line) for line in (env.output / "metrics.jsonl").read_text().splitlines()]
    start = 0
    for step, (event, group, lang) in enumerate(zip(events, steps, ("ko", "ja", "zh", "ko")), 1):
        row = grpo_row(env.pools[lang], 11, step, 0)
        sampled = [reward("abcd", GRPO_TEXT[c[0]] * (2 if c[0] == 3 else 1), lang) if c[-1] == GRPO_EOS
                   else -1.0 for c in group]
        assert (event["id"], event["rewards"]) == (row["id"], sampled)  # the four actual samples
        assert len(event["completions"]) == 4
        assert event["reference_reward"] == 1.0
        assert event["advantages"] == advantages(sampled + [1.0])
        assert (event["draws"], event["skipped"]) == (1, [])
        assert event["exhausted"] == (not informative(sampled))
        assert event["zero_advantage"] == (group is EQUAL)
        if group in (IMPERFECT, TRUNCATED):  # equal but wrong: the reference alone is pushed up
            assert event["advantages"][-1] > 0 and all(x < 0 for x in event["advantages"][:4])
        # Old, parent and policy likelihoods: four samples then the transcript, same context.
        used = env.targets[start:env.updates[step - 1]]
        members = [tuple(c[:2]) for c in group] + [reference]
        assert used == [(int(row["audio"][:-4]), c) for c in members] * 3
        start = env.updates[step - 1]
        step_calls = calls[5 * (step - 1):5 * step]
        assert [advantage for _, _, advantage in step_calls] == event["advantages"]
        assert all(all(x < 0 for x in old) for old, _, _ in step_calls[:4])  # samples keep their old policy
        old, parent, _ = step_calls[4]
        assert old == [0.0] * len(reference)  # only the transcript's behaviour log-probability is zero
        assert all(x < 0 for x in parent) and len(parent) == len(reference)
        probability = event["reference_mean_probability"]
        assert 0 < probability < 1  # measured before the behaviour log-probability is replaced
        parent_probability = sum(math.exp(x) for x in parent) / len(parent)
        if step == 1:  # the policy still equals the parent: same likelihoods
            assert probability == pytest.approx(parent_probability, rel=1e-5)
            assert all(o == pytest.approx(p, rel=1e-6) for o, p, _ in step_calls[:4] for o, p in zip(o, p))
        else:  # the parent stays fixed while the policy has moved
            assert probability != pytest.approx(parent_probability, rel=1e-6)
        # Gradients are averaged over five members before the global norm.
        grads = member_grads[5 * (step - 1):5 * step]
        total = sum(float(mx.sum(sum(g[name] for g in grads) ** 2).item()) for name in grads[0])
        assert event["gradient_norm"] == pytest.approx(math.sqrt(total) / 5, rel=1e-4)
        assert event["loss"] == pytest.approx(sum(losses[5 * (step - 1):5 * step]) / 5)
    assert json.loads((env.output / "run.json").read_text())["contract"]["settings"]["reference_member"] is True


@pytest.mark.parametrize("first", [False, True])
def test_reference_member_change_is_refused_on_resume_before_model_load(monkeypatch, tmp_path, first):
    env = fake_grpo_runtime(monkeypatch, tmp_path, [*MIXED])
    flag = ["--reference-member"]
    with env.cpu():
        env.trainer.run(env.configure(["--steps", "1", *(flag if first else [])]))
    loads = []
    monkeypatch.setattr(sys.modules["mlx_audio.stt.utils"], "load_model", loads.append)
    with pytest.raises(ValueError, match="identical inputs/settings/code"):
        env.trainer.run(env.configure(["--steps", "2", "--resume", *([] if first else flag)]))
    assert not loads


def test_grpo_default_path_matches_the_pre_reference_member_trainer(monkeypatch, tmp_path):
    """Flag off, the trainer reproduces the pre-feature trainer's rows, rollouts, metrics and tensors."""
    import json
    import subprocess
    from types import ModuleType
    mx = pytest.importorskip("mlx.core")
    base = "4272251005d5b837cd238364a1567fd0870f276a"
    shown = subprocess.run(["git", "-C", str(DIRECTORY), "show", f"{base}:training/qwen_asr/train.py"],
                           capture_output=True, text=True)
    if shown.returncode:
        pytest.skip(f"pre-feature trainer {base} unavailable")
    script = [*MIXED, *EQUAL, [2, GRPO_EOS], [1, GRPO_EOS], [3, 3, 3], [2, GRPO_EOS]]
    env = fake_grpo_runtime(monkeypatch, tmp_path, script)
    previous = ModuleType("qwen_train_base")
    previous.__file__ = str(DIRECTORY / "train.py")  # same code/scorer paths
    exec(compile(shown.stdout, previous.__file__, "exec"), previous.__dict__)
    previous.training_targets = env.trainer.training_targets  # same likelihood recording
    observed = {}
    for name, trainer in (("current", env.trainer), ("base", previous)):
        env.remaining[:] = script
        del env.seeds[:], env.targets[:], env.contexts[:]
        env.configure(["--steps", "3", "--learning-rate", "0.05", "--save-every", "1"])
        args = trainer.arguments()
        args.output = tmp_path / name
        args.output.mkdir()
        with env.cpu():
            trainer.run(args)
        assert not env.remaining
        events = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
        for event in events:
            assert "reference_reward" not in event and "reference_mean_probability" not in event
            del event["seconds"], event["peak_memory_bytes"]
        tensors = {f"{step}/{kind}": mx.load(str(args.output / f"step-{step:06d}/{kind}.safetensors"))
                   for step in (1, 2, 3) for kind in ("adapters", "optimizer")}
        dev = json.loads((args.output / "dev-step-000003.json").read_text())["rows"]
        observed[name] = (events, tensors, dev, list(env.seeds), list(env.targets), list(env.contexts))
    (events, tensors, dev, seeds, targets, contexts), base_run = observed["current"], observed["base"]
    assert events == base_run[0]
    assert tensors.keys() == base_run[1].keys()
    for key, arrays in tensors.items():
        assert arrays.keys() == base_run[1][key].keys()
        assert all(mx.array_equal(arrays[n], base_run[1][key][n]).item() for n in arrays), key
    assert (dev, seeds, targets, contexts) == base_run[2:]
