"""FLEURS written-form intake exclusions and the frozen KSS probe (CPU only)."""
import io
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy.signal")

DIRECTORY = Path(__file__).resolve().parents[1] / "training/qwen_asr"
sys.path.insert(0, str(DIRECTORY))
import prepare_fleurs_style as prep  # noqa: E402
from common import sha256  # noqa: E402


@pytest.fixture(autouse=True)
def stable_resources(monkeypatch):
    monkeypatch.setattr(prep, "MIN_FREE", 0)
    monkeypatch.setattr(prep, "memory_pressure_normal", lambda: True)


def tone(hz, seconds=2.0, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, 0.1 * np.sin(2 * np.pi * hz * t), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_parquet(path, columns, row_group_size=2):
    table = pa.table({k: v for k, v in columns.items()})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def frozen_split(tmp_path):
    """Frozen dev directory plus a protected test manifest, built like the real ones."""
    dev_dir = tmp_path / "dev-source"
    (dev_dir / "wav").mkdir(parents=True)
    wav = dev_dir / "wav/fleurs-dev-ko-00000.wav"
    sf.write(wav, sf.read(io.BytesIO(tone(300)), dtype="float32")[0], 16000, subtype="PCM_16")
    dev = [{"id": "fleurs-validation/ko/0", "language": "ko", "speaker": "", "audio": "wav/fleurs-dev-ko-00000.wav",
            "text": "검증 문장입니다", "seconds": 2.0, "sha256": sha256(wav)}]
    (dev_dir / "dev.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dev))
    (dev_dir / "data.json").write_text("{}")
    protected = tmp_path / "test.jsonl"
    protected.write_text(json.dumps({"id": 900, "language": "ja", "reference": "試験の文です。",
                                     "sha256": "0" * 64}, ensure_ascii=False) + "\n")
    return dev_dir, protected, dev[0]["sha256"]


def test_fleurs_intake_excludes_heldout_ids_text_audio_and_probe(tmp_path):
    dev_dir, protected, _ = frozen_split(tmp_path)
    probe = tmp_path / "probe.jsonl"
    probe.write_text(json.dumps({"text": "프로브 문장.", "sha256": "1" * 64}, ensure_ascii=False) + "\n")
    audio = lambda hz, s=2.0: {"bytes": tone(hz, s), "path": None}
    ko = write_parquet(tmp_path / "ko.parquet", {
        "id": [1, 2, 3, 4, 5, 6, 1],
        "audio": [audio(440), audio(450), audio(460), audio(300), audio(470, 0.5), audio(480), audio(440)],
        "raw_transcription": ["2024년에  열렸다.", "검증 문장입니다!", "다른 문장", "새 문장", "짧다", "프로브 문장", "또 읽기."],
        "transcription": ["2024년에 열렸다", "검증 문장입니다", "다른 문장", "새 문장", "짧다", "프로브 문장", "또 읽기"],
    })
    ja = write_parquet(tmp_path / "ja.parquet", {
        "id": [7, 900], "audio": [audio(500), audio(510)],
        "raw_transcription": ["日本語の文。", "別の訳文。"], "transcription": ["日本語の文", "別の訳文"],
    })
    out = tmp_path / "data"
    blocked = {3, 900}  # one validation id from any language, one test id
    train, dev, meta = prep.build_train(out, {"ko": ko, "ja": ja}, blocked, dev_dir, protected, [probe], {})

    assert [r["id"] for r in train] == ["fleurs-train/ko/0", "fleurs-train/ja/0"]
    assert train[0]["text"] == "2024년에 열렸다."  # written form, whitespace only normalized
    assert {r["speaker"] for r in train} == {""}  # unknown, not invented
    assert meta["rejected"] == {"ko/heldout_text": 1, "ko/heldout_source_sentence_id": 1,
                                "ko/heldout_audio": 1, "ko/duration_or_nonfinite": 1,
                                "ko/probe_text": 1, "ko/train_audio_duplicate": 1,
                                "ja/heldout_source_sentence_id": 1}
    assert meta["source_sentence_ids"]["train_shared_across_languages"] == {"ja&ko": 0}
    assert (out / "dev.jsonl").read_bytes() == (dev_dir / "dev.jsonl").read_bytes()
    assert sorted(p.name for p in (out / "wav").iterdir()) == [
        "fleurs-dev-ko-00000.wav", "fleurs-train-ja-00000.wav", "fleurs-train-ko-00000.wav"]


def kss_files(tmp_path):
    texts = ["사과 3개를 샀다.", "책을 읽었다.", "같은 문장.", "같은 문장.", "학습 문장.",
             "버스는 10시에 온다.", "비가 온다.", "눈이 온다.", "검증 문장입니다", "바람이 분다."]
    audio = [{"bytes": tone(200 + 20 * i, rate=44100), "path": None} for i in range(len(texts))]
    first = write_parquet(tmp_path / "train-00000.parquet", {
        "audio": audio[:5], "original_script": texts[:5], "duration": [2.0] * 5}, row_group_size=3)
    second = write_parquet(tmp_path / "train-00001.parquet", {
        "audio": audio[5:], "original_script": texts[5:], "duration": [2.0] * 5}, row_group_size=3)
    return [("data/train-00000.parquet", first), ("data/train-00001.parquet", second)]


def test_kss_probe_is_balanced_unique_excluded_and_deterministic(tmp_path):
    dev_dir, protected, _ = frozen_split(tmp_path)
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"text": "학습 문장", "sha256": "2" * 64}, ensure_ascii=False) + "\n")
    files = kss_files(tmp_path)
    runs = [prep.build_probe(tmp_path / f"probe{i}", files, dev_dir, protected, [pool],
                             seed=7, per_cohort=2, reserve=1, downloads={}) for i in (1, 2)]
    assert [r["id"] for r in runs[0]] == [r["id"] for r in runs[1]]
    probe = runs[0]
    cohorts = [r["cohort"] for r in probe]
    assert cohorts == ["digit", "digit", "other", "other"]  # all available digit rows, then others
    texts = {r["text"] for r in probe}
    assert texts.isdisjoint({"같은 문장.", "학습 문장.", "검증 문장입니다"})
    assert len({r["sha256"] for r in probe}) == len(probe)
    meta = json.loads((tmp_path / "probe1/probe.json").read_text())
    assert meta["eligible_population"] == {"digit": 2, "other": 4}
    assert meta["rejected"] == {"repeated_text": 2, "pool_text": 1, "heldout_text": 1}
    assert len(list((tmp_path / "probe1/wav").iterdir())) == len(probe)
    assert prep.verify_probe(tmp_path / "probe1") == probe


def test_kss_probe_refuses_an_incomplete_cohort(tmp_path):
    dev_dir, protected, _ = frozen_split(tmp_path)
    output = tmp_path / "probe"
    with pytest.raises(ValueError, match="insufficient digit clips"):
        prep.build_probe(output, kss_files(tmp_path), dev_dir, protected, [],
                         seed=7, per_cohort=3, reserve=1, downloads={})
    assert not (output / "probe.json").exists()


def test_guard_stops_on_sustained_memory_pressure(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "memory_pressure_normal", lambda: False)
    monkeypatch.setattr(prep.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="memory pressure"):
        prep.guard(tmp_path, patience=0)
