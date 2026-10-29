"""Source transcript parsing: one Common Voice TSV line is one training target."""
import csv
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/qwen_asr"))
from common import read_tsv

HEADER = "client_id\tpath\tsentence\tup_votes\tdown_votes\tlocale\n"
# An opening quote that is never closed, as in the pinned CV17 Japanese train
# TSV, followed by ordinary rows in several scripts.
UNBALANCED = ("a\tcommon_voice_ja_1.mp3\t\"引用が閉じられていない文\t2\t0\tja\n"
              "b\tcommon_voice_ja_2.mp3\t次の行は独立した文です。\t2\t0\tja\n"
              "c\tcommon_voice_zh-CN_3.mp3\t这是另一句话。\t2\t0\tzh-CN\n"
              "d\tcommon_voice_en_4.mp3\t\"Quoted,\" she said.\t2\t0\ten\n"
              "e\tcommon_voice_ko_5.mp3\t마지막 문장입니다.\t2\t0\tko\n")


def write(tmp_path, text):
    path = tmp_path / "train.tsv"
    path.write_text(HEADER + text, encoding="utf-8")
    return path


def test_default_csv_quoting_merges_lines_after_an_unbalanced_quote(tmp_path):
    with write(tmp_path, UNBALANCED).open(newline="") as stream:
        legacy = list(csv.DictReader(stream, delimiter="\t"))
    # The next quote character (inside the English row) closes the field, so
    # three ordinary records vanish into the first target.
    assert [r["client_id"] for r in legacy] == ["a", "e"]
    assert legacy[0]["sentence"].count("\n") == 3


def test_each_physical_line_keeps_its_own_literal_sentence(tmp_path):
    rows = read_tsv(write(tmp_path, UNBALANCED))
    assert [r["sentence"] for r in rows] == [
        "\"引用が閉じられていない文", "次の行は独立した文です。", "这是另一句话。",
        "\"Quoted,\" she said.", "마지막 문장입니다."]
    assert [r["locale"] for r in rows] == ["ja", "ja", "zh-CN", "en", "ko"]


@pytest.mark.parametrize("line", [
    "f\tcommon_voice_ja_6.mp3\t文\tタブ\t2\t0\tja\n",  # stray tab: one field too many
    "f\tcommon_voice_ja_6.mp3\t文\t2\t0\n",  # one field too few
])
def test_records_with_the_wrong_field_count_stop_intake(tmp_path, line):
    with pytest.raises(ValueError, match="malformed TSV record"):
        read_tsv(write(tmp_path, UNBALANCED + line))


def test_empty_trailing_fields_are_still_well_formed(tmp_path):
    row, = read_tsv(write(tmp_path, "f\tcommon_voice_ja_6.mp3\t文\t\t\t\n"))
    assert row["sentence"] == "文" and row["locale"] == ""


def test_expansion_assigns_each_japanese_clip_its_own_sentence(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    pytest.importorskip("pyarrow")
    from common import atomic_json, load_manifests, sha256
    import expand_clean
    from expand_clean import build
    monkeypatch.setattr(expand_clean.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * (1 << 30)))
    from prefetch import SOURCES

    root, core, output = (tmp_path / name for name in ("root", "core", "output"))
    for folder in (root, core / "wav", output):
        folder.mkdir(parents=True)
    rng = np.random.default_rng(7)

    def audio_bytes():
        stream = io.BytesIO()
        sf.write(stream, rng.uniform(-0.1, 0.1, 32000), 16000, format="WAV", subtype="PCM_16")
        return stream.getvalue()

    metadata = {}
    for split in ("train", "dev"):
        wav = core / "wav" / f"{split}.wav"
        wav.write_bytes(audio_bytes())
        line = json.dumps(dict(id=split, language="en", text=f"core {split}", speaker=split,
                               audio=f"wav/{split}.wav", sha256=sha256(wav), seconds=2,
                               source="fixture", source_revision="fixture", license="fixture"))
        (core / f"{split}.jsonl").write_text(line + "\n")
        metadata[f"{split}_sha256"] = sha256(core / f"{split}.jsonl")
    protected = root / "test.jsonl"
    protected.write_text(json.dumps(dict(reference="sealed test", sha256="sealed")) + "\n")
    atomic_json(core / "data.json", {**metadata, "protected_manifest_sha256": sha256(protected)})

    completed = {}

    def cached(key, content):
        path = root / str(len(completed))
        path.write_bytes(content)
        source = SOURCES[key.split("/")[0]]
        completed[key] = dict(path=str(path), sha256=sha256(path), source=source[0],
                              revision=source[1], license=source[2])

    japanese = "".join(line for line in UNBALANCED.splitlines(True) if line.endswith("\tja\n"))
    cached("cv17/transcript/ja/train.tsv", (HEADER + japanese).encode())
    cached("cv17/transcript/ja/dev.tsv", HEADER.encode())
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name in ("common_voice_ja_1.mp3", "common_voice_ja_2.mp3"):
            data = audio_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    cached("cv17/audio/ja/train/00000.tar", archive.getvalue())
    cached("aishell/data_aishell/transcript/aishell_transcript_v0.8.txt", b"")
    atomic_json(root / "prefetch.json", dict(status="complete", completed=completed))

    build(SimpleNamespace(root=root, core=core, output=output, protected_test=protected,
                          aishell_hours=40, seed=7))
    train, _, _ = load_manifests(output, protected)
    texts = {r["id"]: r["text"] for r in train}
    assert texts == {"train": "core train", "cv17/common_voice_ja_1": "\"引用が閉じられていない文",
                     "cv17/common_voice_ja_2": "次の行は独立した文です。"}
