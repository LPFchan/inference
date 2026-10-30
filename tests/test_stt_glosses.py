"""Translator glosses scored as optional reference spans by selection.py --glosses."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/qwen_asr"))
import glosses  # noqa: E402
import selection  # noqa: E402


def row(language, reference, hypothesis, found=None):
    value = {"id": "x", "language": language, "reference": reference, "hypothesis": hypothesis}
    if found is not None:
        value["glosses"] = found
    return value


def test_only_latin_spans_without_native_script_are_glosses():
    raw = "罗克汉普顿（Rockhampton）在（尤其是爪哇椰子糖）和(例えば、APS系)之间。Ski（\"Ski Bums\"）"
    reference = "罗 克 汉 普 顿 rockhampton 在 尤 其 是 爪 哇 椰 子 糖 和 例えば aps系 之 间 ski ski bums"
    assert glosses.glosses(raw, reference) == ["rockhampton", "ski bums"]


def test_repeated_glosses_must_each_occur_in_the_reference():
    assert glosses.glosses("A(Super G) B(Super G)", "a super g b super g") == ["super g", "super g"]
    with pytest.raises(ValueError, match="not in the reference"):
        glosses.glosses("A(Super G) B(Super G)", "a super g b")
    with pytest.raises(ValueError, match="not in the reference"):
        glosses.glosses("크레젤(Maciej Krezel)과", "크레젤과")


@pytest.mark.parametrize("hypothesis", ["罗克汉普顿在", "罗克汉普顿（Rockhampton）在", "罗克汉普顿 Rockhampton 在"])
def test_a_gloss_may_be_written_or_left_out(hypothesis):
    case = row("zh", "罗 克 汉 普 顿 rockhampton 在", hypothesis, ["rockhampton"])
    assert selection.error_counts(case) == (0, 6)


def test_a_misspelled_gloss_is_scored_against_the_closer_variant():
    case = row("zh", "罗 克 汉 普 顿 rockhampton 在", "罗克汉普顿rokhamptn在", ["rockhampton"])
    assert selection.error_counts(case) == (2, 6)  # two deletions, not nine insertions
    assert selection.error_counts(row("zh", "罗 克 汉 普 顿 rockhampton 在", "罗克汉普顿rokhamptn在")) == (2, 17)


def test_korean_glued_gloss_drops_without_splitting_the_word():
    case = row("ko", "마키 크레젤maciej krezel과 함께", "마키 크레젤과 함께", ["maciej krezel"])
    assert selection.error_counts(case) == (0, 3)
    kept = row("ko", "마키 크레젤maciej krezel과 함께", "마키 크레젤maciej krezel과 함께", ["maciej krezel"])
    assert selection.error_counts(kept) == (0, 3)


def test_each_gloss_is_optional_on_its_own():
    reference = "a super g b super g c"
    for hypothesis in ("a super g b c", "a b super g c", "a b c", reference):
        assert selection.error_counts(row("ko", reference, hypothesis, ["super g", "super g"])) == (0, 3)


def test_glosses_match_whole_latin_words_left_to_right():
    assert selection.gloss_spans("wifi 和 ai 芯 片", ["ai"]) == [(7, 9)]
    assert selection.gloss_spans("크레젤maciej krezel과", ["maciej krezel"]) == [(3, 16)]
    with pytest.raises(ValueError, match="not in the reference"):
        selection.gloss_spans("said", ["ai"])
    with pytest.raises(ValueError, match="not in the reference"):
        selection.gloss_spans("b super g a", ["a", "super g"])


def test_without_glosses_the_counts_are_unchanged():
    case = row("ko", "마키 크레젤maciej krezel과 함께", "마키 크레젤과 함께")
    assert selection.error_counts(case) == (2, 4)
    assert selection.error_counts({**case, "glosses": []}) == (2, 4)


def evaluation(tmp_path, name, rows):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"rows": rows}))
    return path


def test_gloss_file_must_cover_the_same_clips_and_references(tmp_path):
    rows = [{"id": "a", "language": "zh", "reference": "罗 克 rockhampton", "hypothesis": "罗克"}]
    path = evaluation(tmp_path, "e", rows)
    good = {"a": {"language": "zh", "reference": "罗 克 rockhampton", "glosses": ["rockhampton"]}}
    assert selection.read_evaluation(path, good)["a"]["glosses"] == ["rockhampton"]
    assert "glosses" not in selection.read_evaluation(path)["a"]
    with pytest.raises(ValueError, match="different evaluation set"):
        selection.read_evaluation(path, {**good, "b": good["a"]})
    with pytest.raises(ValueError, match="reference differs"):
        selection.read_evaluation(path, {"a": {**good["a"], "reference": "罗 克"}})


def test_build_reads_the_pinned_fleurs_rows_and_leaves_english_scored(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    snapshot = tmp_path / "snapshot"
    raw = {"ko": ("마키 크레젤(Maciej Krezel)과", "마키 크레젤maciej krezel과"),
           "ja": ("NSW（New South Wales）州", "nsw new south wales 州"),
           "zh": ("罗克汉普顿（Rockhampton）", "罗 克 汉 普 顿 rockhampton")}
    for lang, (raw_text, text) in raw.items():
        folder = snapshot / glosses.CONFIGS[lang] / "validation"
        folder.mkdir(parents=True)
        pq.write_table(pa.table({"transcription": ["unused", text], "raw_transcription": ["unused", raw_text]}),
                       folder / "0000.parquet")
    manifest = tmp_path / "dev.jsonl"
    dev = [{"id": f"fleurs-validation/{lang}/1", "language": lang, "text": text} for lang, (_, text) in raw.items()]
    dev.append({"id": "fleurs-validation/en/0", "language": "en", "text": "a satellite (especially gps) helps"})
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dev))
    result = glosses.build(manifest, snapshot)
    assert {i: r["glosses"] for i, r in result["rows"].items()} == {
        "fleurs-validation/ko/1": ["maciej krezel"], "fleurs-validation/ja/1": ["new south wales"],
        "fleurs-validation/zh/1": ["rockhampton"], "fleurs-validation/en/0": []}
    assert set(result["sources"]) == {"ko_kr", "ja_jp", "cmn_hans_cn"} and result["rule"].startswith("FLEURS raw")
    dev[0]["text"] = "changed"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dev))
    with pytest.raises(ValueError, match="does not match its FLEURS row"):
        glosses.build(manifest, snapshot)
