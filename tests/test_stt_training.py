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
