#!/usr/bin/env python3
"""CPU-only helpers for language-mean SFT token weighting and train.py's teacher-forced TRAIN NLL probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import wave

from common import LANGUAGES, atomic_json, load_manifests, sentence_epoch_draw, sentence_epoch_order, sentence_key, sha256

MODE = "language-mean"
BOOTSTRAP, BOOTSTRAP_SEED = 400, 20261030
PRIMARY, DIAGNOSTIC = "zh", ("en",)
SCOPE = ("TRAIN-only teacher-forced NLL of one adapter on sentences its own run never drew in that language. "
         "The same content may have been drawn in another language. Not held-out accuracy or an adoption gate.")
RECORD = ("index", "id", "language", "sentence", "reference_sha256")


def reference_rows(reference, data, protected, train, metadata):
    """Validate the full reference snapshot; TRAIN must be a matching subset of it.

    A row matches when every field except its audio location is equal; the audio
    hash binds the content. Returns the reference TRAIN rows and their binding.
    """
    reference = Path(reference)
    if reference.resolve() == Path(data).resolve():
        rows, meta = train, metadata
    else:
        rows, _, meta = load_manifests(reference, protected)
    for key in ("dev_sha256", "protected_manifest_sha256"):
        if meta[key] != metadata[key]:
            raise ValueError(f"reference snapshot has a different {key}")
    known = {row["id"]: row for row in rows}

    def identity(row):
        return {key: value for key, value in row.items() if key != "audio"}
    for row in train:
        if row["id"] not in known or identity(known[row["id"]]) != identity(row):
            raise ValueError(f"TRAIN row is not in the reference snapshot: {row['id']}")
    return rows, {"mode": MODE, "reference_train_sha256": meta["train_sha256"],
                  "reference_data_sha256": sha256(reference / "data.json")}


def statistics(rows, count, binding, eos):
    """Per-language mean token count (transcript plus EOS, as SFT supervises) over every reference row."""
    totals = {lang: [0, 0] for lang in LANGUAGES}
    for row in rows:
        totals[row["language"]][0] += 1
        totals[row["language"]][1] += count(row["text"])
    if not all(recordings for recordings, _ in totals.values()):
        raise ValueError("reference TRAIN needs all four languages")
    return {**binding, "eos_token_id": eos, "languages": {
        lang: {"recordings": recordings, "tokens": tokens, "mean": tokens / recordings}
        for lang, (recordings, tokens) in totals.items()}}


def freeze_statistics(path, stats, published):
    """Write the statistics once; afterwards a recomputation must match them exactly.

    Missing statistics are refused once the run has published an update.
    """
    path = Path(path)
    if path.exists():
        try:
            saved = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError(f"corrupt token-weight statistics: {path}") from None
        if saved != stats:
            raise ValueError("token-weight statistics differ from this run's saved statistics")
    elif published:
        raise ValueError("missing token-weight statistics for a run with published updates")
    else:
        atomic_json(path, stats)
    return {lang: value["mean"] for lang, value in stats["languages"].items()}


def language_mean_objective(nll, mean):
    """Weighted objective sum(nll) / language mean, and the unweighted per-clip mean from the same forward."""
    return nll.sum() / mean, nll.mean()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def probe_plan(path, train, data, checkpoint, schedule):
    """Verify a frozen per-seed TRAIN probe against the adapter's own sentence-epoch draws.

    Every clip must be a TRAIN row of its language with a valid WAV whose sentence
    the checkpoint's run never drew in that language. Draws of the same sentence
    in another language are recorded per clip, not refused.
    """
    path = Path(path)
    frozen = json.loads(path.read_text())
    contract = checkpoint["contract"]
    settings = contract["settings"]
    if (checkpoint["stage"] != "sft" or settings.get("sampler") != "sentence-epoch"
            or contract.get("init_adapter_sha256") is not None):
        raise ValueError("TRAIN NLL probe needs an adapter from one fresh sentence-epoch SFT run")
    if frozen["train_sha256"] != contract["data"]["train_sha256"]:
        raise ValueError("probe plan was frozen from other TRAIN data")
    seed = settings["seed"]
    probe = frozen["probes"].get(str(seed))
    if probe is None or set(probe["languages"]) != set(LANGUAGES):
        raise ValueError(f"probe plan has no four-language entry for seed {seed}")
    ids = [clip for lang in LANGUAGES for clip in probe["languages"][lang]["ids"]]
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != probe["ids_sha256"] or len(set(ids)) != len(ids):
        raise ValueError("changed or duplicated probe plan IDs")
    pools = {lang: [row for row in train if row["language"] == lang] for lang in LANGUAGES}
    order = sentence_epoch_order(pools, seed)
    drawn = {}
    for index in range(checkpoint["step"] * settings["accumulate"]):
        row = sentence_epoch_draw(order, index, schedule)
        drawn.setdefault(sentence_key(row), set()).add(row["language"])
    rows = {row["id"]: row for row in train}
    plan, seen = [], set()
    for lang in LANGUAGES:
        for clip in probe["languages"][lang]["ids"]:
            row = rows.get(clip)
            if row is None or row["language"] != lang:
                raise ValueError(f"probe clip is not a {lang} TRAIN row: {clip}")
            key = sentence_key(row)
            if lang in drawn.get(key, ()) or (lang, key) in seen:
                raise ValueError(f"probe sentence was drawn, or repeated, in its language: {clip}")
            seen.add((lang, key))
            with wave.open(str(Path(data) / row["audio"])) as audio:
                layout = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
                seconds = audio.getnframes() / audio.getframerate()
            if layout != (1, 2, 16000) or "seconds" not in row or abs(seconds - row["seconds"]) > 1e-3:
                raise ValueError(f"unexpected probe audio format or duration: {clip}")
            plan.append({"index": len(plan), "id": clip, "language": lang, "sentence": list(key),
                         "reference_sha256": text_sha256(row["text"]),
                         "drawn_in_other_languages": sorted(drawn.get(key, set()) - {lang})})
    return plan, {"plan_sha256": sha256(path), "seed": seed, "ids_sha256": probe["ids_sha256"],
                  "checkpoint": {"step": checkpoint["step"], "accumulate": settings["accumulate"],
                                 "sampler": settings["sampler"]}}


def _digest(record):
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def record(clip, contract_sha256, tokens, nll_sum):
    if not math.isfinite(nll_sum):
        raise FloatingPointError("nonfinite TRAIN NLL; record not published")
    row = {**{key: clip[key] for key in RECORD}, "tokens": tokens, "nll_sum": nll_sum,
           "contract_sha256": contract_sha256}
    row["record_sha256"] = _digest(row)
    return row


def saved_records(path, plan, contract_sha256):
    """Completed clips; reject anything edited, duplicated, foreign or from another contract.

    Only a torn final line from an interrupted write is dropped.
    """
    path = Path(path)
    data = path.read_bytes() if path.exists() else b""
    lines = data.splitlines()
    repair = bool(data and not data.endswith(b"\n"))
    if repair:
        try:
            json.loads(lines[-1])
        except (json.JSONDecodeError, UnicodeDecodeError):
            lines.pop()
    expected = {clip["id"]: clip for clip in plan}
    saved = {}
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError(f"corrupt TRAIN NLL record at line {number}") from None
        clip = expected.get(row.get("id")) if isinstance(row, dict) else None
        if (clip is None or row["id"] in saved or row.get("contract_sha256") != contract_sha256
                or any(row.get(key) != clip[key] for key in RECORD)
                or not (isinstance(row.get("tokens"), int) and row["tokens"] >= 1)
                or not (isinstance(row.get("nll_sum"), float) and math.isfinite(row["nll_sum"]))
                or row.get("record_sha256") != _digest(row)):
            raise ValueError(f"saved TRAIN NLL record drift at line {number}")
        saved[row["id"]] = row
    if repair:
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(b"".join(line + b"\n" for line in lines))
        temporary.replace(path)
    return saved


def summary(plan, saved):
    """Token-pooled NLL per language (sum NLL / sum tokens) and every clip's record; languages never pooled."""
    records = [{"id": clip["id"], "language": clip["language"], "tokens": saved[clip["id"]]["tokens"],
                "nll_sum": saved[clip["id"]]["nll_sum"], "drawn_in_other_languages": clip["drawn_in_other_languages"]}
               for clip in plan]
    languages = {}
    for lang in LANGUAGES:
        rows = [r for r in records if r["language"] == lang]
        tokens, total = sum(r["tokens"] for r in rows), math.fsum(r["nll_sum"] for r in rows)
        languages[lang] = {"clips": len(rows), "tokens": tokens, "nll_sum": total,
                           "token_pooled_nll": total / tokens if tokens else None,
                           "clips_drawn_in_another_language": sum(bool(r["drawn_in_other_languages"]) for r in rows)}
    return {"scope": SCOPE, "nll": "teacher-forced reference tokens plus EOS at temperature 1; "
                                   "sum over a clip's tokens, pooled per language by token count",
            "languages": languages, "records": records}


def contract_sha256(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def paired(pairs):
    """Candidate-minus-twin token-pooled NLL and a paired clip bootstrap over (tokens, candidate, twin)."""
    if not pairs:
        return {"clips": 0, "tokens": 0, "difference": None, "ci95": None}

    def pooled(chosen, side):
        return math.fsum(p[side] for p in chosen) / sum(p[0] for p in chosen)
    result = {"clips": len(pairs), "tokens": sum(p[0] for p in pairs), "candidate_nll": pooled(pairs, 1),
              "twin_nll": pooled(pairs, 2)}
    result["difference"] = result["candidate_nll"] - result["twin_nll"]
    result["ci95"] = None
    if len(pairs) >= 2:
        rng = random.Random(BOOTSTRAP_SEED)
        deltas = sorted(pooled(s, 1) - pooled(s, 2) for s in (rng.choices(pairs, k=len(pairs))
                                                               for _ in range(BOOTSTRAP)))
        result["ci95"] = [deltas[int(0.025 * BOOTSTRAP)], deltas[min(BOOTSTRAP - 1, int(0.975 * BOOTSTRAP))]]
    return result


def compare(candidate, twin):
    """Paired readout of two finished TRAIN NLL summaries that differ only in the adapter."""
    for report in (candidate, twin):
        if contract_sha256(report["contract"]) != report["contract_sha256"]:
            raise ValueError("TRAIN NLL summary contract digest mismatch")
    strip = lambda contract: {k: v for k, v in contract.items() if k != "init_adapter_sha256"}
    if strip(candidate["contract"]) != strip(twin["contract"]):
        raise ValueError("TRAIN NLL summaries differ in more than the adapter")
    if candidate["contract"]["init_adapter_sha256"] == twin["contract"]["init_adapter_sha256"]:
        raise ValueError("candidate and twin are the same adapter")
    identity = lambda report: [(r["id"], r["language"], r["tokens"]) for r in report["records"]]
    if identity(candidate) != identity(twin):
        raise ValueError("TRAIN NLL summaries cover different clips")
    pairs = {lang: [] for lang in LANGUAGES}
    for c, t in zip(candidate["records"], twin["records"]):
        pairs[c["language"]].append((c["tokens"], c["nll_sum"], t["nll_sum"], c["id"]))
    languages = {lang: paired(rows) for lang, rows in pairs.items()}
    ranked = sorted(pairs[PRIMARY], key=lambda p: (p[0], p[3]))
    third = len(ranked) // 3
    primary = languages[PRIMARY]
    if primary["ci95"] is None:
        verdict = "untestable"
    elif primary["difference"] >= 0:
        verdict = "falsified"
    else:
        verdict = "supported" if primary["ci95"][1] < 0 else "inconclusive"
    return {"scope": SCOPE + " One seed's candidate against its same-seed twin.",
            "rule": f"{PRIMARY} candidate-minus-twin token-pooled NLL: CI95 upper < 0 supported, point >= 0 "
                    "falsified, otherwise inconclusive. The mechanism is supported only if every trained seed is.",
            "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED, "unit": "clip, paired"},
            "verdict": verdict, "primary": PRIMARY, "diagnostic": list(DIAGNOSTIC),
            "report_only": [lang for lang in LANGUAGES if lang != PRIMARY and lang not in DIAGNOSTIC],
            "candidate": {"init_adapter_sha256": candidate["contract"]["init_adapter_sha256"],
                          "checkpoint": candidate.get("checkpoint")},
            "twin": {"init_adapter_sha256": twin["contract"]["init_adapter_sha256"],
                     "checkpoint": twin.get("checkpoint")},
            "languages": languages,
            f"{PRIMARY}_thirds_by_tokens": None if not third else {
                "shortest": paired(ranked[:third]), "longest": paired(ranked[-third:])}}


def main():
    p = argparse.ArgumentParser(description="Compare a candidate's TRAIN NLL summary with its same-seed twin's.")
    p.add_argument("command", choices=("compare",))
    p.add_argument("candidate", type=Path)
    p.add_argument("twin", type=Path)
    p.add_argument("--output", type=Path, help="write the comparison here; an existing file is never replaced")
    a = p.parse_args()
    result = compare(*(json.loads(path.read_text()) for path in (a.candidate, a.twin)))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if a.output:
        with a.output.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
