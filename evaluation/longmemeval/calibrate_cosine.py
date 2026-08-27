"""Fit the cosine threshold on stored reader-sweep signals.

`reader_sweep.py` stores four signals per answer -- numeric F1, period F1,
cosine, and the judge ballot -- so the operating point of the cosine fallback
can be chosen from data instead of inherited. The 0.62 default came from ECT-QA,
where gold is a list of figures; LongMemEval gold is terse prose against verbose
answers, which is a different distribution and has no reason to share a
threshold.

What this can and cannot do. It measures *agreement with the panel*, not
correctness. This repository has twice found panels to be the weaker instrument
-- on ECT-QA a panel scored six points below a deterministic metric over
identical answers -- so a threshold fitted here inherits whatever bias the panel
has. Read the separation statistics rather than the argmax: if the two
distributions barely separate, no threshold is defensible and the honest
conclusion is that cosine does not measure this, which is a result in itself.

Deterministically-scored answers are excluded. Where numeric or period F1 fired,
cosine was never consulted, so including them would fit the threshold on rows it
will never be applied to.

    python evaluation/longmemeval/calibrate_cosine.py reader_sweep.json
"""
import argparse
import json
import pathlib
import statistics


def rows_from(path: pathlib.Path):
    """Every answer scored by cosine, with the panel's verdict beside it."""
    d = json.loads(path.read_text())
    out = []
    for inst in d["instances"]:
        for arm, v in inst["arms"].items():
            for r in v["runs"]:
                if r["metric"] == "cosine" and isinstance(r.get("value"), (int, float)):
                    out.append({"arm": arm, "type": inst["type"],
                                "cos": float(r["value"]), "judge": bool(r["judge"])})
    return d, out


def at(rows, thr):
    tp = sum(1 for r in rows if r["cos"] >= thr and r["judge"])
    fp = sum(1 for r in rows if r["cos"] >= thr and not r["judge"])
    fn = sum(1 for r in rows if r["cos"] < thr and r["judge"])
    tn = sum(1 for r in rows if r["cos"] < thr and not r["judge"])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(rows) if rows else 0.0
    tpr = rec
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "youden": tpr - fpr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="a reader_sweep.py output file")
    ap.add_argument("--current", type=float, default=0.62,
                    help="the threshold in force, reported for comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_absolute():
        path = pathlib.Path(__file__).resolve().parent / path
    d, rows = rows_from(path)
    if not rows:
        raise SystemExit("no cosine-scored answers in that file")

    pos = [r["cos"] for r in rows if r["judge"]]
    neg = [r["cos"] for r in rows if not r["judge"]]
    print(f"{len(rows)} cosine-scored answers "
          f"({len(pos)} panel-correct, {len(neg)} panel-incorrect)")
    if not pos or not neg:
        raise SystemExit("one class is empty; nothing to separate")

    mp, mn = statistics.mean(pos), statistics.mean(neg)
    print(f"\n  mean cosine, panel-correct    {mp:.3f}"
          f"  (sd {statistics.pstdev(pos):.3f})")
    print(f"  mean cosine, panel-incorrect  {mn:.3f}"
          f"  (sd {statistics.pstdev(neg):.3f})")
    print(f"  separation                    {mp - mn:+.3f}")
    # Effect size says whether any threshold can work, which the argmax does not.
    sd = statistics.pstdev(pos + neg) or 1e-9
    dprime = (mp - mn) / sd
    print(f"  standardised separation       {dprime:+.2f}"
          f"   ({'usable' if abs(dprime) >= 0.8 else 'weak — see the note below'})")

    grid = [round(0.30 + 0.01 * i, 2) for i in range(66)]
    scored = [at(rows, t) for t in grid]
    best_f1 = max(scored, key=lambda s: s["f1"])
    best_acc = max(scored, key=lambda s: s["accuracy"])
    best_j = max(scored, key=lambda s: s["youden"])
    cur = at(rows, args.current)

    print(f"\n  {'':16}{'thr':>6}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}{'J':>8}")
    for name, s in (("current", cur), ("best accuracy", best_acc),
                    ("best F1", best_f1), ("best Youden J", best_j)):
        print(f"  {name:<16}{s['thr']:>6.2f}{s['accuracy']:>8.3f}"
              f"{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}"
              f"{s['youden']:>8.3f}")

    print("\n  agreement across the grid (every 0.05):")
    for s in scored:
        if abs(s["thr"] * 100 % 5) < 1e-6:
            bar = "#" * int(s["accuracy"] * 40)
            print(f"    {s['thr']:.2f}  {s['accuracy']:.3f}  {bar}")

    if abs(dprime) < 0.8:
        print("\n  NOTE: the two distributions overlap heavily. The argmax above is\n"
              "  fitting noise; no cosine threshold separates panel-correct from\n"
              "  panel-incorrect on this data. Report that, do not ship the number.")

    payload = {"source": str(path.name), "n": len(rows),
               "mean_correct": mp, "mean_incorrect": mn, "separation": mp - mn,
               "d_prime": dprime, "current": cur, "best_f1": best_f1,
               "best_accuracy": best_acc, "best_youden": best_j,
               "grid": scored,
               "caveat": "agreement with an LLM panel, not with ground truth"}
    out = pathlib.Path(args.out) if args.out else path.with_name(
        path.stem + "_calibration.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
