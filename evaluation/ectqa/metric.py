"""Unit-aware numeric F1 for ECT-QA -- TESTED AND REJECTED.

Kept so the negative result is reproducible, not wired into run.py.

The idea: precision suffers because a correct answer that explains itself
carries figures it never asserted as the requested value, so restrict the
precision denominator to answer figures whose unit class the gold uses.

The measurement, offline over the 68-question probe set: +0.02 on the control
arm and -0.06 on the answer-line arm -- flat at best, and negative where the
class test rejected a genuine match ("33.9%" in gold against a bare "33.9" in
the answer). The contamination turns out to be same-class: an answer about
quarterly revenue quotes other quarters' revenue, all of it currency. Unit
classes cannot separate what the model asserted from what it mentioned.

Original docstring follows.

"""
"""Unit-aware numeric F1 for ECT-QA.

ECT-QA gold answers are lists of figures, so matching figures is the
measurement and everything else is proxy. The weakness of a plain figure-set
F1 is precision: a correct answer that explains itself carries numbers it
never asserted as the requested value -- a growth rate quoted beside a
revenue figure, a headcount beside a margin -- and each one costs precision
against a gold that holds four values.

Measured: on the 68-question set, scoring only a model's final bare figures
line rather than its prose lifted accuracy 16 points with the retrieval held
identical. The engine was right and the ruler was wrong. Extracting that line
requires instructing the model to emit one, which measurably degrades its
refusals, so the fix belongs in the metric instead.

The fix is unit classes. A figure carries a class from its notation --
percent, currency, or bare count -- and a magnitude from any scale word
attached to it. Precision is then computed only over answer figures whose
class the gold actually uses: a percentage in the prose cannot dilute a gold
made of dollar amounts, because it was never a candidate for that answer.
Recall is unchanged, and matching stays one-to-one so an answer cannot cover
four distinct gold values by repeating one figure four times.
"""
import re

# Scale words bind to the figure they follow, so "3.3 billion" and
# "3,300 million" compare equal.
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mm": 1e6,
          "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}

# Chronology and citations are not figures. Without this a prose answer citing
# "fiscal 2023-q2 [1]" contributes 2023, 2 and 1 as values.
_NOT_A_FIGURE = re.compile(
    r"\[\d+\]"
    r"|\b(?:19|20)\d{2}-q[1-4]\b"
    r"|\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b"
    r"|\bq[1-4]\s+(?:19|20)\d{2}\b"
    r"|\b(?:fy\s*)?(?:19|20)\d{2}\b"
    r"|\bq[1-4]\b",
    re.I)

_FIGURE = re.compile(
    r"(?P<neg>negative\s+|minus\s+|-)?"
    r"(?P<cur>[$€£¥]|\bRMB\b|\bUSD\b|\bEUR\b)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|trillion|mm|bn|[kmb])?\s*"
    r"(?P<pct>%|\bpercent\b|\bpercentage points?\b|\bbps\b)?",
    re.I)


def figures(text: str):
    """(class, value) pairs asserted as figures in *text*.

    class is "pct", "cur" or "num". Currency symbol is not part of the class:
    a gold in RMB and an answer in RMB agree, and cross-currency confusion is
    a retrieval error the value comparison will catch anyway.
    """
    cleaned = _NOT_A_FIGURE.sub(" ", text or "")
    out = []
    for m in _FIGURE.finditer(cleaned):
        raw = m.group("num")
        if not raw:
            continue
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if m.group("scale"):
            val *= _SCALE[m.group("scale").lower()]
        if m.group("neg"):
            val = -val
        if m.group("pct"):
            cls = "pct"
        elif m.group("cur") or m.group("scale"):
            cls = "cur"
        else:
            cls = "num"
        out.append((cls, val))
    return out


def _match(a, b, rel_tol=0.01, abs_tol=0.05):
    """Same class, and equal within tolerance. Scale-only differences match."""
    (ca, va), (cb, vb) = a, b
    if ca != cb and not {ca, cb} <= {"cur", "num"}:
        return False
    return abs(va - vb) <= max(abs_tol, abs(va) * rel_tol)


def unit_aware_f1(gold: str, answer: str, rel_tol=0.01, abs_tol=0.05):
    """F1 over figures, precision restricted to the classes gold uses.

    Returns None when gold asserts no figure, so the caller can fall through
    to whatever it uses for narrative golds.
    """
    g = figures(gold)
    if not g:
        return None
    a = figures(answer)
    if not a:
        return 0.0

    # One-to-one greedy matching: an answer figure is spent once, so repeating
    # a value cannot cover several distinct gold values.
    unused = list(a)
    matched = 0
    for gv in g:
        for i, av in enumerate(unused):
            if _match(gv, av, rel_tol, abs_tol):
                del unused[i]
                matched += 1
                break
    recall = matched / len(g)

    # Precision candidates: only answer figures in a class gold actually uses.
    gold_classes = {c for c, _ in g}
    if gold_classes <= {"cur", "num"}:
        gold_classes |= {"cur", "num"}
    candidates = [x for x in a if x[0] in gold_classes]
    if not candidates:
        return 0.0
    unused = list(g)
    hit = 0
    for av in candidates:
        for i, gv in enumerate(unused):
            if _match(gv, av, rel_tol, abs_tol):
                del unused[i]
                hit += 1
                break
    precision = hit / len(candidates)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
