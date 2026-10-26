"""A quarter written against a fiscal year is a quarter, not a fiscal year.

Earnings transcripts write the specific period inside the fiscal one -- "Q3
FY2023", "FY 2020 Q4", "Q1 FY 2022". Two things go wrong if the scorer reads
those naively, and both were measured over the 3,464 stored answers in
evaluation/ectqa/results*.json before this guard existed:

  * An unguarded fiscal-year alternative wins, because `asserted_period`
    takes the FIRST match in the span it examines and "FY 2020" precedes
    "Q4". The asserted quarter was replaced by its enclosing fiscal year in
    40 answers -- the error `asserted_period` exists to prevent.

  * If the fiscal marker is merely suppressed, nothing matches at all: the
    quarter branch wants the year immediately after the quarter and "FY" sits
    between them. `asserted_period` then walks past the model's own pick and
    grades it against the comparator it quoted next. That is what produced
    {2021-q4} for an answer whose emphasised choice was **Q1 FY 2022**.

So the quarter and half branches absorb an optional fiscal marker, and the
fiscal-year branch matches only when it stands alone.
"""

import importlib.util
from pathlib import Path

RUN_PY = Path(__file__).resolve().parent.parent / "evaluation" / "ectqa" / "run.py"
_spec = importlib.util.spec_from_file_location("ectqa_run_fy", RUN_PY)
_run = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_run)
except SystemExit:
    pass

_periods = _run._periods
asserted_period = _run.asserted_period


class TestAQuarterInsideAFiscalYear:
    """The specific period must win over the one enclosing it."""

    def test_quarter_against_a_fiscal_year_is_the_quarter(self):
        assert _periods("Q3 FY2023") == {"2023-q3"}
        assert _periods("Q3 FY 2023") == {"2023-q3"}
        assert _periods("Q4 FY 2020") == {"2020-q4"}

    def test_fiscal_year_first_is_still_the_quarter(self):
        assert _periods("FY 2020 Q4") == {"2020-q4"}

    def test_a_half_against_a_fiscal_year_is_the_half(self):
        assert _periods("H1 FY2023") == {"2023-h1"}

    def test_no_fiscal_token_leaks_alongside_the_quarter(self):
        # The regression: {"2023-fy", "2023-q3"} would score the answer
        # against a period it never asserted.
        assert "2023-fy" not in _periods("Q3 FY2023")


class TestABareFiscalYearStillWorks:
    """The feature the contributor added must survive the guard."""

    def test_a_fiscal_year_on_its_own(self):
        assert _periods("FY2022") == {"2022-fy"}
        assert _periods("FY 2022") == {"2022-fy"}
        assert _periods("fy-2022") == {"2022-fy"}

    def test_two_digit_fiscal_years_are_still_refused(self):
        assert _periods("FY22") == set()

    def test_prose_fiscal_is_not_a_token(self):
        assert _periods("fiscal 2022 was flat") == set()


class TestAssertedPeriodPrefersTheAnswerOverItsComparators:

    def test_the_emphasised_pick_wins_over_a_later_comparator(self):
        """Verbatim shape of the answer that scored against its own
        comparison figure: the pick is emphasised first, a comparator is
        emphasised later, and the scorer must take the first."""
        answer = (
            "Western Digital reported its highest non-GAAP gross margin in "
            "**Q1 FY 2022:** 33.9%, against **Q4 2021** at 32.1%."
        )
        assert asserted_period(answer) == {"2022-q1"}

    def test_a_fiscal_year_prefix_does_not_displace_the_quarter(self):
        answer = "**Western Digital Corporation - FY 2020 Q4 (April-June 2020)**"
        assert asserted_period(answer) == {"2020-q4"}

    def test_an_answer_asserting_only_a_fiscal_year_keeps_it(self):
        assert asserted_period("**FY2022** was the strongest year.") == {"2022-fy"}
