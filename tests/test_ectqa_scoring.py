import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_PY = HERE.parent / "evaluation" / "ectqa" / "run.py"

_spec = importlib.util.spec_from_file_location("ectqa_run", RUN_PY)
_run = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_run)
except SystemExit:
    pass

asserted_period = _run.asserted_period
_periods = _run._periods
_numbers = _run._numbers


def test_period_normalization():
    # Quarters
    assert _periods("Q1 2022") == {"2022-q1"}
    assert _periods("2022-Q1") == {"2022-q1"}
    assert _periods("q3-2023") == {"2023-q3"}

    # Months
    assert _periods("March 2022") == {"2022-03"}
    assert _periods("Mar 2022") == {"2022-03"}
    assert _periods("2022-03") == {"2022-03"}
    assert _periods("jan, 2022") == {"2022-01"}
    assert _periods("Dec 2020") == {"2020-12"}

    # Half-years
    assert _periods("H1 2022") == {"2022-h1"}
    assert _periods("2022-H1") == {"2022-h1"}
    assert _periods("h2 2023") == {"2023-h2"}

    # Fiscal years
    assert _periods("FY2022") == {"2022-fy"}
    assert _periods("FY 2022") == {"2022-fy"}
    assert _periods("fy2023") == {"2023-fy"}


def test_invalid_forms_rejected():
    assert _periods("Q5 2022") == set()
    assert _periods("H3 2022") == set()
    assert _periods("2022-13") == set()
    assert _periods("2022-00") == set()
    assert _periods("FY22") == set()
    assert _periods("March 3, 2022") == set()
    assert _periods("2022-03-99") == set()


def test_asserted_period_extraction():
    # Quarters
    assert asserted_period("The peak was in Q1 2022") == {"2022-q1"}
    
    # Months
    assert asserted_period("The highest revenue was in March 2022") == {"2022-03"}
    
    # Half-years
    assert asserted_period("Most losses occurred in H1 2022") == {"2022-h1"}
    
    # Fiscal years
    assert asserted_period("Peak revenue was in FY2022") == {"2022-fy"}


def test_asserted_period_context():
    # Answer vs context differentiation
    assert asserted_period("The highest margin was in Q3 2022, compared with Q2 2022.") == {"2022-q3"}
    assert asserted_period("The highest revenue was in March 2022, compared with February 2022.") == {"2022-03"}
    assert asserted_period("Most losses occurred in H1 2022, compared with H2 2021.") == {"2022-h1"}
    assert asserted_period("Peak revenue was in FY2022, compared with FY2021.") == {"2022-fy"}

    # Constraint behavior check
    # Falls back to first sentence which yields both
    assert asserted_period("The highest margin prior to Q2 2022 was Q1 2022.") == {"2022-q2", "2022-q1"}


def test_numeric_leakage():
    # Quarter chronology should remain stripped
    assert _numbers("Revenue was $12M in Q3 2022") == [12.0]
    
    # Half-year should not leak "1" or "2"
    assert _numbers("Revenue was $12M in H1 2022") == [12.0]
    assert _numbers("Revenue was 12 million in H1 2022") == [12.0]
    
    # Fiscal year should not leak year
    assert _numbers("Revenue was $30B in FY2022") == [30.0]
    
    # Months should not leak year, and month name shouldn't yield numbers
    assert _numbers("Revenue was $12M in March 2022") == [12.0]
    assert _numbers("Revenue increased 5% in March 2022") == [5.0]
    
    # ISO month form should not leak
    assert _numbers("Revenue was $12M in 2022-03") == [12.0]
