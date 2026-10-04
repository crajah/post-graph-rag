from evaluation.ectqa.run import _numbers, period_f1


def test_period_f1_supports_month():
    assert period_f1("March 2022", "March 2022") == 1.0


def test_period_f1_supports_iso_month():
    assert period_f1("2022-03", "2022-03") == 1.0


def test_period_f1_supports_half_year():
    assert period_f1("H1 2022", "H1 2022") == 1.0


def test_period_f1_supports_fiscal_year():
    assert period_f1("FY2022", "FY2022") == 1.0


def test_half_year_is_not_scored_as_a_number():
    assert _numbers("Revenue increased in H1 2022 from 120 to 150.") == [
        120.0,
        150.0,
    ]


def test_period_f1_matches_month_formats():
    assert period_f1("March 2022", "2022-03") == 1.0


def test_period_f1_matches_fiscal_year_spacing():
    assert period_f1("FY2022", "FY 2022") == 1.0


def test_period_f1_does_not_match_different_months():
    assert period_f1("March 2022", "April 2022") == 0.0
