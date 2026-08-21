"""Tests for scripts.validate_project."""

from scripts.validate_project import ValidationReport, run_validation


def test_validation_report_all_passed_true_when_empty():
    report = ValidationReport()
    assert report.all_passed


def test_validation_report_all_passed_false_on_any_failure():
    report = ValidationReport()
    report.add("check_a", True)
    report.add("check_b", False, "something broke")
    assert not report.all_passed


def test_validation_report_print_summary_does_not_raise(capsys):
    report = ValidationReport()
    report.add("check_a", True)
    report.add("check_b", False, "detail here")
    report.print_summary()
    captured = capsys.readouterr()
    assert "check_a" in captured.out
    assert "check_b" in captured.out


def test_run_validation_passes_in_this_environment():
    """End-to-end: every check should pass in a correctly configured
    environment with all dependencies installed (this test environment)."""
    result = run_validation()
    assert result is True
