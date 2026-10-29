"""check_gate fails a run on a broken budget or a drop from the baseline, and only then."""

from workout_agent.evals.gate import check_gate

BASELINE = {
    "verified": 1.0,
    "all_checks_pass": 1.0,
    "kind_accuracy": 1.0,
    "trajectory_accuracy": 1.0,
    "mean_tokens_per_turn": 2000.0,
    "p50_seconds": 3.0,
}


def run(**changes):
    return {**BASELINE, **changes}


def test_a_run_equal_to_the_baseline_passes():
    assert check_gate(run(), BASELINE, timing=True) == []


def test_an_unverified_reply_fails_with_no_slack():
    [failure] = [f for f in check_gate(run(verified=0.97), BASELINE) if "faithfulness" in f]
    assert "0.97" in failure


def test_trajectory_accuracy_at_the_budget_fails():
    assert any(
        "trajectory_accuracy" in f for f in check_gate(run(trajectory_accuracy=0.95), BASELINE)
    )


def test_a_small_drop_in_kind_accuracy_is_within_slack():
    assert check_gate(run(kind_accuracy=0.96), BASELINE) == []


def test_a_larger_drop_in_kind_accuracy_is_a_regression():
    failures = check_gate(run(kind_accuracy=0.9), BASELINE)
    assert failures == ["regression: kind_accuracy 0.90, baseline 1.00"]


def test_tokens_over_the_baseline_margin_fail_but_not_under_it():
    assert check_gate(run(mean_tokens_per_turn=2400.0), BASELINE) == []
    assert any("tokens" in f for f in check_gate(run(mean_tokens_per_turn=2600.0), BASELINE))


def test_tokens_over_the_slow_path_budget_fail_even_if_the_baseline_was_high():
    baseline = {**BASELINE, "mean_tokens_per_turn": 9000.0}
    failures = check_gate(run(mean_tokens_per_turn=8500.0), baseline)
    assert any("limit 8000" in f for f in failures)


def test_latency_is_ignored_unless_timing_is_asked_for():
    slow = run(p50_seconds=9.0)
    assert check_gate(slow, BASELINE) == []
    assert len(check_gate(slow, BASELINE, timing=True)) == 2  # over the budget and the baseline
