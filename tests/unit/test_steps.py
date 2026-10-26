from workout_agent.agents.guardrails import page_result
from workout_agent.ui.steps import MAX_SHOWN_CHARS, MAX_TABLE_ROWS, format_args, format_result


def test_args_are_plain_json():
    assert format_args({}) == "{}"
    assert format_args({"query": "bench press"}) == '{\n  "query": "bench press"\n}'


def test_the_status_line_comes_first_then_the_result():
    out = format_result({"status": "ok", "latest_on": "2026-02-09"})

    assert out.startswith("✓ ok\n\n")
    assert '"latest_on": "2026-02-09"' in out


def test_a_list_of_flat_rows_is_a_table():
    out = format_result(
        {
            "status": "ambiguous",
            "matches": [{"exercise_id": 2, "name": "A|B"}, {"exercise_id": 3, "name": "C"}],
        }
    )

    assert out.startswith("ambiguous\n\n| exercise_id | name |")
    assert "| 2 | A\\|B |" in out


def test_a_long_list_shows_a_row_limit_and_a_count_of_the_rest():
    days = [{"day_id": i, "name": f"Day {i}"} for i in range(MAX_TABLE_ROWS + 3)]

    out = format_result({"status": "ok", "days": days})

    assert "Day 11" in out and "Day 12" not in out
    assert "… 3 more rows" in out


def test_planned_day_is_a_summary_without_the_nested_sets():
    out = format_result(
        {
            "status": "ok",
            "day_id": 1,
            "day_name": "Push",
            "exercises": [
                {
                    "exercise_name": "Bench",
                    "target_reps": [8, 8],
                    "last_on": "2026-02-09",
                    "last_sets": [{"set_number": 1, "weight_kg": "80.00", "reps": 8}],
                }
            ],
        }
    )

    assert "| Bench | 2 | 8, 8 | 2026-02-09 |" in out
    assert "weight_kg" not in out


def test_a_very_long_result_is_cut_for_the_page():
    out = format_result({"status": "ok", "note": "x" * 5000})

    assert len(out) < MAX_SHOWN_CHARS + 50
    assert out.endswith("(cut for the page)")


def test_a_result_with_only_a_status_is_just_the_status_line():
    assert format_result({"status": "retry"}) == "retry"


def test_page_result_drops_keys_outside_the_contract():
    shown = page_result(
        "planned_day",
        {
            "status": "ok",
            "day_id": 1,
            "day_name": "Push",
            "date_of_birth": "1990-01-01",
            "exercises": [
                {
                    "exercise_id": 2,
                    "exercise_name": "Bench",
                    "timed": False,
                    "target_reps": [8],
                    "last_on": None,
                    "body_weight": 60,
                    "last_sets": [{"set_number": 1, "weight_kg": "80.00", "reps": 8, "bmi": 22}],
                }
            ],
        },
    )

    assert "date_of_birth" not in shown
    [exercise] = shown["exercises"]
    assert "body_weight" not in exercise
    assert exercise["last_sets"] == [{"set_number": 1, "weight_kg": "80.00", "reps": 8}]


def test_page_result_for_an_unknown_tool_shows_only_the_status():
    assert page_result("mystery", {"status": "ok", "secret": 1}) == {"status": "ok"}
