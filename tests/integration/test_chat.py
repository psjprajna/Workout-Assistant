"""Integration tests: handle_message end to end (ask, pick an option, verify) with a fake router."""

from datetime import date
from decimal import Decimal

from workout_agent.workflow.chat import RouteDecision, Session, handle_message


def fake_route(decision: RouteDecision):
    """A stand-in for the model: always returns the same decision."""

    async def route(text: str, has_pending: bool) -> RouteDecision:
        return decision

    return route


async def _seed(conn):
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [(2, "Barbell Bench Press"), (8, "Barbell Incline Bench Press"), (112, "Barbell Curl")],
    )
    await conn.executemany(
        "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
        "values ($1, $2, $3)",
        [(10, 2, date(2026, 2, 8)), (11, 8, date(2026, 2, 9)), (12, 112, date(2026, 2, 10))],
    )
    await conn.executemany(
        "insert into sets (exercise_log_id, set_number, weight_kg, weight_raw, reps) "
        "values ($1, $2, $3, $4, $5)",
        [
            (10, 1, Decimal("80.00"), Decimal("80.00"), 8),
            (11, 1, Decimal("60.00"), Decimal("60.00"), 10),
            (12, 1, Decimal("30.00"), Decimal("30.00"), 12),
        ],
    )


async def test_ambiguous_exercise_asks_and_remembers_options(conn):
    await _seed(conn)
    session = Session()
    route = fake_route(RouteDecision(intent="last_session", exercise_text="bench press"))
    reply = await handle_message(conn, session, "last time I did bench press?", route)
    assert "Which one" in reply
    assert len(session.pending) == 2


async def test_picking_an_option_answers_and_clears_pending(conn):
    await _seed(conn)
    session = Session()
    ask = fake_route(RouteDecision(intent="last_session", exercise_text="bench press"))
    await handle_message(conn, session, "last time I did bench press?", ask)

    pick = fake_route(RouteDecision(intent="pick_option", option_number=1))
    reply = await handle_message(conn, session, "1", pick)
    assert "kg" in reply
    assert session.pending is None


async def test_option_number_out_of_range_asks_again_and_keeps_pending(conn):
    await _seed(conn)
    session = Session()
    ask = fake_route(RouteDecision(intent="last_session", exercise_text="bench press"))
    await handle_message(conn, session, "last time I did bench press?", ask)

    pick = fake_route(RouteDecision(intent="pick_option", option_number=9))
    reply = await handle_message(conn, session, "9", pick)
    assert "Which one" in reply
    assert len(session.pending) == 2


async def test_pick_option_with_nothing_pending_does_not_crash(conn):
    await _seed(conn)
    session = Session()
    pick = fake_route(RouteDecision(intent="pick_option", option_number=1))
    reply = await handle_message(conn, session, "1", pick)
    assert "kg" not in reply


async def test_unique_exercise_answers_directly(conn):
    await _seed(conn)
    session = Session()
    route = fake_route(RouteDecision(intent="last_session", exercise_text="curl"))
    reply = await handle_message(conn, session, "last curl?", route)
    assert "Barbell Curl" in reply
    assert "30.00 kg x 12" in reply
    assert session.pending is None


async def test_unknown_exercise_says_not_in_your_data(conn):
    await _seed(conn)
    session = Session()
    route = fake_route(RouteDecision(intent="last_session", exercise_text="zzz"))
    reply = await handle_message(conn, session, "last time I did zzz?", route)
    assert "not in your data" in reply


async def test_unknown_intent_states_what_it_can_do(conn):
    await _seed(conn)
    session = Session()
    route = fake_route(RouteDecision(intent="unknown"))
    reply = await handle_message(conn, session, "what's the weather?", route)
    assert "last session" in reply