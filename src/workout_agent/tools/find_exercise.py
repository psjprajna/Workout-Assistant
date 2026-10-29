"""
Find the exercise the user means, as one of: ok, ambiguous, no_rows.

The search runs in stages, cheapest and surest first:
  1. tidy the phrase: lowercase, hyphens to spaces, "db" -> "dumbbell"…
  2. phrases about a day or a workout ("push day") are not exercise names: no_rows
  3. whole-word match on exercise names and their other names (exercise_aliases);
     a phrase that only names muscles ("lats") also gets exercises tagged with them
  4. when the words match nothing: fix misspelt words ("dumbell" -> "dumbbell", pg_trgm)
     and match again, and search by meaning (Qdrant); both are offered, never assumed
Stages 1 to 4 are plain code and SQL, except search by meaning, which calls
the embedding model. gather() collects the candidates and decide() picks the answer.
decide() is a pure function, so scripts/sweep_search.py can replay it offline.
"""
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import asyncpg

from workout_agent.embeddings.search import MIN_SCORE, ScoredExercise

# Takes the user's phrase, returns exercises by meaning, best first. The app passes
# search_exercises bound to Qdrant and the embedder; tests pass a fake.
Searcher = Callable[[str], Awaitable[list[ScoredExercise]]]

# At most this many "did you mean…?" options from the typo and meaning stages.
MAX_SUGGESTIONS = 3

# At most this many options for a muscle phrase ("something for my back").
MAX_MUSCLE_OPTIONS = 5

# Trigram similarity a misspelt word needs to its correction. Measured on the real names:
# typos score 0.45 to 0.73 ("dumbel"->dumbbell 0.45, "lungs"->lunge 0.50, "dumbell" 0.70);
# chat words stay at 0.29 or below ("tomorrow" 0.23, "thanks" 0.17, "asdfgh" 0.14).
SPELL_MIN = 0.45

# Gym shorthand, expanded before matching. Whole words only: "bb" never touches "dumbbell".
ABBREVIATIONS = {
    "db": "dumbbell",
    "dbs": "dumbbell",
    "bb": "barbell",
    "kb": "kettlebell",
    "bw": "bodyweight",
    "alt": "alternating",
}

# A phrase with one of these words is about a day or a plan, not one exercise.
NOT_AN_EXERCISE = frozenset({"day", "days", "workout", "workouts", "routine", "session", "program"})

# Filler dropped before the whole-word match ("something for my back" -> "back").
STOP_WORDS = frozenset(
    {"a", "an", "the", "for", "my", "me", "i", "to", "of", "on", "with", "and", "some",
     "something", "exercise", "exercises", "do", "what", "good"}
)

# Body parts. A phrase made only of these names a muscle, not an exercise, so a single
# partial name match ("forearms" -> "Modified Push-Up to Forearms") is never assumed.
MUSCLE_WORDS = frozenset(
    {"chest", "pec", "pecs", "ab", "abs", "core", "oblique", "obliques", "lat", "lats",
     "back", "trap", "traps", "shoulder", "shoulders", "delt", "delts", "bicep", "biceps",
     "tricep", "triceps", "forearm", "forearms", "arm", "arms", "glute", "glutes", "quad",
     "quads", "hamstring", "hamstrings", "calf", "calves", "leg", "legs", "thigh", "thighs",
     "upper", "lower", "rear", "front", "side", "inner", "outer", "mid", "middle",
     "adductor", "adductors", "abductor", "abductors", "hip", "hips", "flexor", "flexors"}
)

# Words people use for a muscle group -> the catalog muscle names they cover
# (config/exercise_catalog.toml). Any other muscle word must appear in the muscle name.
MUSCLE_GROUPS = {
    "back": ("back", "lats"),
    "pec": ("chest",),
    "pecs": ("chest",),
    "core": ("abs", "obliques"),
    "arm": ("biceps", "triceps", "forearms"),
    "arms": ("biceps", "triceps", "forearms"),
    "leg": ("quads", "hamstrings", "glutes", "calves"),
    "legs": ("quads", "hamstrings", "glutes", "calves"),
    "delt": ("delts", "shoulders"),
    "delts": ("delts", "shoulders"),
}

# Irregular plurals the s/es rule in _word_pattern does not cover.
SINGULAR = {"calves": "calf", "flyes": "fly", "flies": "fly"}

# The same tidy-up as normalise(), in SQL, so names compare like the query.
NORMALISED_TEXT = (
    "regexp_replace(regexp_replace(lower(t.text), '[-''’]', ' ', 'g'), '\\s+', ' ', 'g')"
)

# Every searchable text: the name, then each other name.
SEARCH_TEXTS = f"""
    texts as (
        select exercise_id, name as text, true as is_name from exercises
        union all
        select exercise_id, alias as text, false as is_name from exercise_aliases
    ),
    normalised as (
        select t.exercise_id, {NORMALISED_TEXT} as text, t.is_name from texts t
    )
"""


@dataclass
class ExerciseMatch:
    exercise_id: int
    name: str


@dataclass
class FindResult:
    status: Literal["ok", "ambiguous", "no_rows"]
    matches: list[ExerciseMatch]


Step = Literal["guard", "exact", "muscle", "text", "fuzzy", "meaning", "none"]


@dataclass
class TextHit:
    exercise_id: int
    name: str
    exact: bool  # the whole phrase equals the name
    usage_count: int
    exact_alias: bool = False  # the whole phrase equals one of its other names


@dataclass
class Query:
    phrase: str  # normalised, abbreviations expanded
    words: list[str]  # phrase without filler words
    muscle_only: bool
    not_an_exercise: bool


@dataclass
class Candidates:
    """Everything decide() needs. Saved in eval results so cutoffs can be replayed."""

    query: Query
    text: list[TextHit]
    muscle: list[TextHit] = field(default_factory=list)  # tagged with the muscles named
    corrected: list[str] = field(default_factory=list)  # words after spelling fixes
    fuzzy: list[TextHit] = field(default_factory=list)  # whole-word match on `corrected`
    meaning: list[ScoredExercise] = field(default_factory=list)


@dataclass(frozen=True)
class Thresholds:
    min_score: float = MIN_SCORE


@dataclass
class Decision:
    result: FindResult
    step: Step  # which stage produced the answer, for evals and logs


def normalise(query: str) -> str:
    """Lowercase, hyphens, apostrophes and control characters to spaces, expand gym shorthand."""
    # A NUL byte in the model's query made Postgres raise instead of the tool returning a value.
    text = re.sub(r"[-'’\x00-\x1f\x7f]", " ", query.lower())
    return " ".join(ABBREVIATIONS.get(word, word) for word in text.split())


def parse_query(query: str) -> Query:
    phrase = normalise(query)
    words = [w for w in phrase.split() if w not in STOP_WORDS] or phrase.split()
    return Query(
        phrase=phrase,
        words=words,
        muscle_only=bool(words) and all(w in MUSCLE_WORDS for w in words),
        not_an_exercise=any(w in NOT_AN_EXERCISE for w in phrase.split()),
    )


def _pg_regex_escape(text: str) -> str:
    """Escape PostgreSQL regex metacharacters in user text."""
    return re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", text)


def _forms(word: str) -> list[str]:
    """The word and its likely singulars: "raises" -> raises, raise, rais."""
    forms = {word, SINGULAR.get(word, word)}
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        forms.add(word[:-1])
        if word.endswith("es"):
            forms.add(word[:-2])
    return sorted(forms)


def _phrase_forms(phrase: str) -> list[str]:
    """The phrase with its last word in each likely singular: "calf raises" -> calf raise."""
    *head, last = phrase.split()
    return [" ".join([*head, form]) for form in _forms(last)]


def _word_pattern(word: str, *more: str) -> str:
    r"""A whole-word regex that also accepts the singular or plural: "raises" -> Raise.

    `more` are other words that count as a match. \m and \M are PostgreSQL's start-
    and end-of-word markers.
    """
    forms = sorted({f for w in (word, *more) for f in _forms(w)})
    alternatives = "|".join(_pg_regex_escape(f) for f in forms)
    return rf"\m(?:{alternatives})(?:e?s)?\M"


def _is_known(word: str, vocabulary: set[str]) -> bool:
    """The word, or its singular or plural, is a word of some name."""
    return any(f in vocabulary or f + "s" in vocabulary or f + "es" in vocabulary
               for f in _forms(word))


async def _corrected_words(conn: asyncpg.Connection, words: list[str]) -> list[str]:
    """Replace each word no name uses with the closest word that one does, if close enough."""
    vocabulary = {
        r["word"]
        for r in await conn.fetch(
            f"""
            with {SEARCH_TEXTS}
            select distinct w as word
            from normalised, regexp_split_to_table(text, '[^a-z0-9]+') w
            where w <> ''
            """
        )
    }
    unknown = [w for w in words if not _is_known(w, vocabulary)]
    if not unknown:
        return words
    rows = await conn.fetch(
        """
        select q.word, best.word as fix, best.score
        from unnest($1::text[]) q(word)
        cross join lateral (
            select v.word, similarity(q.word, v.word) as score
            from unnest($2::text[]) v(word)
            order by score desc, v.word
            limit 1
        ) best
        """,
        unknown,
        sorted(vocabulary),
    )
    fixes = {r["word"]: r["fix"] for r in rows if r["score"] >= SPELL_MIN}
    return [fixes.get(w, w) for w in words]


async def _text_hits(
    conn: asyncpg.Connection, query: Query, words: list[str] | None = None
) -> list[TextHit]:
    """Exercises whose name, or one other name, contains every word as a whole word."""
    rows = await conn.fetch(
        f"""
        with {SEARCH_TEXTS},
        hits as (
            select exercise_id,
                   bool_or(is_name and text = any($2)) as exact,
                   bool_or(not is_name and text = any($2)) as exact_alias
            from normalised
            where text ~ all($1)
            group by exercise_id
        )
        select e.exercise_id, e.name, h.exact, h.exact_alias,
               (select count(*) from exercise_logs l
                where l.exercise_id = e.exercise_id) as usage_count
        from hits h join exercises e using (exercise_id)
        order by h.exact desc, h.exact_alias desc, usage_count desc, length(e.name), e.name
        """,
        [_word_pattern(w) for w in (words or query.words)],
        _phrase_forms(query.phrase),
    )
    return [
        TextHit(r["exercise_id"], r["name"], r["exact"], r["usage_count"], r["exact_alias"])
        for r in rows
    ]


async def _muscle_hits(conn: asyncpg.Connection, query: Query) -> list[TextHit]:
    """Exercises with a muscle naming every word, main muscle first, then most logged."""
    rows = await conn.fetch(
        """
        select e.exercise_id, e.name, min(m.position) as position,
               (select count(*) from exercise_logs l
                where l.exercise_id = e.exercise_id) as usage_count
        from exercise_muscles m join exercises e using (exercise_id)
        where m.muscle ~ all($1)
        group by e.exercise_id, e.name
        order by position, usage_count desc, length(e.name), e.name
        """,
        [_word_pattern(w, *MUSCLE_GROUPS.get(w, ())) for w in query.words],
    )
    return [TextHit(r["exercise_id"], r["name"], False, r["usage_count"]) for r in rows]


async def gather(
    conn: asyncpg.Connection,
    query: str,
    search: Searcher | None = None,
    everything: bool = False,
) -> Candidates:
    """Collect candidates from each stage. Later stages run only if decide() may need them;
    `everything=True` runs them all, so an eval can record every score."""
    parsed = parse_query(query)
    if not parsed.words or (parsed.not_an_exercise and not everything):
        return Candidates(parsed, text=[])

    text = await _text_hits(conn, parsed)
    muscle = await _muscle_hits(conn, parsed) if parsed.muscle_only else []
    corrected: list[str] = []
    fuzzy: list[TextHit] = []
    if not text or everything:
        corrected = await _corrected_words(conn, parsed.words)
        if corrected != parsed.words:
            fuzzy = await _text_hits(conn, parsed, corrected)
    needs_meaning = not text or parsed.muscle_only or everything
    meaning = await search(parsed.phrase) if search and needs_meaning else []
    return Candidates(parsed, text, muscle, corrected, fuzzy, meaning)


def _merge(*groups: list[ExerciseMatch]) -> list[ExerciseMatch]:
    seen: set[int] = set()
    merged = []
    for group in groups:
        for match in group:
            if match.exercise_id not in seen:
                seen.add(match.exercise_id)
                merged.append(match)
    return merged


def _matches(hits: list[ScoredExercise] | list[TextHit]) -> list[ExerciseMatch]:
    return [ExerciseMatch(h.exercise_id, h.name) for h in hits]


DEFAULT_THRESHOLDS = Thresholds()


def decide(c: Candidates, t: Thresholds = DEFAULT_THRESHOLDS) -> Decision:
    """Turn the candidates into ok / ambiguous / no_rows. Pure: no database, no network."""
    if not c.query.words:
        return Decision(FindResult("no_rows", []), "none")
    if c.query.not_an_exercise:
        return Decision(FindResult("no_rows", []), "guard")

    meaning = [h for h in c.meaning if h.score >= t.min_score][:MAX_SUGGESTIONS]

    if c.text:
        exact = [h for h in c.text if h.exact]
        # Exactly one exercise has this name: no doubt.
        if len(exact) == 1:
            return Decision(FindResult("ok", _matches(exact)), "exact")
        # Several share it (two Ab Rollouts): ask about just those.
        if exact:
            return Decision(FindResult("ambiguous", _matches(exact)), "exact")
        # One of its other names ("rdl", "pec deck"). Ours, not the user's words, so it is
        # only assumed when nothing else matches; otherwise it is asked, listed first.
        if any(h.exact_alias for h in c.text):
            status = "ok" if len(c.text) == 1 else "ambiguous"
            return Decision(FindResult(status, _matches(c.text)), "exact")

    # A phrase naming only muscles ("forearms") is never assumed to be one exercise, even
    # if a name contains the word: offer the exercises tagged with it, then name matches.
    if c.query.muscle_only:
        options = _merge(_matches(c.muscle), _matches(c.text), _matches(meaning))
        if options:
            step: Step = "muscle" if c.muscle else "text" if c.text else "meaning"
            return Decision(FindResult("ambiguous", options[:MAX_MUSCLE_OPTIONS]), step)
        return Decision(FindResult("no_rows", []), "none")

    if c.text:
        status = "ok" if len(c.text) == 1 else "ambiguous"
        return Decision(FindResult(status, _matches(c.text)), "text")

    # No name contains the words. Spelling fixes first, then close meanings; always asked
    # ("did you mean…?"), never assumed, even when only one is left.
    fuzzy = c.fuzzy[:MAX_SUGGESTIONS]
    options = _merge(_matches(fuzzy), _matches(meaning))[:MAX_SUGGESTIONS]
    if not options:
        return Decision(FindResult("no_rows", []), "none")
    return Decision(FindResult("ambiguous", options), "fuzzy" if fuzzy else "meaning")


async def find_exercise(
    conn: asyncpg.Connection, query: str, search: Searcher | None = None
) -> FindResult:
    """Turn what the user typed into one of: ok, ambiguous, no_rows."""
    return decide(await gather(conn, query, search)).result
