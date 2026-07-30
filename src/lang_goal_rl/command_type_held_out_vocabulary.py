"""Stage 11's held-out command-type vocabulary: never trained on, eval-only.

`command_type_vocabulary.build_command_type_training_set` is what
`command_type_classifier.train_command_type_classifier` regresses against.
This module is the disjoint counterpart: a second labeled set, same 5
`CommandType` classes, that the experiment-runner uses to measure whether
the trained classifier actually generalizes to unseen phrasing rather than
memorizing the training set. Every phrasing here is checked (see
`tests/lang_goal_rl/test_command_type_held_out_vocabulary.py`) to have zero
exact-text overlap and zero trivial single-word-edit overlap with the
training vocabulary -- the same disjointness guard
`held_out_paraphrases.py` and `augmented_training_vocabulary.py` already use
for stage 3/4's region vocabulary.

`GOTO_NAMED_REGION` no longer reuses `held_out_paraphrases.HELD_OUT_PARAPHRASES`
(2026-07-24 fix): those 14 phrases are written in Phase 1's directional-verb
convention ("push your arm out in front of you", "swing your arm over to the
left") -- the same convention that collided with MOVE and caused MOVE's 0%
held-out accuracy (see `command_type_vocabulary.py`'s docstring for the full
diagnosis). This module now writes its own fresh, disjoint
absolute-destination-framed held-out phrasings for `GOTO_NAMED_REGION`,
matching `command_type_vocabulary.goto_named_region_examples`'s "name a
place" convention instead.
"""

from __future__ import annotations

from lang_goal_rl.command_type_vocabulary import CommandType, LabeledCommandExample

_MOVE_HELD_OUT: dict[str, tuple[str, ...]] = {
    "reach forward": ("scoot forward by a small margin", "advance just a fraction from where you currently are"),
    "reach back": ("scoot backward by a small margin", "retreat just a fraction from where you currently are"),
    "reach left": ("scoot leftward by a small margin", "veer left just a fraction from where you currently are"),
    "reach right": ("scoot rightward by a small margin", "veer right just a fraction from where you currently are"),
    "reach up high": ("scoot upward by a small margin", "climb just a fraction from where you currently are"),
    "reach down low": ("scoot downward by a small margin", "descend just a fraction from where you currently are"),
}
"""2 held-out phrasings per `command_grammar.KNOWN_DIRECTIONS` entry. Same
"incremental motion from wherever the robot currently is" convention as
`command_type_vocabulary._MOVE_TEMPLATES` (magnitude/degree cue and/or
"from here" framing), but with verbs (scoot, advance, retreat, veer, climb,
descend) and templates distinct from the training set -- disjointness is
checked in `tests/lang_goal_rl/test_command_type_held_out_vocabulary.py`.
Deliberately avoids "angle"/"swing" -- the exact words that, in the original
version of this dict, collided with Phase 1's directional region-naming
convention and caused a 0% held-out MOVE accuracy (see
`command_type_vocabulary.py`'s docstring)."""

_GOTO_HELD_OUT_PLACE_PHRASES: dict[str, str] = {
    "center": "the middle of the workspace",
    "reach forward": "the front region of the workspace",
    "reach back": "the back region of the workspace",
    "reach left": "the left region of the workspace",
    "reach right": "the right region of the workspace",
    "reach up high": "the upper region of the workspace",
    "reach down low": "the lower region of the workspace",
}
"""One fresh place-noun-phrase per region name, worded distinctly from
`command_type_vocabulary._GOTO_PLACE_PHRASES`'s training variants -- names a
place, never a direction of travel, same as the training convention."""

_GOTO_NAMED_REGION_HELD_OUT: tuple[str, ...] = tuple(
    text.format(place=place)
    for place in _GOTO_HELD_OUT_PLACE_PHRASES.values()
    for text in ("position yourself near {place}", "work your way over to {place}")
)
"""14 held-out phrasings (2 per region x 7 regions), built from
`_GOTO_HELD_OUT_PLACE_PHRASES` with verb frames ("position yourself near",
"work your way over to") that appear nowhere in
`command_type_vocabulary._GOTO_TEMPLATES` -- fresh wording, not a
recombination of the training set's own templates. Replaces the old
`held_out_paraphrases.HELD_OUT_PARAPHRASES` reuse; see this module's
docstring for why that was unsafe."""

_STOP_HELD_OUT: tuple[str, ...] = (
    "cut it out immediately",
    "quit right where you are",
    "no more movement please",
    "settle down and remain still",
    "halt any further motion",
    "just stay exactly put",
    "cease all movement now",
    "power down and stay still",
)

_RESET_HELD_OUT: tuple[str, ...] = (
    "wipe the slate clean",
    "revert to the very beginning",
    "kick things off again",
    "begin the run over",
    "restart everything from scratch",
    "go back to time zero",
    "erase progress and begin anew",
    "power everything back to zero",
)

_UNSUPPORTED_HELD_OUT: tuple[str, ...] = (
    "assemble the toy blocks",
    "fetch me a coffee please",
    "what's on the news today",
    "recite a poem for me",
    "purple triangles whisper loudly",
    "check the humidity level outside",
    "schedule a meeting for tomorrow",
    "fold the laundry neatly please",
    "what movie should i watch tonight",
    "calculate the square root of nine",
)


def goto_named_region_held_out_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the held-out GOTO_NAMED_REGION phrasings, each labeled `CommandType.GOTO_NAMED_REGION`."""
    return tuple(
        LabeledCommandExample(text, CommandType.GOTO_NAMED_REGION) for text in _GOTO_NAMED_REGION_HELD_OUT
    )


def move_held_out_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the held-out MOVE phrasings (2 per direction), each labeled `CommandType.MOVE`."""
    return tuple(
        LabeledCommandExample(text, CommandType.MOVE)
        for phrasings in _MOVE_HELD_OUT.values()
        for text in phrasings
    )


def stop_held_out_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the held-out STOP phrasings, each labeled `CommandType.STOP`."""
    return tuple(LabeledCommandExample(text, CommandType.STOP) for text in _STOP_HELD_OUT)


def reset_held_out_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the held-out RESET phrasings, each labeled `CommandType.RESET`."""
    return tuple(LabeledCommandExample(text, CommandType.RESET) for text in _RESET_HELD_OUT)


def unsupported_held_out_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the held-out UNSUPPORTED phrasings, each labeled `CommandType.UNSUPPORTED`."""
    return tuple(LabeledCommandExample(text, CommandType.UNSUPPORTED) for text in _UNSUPPORTED_HELD_OUT)


def build_command_type_held_out_set() -> tuple[LabeledCommandExample, ...]:
    """Assemble all 5 classes' held-out examples into one evaluation set.

    Returns:
        Every example from `goto_named_region_held_out_examples`,
        `move_held_out_examples`, `stop_held_out_examples`,
        `reset_held_out_examples`, and `unsupported_held_out_examples`,
        combined. Unshuffled -- unlike the training set, order doesn't
        matter for an accuracy evaluation.

    """
    return (
        *goto_named_region_held_out_examples(),
        *move_held_out_examples(),
        *stop_held_out_examples(),
        *reset_held_out_examples(),
        *unsupported_held_out_examples(),
    )
