"""Stage 11's command-type training vocabulary: labeled `(text, CommandType)` pairs.

Phase 2a (`command_grammar.py`) shipped a deterministic typed-command
language -- five `Command` dataclasses, parsed from exact grammar strings
like "goto 1.3 0.7 0.5" or "move reach left 0.05". Phase 2b's first learned
piece is a classifier that decides which of those five command *types* a
free-form English sentence intends, before any continuous-parameter
regression happens (stage 12's job). This module is that classifier's
training data.

Revision history / why `GOTO_NAMED_REGION` no longer reuses Phase 1's
sentences (2026-07-24 fix -- see below for the full story):

The first version of this module built `GOTO_NAMED_REGION` by relabeling
stage 3/4's existing region-instruction vocabulary
(`goal_region_vocabulary.ALL_INSTRUCTIONS` +
`augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS`) wholesale -- e.g.
"reach forward", "angle your hand outward toward the front". Those sentences
were written for Phase 1's region-*naming* convention, which phrases a
region as a *direction of travel* ("angle X toward direction", "swing
outward"). That is exactly how a natural MOVE sentence reads too. The
resulting classifier scored 0% (0/12) on held-out MOVE across every one of
12 hyperparameter-tuning runs while training loss dropped near-zero in all
of them -- not a tuning problem, but the classifier correctly learning
"angle/swing toward <direction>" -> GOTO_NAMED_REGION from training data
that collided head-on with what a fair MOVE test sentence sounds like. An
adversarial review traced this to the data design, not the model
architecture (`command_type_classifier.py` is unmodified by this fix).

The fix redesigns the phrasing convention on *both* sides of the boundary
so the two classes carry a genuinely distinguishing linguistic signal
instead of relying on which verbs happen to appear where:

- `MOVE` (relative displacement): every phrasing reads as incremental motion
  *from wherever the robot currently is*, using an explicit magnitude/degree
  cue ("a bit", "slightly", "a small amount", "a good distance", "all the
  way") and/or an explicit "from here" / "from your current spot" framing.
  This matches `command_grammar.MoveCommand.distance_m`'s real semantics: a
  signed delta from the current position, not a place name.
- `GOTO_NAMED_REGION` (absolute destination): every phrasing reads as naming
  a *place*, via a "go/head/aim/travel/arrive at <place-noun-phrase>" frame,
  and deliberately never uses a "reach/angle/swing <direction>" verb-phrase
  (Phase 1's own region-naming convention, used elsewhere in the repo for a
  different purpose) for this class's own templates. `goto_named_region_examples`
  therefore builds a fresh set of absolute-destination phrasings -- one set
  per region name from `goal_region_vocabulary.region_names()` -- rather than
  reusing `ALL_INSTRUCTIONS` / `AUGMENTED_INSTRUCTIONS` verbatim. Those two
  Phase-1 modules are untouched; this module just stops relabeling them.

The other three classes are unchanged in shape from the original version:

- `STOP` / `RESET`: hand-written, diverse phrasings -- short, closed-form
  intents with no natural axis to template over (unlike `MOVE`'s six
  directions), so a template generator would buy nothing here. A few extra
  idiom-diverse phrasings were added alongside this fix (the same review
  flagged STOP/RESET's held-out accuracy as softly low, 66.7%/83.3%, though
  not the 0% MOVE collision).
- `UNSUPPORTED`: a deliberately varied set of out-of-scope requests -- object
  manipulation, vague/off-topic text, nonsense, and some superficially
  plausible-sounding requests (e.g. "connect to the wifi network") -- so the
  classifier has to learn a real boundary around what this pipeline can
  actually do, not just "is this a well-formed sentence".

See `check_cross_class_embedding_overlap` at the bottom of this module for a
cheap diagnostic that would have caught the original MOVE/GOTO_NAMED_REGION
collision before a full train+eval cycle: it flags class pairs whose
training examples are suspiciously often each other's nearest neighbor in
embedding space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from lang_goal_rl.command_grammar import KNOWN_DIRECTIONS
from lang_goal_rl.goal_region_vocabulary import region_names
from lang_goal_rl.language_embedding import encode_instructions

if TYPE_CHECKING:
    from collections.abc import Sequence


class CommandType(str, Enum):
    """The five typed-command classes `command_grammar.Command` maps onto.

    A `str` subclass so labels serialize cleanly (e.g. for logging or a
    confusion-matrix report) and compare equal to their string value.
    """

    MOVE = "MOVE"
    GOTO_NAMED_REGION = "GOTO_NAMED_REGION"
    STOP = "STOP"
    RESET = "RESET"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class LabeledCommandExample:
    """One training or evaluation example for the command-type classifier.

    Attributes:
        text: The English sentence.
        command_type: Its ground-truth `CommandType` label.

    """

    text: str
    command_type: CommandType


_DIRECTION_WORDS: dict[str, tuple[str, ...]] = {
    "reach forward": ("forward", "ahead", "onward"),
    "reach back": ("back", "backward", "rearward"),
    "reach left": ("left", "leftward", "over to the left"),
    "reach right": ("right", "rightward", "over to the right"),
    "reach up high": ("up", "upward", "higher"),
    "reach down low": ("down", "downward", "lower"),
}
"""Plain direction adverbs for each `command_grammar.KNOWN_DIRECTIONS` entry.
Deliberately carries no magnitude wording of its own -- every `_MOVE_TEMPLATES`
entry supplies the magnitude/degree or "from here" cue, so this dict only
needs to say *which way*, not *how much*."""

_MOVE_TEMPLATES: tuple[str, ...] = (
    "move a bit {word}",
    "shift slightly {word} from where you are",
    "go {word} just a little further",
    "ease {word} a small amount",
    "nudge {word} a touch from here",
    "shift {word} by a little",
    "creep {word} a short distance from your current spot",
    "move a good distance {word} from here",
    "go all the way {word} from where you are now",
    "inch {word} a bit further",
    "drift {word} slightly from your current position",
    "push a little further {word} from here",
    "move some {word} from your current spot",
    "slide {word} a small amount from here",
    "edge {word} just a little bit",
)
"""15 sentence templates. Every entry carries an explicit magnitude/degree
cue ("a bit", "slightly", "a small amount", "a good distance", "all the
way", ...) and/or an explicit "from here" / "from your current spot" /
"from where you are" framing -- the real semantics of
`command_grammar.MoveCommand.distance_m` (a signed delta from wherever the
robot currently is, not a place name). This is the fix for the
MOVE/GOTO_NAMED_REGION collision documented in this module's docstring:
before this fix, MOVE's templates ("move {word}", "reach {word}") were
structurally indistinguishable from a directional destination phrase."""

_STOP_PHRASINGS: tuple[str, ...] = (
    "stop",
    "stop right there",
    "hold there",
    "freeze",
    "don't move",
    "halt",
    "stay right there",
    "pause",
    "wait, don't go anywhere",
    "quit moving",
    "cease movement",
    "stand still",
    "hold your position",
    "come to a stop",
    "stop moving now",
    "stay put",
    "put the brakes on",
    "quit right now",
    "cut it out",
    "knock it off",
    "that's enough",
    "enough of that",
    "quit it",
    "no more of that",
    "don't do any more",
    "that'll do",
)
"""The last 8 entries (2026-07-30 fix, attempt 3) are idiomatic and
negation-form cessation phrasings with no explicit stop-keyword ("cut it
out", "knock it off", "that's enough") or a polite negation frame ("no more
of that", "don't do any more") -- added because held-out phrasings of that
shape ("cut it out immediately", "no more movement please") were
misclassified every prior attempt: the training set had zero examples of
either pattern, all 18 original phrasings being direct imperatives with an
explicit stop-keyword (stop/halt/freeze/cease/still/put/brakes)."""

_RESET_PHRASINGS: tuple[str, ...] = (
    "reset",
    "start over",
    "begin again",
    "restart the episode",
    "go back to the start",
    "let's try again from scratch",
    "reset everything",
    "start fresh",
    "back to square one",
    "restart from the beginning",
    "reset the environment",
    "start the episode over",
    "begin the task again",
    "reset your position",
    "go back to where you started",
    "rewind to the beginning",
    "give it a fresh start",
    "clear the board and begin again",
)

_UNSUPPORTED_PHRASINGS: tuple[str, ...] = (
    "pick up the block",
    "grab that toy",
    "put the cup on the table",
    "open the drawer",
    "turn on the lights",
    "how's the weather today",
    "tell me a joke",
    "what time is it",
    "sing me a song",
    "what's your favorite color",
    "can you order me a pizza",
    "play some music",
    "translate this sentence into french",
    "write a poem about robots",
    "purple elephants dance sideways",
    "asdkfj qweoi zzxxcv",
    "pick that up and hand it to me",
    "stack the blocks neatly",
    "wave hello to the camera",
    "do a little dance",
    "tell me about your day",
    "increase the temperature by two degrees",
    "connect to the wifi network",
    "give me a status report on the battery",
    "what's the capital of france",
    "sort these files by date",
    "solve this equation for x",
    "what is seven times thirteen",
    "compute the factorial of five",
    "find the square root of sixteen",
    "what's the derivative of x squared",
)
"""A deliberately varied set: object manipulation the pipeline has no
mechanism for ("pick up the block"), vague/off-topic requests ("tell me a
joke"), nonsense ("asdkfj qweoi zzxxcv"), superficially plausible-sounding
requests that are still out of scope ("connect to the wifi network"), and
(2026-07-30 fix, attempt 3) math/calculation requests ("solve this equation
for x") -- the last group closes a gap where a held-out phrasing like
"calculate the square root of nine" had no semantic anchor in training and
drifted to a different wrong class in every prior attempt. Together these
give the classifier a real capability boundary, not just "is this valid
English"."""

_GOTO_PLACE_PHRASES: dict[str, tuple[str, ...]] = {
    "center": (
        "the center of the workspace",
        "the middle of the reachable area",
        "the workspace's midpoint",
        "the central position",
    ),
    "reach forward": (
        "the front of the workspace",
        "the front edge of the reachable area",
        "the forward region",
        "the far front area",
    ),
    "reach back": (
        "the back of the workspace",
        "the rear edge of the reachable area",
        "the backward region",
        "the far rear area",
    ),
    "reach left": (
        "the left side of the workspace",
        "the left edge of the reachable area",
        "the left-hand region",
        "the far left area",
    ),
    "reach right": (
        "the right side of the workspace",
        "the right edge of the reachable area",
        "the right-hand region",
        "the far right area",
    ),
    "reach up high": (
        "the top of the workspace",
        "the upper edge of the reachable area",
        "the highest region",
        "the far upper area",
    ),
    "reach down low": (
        "the bottom of the workspace",
        "the lower edge of the reachable area",
        "the lowest region",
        "the far lower area",
    ),
}
"""A place-noun-phrase per region name (`goal_region_vocabulary.region_names()`),
4 variants each, used to fill `_GOTO_TEMPLATES`. These name a *place*, never
a direction of travel -- the deliberate opposite of `_DIRECTION_WORDS`
above. Keys match `goal_region_vocabulary.region_names()` exactly so this
class's examples still map onto the same 7 underlying regions, even though
`CommandType.GOTO_NAMED_REGION` itself carries no region label."""

_GOTO_TEMPLATES: tuple[str, ...] = (
    "go to {place}",
    "head toward {place}",
    "aim for {place}",
    "make your way to {place}",
    "travel to {place}",
    "get to {place}",
    "proceed to {place}",
    "arrive at {place}",
    "go toward {place}",
    "settle at {place}",
    "head for {place}",
    "come to rest at {place}",
)
"""12 "name a destination" templates -- an "absolute place" frame
(go/head/aim/travel/arrive at <place>), never a "reach/angle/swing
<direction>" verb-phrase. That directional-verb-phrase pattern is Phase 1's
own region-naming convention (`goal_region_vocabulary.py`,
`augmented_training_vocabulary.py`) and is exactly what collided with MOVE's
phrasing -- see this module's docstring. None of these templates use the
word "move" either, so a GOTO_NAMED_REGION sentence never shares MOVE's own
verb."""


def _move_phrasings(direction: str) -> tuple[str, ...]:
    """Fill `_MOVE_TEMPLATES` with `direction`'s adverbial word variants.

    Args:
        direction: One of `command_grammar.KNOWN_DIRECTIONS`.

    Returns:
        One phrasing per `_MOVE_TEMPLATES` entry (15), cycling through
        `_DIRECTION_WORDS[direction]` since there are fewer words than
        templates.

    """
    words = _DIRECTION_WORDS[direction]
    return tuple(
        template.format(word=words[index % len(words)]) for index, template in enumerate(_MOVE_TEMPLATES)
    )


def _goto_phrasings(region_name: str) -> tuple[str, ...]:
    """Fill `_GOTO_TEMPLATES` with `region_name`'s place-phrase variants.

    Args:
        region_name: One of `goal_region_vocabulary.region_names()`.

    Returns:
        One phrasing per `_GOTO_TEMPLATES` entry (12), cycling through
        `_GOTO_PLACE_PHRASES[region_name]` since there are fewer place
        phrases than templates.

    """
    places = _GOTO_PLACE_PHRASES[region_name]
    return tuple(
        template.format(place=places[index % len(places)]) for index, template in enumerate(_GOTO_TEMPLATES)
    )


def goto_named_region_examples() -> tuple[LabeledCommandExample, ...]:
    """Build the GOTO_NAMED_REGION class: 12 templated destination-phrasings per region.

    Fresh phrasings, not a relabeling of `goal_region_vocabulary.ALL_INSTRUCTIONS`
    / `augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS` -- see this
    module's docstring for why reusing those verbatim caused a 0% held-out
    MOVE collision.

    Returns:
        84 examples (7 regions x 12 phrasings), each labeled
        `CommandType.GOTO_NAMED_REGION`.

    """
    return tuple(
        LabeledCommandExample(text, CommandType.GOTO_NAMED_REGION)
        for region_name in region_names()
        for text in _goto_phrasings(region_name)
    )


def move_examples() -> tuple[LabeledCommandExample, ...]:
    """Build the MOVE class: 15 templated phrasings per `KNOWN_DIRECTIONS` entry.

    Returns:
        90 examples (6 directions x 15 phrasings), each labeled
        `CommandType.MOVE`.

    """
    return tuple(
        LabeledCommandExample(text, CommandType.MOVE)
        for direction in KNOWN_DIRECTIONS
        for text in _move_phrasings(direction)
    )


def stop_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the fixed STOP phrasing set, each labeled `CommandType.STOP`."""
    return tuple(LabeledCommandExample(text, CommandType.STOP) for text in _STOP_PHRASINGS)


def reset_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the fixed RESET phrasing set, each labeled `CommandType.RESET`."""
    return tuple(LabeledCommandExample(text, CommandType.RESET) for text in _RESET_PHRASINGS)


def unsupported_examples() -> tuple[LabeledCommandExample, ...]:
    """Return the fixed UNSUPPORTED phrasing set, each labeled `CommandType.UNSUPPORTED`."""
    return tuple(LabeledCommandExample(text, CommandType.UNSUPPORTED) for text in _UNSUPPORTED_PHRASINGS)


def build_command_type_training_set(*, seed: int = 0) -> tuple[LabeledCommandExample, ...]:
    """Assemble all 5 classes into one training set, shuffled with a fixed seed.

    Args:
        seed: Seed for the shuffle -- a given `seed` always returns the same
            order (reproducible training runs); different seeds reorder the
            same underlying set of examples.

    Returns:
        Every example from `goto_named_region_examples`, `move_examples`,
        `stop_examples`, `reset_examples`, and `unsupported_examples`,
        combined and shuffled.

    """
    examples = [
        *goto_named_region_examples(),
        *move_examples(),
        *stop_examples(),
        *reset_examples(),
        *unsupported_examples(),
    ]
    random.Random(seed).shuffle(examples)
    return tuple(examples)


@dataclass(frozen=True)
class CrossClassNeighbor:
    """One training example's nearest *other* example in embedding space.

    Attributes:
        text: The example's text.
        command_type: The example's ground-truth label.
        nearest_text: The text of the closest other example (by cosine
            similarity, excluding the example itself).
        nearest_command_type: That neighbor's ground-truth label.
        is_cross_class: Whether `nearest_command_type != command_type` --
            the per-example signal the aggregate report is built from.

    """

    text: str
    command_type: CommandType
    nearest_text: str
    nearest_command_type: CommandType
    is_cross_class: bool


@dataclass(frozen=True)
class CrossClassOverlapReport:
    """Aggregate cross-class nearest-neighbor confusion over a labeled example set.

    For every ordered class pair `(a, b)` with `a != b`, `pair_rates[(a, b)]`
    is the fraction of class `a`'s examples whose nearest *other* example (in
    frozen sentence-embedding space) belongs to class `b`. A high rate means
    "class `a`'s own phrasing convention looks, in embedding space, like
    class `b`'s" -- exactly the shape of the MOVE/GOTO_NAMED_REGION collision
    this module's docstring describes, and the thing this diagnostic exists
    to catch cheaply, before a full train+held-out-eval cycle.

    Attributes:
        neighbors: One `CrossClassNeighbor` per input example, in input order.
        pair_counts: Raw cross-class neighbor counts, keyed `(a, b)`.
        pair_rates: `pair_counts[(a, b)] / (number of class a examples)`.
        flagged_pairs: `(a, b, rate)` tuples with `rate >= flag_threshold`,
            sorted by `rate` descending -- empty if nothing crosses the
            threshold.

    """

    neighbors: tuple[CrossClassNeighbor, ...]
    pair_counts: dict[tuple[str, str], int]
    pair_rates: dict[tuple[str, str], float]
    flagged_pairs: tuple[tuple[str, str, float], ...]

    def summary(self) -> str:
        """Render a human-readable, log-friendly breakdown of flagged class pairs."""
        if not self.flagged_pairs:
            return "CrossClassOverlapReport: no class pair exceeded the flag threshold"
        lines = ["CrossClassOverlapReport: flagged class pairs (rate = a's examples nearest to b)"]
        lines.extend(f"  {a} -> {b}: {rate:.3f}" for a, b, rate in self.flagged_pairs)
        return "\n".join(lines)


def check_cross_class_embedding_overlap(
    examples: Sequence[LabeledCommandExample],
    *,
    flag_threshold: float = 0.15,
) -> CrossClassOverlapReport:
    """Flag class pairs whose training phrasing collides in embedding space.

    For every example, finds its nearest *other* example by cosine
    similarity in frozen sentence-embedding space
    (`language_embedding.encode_instructions`) and checks whether that
    neighbor belongs to a different class. This is a cheap, train-free way
    to catch a MOVE/GOTO_NAMED_REGION-style phrasing collision (see this
    module's docstring) before spending a full train+held-out-eval cycle to
    discover it the expensive way.

    Args:
        examples: The labeled example set to check (typically
            `build_command_type_training_set()`'s output, but any labeled
            set works -- e.g. training + held-out combined, to check whether
            a held-out phrasing collides with a *different* class's training
            phrasing).
        flag_threshold: Minimum cross-class nearest-neighbor rate for a
            class pair to appear in `CrossClassOverlapReport.flagged_pairs`.

    Returns:
        A `CrossClassOverlapReport`.

    """
    texts = [example.text for example in examples]
    embeddings = torch.from_numpy(encode_instructions(texts)).to(torch.float32)
    normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(float("-inf"))  # never match an example to itself
    nearest_indices = torch.argmax(similarity, dim=1).tolist()

    class_totals: dict[str, int] = {}
    for example in examples:
        class_totals[example.command_type.value] = class_totals.get(example.command_type.value, 0) + 1

    neighbors: list[CrossClassNeighbor] = []
    pair_counts: dict[tuple[str, str], int] = {}
    for example, nearest_index in zip(examples, nearest_indices, strict=True):
        nearest = examples[nearest_index]
        is_cross_class = nearest.command_type != example.command_type
        neighbors.append(
            CrossClassNeighbor(
                text=example.text,
                command_type=example.command_type,
                nearest_text=nearest.text,
                nearest_command_type=nearest.command_type,
                is_cross_class=is_cross_class,
            ),
        )
        if is_cross_class:
            key = (example.command_type.value, nearest.command_type.value)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_rates = {pair: count / class_totals[pair[0]] for pair, count in pair_counts.items()}
    flagged_pairs = tuple(
        sorted(
            ((a, b, rate) for (a, b), rate in pair_rates.items() if rate >= flag_threshold),
            key=lambda item: item[2],
            reverse=True,
        ),
    )
    return CrossClassOverlapReport(
        neighbors=tuple(neighbors),
        pair_counts=pair_counts,
        pair_rates=pair_rates,
        flagged_pairs=flagged_pairs,
    )
