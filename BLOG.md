# Building a Robot That Takes English Instructions Live

A devlog for [lang-goal-rl](https://github.com/graylayer-labs/lang-goal-rl) —
teaching a simulated robot arm to take ad-hoc English instructions *while it's
already moving*, and re-target on the fly. This is the casual version of the
story. For the structured, all-numbers version, see
[`STATUS.md`](STATUS.md); for the staged technical record, see
[`ROADMAP.md`](ROADMAP.md).

---

## 2026-07-26 — Day one: the robot I planned to use wouldn't even install

The original plan was `panda-gym` — it's the more feature-complete simulated
arm, it's what a chunk of the robot-learning literature actually uses, and I
had a whole staged roadmap written around it and Gymnasium-Robotics's Fetch
suite as alternates.

Then I tried to actually build it. `panda-gym` depends on `pybullet`, and
`pybullet` refused to build from source on macOS arm64. Not a version
mismatch I could pin around — a real build failure, first command of the
project.

So: swap to `gymnasium-robotics`'s `FetchReach-v4`, a MuJoCo-backed
free-space reaching task, and don't look back. In hindsight this was the
right call for reasons that had nothing to do with the build error — Reach
is the *simplest* task in the Fetch suite (no grabbing, no pushing, no
contact physics), which meant every stage after this one could focus
entirely on the actual thesis (language → continuous goal → live re-goaling)
instead of fighting simulated contact dynamics on top of it. Locked that in
explicitly as scope, not as a thing to quietly walk back later: this project
proves the *mechanism*, not that it survives harder manipulation tasks.
That's honest future work, not a hidden gap.

By the end of day one: SAC + Hindsight Experience Replay, given the exact
xyz coordinates of a goal, hits 100% success. Robot goes where it's told.
Stage 1, done.

---

## 2026-07-26 — Stage 3's four-attempt saga, or: how I spent three rounds fixing something that wasn't broken

Stage 3 is the first one where an English sentence has to actually mean
something to the robot: turn a sentence into a point in the same latent
goal-space stage 2 already proved out, and see if the robot still gets
there.

**Attempt 1: near-total failure.** ~0.2% success. Dug into it and found the
sentence-to-goal converter was outputting numbers 5-10x the scale the policy
had ever seen during training — the network had learned to keep different
sentences far apart from each other, which it did (that check passed
cleanly), but nothing in its loss function cared *where* those points sat
relative to what the robot actually understands. Right shape, wrong
neighborhood.

**Attempt 2: fixed the scale.** Added a term that pulls the output back into
the right range. Result: ~7%. Better, but nowhere close to done. Looked
closer and found a second problem — the *direction* was still off, just not
by as much.

**Attempt 3: fixed the direction too.** Retrained to point straight at each
instruction's exact target instead of a noisy moving estimate. The network
now matched its target almost perfectly — and RL success capped out around
16%, even for the sentence it did best on. That's the moment this stopped
looking like a bug and started looking like a wall. Maybe you just can't get
a fixed point to reliably land inside a 5cm success radius from a sentence
that describes a whole *region* of space?

**Attempt 4 — the actual answer.** It wasn't the robot, and it wasn't the
sentence converter. It was the test. "Reach up high" describes a region, not
one exact spot — but the eval was picking a fresh random point *inside* that
region on every single check and demanding the robot land exactly there,
when the robot had only ever been given one fixed point to aim for the whole
time. That's not a hard problem, it's close to a geometrically impossible
one, no matter how good the underlying model is. The math for "how often
would a fixed point happen to land within 5cm of a random point in a region
this wide" predicts almost exactly the ~16% ceiling attempt 3 hit. Changed
one line — judge success against the point the sentence actually points to,
not a random spot nearby — and the result jumped straight to **1.000**,
matching the coordinate-only baseline exactly. Zero retraining needed. The
model had been right since attempt 3. The test was wrong the whole time.

That one stung a little, honestly — three attempts improving the wrong
thing before I asked "wait, what is this eval actually checking?" But it's
also the best kind of bug to find, because the fix was one line and the
lesson (know whether your ground truth is a fixed point or a resampled
region before you start "fixing" a model) carried straight into the next
stage.

---

## 2026-07-26 — Stage 4: the *other* four-attempt saga, and this time the model really was broken

Stage 4 asks a harder question: does this work on sentences the robot's
never seen? Wrote 14 brand-new paraphrases, split across the same seven
regions, and ran them through the exact same pipeline stage 3 just proved
works.

**Attempt 1: it doesn't generalize at all.** ~2% RL success, barely above
random region-guessing on classification. Plotted where the training
sentences and the new ones land in the model's internal space, and they
occupy almost completely separate territory — the visual signature of a
model that memorized 14 exact answers instead of learning a rule. Made
sense on reflection: a ~25,000-parameter network trained on 14 examples has
every incentive to just memorize them.

Before spending a training cycle "fixing" this, I ran a sanity check with
zero training involved: skip the learned model entirely, just borrow the
answer from whichever of the 14 known sentences a new one resembles most.
That got 10/14 right, versus the trained model's 4/14. Confirmed cleanly:
the raw language understanding underneath has the signal, the small
trained layer on top of it was throwing it away.

**Attempt 2: more training data.** Retrained on 70 sentences instead of 14.
Classification jumped a lot (right region 9/14 times instead of 4/14) —
but actual task success barely moved, ~10% instead of ~2%. The shape of the
problem had changed underneath me: it wasn't "wrong region" anymore, it was
"right region, not precise enough" — and how forgiving each direction was
of imprecision seemed to vary a lot. One sentence landed *closer* to its
target than another and still failed, while the farther one succeeded
every time.

**Attempt 3: trying to map out exactly how forgiving each direction is.**
Deliberately fed the robot slightly-wrong versions of the right answer at
increasing amounts of wrongness, for each of the seven directions. Expected
a clean story — "direction X tolerates more error than direction Y." Got a
mess instead: some directions recovered *after* apparently collapsing at a
smaller error. Turned out each test only tried one random way of being
wrong at each amount, so what looked like "more wrongness breaks it" was
sometimes just "this particular way of being wrong broke it, a different
way at the same amount wouldn't have." Not a wasted experiment, just a
different, sharper finding than the one I went looking for: it's not just
*how far off* an answer is, it's *which direction*.

**Attempt 4 — the decisive one.** Instead of trying to characterize that
mess more carefully, I asked the more useful question: is this the small
trained model's fault, or is it inherent to the robot's precision needs?
Skipped the trained model one more time, went back to the simple
lookup-the-closest-known-sentence approach from attempt 1's sanity check,
but now with all 84 known sentences (14 original + 70 new) instead of just
14. Real success on brand-new sentences jumped from ~10% to **57%**, and for
the *typical* new sentence it now succeeds every time. That's decisive: the
trained model was the problem all along, not the robot. The fix wasn't more
training — it was deleting the trained layer and using a lookup table
instead.

The honest asterisk, and it matters: this only works because a new sentence
usually resembles *something* in that 84-sentence list closely enough. It
doesn't understand language from scratch. That's a real, logged limitation
going into the later stages, not a footnote I get to skip.

---

## 2026-07-27 — The GIF that looked wrong, and the question it forced me to ask

While putting together demo clips to actually *show* this thing working —
not just report numbers — a couple of them looked suspiciously boring. The
arm barely moved. And two clips that were supposed to show different stages
turned out to be byte-for-byte identical.

Root cause, once I looked: every demo script tried eval seeds in order and
stopped at the very first one that succeeded. FetchReach spawns goals only
centimeters from where the arm starts, so "first success" is very often an
almost invisible nudge — a real success, just a visually useless one to put
in a demo. Fixed it by tracking total gripper travel across the whole
episode, searching a small range of seeds, and picking the most dynamic
real success instead of the first one. Also made sure no two scripts could
land on the exact same seed again.

But the more interesting thing this surfaced wasn't a video bug — it was a
question. Looking through those seed searches, one training episode had a
goal placed unusually close to the robot's resting position. Which raises a
genuinely uncomfortable possibility: what if FetchReach's goals are placed
close enough, often enough, that a robot doing *absolutely nothing* would
already score decently — and every "it worked" result in this entire
project has been measuring an easy task, not real learned behavior?

I didn't want to just wave that away. So I tested it directly: reset the
simulator 500 times and measured how far goals actually spawn (median 14.5cm
— three times farther than the 5cm success threshold), then measured what a
robot that never moves scores under this project's own pass/fail rule
(1.8%), and what a robot moving completely at random scores (0.4%). Then
lined every real result this project has ever reported up against that
floor. The *weakest* result across six stages — 54.8% — is still 30x higher
than doing nothing. The strongest results are 50-plus times higher. That's
not a coincidence, and it's not close. This is now a permanent, re-runnable
check in the repo, not something I get to assert once and move on from.

Worth being honest about why this mattered to actually build and not just
assume: it would have been very easy to skip this, since every prior stage
already had a passing number. The only reason it happened at all is that a
video looked a little too calm.

---

## 2026-07-27 — Stage 6: it all comes together, live

Everything before this stage proved one piece each: understand a sentence,
reach a learned code instead of raw coordinates, generalize to sentences
never seen before, change target mid-task without a reset. Stage 6 wires
every one of those pieces together and actually runs it — live, no special
retraining for the combination.

Three independently-trained robots, each given a typed English instruction,
then a *different* typed instruction partway through the same task, with no
reset in between. On a set of 7 completely new sentences that had never
appeared anywhere in this project before — checked to make sure they weren't
just near-copies of anything already used — it redirected and succeeded 86%
of the time, usually within 3 steps of the new instruction landing. On the
sentences stage 4 had already tested, used here purely as a sanity check
that the live version behaves the same as the offline-tested one, it matched
almost exactly (55% here vs. 57% before). In both cases, the act of
switching mid-task cost nothing measurable on its own — whatever the
switching mechanism achieved on 100% of switches, it delivered every time;
the failures that did happen were the same sentence-recognition gaps stage 4
already knew about, not a new cost of switching live.

Honest framing, because it's the whole point of logging this project
truthfully rather than selling it: this proves the *mechanism* — type a
sentence, the robot goes for it, type a different one, it changes its mind —
works, on the simplest task in its test suite, with a vocabulary of 84 known
sentence patterns. It doesn't prove this scales to truly unlimited English,
and it doesn't prove it survives harder physical tasks like pushing or
grasping. Those are real, open, and logged as future work — not quietly
implied to already be solved.

One more thing worth being precise about: the 55%/86% numbers above came
from a scripted harness (`live_regoal_eval.py`) feeding both instructions
in programmatically — the first at episode start, the second at a fixed
switch step (step 20 of 50) — not a human typing them at a terminal in real
time. The mechanism it drives is the exact same one a person typing at
`interactive_demo.py`'s prompt invokes; only the source of the keystrokes
differs.

That's the project as it stands after Phase 1: seven stages, two real
debugging sagas, one sanity check that turned out to matter more than
expected, and a robot that takes English instructions live and actually
changes its mind when you tell it something new.

---

## 2026-07-28 — Phase 2a: swapping sentences for real commands

Phase 1 proved the core trick — a sentence can pick a target, and the robot
can be told something new mid-task. But every sentence in that whole
project snapped to one of 7 fixed regions under the hood. "Reach up high"
and "raise your arm as high as it will go" both collapsed to the same
single point. That's fine for proving the mechanism, but it's not what
"understand a command" should mean long-term.

Phase 2a's job was to build the layer sentences will eventually plug into,
*without* the language model in the loop yet — deliberately, so any bugs
in this new layer can't be blamed on language understanding, and vice
versa. Four stages: check that "left" on screen actually looks like left
on camera (still waiting on me to actually watch those clips — deferred,
not blocking); prove the robot can shift itself relative to wherever it
already is, not just toward a target it started the episode with; prove
it can chain several of those moves together with no reset in between;
and finally wrap all of that behind a small typed command language
(`goto`, `move`, `waypoints`, `stop`, `reset`) that a future language
layer — or a human typing directly — can drive.

All three of the stages that didn't need my own eyes came back clean.
Relative moves: perfect, in every direction, at every distance tried.
Waypoint chains: the first attempt at this one only tested a single
trained robot, which the review process correctly flagged as not enough
evidence — rerunning across all the healthy robots confirmed zero chains
broke down partway through. The typed-command wrapper: wiring the same
mechanisms behind actual parsed text changed nothing measurable, and it
correctly rejects garbled input instead of guessing what you meant. The
one genuinely new thing tested — does `stop` actually make the robot hold
still — came back with an honest, slightly imperfect answer: it settles to
within a couple centimeters and stays there, rather than drifting further,
which makes sense for a robot that was never trained to just sit still on
command.

Next up is the part that actually needed all of this scaffolding: teaching
a language model to speak this typed-command language instead of picking
from 7 fixed buckets.

---

## 2026-07-30 — Phase 2b, stage 1: the classifier and I disagreed about what a sentence meant

First step of teaching a model to speak the typed-command language: before
it can figure out *where* to move or *how far*, it has to figure out *what
kind* of instruction it just heard — a move, a stop, a reset, "go to that
spot over there," or something it has no business trying to handle at all.
Five buckets, one small classifier sitting on top of the same sentence
embeddings used everywhere else in this project.

First attempt: 0% on one of the five buckets. Not "struggling" — flat
zero, every single time, no matter how long I trained it or how I tweaked
the settings. That's actually a useful signal once you know how to read
it: if more training makes the model *more* confident while it stays
*exactly as wrong*, the model isn't the problem. I was.

Here's what happened. I told the builder to reuse a pile of sentences from
way back near the start of this project — "reach forward," "angle your
hand toward the front" — as training examples for "go to a named spot."
Seemed efficient: why write new sentences when 84 already exist. Except
those sentences were written, deliberately, to sound like *movement
instructions* — that was their whole job at the time. Reusing them to mean
"go to an absolute place" instead of "move in a direction" meant I'd
quietly taught the model that "angle X toward direction" means "absolute
destination," which is exactly backwards from what a real move instruction
sounds like. The model wasn't confused. It learned precisely what I fed
it — I just fed it the wrong thing.

Fixed it by giving each meaning its own honest phrasing: move instructions
now always carry a sense of degree ("a bit," "slightly," "a good
distance"); go-to-a-place instructions now always name an actual
destination ("go to the far left side") instead of a direction. That
alone fixed the 0%.

Which uncovered two much smaller problems the big one had been hiding:
the model had never seen a stop command phrased as an idiom ("cut it
out") instead of the word "stop," and had never seen a math question at
all, so one showed up as a random guess instead of correctly getting
flagged as "not something I can do." A dozen or so extra example
sentences later, both were fixed too.

One small leftover, and I'm leaving it alone on purpose: two "restart"
phrasings now get mildly confused with the newly-taught stop idioms. It
doesn't break anything that matters yet, and chasing it further right now
would be polishing a corner nobody's using yet. Noted, not fixed —
that's a deliberate call, not an oversight.

Honest note for anyone reading this as "AI built a robot classifier
flawlessly": it took three rounds, and the first failure was a mistake I
made in how I set up the task, not a bug the model introduced. That's
normal. The part worth being proud of isn't "it worked first try" — it's
that the failure mode was diagnosable (0% + falling training loss = look
at your data, not your hyperparameters) and every fix was checked by an
independent pass before I called it done, twice.
