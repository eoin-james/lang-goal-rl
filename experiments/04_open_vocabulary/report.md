# Stage 4: Open vocabulary

## In plain English

This stage tests whether the robot can follow instructions phrased in new
ways it never trained on -- not just the exact sentences it learned, but
paraphrases like "push your arm out in front of you" instead of "move your
hand forward." Getting there took four attempts. **Attempt 1** trained a
small neural network to convert sentences into goal locations, but it
memorized its 14 training sentences instead of learning the underlying
pattern -- brand-new phrasings landed in almost the wrong place every time,
and real task success collapsed to near-zero. A **quick zero-training sanity
check** in between confirmed that diagnosis: just looking up the nearest
known sentence directly, with no trained network at all, already worked
better than the trained network did. **Attempt 2** applied the obvious fix --
retrain on far more example sentences (70 instead of 14) -- which markedly
improved the network's ability to *classify* a new sentence into the right
region, but barely moved the real task success rate, revealing a second,
deeper problem. **Attempt 3** tried to measure exactly how much imprecision
the robot's trained behavior could tolerate per region, hoping to explain
that gap, but the measurement itself turned out to be too noisy to draw firm
conclusions from -- it pointed at "direction matters, not just distance,"
but couldn't be trusted as a clean per-region number. **Attempt 4** made the
decisive change: instead of trying to fix or retrain the neural network at
all, it removed the network entirely and replaced it with a simple
nearest-match lookup against a combined 84-sentence reference list -- and
that is what finally worked, lifting real task success on brand-new
sentences from near-zero to a majority pass rate.

## Result

**Passed on attempt 4 -- real task success on brand-new, never-seen
sentences jumped from ~2-10% (attempts 1-2, trained neural network) to a
mean of 57% (median 100%) after replacing the trained network with a simple
nearest-match lookup, with zero additional training of any kind.**

*(No chart in this report captures the attempt-4 result specifically --
every chart on file was generated for an earlier, failed attempt and would
misrepresent the final outcome if shown here. See the "Full evidence"
section below for the complete number-by-number breakdown of what changed
between attempts.)*

## How this was tested

Across all four attempts, the same test was reused for an apples-to-apples
comparison: take 14 instructions the system had never seen during training
(paraphrases of the original training sentences, e.g. "swing your arm over
to the left" instead of "move your hand to the left"), convert each into a
target location using whichever method that attempt was testing, and run
the robot's existing, already-trained control policy toward that target for
50 episodes per instruction, across 3 random seeds (42 success-rate samples
total per attempt). "Success" means the robot's hand ends up within 5cm of
the correct target region. A second, separate check ("semantic-neighbor
classification") asked a narrower geometry-only question with no robot
control involved: does a new sentence's converted location land closer to
the *correct* region than to any wrong one? That second check is a proxy,
not the real test -- as the attempts below show, doing well on it did not
reliably predict doing well on the real robot task, which is exactly the gap
the four attempts trace out.

---
## Full evidence
The complete technical record — proof gate, full result tables, charts,
raw logs, anomalies, known-risks cross-check, and the reviewer
verdict — lives in [`evidence.md`](evidence.md).
