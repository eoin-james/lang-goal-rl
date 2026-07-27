# Open Questions and Concerns

This file collects points that should be clarified before presenting the
project as portfolio-ready. It is a working list, not a claim that the
underlying results are invalid.

## Results language to standardize

- **Stage 4 versus stage 6:** Stage 4 reports 57.1% mean success on 14 held-out
  paraphrases. Stage 6 reports 85.7% on a separate set of 7 brand-new
  phrasings. These are different evaluations and should not be combined into
  one headline number.
- **Number of novel phrases:** Use 14 when discussing the stage 4 held-out
  evaluation and 7 when discussing the stage 6 live-interface evaluation.
- **Meaning of "open vocabulary":** The method performs nearest-neighbor lookup
  against 84 reference sentences. It accepts arbitrary input text, but the
  evidence does not establish broad, unrestricted language understanding.
- **Meaning of "live":** Clarify whether the published evidence comes from a
  person entering instructions interactively, a scripted simulation of that
  interaction, or both.
- **Stage count:** The repository contains stages 0 through 6. Describe this as
  seven stages total, or six experimental stages plus the setup/audit stage,
  consistently.

## Research concerns

- How quickly does performance degrade as a new instruction moves farther from
  the 84-sentence reference set?
- Do the results hold across more than seven spatial regions and beyond
  free-space reaching?
- How does the approach behave with ambiguous, contradictory, compositional, or
  out-of-distribution instructions?
- Can the system detect low-confidence language matches instead of confidently
  selecting the nearest known instruction?
- Does the 85.7% stage 6 result hold with more novel instructions, policies, and
  random seeds?
- Would a stronger language-to-goal method improve coverage without repeating
  the memorization failure found in the learned projection?

## Reproducibility concerns

- There is no single top-level command that reproduces a selected stage from a
  clean checkout.
- Required checkpoints, model downloads, hardware expectations, and approximate
  runtime should be explicit in the README.
- Demo artifacts should record the exact command, instruction sequence,
  checkpoint, environment seed, and outcome.
- Absolute paths appear in some saved logs. Confirm that no reproduction step
  relies on the original machine layout.
- Clarify which reported results can be reproduced from committed artifacts and
  which require retraining.

## Demo and presentation concerns

- A viewer cannot currently see the English instruction alongside the robot
  motion in the GIFs.
- Demo selection favors successful episodes with visible movement. This is
  documented and useful for presentation, but aggregate success rates and the
  selection rule should remain adjacent to each demo.
- The main README should distinguish the strongest capstone result from the
  broader and more conservative stage 4 result.
- Claims such as "understands English," "open vocabulary," and "real-time"
  should be defined narrowly enough to match the measured evidence.

## Decisions to make

- Choose the primary portfolio headline metric: stage 4's broader 14-phrase
  evaluation or stage 6's smaller live-control evaluation.
- Decide whether the next priority is a reproduction CLI, contextualized demos,
  or a README rewrite.
- Decide whether v0.1.0 represents the completed research record or a later
  portfolio-ready release.
- Decide how much additional evaluation is required before calling the project
  portfolio-ready.

## Already addressed

- Trivial do-nothing and random-policy baselines show that success is not caused
  by an accidentally easy task.
- The stage 4 memorization failure was investigated and reported rather than
  hidden.
- Demo selection and duplicate-video issues are documented in `demos/README.md`.
- Scope limits—84 reference sentences, seven regions, and free-space
  reaching—are stated in the repository.
