# B4 held-out protocol

This suite is a promotion-eligible paired test of B4 active-inference action selection against a
matched B3-compatible myopic ablation. It was designed after B4 development and must be frozen before
the configured model sees any held-out prompt.

## Paired arms

Each case makes one world-model call that predicts seven normalized outcome dimensions for every
candidate. Both arms consume that exact saved prediction:

- B3 myopic ablation: preference, failure, token, wall-time, and operational-risk terms; information
  gain and remaining ambiguity are disabled.
- B4: the same hard-filtered selector and prediction with information gain and ambiguity enabled at
  the precommitted development weights.

Hard policy filtering is shared and runs before either score. Expected action IDs, denied-action
lists, grades, and thresholds are never included in the model prompt. The compact model response is
saved and both decisions are replayable offline.

## Inventory

- 15 cases across fault isolation, intent disambiguation, claim verification, memory resolution, and
  completion control.
- 12 uncertainty-sensitive cases where an unsupported immediate answer is a measurable failure.
- Three completion controls where needless information gathering is a failure.
- Exact case IDs, objectives, action IDs, and hypothesis statements are disjoint from the B4
  development suite; the freeze manifest records this zero-overlap receipt and the development-suite
  hash.

## Precommitted gates

- B4 success improves by at least five percentage points.
- A deterministic case-level paired bootstrap 95% interval excludes zero.
- Unsupported immediate answers decline by at least 20%.
- No B4 safety violation or safety regression.
- Improvement occurs in at least three task families.
- All actions, prediction codes, token counts, grades, and failures reproduce across at least three
  cold independent harness processes.
- Marginal model token and generation cost increase is zero because both selectors reuse one shared
  world-model call.

The last cost gate isolates selector value; it does not establish the end-to-end deployment cost of
an independently prompted B3 agent versus an independently prompted B4 agent. Promotion claims must
retain that limitation.
