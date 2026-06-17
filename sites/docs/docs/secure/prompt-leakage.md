---
title: Prompt Leakage
description: Prevent held-out prompts from becoming training lessons.
---

# Prompt Leakage

A held-out probe is only a fair test if the model has never trained on its exact
prompt form. The prompt-leakage audit enforces that separation: it checks that no
training lesson reuses the verbatim prompt of a protected held-out evaluation. It
is one of the promotion-gate audits a run must pass before any weight update is
promoted (see [Self-improvement loop](../learn/self-improvement-loop.md)).

## Facts may be learned; protected prompts may not

QuarkLM is allowed to learn a held-out *fact* — through ordinary fact-style
lessons admitted to the corpus, the knowledge can be consolidated like any other.
What it must not train on is the *exact prompt string* used to evaluate that fact
in a held-out set. The held-out prompt form is the measuring instrument. If a
lesson copies it verbatim, a later "correct" answer no longer distinguishes
generalization from memorization of the test, and the held-out number stops
meaning anything.

The audit therefore operates on exact prompt strings, not on facts or topics. A
paraphrased lesson about the same fact is fine; a byte-for-byte copy of the
evaluation prompt is a leak.

## How the audit works

`audit_prompt_leakage` (in `self_improvement_audits`) compares lesson files
against one protected evaluation file:

1. Read the protected eval file and collect the set of its `prompt` strings.
2. Scan each lesson file line by line. A lesson record whose `prompt` exactly
   matches a protected eval prompt is recorded as leaked.
3. If a named lesson file does not exist, that absence is itself recorded as a
   leak (`<missing lesson file>`), so a missing source cannot silently pass.

The audit reports the eval source, the lesson sources, the list of leaked
prompts, and `passed` — which is true only when no leak was found.

```text
protected eval prompts ─┐
                        ├─ exact-match scan ─> leaked_prompts (must be empty)
lesson file prompts ────┘
```

## Protected prompt sets

`audit_all_protected_prompts` runs the audit over two protected sets:

| Set | Source | Scope |
| --- | --- | --- |
| Held-out facts | `evals/heldout.jsonl` | All records in the file. |
| Held-out ownership | `evals/owner.jsonl` | Only records whose `id` contains `-heldout-`. |

The ownership file mixes known and held-out probes, so the audit filters it to
the held-out portion (`protected_id_contains="-heldout-"`). The known ownership
prompts are answerable and may appear in lessons; only the held-out ones are
protected.

These two audits surface in the run report as the named checks
`heldout_prompt_leakage` and `owner_heldout_prompt_leakage`.

## What promotion requires

Both checks must report `passed: true` — zero leaked protected prompts — for a
run to clear the promotion gate. They sit alongside the generated-probes,
closed-world verifier, constraint-first, forgetting, and exact-eval audits; a run
is not promoted because it completed, only because every required audit passed
(see [Self-improvement loop](../learn/self-improvement-loop.md)).

## Why it exists

This audit protects the integrity of the evidence, not the weights themselves.
QuarkLM keeps `memory-served` and `weight-consolidated` as separate rails, and it
reports held-out numbers as out-of-sample evidence about the learning path. If
the held-out prompt could leak into training, that evidence would quietly
collapse into memorization of the test. The leakage audit keeps the held-out sets
honest, in the same spirit as the [purity boundary](./purity-boundary.md):
training may only consume admitted material, and the test set is not it.
