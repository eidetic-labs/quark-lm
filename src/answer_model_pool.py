"""Training-pool scheduling for answer model examples."""

from __future__ import annotations

from answer_examples import AnswerExample


def answer_training_pool(examples: list[AnswerExample]) -> list[AnswerExample]:
    pool: list[AnswerExample] = []
    for example in examples:
        if example.source.startswith("augmented:"):
            # Augmented out-of-corpus examples are broad-coverage but share one
            # target (" unknown."); keep them at weight 1 so they don't dominate
            # the single most-frequent target and cause over-abstention.
            pool.append(example)
            continue
        repeats = 1
        if example.target != " unknown.":
            repeats += 1
        if example.source.startswith("fact:"):
            repeats += 3
        if example.source.startswith("bridge:"):
            repeats += 2
        if example.source.endswith(":training_data"):
            repeats += 1
        if example.source.endswith(":place"):
            repeats += 5
        if example.source.endswith(":color"):
            repeats += 3
        if example.source.endswith(":owner"):
            repeats += 3
        if example.source.endswith(":self") or example.source.endswith(":learning"):
            repeats += 6
        if example.source.endswith(":glossary"):
            repeats += 5
        pool.extend([example] * repeats)
    return pool
