"""Mode names for basic branch direct-answer objectives."""

from __future__ import annotations


BASIC_BRANCH_DIRECT_ANSWER_MODES = frozenset(
    {
        "branch-repair-unlikelihood",
        "periodic-branch-repair-unlikelihood",
        "branch-collapse-unlikelihood",
        "periodic-branch-collapse-unlikelihood",
        "branch-batch-contrast-unlikelihood",
        "periodic-branch-batch-contrast-unlikelihood",
        "branch-diversity-unlikelihood",
        "periodic-branch-diversity-unlikelihood",
        "branch-target-softmax-unlikelihood",
        "periodic-branch-target-softmax-unlikelihood",
        "branch-target-margin-unlikelihood",
        "periodic-branch-target-margin-unlikelihood",
        "branch-hidden-projection-margin-unlikelihood",
    }
)
