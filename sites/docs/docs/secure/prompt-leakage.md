---
title: Prompt Leakage
description: Prevent held-out prompts from becoming training lessons.
---

# Prompt Leakage

Held-out prompts must not be copied into lesson files. QuarkLM can learn held-out
facts through fact-style lessons, but it should not train on the exact held-out
evaluation prompt forms.

The self-improvement report audits protected prompt leakage for:

- held-out facts
- held-out ownership prompts

The expected result is zero leaked protected prompts.
