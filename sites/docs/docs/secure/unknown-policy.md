---
title: Unknown Policy
description: How QuarkLM should answer outside the corpus.
---

# Unknown Policy

When a fact is outside the admitted corpus, QuarkLM should answer:

```text
unknown.
```

For training-data status questions, known admitted facts answer `yes`; facts
outside the corpus answer `no`.

This rule keeps the prototype honest. It should not invent answers from nearby
surface forms or from the surrounding world.
