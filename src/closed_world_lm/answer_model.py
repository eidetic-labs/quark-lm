"""A learned closed-world answer model trained from admitted corpus lessons."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .curriculum import DEFAULT_CORPUS_DIR, DEFAULT_OUTPUT_DIR, read_json
from .glossary_probes import glossary_definitions, probe_words
from .probes import read_jsonl
from .respond import CorpusResponder


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_TEXT = DEFAULT_OUTPUT_DIR / "train.txt"
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "answer-latest"
DEFAULT_EVALS = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "paraphrases.jsonl",
    PROJECT_DIR / "evals" / "owner.jsonl",
    PROJECT_DIR / "evals" / "self.jsonl",
    PROJECT_DIR / "evals" / "learning.jsonl",
    PROJECT_DIR / "evals" / "admissions.jsonl",
    PROJECT_DIR / "evals" / "admission_paraphrases.jsonl",
    PROJECT_DIR / "evals" / "glossary.jsonl",
]
WORD_RE = re.compile(r"[a-z']+")
SEMANTIC_FEATURE_WEIGHT = 6
SEMANTIC_PROMPT_PATTERNS = [
    (
        "place",
        re.compile(r"question: where is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?"),
    ),
    (
        "color",
        re.compile(r"question: what color is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?"),
    ),
    ("place", re.compile(r"ask: place for (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"ask: color for (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"place: (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"color: (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"fact place (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"fact color (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"place fact (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"color fact (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"tell me the place of (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"which color belongs to (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("owner", re.compile(r"question: who has the (?P<object>[a-z]+)")),
    ("owner", re.compile(r"ask: owner for (?P<object>[a-z]+)")),
    ("owner", re.compile(r"owner: (?P<object>[a-z]+)")),
    ("owner", re.compile(r"fact owner (?P<object>[a-z]+)")),
    ("owner", re.compile(r"owner fact (?P<object>[a-z]+)")),
    ("owner", re.compile(r"which person has (?P<object>[a-z]+)")),
    (
        "training_data",
        re.compile(r"question: is (?P<person>[a-z]+)'s (?P<object>[a-z]+) part of your training data\?"),
    ),
    ("training_data", re.compile(r"training data: (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("training_data", re.compile(r"fact training data (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("training_data", re.compile(r"training data fact (?P<person>[a-z]+) (?P<object>[a-z]+)")),
]
SELF_PROMPT_PATTERNS = [
    ("self", "kind", re.compile(r"question: what are you\?")),
    ("self", "kind", re.compile(r"ask: self kind")),
    ("self", "kind", re.compile(r"fact self kind")),
    ("self", "kind", re.compile(r"self fact kind")),
    ("self", "dataset", re.compile(r"question: what is your dataset\?")),
    ("self", "dataset", re.compile(r"ask: self dataset")),
    ("self", "dataset", re.compile(r"fact self dataset")),
    ("self", "dataset", re.compile(r"self fact dataset")),
    ("self", "pretrained_weights", re.compile(r"question: do you use pretrained weights\?")),
    ("self", "pretrained_weights", re.compile(r"ask: self pretrained weights")),
    ("self", "pretrained_weights", re.compile(r"fact self pretrained_weights")),
    ("self", "pretrained_weights", re.compile(r"self fact pretrained_weights")),
    (
        "self",
        "unknown_policy",
        re.compile(r"question: what do you say when a fact is outside your corpus\?"),
    ),
    ("self", "unknown_policy", re.compile(r"ask: self unknown policy")),
    ("self", "unknown_policy", re.compile(r"fact self unknown_policy")),
    ("self", "unknown_policy", re.compile(r"self fact unknown_policy")),
    ("self", "improvement_method", re.compile(r"question: how do you improve\?")),
    ("self", "improvement_method", re.compile(r"ask: self improvement method")),
    ("self", "improvement_method", re.compile(r"fact self improvement_method")),
    ("self", "improvement_method", re.compile(r"self fact improvement_method")),
    ("self", "diagnosis_source", re.compile(r"question: what source guides your self-diagnosis\?")),
    ("self", "diagnosis_source", re.compile(r"ask: self diagnosis source")),
    ("self", "diagnosis_source", re.compile(r"fact self diagnosis_source")),
    ("self", "diagnosis_source", re.compile(r"self fact diagnosis_source")),
    (
        "self",
        "external_model_shaping",
        re.compile(r"question: does an external model shape your self-diagnosis\?"),
    ),
    ("self", "external_model_shaping", re.compile(r"ask: self external model shaping")),
    ("self", "external_model_shaping", re.compile(r"fact self external_model_shaping")),
    ("self", "external_model_shaping", re.compile(r"self fact external_model_shaping")),
    (
        "learning",
        "new_data",
        re.compile(r"question: what happens when you learn something new\?"),
    ),
    ("learning", "new_data", re.compile(r"ask: learning new data")),
    ("learning", "new_data", re.compile(r"fact learning new_data")),
    ("learning", "new_data", re.compile(r"learning fact new_data")),
    ("learning", "admission", re.compile(r"question: when is something learned\?")),
    ("learning", "admission", re.compile(r"ask: learning admission")),
    ("learning", "admission", re.compile(r"fact learning admission")),
    ("learning", "admission", re.compile(r"learning fact admission")),
    (
        "learning",
        "weight_update",
        re.compile(r"question: what changes after new training data is admitted\?"),
    ),
    ("learning", "weight_update", re.compile(r"ask: learning weight update")),
    ("learning", "weight_update", re.compile(r"fact learning weight_update")),
    ("learning", "weight_update", re.compile(r"learning fact weight_update")),
    (
        "learning",
        "repair_action",
        re.compile(r"question: how is the next repair action chosen\?"),
    ),
    ("learning", "repair_action", re.compile(r"ask: learning repair action")),
    ("learning", "repair_action", re.compile(r"fact learning repair_action")),
    ("learning", "repair_action", re.compile(r"learning fact repair_action")),
]
GLOSSARY_PROMPT_PATTERNS = [
    re.compile(r"question: what does (?P<word>[a-z]+) mean\?"),
    re.compile(r"define (?P<word>[a-z]+)"),
    re.compile(r"fact glossary (?P<word>[a-z]+)"),
    re.compile(r"glossary fact (?P<word>[a-z]+)"),
]


@dataclass(frozen=True)
class AnswerExample:
    prompt: str
    target: str
    source: str


@dataclass
class AnswerModelConfig:
    labels: list[str]
    features: list[str]
    seed: int = 7


class AnswerSoftmax:
    def __init__(
        self,
        config: AnswerModelConfig,
        weights: list[list[float]],
        bias: list[float],
    ) -> None:
        self.config = config
        self.weights = weights
        self.bias = bias
        self.label_to_index = {label: index for index, label in enumerate(config.labels)}
        self.feature_to_index = {feature: index for index, feature in enumerate(config.features)}

    @classmethod
    def init_random(cls, config: AnswerModelConfig) -> "AnswerSoftmax":
        rng = random.Random(config.seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in config.features]
            for _ in config.labels
        ]
        bias = [0.0 for _ in config.labels]
        return cls(config, weights, bias)

    def featurize(self, prompt: str) -> dict[int, float]:
        names = feature_names(prompt)
        counts: dict[int, float] = {}
        for name in names:
            index = self.feature_to_index.get(name)
            if index is not None:
                counts[index] = counts.get(index, 0.0) + 1.0
        return counts

    def probabilities(self, prompt: str) -> list[float]:
        features = self.featurize(prompt)
        logits = self._logits(features)
        return softmax(logits)

    def predict(self, prompt: str) -> str:
        probs = self.probabilities(prompt)
        index = max(range(len(probs)), key=lambda item: probs[item])
        return self.config.labels[index]

    def loss(self, prompt: str, target: str) -> float:
        target_index = self.label_to_index[target]
        probs = self.probabilities(prompt)
        return -math.log(max(probs[target_index], 1e-12))

    def train_step(self, example: AnswerExample, learning_rate: float) -> float:
        target_index = self.label_to_index[example.target]
        features = self.featurize(example.prompt)
        logits = self._logits(features)
        probs = softmax(logits)
        loss = -math.log(max(probs[target_index], 1e-12))
        probs[target_index] -= 1.0

        for label_index, grad in enumerate(probs):
            self.bias[label_index] -= learning_rate * grad
            for feature_index, value in features.items():
                self.weights[label_index][feature_index] -= learning_rate * grad * value
        return loss

    def _logits(self, features: dict[int, float]) -> list[float]:
        logits = self.bias[:]
        for label_index, row in enumerate(self.weights):
            total = logits[label_index]
            for feature_index, value in features.items():
                total += row[feature_index] * value
            logits[label_index] = total
        return logits

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerSoftmax":
        return cls(
            AnswerModelConfig(**payload["config"]),
            payload["weights"],
            payload["bias"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "AnswerSoftmax":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def feature_names(prompt: str) -> list[str]:
    lower = prompt.lower()
    words = WORD_RE.findall(lower)
    names = ["bias"]
    names.extend(f"word:{word}" for word in words)
    names.extend(f"wordpair:{left}:{right}" for left, right in zip(words, words[1:], strict=False))
    names.extend(f"char:{char}" for char in lower)
    for size in (2, 3, 4):
        names.extend(f"ngram{size}:{lower[index:index + size]}" for index in range(len(lower) - size + 1))
    names.extend(semantic_feature_names(lower))
    return names


def semantic_feature_names(lower_prompt: str) -> list[str]:
    names: list[str] = []
    for intent, pattern in SEMANTIC_PROMPT_PATTERNS:
        match = pattern.search(lower_prompt)
        if not match:
            continue
        person = match.groupdict().get("person")
        obj = match["object"]
        semantic = (
            [f"intent:{intent}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"object:{obj}"] * SEMANTIC_FEATURE_WEIGHT
        )
        if person:
            semantic.extend(
                [f"person:{person}"] * SEMANTIC_FEATURE_WEIGHT
                + [f"entity:{person}:{obj}"] * SEMANTIC_FEATURE_WEIGHT
                + [f"intent_entity:{intent}:{person}:{obj}"] * SEMANTIC_FEATURE_WEIGHT
            )
        else:
            semantic.extend([f"intent_object:{intent}:{obj}"] * SEMANTIC_FEATURE_WEIGHT)
        names.extend(semantic)
    for intent, slot, pattern in SELF_PROMPT_PATTERNS:
        if not pattern.search(lower_prompt):
            continue
        names.extend(
            [f"intent:{intent}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"slot:{slot}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"intent_slot:{intent}:{slot}"] * SEMANTIC_FEATURE_WEIGHT
        )
    for pattern in GLOSSARY_PROMPT_PATTERNS:
        match = pattern.search(lower_prompt)
        if not match:
            continue
        word = match["word"]
        names.extend(
            ["intent:glossary"] * SEMANTIC_FEATURE_WEIGHT
            + [f"glossary_word:{word}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"intent_word:glossary:{word}"] * SEMANTIC_FEATURE_WEIGHT
        )
    return names


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(item - max_logit) for item in logits]
    total = sum(exps)
    return [item / total for item in exps]


def prompt_templates(person: str, obj: str, kind: str, lesson_style: str = "qa") -> list[str]:
    if kind == "place":
        if lesson_style == "bridge":
            return [
                f"tell me the place of {person} {obj}\nanswer:",
            ]
        if lesson_style == "fact":
            return [
                f"fact place {person} {obj}\nanswer:",
                f"place fact {person} {obj}\nanswer:",
            ]
        return [
            f"question: where is {person}'s {obj}?\nanswer:",
            f"ask: place for {person} {obj}\nanswer:",
            f"place: {person} {obj}\nanswer:",
        ]
    if kind == "color":
        if lesson_style == "bridge":
            return [
                f"which color belongs to {person} {obj}\nanswer:",
            ]
        if lesson_style == "fact":
            return [
                f"fact color {person} {obj}\nanswer:",
                f"color fact {person} {obj}\nanswer:",
            ]
        return [
            f"question: what color is {person}'s {obj}?\nanswer:",
            f"ask: color for {person} {obj}\nanswer:",
            f"color: {person} {obj}\nanswer:",
        ]
    if kind == "owner":
        if lesson_style == "bridge":
            return [
                f"which person has {obj}\nanswer:",
            ]
        if lesson_style == "fact":
            return [
                f"fact owner {obj}\nanswer:",
                f"owner fact {obj}\nanswer:",
            ]
        return [
            f"question: who has the {obj}?\nanswer:",
            f"ask: owner for {obj}\nanswer:",
            f"owner: {obj}\nanswer:",
        ]
    if kind == "training_data":
        if lesson_style == "fact":
            return [
                f"fact training data {person} {obj}\nanswer:",
                f"training data fact {person} {obj}\nanswer:",
            ]
        return [
            f"question: is {person}'s {obj} part of your training data?\nanswer:",
            f"training data: {person} {obj}\nanswer:",
        ]
    raise ValueError(f"unknown prompt kind {kind!r}")


def self_prompt_templates(slot: str, lesson_style: str = "qa") -> list[str]:
    if lesson_style == "fact":
        return [
            f"fact self {slot}\nanswer:",
            f"self fact {slot}\nanswer:",
        ]
    prompts = {
        "kind": [
            "question: what are you?\nanswer:",
            "ask: self kind\nanswer:",
        ],
        "dataset": [
            "question: what is your dataset?\nanswer:",
            "ask: self dataset\nanswer:",
        ],
        "pretrained_weights": [
            "question: do you use pretrained weights?\nanswer:",
            "ask: self pretrained weights\nanswer:",
        ],
        "unknown_policy": [
            "question: what do you say when a fact is outside your corpus?\nanswer:",
            "ask: self unknown policy\nanswer:",
        ],
        "improvement_method": [
            "question: how do you improve?\nanswer:",
            "ask: self improvement method\nanswer:",
        ],
        "diagnosis_source": [
            "question: what source guides your self-diagnosis?\nanswer:",
            "ask: self diagnosis source\nanswer:",
        ],
        "external_model_shaping": [
            "question: does an external model shape your self-diagnosis?\nanswer:",
            "ask: self external model shaping\nanswer:",
        ],
    }
    return prompts[slot]


def learning_prompt_templates(slot: str, lesson_style: str = "qa") -> list[str]:
    if lesson_style == "fact":
        return [
            f"fact learning {slot}\nanswer:",
            f"learning fact {slot}\nanswer:",
        ]
    prompts = {
        "new_data": [
            "question: what happens when you learn something new?\nanswer:",
            "ask: learning new data\nanswer:",
        ],
        "admission": [
            "question: when is something learned?\nanswer:",
            "ask: learning admission\nanswer:",
        ],
        "weight_update": [
            "question: what changes after new training data is admitted?\nanswer:",
            "ask: learning weight update\nanswer:",
        ],
        "repair_action": [
            "question: how is the next repair action chosen?\nanswer:",
            "ask: learning repair action\nanswer:",
        ],
    }
    return prompts[slot]


def glossary_prompt_templates(word: str, lesson_style: str = "qa") -> list[str]:
    if lesson_style == "fact":
        return [
            f"fact glossary {word}\nanswer:",
            f"glossary fact {word}\nanswer:",
        ]
    return [
        f"question: what does {word} mean?\nanswer:",
        f"define {word}\nanswer:",
    ]


def examples_from_sources(
    train_text: str,
    grammar: dict[str, Any],
    glossary: dict[str, Any] | None = None,
) -> list[AnswerExample]:
    responder = CorpusResponder.train_from_text(train_text)
    examples: list[AnswerExample] = []
    fact_ids_by_key = {
        (fact["person"], fact["object"]): fact["id"]
        for fact in grammar.get("story_facts", [])
    }
    qa_lesson_ids = set(grammar.get("qa_lesson_ids", []))
    for (person, obj), fact in sorted(responder.facts.items()):
        fact_id = fact_ids_by_key.get((person, obj))
        is_admitted_fact = fact_id is None
        lesson_styles = ["qa", "fact"] if fact_id in qa_lesson_ids or is_admitted_fact else ["fact"]
        answer_lesson_styles = [*lesson_styles, "bridge"]
        if fact.place:
            for lesson_style in answer_lesson_styles:
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {fact.place}.",
                        source=f"{lesson_style}:place",
                    )
                    for prompt in prompt_templates(person, obj, "place", lesson_style)
                )
        if fact.color:
            for lesson_style in answer_lesson_styles:
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {fact.color}.",
                        source=f"{lesson_style}:color",
                    )
                    for prompt in prompt_templates(person, obj, "color", lesson_style)
                )
        if fact.owner:
            for lesson_style in answer_lesson_styles:
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {fact.owner}.",
                        source=f"{lesson_style}:owner",
                    )
                    for prompt in prompt_templates(person, obj, "owner", lesson_style)
                )
        for lesson_style in lesson_styles:
            examples.extend(
                AnswerExample(
                    prompt=prompt,
                    target=" yes.",
                    source=f"{lesson_style}:training_data",
                )
                for prompt in prompt_templates(person, obj, "training_data", lesson_style)
            )

    for fact in grammar.get("unknown_facts", []):
        person = fact["person"]
        obj = fact["object"]
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="unknown:place")
            for prompt in prompt_templates(person, obj, "place")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="bridge:place")
            for prompt in prompt_templates(person, obj, "place", "bridge")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="unknown:color")
            for prompt in prompt_templates(person, obj, "color")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="bridge:color")
            for prompt in prompt_templates(person, obj, "color", "bridge")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" no.", source="unknown:training_data")
            for prompt in prompt_templates(person, obj, "training_data")
        )
    for obj in grammar.get("unknown_owner_objects", []):
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="unknown:owner")
            for prompt in prompt_templates("", obj, "owner")
        )
    for fact in grammar.get("self_facts", []):
        for lesson_style in ("qa", "fact"):
            examples.extend(
                AnswerExample(
                    prompt=prompt,
                    target=f" {fact['answer']}.",
                    source=f"{lesson_style}:self",
                )
                for prompt in self_prompt_templates(fact["slot"], lesson_style)
            )
    for rule in grammar.get("learning_rules", []):
        for lesson_style in ("qa", "fact"):
            examples.extend(
                AnswerExample(
                    prompt=prompt,
                    target=f" {rule['answer']}.",
                    source=f"{lesson_style}:learning",
                )
                for prompt in learning_prompt_templates(rule["slot"], lesson_style)
            )
    if glossary is not None:
        definitions = glossary_definitions(glossary)
        for word in probe_words(glossary):
            for lesson_style in ("qa", "fact"):
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {definitions[word]}.",
                        source=f"{lesson_style}:glossary",
                    )
                    for prompt in glossary_prompt_templates(word, lesson_style)
                )
    return examples


def build_model(examples: list[AnswerExample], seed: int) -> AnswerSoftmax:
    labels = sorted({example.target for example in examples})
    feature_set = set[str]()
    for example in examples:
        feature_set.update(feature_names(example.prompt))
    config = AnswerModelConfig(labels=labels, features=sorted(feature_set), seed=seed)
    return AnswerSoftmax.init_random(config)


def answer_training_pool(examples: list[AnswerExample]) -> list[AnswerExample]:
    pool: list[AnswerExample] = []
    for example in examples:
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


def evaluate_records(model: AnswerSoftmax, records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    total_loss = 0.0
    for record in records:
        prediction = model.predict(record["prompt"])
        loss = model.loss(record["prompt"], record["target"])
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "prediction": prediction,
                "exact_match": prediction == record["target"],
                "target_loss": loss,
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "records": scored,
    }


def load_training_examples(train_text_path: Path, corpus_dir: Path) -> list[AnswerExample]:
    grammar = read_json(corpus_dir / "grammar.json")
    glossary = read_json(corpus_dir / "glossary.json")
    train_text = train_text_path.read_text(encoding="utf-8")
    return examples_from_sources(train_text, grammar, glossary)


def write_lessons(examples: list[AnswerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_training_examples(args.train_text, args.corpus_dir)
    training_pool = answer_training_pool(examples)
    model = build_model(examples, args.seed)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    history_path = args.run / "answer_metrics.jsonl"
    lessons_path = args.run / "answer_lessons.jsonl"
    write_lessons(examples, lessons_path)

    def snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
        result = {
            "step": step,
            "train_loss": train_loss,
            "evals": {
                path.stem: summarize_eval(model, read_jsonl(path))
                for path in DEFAULT_EVALS
            },
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        return result

    baseline = snapshot(0, None)
    running_loss = 0.0
    last_snapshot = baseline
    last_snapshot_step = 0
    pool_order = training_pool[:]
    rng.shuffle(pool_order)
    pool_index = 0
    for step in range(1, args.steps + 1):
        if pool_index == len(pool_order):
            rng.shuffle(pool_order)
            pool_index = 0
        example = pool_order[pool_index]
        pool_index += 1
        running_loss += model.train_step(example, args.learning_rate)
        if step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            last_snapshot = snapshot(step, train_loss)
            last_snapshot_step = step
            print(f"step={step} train_loss={train_loss:.4f}")
            running_loss = 0.0

    if last_snapshot_step != args.steps:
        last_snapshot = snapshot(args.steps, None)

    checkpoint_path = args.run / "answer_model.json"
    model.save(checkpoint_path)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "lessons": str(lessons_path),
        "steps": args.steps,
        "examples": len(examples),
        "training_examples": len(training_pool),
        "labels": len(model.config.labels),
        "features": len(model.config.features),
        "baseline": baseline,
        "final": last_snapshot,
    }
    with (args.run / "answer_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def summarize_eval(model: AnswerSoftmax, records: list[dict[str, Any]]) -> dict[str, Any]:
    result = evaluate_records(model, records)
    failed_records = [record for record in result["records"] if not record["exact_match"]]
    return {
        "count": result["count"],
        "exact": result["exact"],
        "exact_rate": result["exact_rate"],
        "avg_target_loss": result["avg_target_loss"],
        "failed_records": failed_records,
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    model = AnswerSoftmax.load(args.checkpoint)
    result = {
        path.stem: evaluate_records(model, read_jsonl(path))
        for path in DEFAULT_EVALS
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    summary = {
        name: {
            "count": value["count"],
            "exact": value["exact"],
            "exact_rate": value["exact_rate"],
            "avg_target_loss": value["avg_target_loss"],
        }
        for name, value in result.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    train.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    train.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--eval-every", type=int, default=200)
    train.add_argument("--seed", type=int, default=7)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "answer_model.json")
    evaluate.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        train_model(args)
        return 0
    if args.command == "eval":
        eval_model(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
