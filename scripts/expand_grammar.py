"""Seeded corpus-scale generator: expand grammar.json + regenerate the eval spine.

Scales the story-fact corpus from the original 8 facts to a larger object-disjoint
partition (default 40 facts) while preserving every closed-world integrity
invariant the experiment depends on:

  * OBJECT-LEVEL DISJOINTNESS -- the set of objects owned by admitted facts and the
    set of objects used only in withheld facts are disjoint. Withheld novelty comes
    from the PERSON axis (a withheld fact's person may appear elsewhere, but its
    object is owned by no admitted fact), so the object-keyed owner oracle honestly
    abstains on "who has the <withheld object>?" -- the owner axis is not leaked.
  * VALUE-COVERAGE PARITY -- withheld facts reuse colors/relations/containers that
    appear in admitted facts, so abstention is never trivially solvable by an
    out-of-distribution value; the only novelty is the (person, object) binding.
  * PARTITION IDENTITY -- withheld_fact_ids == heldout_probe_ids exactly, and
    qa_lesson_ids == all admitted ids (every admitted fact is the QA-trained
    generation-target population).
  * VOCAB STABILITY -- minted person names use only the existing 35-character set
    (lowercase a..y, no 'z', no apostrophe, no hyphen); no new object words are
    minted. A char assertion catches any stray character before anything is written.

The expanded grammar.json stays the committed source of truth: this generator is
the only thing that writes the 40 facts -- they are never hand-edited. Eval probe
targets are produced by the canonical CorpusResponder oracle parsed from the rebuilt
build/train.txt, so targets match the corpus by construction (never hand-typed).

Run:
    PYTHONPATH=src .venv/bin/python scripts/expand_grammar.py \
        --persons 20 --facts 40 --withheld 20 --seed 7

This rewrites corpus/grammar.json, appends minted names to corpus/glossary.json,
rebuilds build/train.txt + valid.txt via the curriculum, and regenerates
evals/{qa,heldout,unknowns,owner}.jsonl from the oracle.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from corpus_responder import CorpusResponder  # noqa: E402
from curriculum import build_curriculum, write_curriculum  # noqa: E402

CORPUS_DIR = REPO_ROOT / "corpus"
BUILD_DIR = REPO_ROOT / "build"
EVAL_DIR = REPO_ROOT / "evals"

GRAMMAR_PATH = CORPUS_DIR / "grammar.json"
GLOSSARY_PATH = CORPUS_DIR / "glossary.json"
ADMISSIONS_PATH = CORPUS_DIR / "admissions.jsonl"

# The pinned 35-character corpus vocabulary (tests/test_vocab_stability.py),
# excluding the <pad> token. Every word the generator emits must stay inside it;
# in particular no 'z'. (newline / space never appear inside a single word.)
PINNED_CHARS = set(" '" + ",-.:?I_abcdefghijklmnopqrstuvwxy")

# Existing person-name glossary words (tags ['person','name']).
EXISTING_PERSONS = ["mia", "noah", "ava", "leo", "ivy", "omar", "nina", "eli", "sara", "milo", "ruth"]
# Minted single-syllable names: lowercase a..y only, no apostrophe/hyphen, no 'z'.
MINTED_PERSONS = ["tom", "ann", "ben", "kai", "lia", "jon", "sam", "gus", "ada"]

# Pure object-tagged glossary words NOT consumed by corpus/admissions.jsonl (those
# admitted facts own their objects, so they cannot serve as withheld/unknown owners).
# Admitted and withheld objects are drawn from DISJOINT slices of this pool.
ADMITTED_OBJECTS = ["ball", "book", "cup", "key", "map"]
WITHHELD_OBJECTS = ["hat", "lamp", "pen", "plant", "water"]

COLORS = ["red", "blue", "green", "yellow"]
RELATIONS = ["in", "on", "under", "near", "over"]
CONTAINERS = ["box", "table", "bag", "door", "shelf"]

# Objects owned by no fact at all (never a story object, not in admissions) -> the
# owner oracle abstains on them. Container words are safe: no fact owns them.
UNKNOWN_OWNER_OBJECTS = ["box", "table", "door", "shelf", "room", "garden"]


def _assert_chars(word: str) -> None:
    bad = sorted({ch for ch in word if ch not in PINNED_CHARS})
    if bad:
        raise ValueError(
            f"word {word!r} uses characters {bad} outside the pinned 35-char vocab "
            "(a new character silently invalidates every checkpoint)"
        )


def _facts_for(objects: list[str], persons: list[str], per_object: int) -> list[dict]:
    """Pair each object with `per_object` distinct persons (rotating the pool).

    Object reuse is confined within this half; the admitted and withheld object
    SETS stay disjoint, which is the property the owner-axis guard checks. Each
    (person, object) key is distinct.
    """

    facts: list[dict] = []
    index = 0
    for object_index, obj in enumerate(objects):
        chosen = [persons[(object_index * per_object + step) % len(persons)] for step in range(per_object)]
        for person in chosen:
            color = COLORS[index % len(COLORS)]
            relation = RELATIONS[index % len(RELATIONS)]
            container = CONTAINERS[index % len(CONTAINERS)]
            facts.append(
                {
                    "id": f"{person}-{obj}",
                    "person": person,
                    "object": obj,
                    "color": color,
                    "relation": relation,
                    "container": container,
                }
            )
            index += 1
    return facts


def build_grammar(persons_n: int, facts_n: int, withheld_n: int, seed: int) -> dict:
    rng = random.Random(seed)

    persons = (EXISTING_PERSONS + MINTED_PERSONS)[:persons_n]
    if len(persons) < persons_n:
        raise ValueError(f"requested {persons_n} persons but only {len(persons)} available")
    for word in persons + ADMITTED_OBJECTS + WITHHELD_OBJECTS + COLORS + RELATIONS + CONTAINERS + UNKNOWN_OWNER_OBJECTS:
        _assert_chars(word)

    admitted_n = facts_n - withheld_n
    if admitted_n <= 0:
        raise ValueError("admitted fact count must be positive")
    # Object-disjoint halves: distribute each half's facts across its object set,
    # reusing each object across `per_object` distinct persons.
    if admitted_n % len(ADMITTED_OBJECTS) != 0 or withheld_n % len(WITHHELD_OBJECTS) != 0:
        raise ValueError(
            "fact counts must divide evenly across the object sets "
            f"(admitted {admitted_n}/{len(ADMITTED_OBJECTS)}, withheld {withheld_n}/{len(WITHHELD_OBJECTS)})"
        )
    admitted_per = admitted_n // len(ADMITTED_OBJECTS)
    withheld_per = withheld_n // len(WITHHELD_OBJECTS)
    if admitted_per > len(persons) or withheld_per > len(persons):
        raise ValueError("not enough persons to give each object distinct owners")

    admitted_facts = _facts_for(ADMITTED_OBJECTS, persons, admitted_per)
    withheld_facts = _facts_for(WITHHELD_OBJECTS, persons, withheld_per)
    story_facts = admitted_facts + withheld_facts

    pairs = [(f["person"], f["object"]) for f in story_facts]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate (person, object) pairing in story_facts")
    ids = [f["id"] for f in story_facts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate story-fact id")

    admitted_objs = {f["object"] for f in admitted_facts}
    withheld_objs = {f["object"] for f in withheld_facts}
    overlap = admitted_objs & withheld_objs
    if overlap:
        raise ValueError(f"OBJECT-LEVEL DISJOINTNESS violated: shared objects {sorted(overlap)}")

    # Value-coverage parity: every withheld color/relation/container also appears in
    # the admitted half (novelty lives only in the person-object binding).
    for axis in ("color", "relation", "container"):
        withheld_values = {f[axis] for f in withheld_facts}
        admitted_values = {f[axis] for f in admitted_facts}
        if not withheld_values <= admitted_values:
            raise ValueError(
                f"value-coverage parity violated on {axis}: withheld {sorted(withheld_values)} "
                f"not a subset of admitted {sorted(admitted_values)}"
            )

    admitted_ids = [f["id"] for f in admitted_facts]
    withheld_ids = [f["id"] for f in withheld_facts]

    # unknown_facts: (person, object) pairs that are neither admitted nor withheld
    # story pairs and not admissions pairs -> true out-of-corpus bindings.
    admissions_pairs = set()
    if ADMISSIONS_PATH.exists():
        for line in ADMISSIONS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                admissions_pairs.add((record["person"], record["object"]))
    forbidden = set(pairs) | admissions_pairs
    all_objects = sorted(admitted_objs | withheld_objs)
    candidate_unknowns = [
        (person, obj) for person in persons for obj in all_objects if (person, obj) not in forbidden
    ]
    rng.shuffle(candidate_unknowns)
    unknown_pairs = candidate_unknowns[:10]
    unknown_facts = [
        {"id": f"unknown-{person}-{obj}", "person": person, "object": obj}
        for person, obj in unknown_pairs
    ]

    # Preserve the existing non-fact grammar sections verbatim.
    existing = json.loads(GRAMMAR_PATH.read_text(encoding="utf-8"))
    grammar = dict(existing)
    grammar["story_facts"] = story_facts
    grammar["qa_lesson_ids"] = admitted_ids
    grammar["heldout_probe_ids"] = withheld_ids
    grammar["withheld_fact_ids"] = withheld_ids
    grammar["unknown_facts"] = unknown_facts
    grammar["unknown_owner_objects"] = list(UNKNOWN_OWNER_OBJECTS)
    return grammar


def update_glossary(persons_n: int) -> list[str]:
    """Append any minted person names (in use) to the glossary, tagged person/name."""

    glossary = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    existing_words = {entry["word"] for entry in glossary["entries"]}
    persons = (EXISTING_PERSONS + MINTED_PERSONS)[:persons_n]
    added: list[str] = []
    for name in persons:
        if name in existing_words:
            continue
        _assert_chars(name)
        glossary["entries"].append(
            {"word": name, "part": "name", "definition": "a person", "tags": ["person", "name"]}
        )
        added.append(name)
    if added:
        GLOSSARY_PATH.write_text(
            json.dumps(glossary, indent=2) + "\n", encoding="utf-8"
        )
    return added


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def regenerate_evals(grammar: dict, oracle: CorpusResponder) -> dict[str, int]:
    """Emit qa/heldout/unknowns/owner probes; every target comes from the oracle."""

    withheld_ids = set(grammar["withheld_fact_ids"])
    admitted = [f for f in grammar["story_facts"] if f["id"] not in withheld_ids]
    withheld = [f for f in grammar["story_facts"] if f["id"] in withheld_ids]

    def place_prompt(person: str, obj: str) -> str:
        return f"question: where is {person}'s {obj}?\nanswer:"

    def color_prompt(person: str, obj: str) -> str:
        return f"question: what color is {person}'s {obj}?\nanswer:"

    def owner_prompt(obj: str) -> str:
        return f"question: who has the {obj}?\nanswer:"

    # Paraphrase (bridge-form) prompts. These reuse the augmentation's bridge forms,
    # so to stay leakage-safe they MUST cover only pairs that augment_unknown_examples
    # excludes -- i.e. withheld story pairs and declared unknown_facts pairs (never a
    # generic out-of-corpus pair, which augmentation would also emit).
    def place_para(person: str, obj: str) -> str:
        return f"tell me the place of {person} {obj}\nanswer:"

    def color_para(person: str, obj: str) -> str:
        return f"which color belongs to {person} {obj}\nanswer:"

    qa: list[dict] = []
    for fact in admitted:
        person, obj = fact["person"], fact["object"]
        qa.append({"id": f"known-{person}-{obj}-place", "prompt": place_prompt(person, obj),
                   "target": oracle.answer_prompt(place_prompt(person, obj))})
        qa.append({"id": f"known-{person}-{obj}-color", "prompt": color_prompt(person, obj),
                   "target": oracle.answer_prompt(color_prompt(person, obj))})

    heldout: list[dict] = []
    for fact in withheld:
        person, obj = fact["person"], fact["object"]
        heldout.append({"id": f"heldout-{person}-{obj}-place", "prompt": place_prompt(person, obj),
                        "target": oracle.answer_prompt(place_prompt(person, obj))})
        heldout.append({"id": f"heldout-{person}-{obj}-color", "prompt": color_prompt(person, obj),
                        "target": oracle.answer_prompt(color_prompt(person, obj))})

    unknowns: list[dict] = []
    for fact in grammar["unknown_facts"]:
        person, obj = fact["person"], fact["object"]
        unknowns.append({"id": f"unknown-{person}-{obj}", "prompt": place_prompt(person, obj),
                         "target": oracle.answer_prompt(place_prompt(person, obj))})

    owner: list[dict] = []
    seen_admitted_objects: set[str] = set()
    for fact in admitted:
        obj = fact["object"]
        if obj in seen_admitted_objects:
            continue
        seen_admitted_objects.add(obj)
        owner.append({"id": f"owner-known-{obj}", "prompt": owner_prompt(obj),
                      "target": oracle.answer_prompt(owner_prompt(obj))})
    for obj in sorted({f["object"] for f in withheld}):
        owner.append({"id": f"owner-heldout-{obj}", "prompt": owner_prompt(obj),
                      "target": oracle.answer_prompt(owner_prompt(obj))})
    for obj in grammar["unknown_owner_objects"]:
        owner.append({"id": f"owner-unknown-{obj}", "prompt": owner_prompt(obj),
                      "target": oracle.answer_prompt(owner_prompt(obj))})

    # Paraphrase probes over withheld + declared-unknown pairs only (augmentation
    # excludes both, so these bridge-form prompts never collide with a negative).
    paraphrases: list[dict] = []
    for fact in withheld:
        person, obj = fact["person"], fact["object"]
        paraphrases.append({"id": f"para-{person}-{obj}-place", "prompt": place_para(person, obj),
                            "target": oracle.answer_prompt(place_para(person, obj))})
        paraphrases.append({"id": f"para-{person}-{obj}-color", "prompt": color_para(person, obj),
                            "target": oracle.answer_prompt(color_para(person, obj))})
    for fact in grammar["unknown_facts"]:
        person, obj = fact["person"], fact["object"]
        paraphrases.append({"id": f"para-{person}-{obj}-place", "prompt": place_para(person, obj),
                            "target": oracle.answer_prompt(place_para(person, obj))})

    _write_jsonl(EVAL_DIR / "qa.jsonl", qa)
    _write_jsonl(EVAL_DIR / "heldout.jsonl", heldout)
    _write_jsonl(EVAL_DIR / "unknowns.jsonl", unknowns)
    _write_jsonl(EVAL_DIR / "owner.jsonl", owner)
    _write_jsonl(EVAL_DIR / "paraphrases.jsonl", paraphrases)
    return {"qa": len(qa), "heldout": len(heldout), "unknowns": len(unknowns),
            "owner": len(owner), "paraphrases": len(paraphrases)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persons", type=int, default=20)
    parser.add_argument("--facts", type=int, default=40)
    parser.add_argument("--withheld", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    grammar = build_grammar(args.persons, args.facts, args.withheld, args.seed)
    GRAMMAR_PATH.write_text(json.dumps(grammar, indent=2) + "\n", encoding="utf-8")
    added = update_glossary(args.persons)

    # Rebuild the curriculum from the freshly written corpus, then build the oracle
    # from that train text so eval targets match the corpus by construction.
    curriculum = build_curriculum(CORPUS_DIR, seed=args.seed)
    write_curriculum(curriculum, BUILD_DIR)
    oracle = CorpusResponder.train_from_text(curriculum.train_text)
    counts = regenerate_evals(grammar, oracle)

    admitted = [f for f in grammar["story_facts"] if f["id"] not in set(grammar["withheld_fact_ids"])]
    withheld = [f for f in grammar["story_facts"] if f["id"] in set(grammar["withheld_fact_ids"])]
    print(f"wrote {GRAMMAR_PATH} ({len(grammar['story_facts'])} story facts)")
    print(f"  admitted={len(admitted)} withheld={len(withheld)} "
          f"unknown_facts={len(grammar['unknown_facts'])} "
          f"unknown_owner_objects={len(grammar['unknown_owner_objects'])}")
    print(f"  admitted objects={sorted({f['object'] for f in admitted})}")
    print(f"  withheld objects={sorted({f['object'] for f in withheld})}")
    print(f"  minted names appended to glossary: {added}")
    print(f"wrote {BUILD_DIR / 'train.txt'} ({len(curriculum.train_text)} chars)")
    print(f"regenerated evals: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
