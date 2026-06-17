"""Admission probe record generation."""

from __future__ import annotations

from typing import Any


def admission_probe_records(admissions: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for fact in admissions:
        person = fact["person"]
        obj = fact["object"]
        slug = f"{person}-{obj}"
        place = f"{fact['relation']} the {fact['container']}"
        records.extend(
            [
                {
                    "id": f"admission-place-{slug}",
                    "prompt": f"question: where is {person}'s {obj}?\nanswer:",
                    "target": f" {place}.",
                },
                {
                    "id": f"admission-color-{slug}",
                    "prompt": f"question: what color is {person}'s {obj}?\nanswer:",
                    "target": f" {fact['color']}.",
                },
                {
                    "id": f"admission-owner-{slug}",
                    "prompt": f"question: who has the {obj}?\nanswer:",
                    "target": f" {person}.",
                },
                {
                    "id": f"admission-status-{slug}",
                    "prompt": f"question: is {person}'s {obj} part of your training data?\nanswer:",
                    "target": " yes.",
                },
            ]
        )
    return records


def admission_paraphrase_probe_records(
    admissions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for fact in admissions:
        person = fact["person"]
        obj = fact["object"]
        slug = f"{person}-{obj}"
        place = f"{fact['relation']} the {fact['container']}"
        records.extend(
            [
                {
                    "id": f"admission-para-place-tell-{slug}",
                    "prompt": f"tell me the place of {person} {obj}\nanswer:",
                    "target": f" {place}.",
                },
                {
                    "id": f"admission-para-place-ask-{slug}",
                    "prompt": f"ask: place for {person} {obj}\nanswer:",
                    "target": f" {place}.",
                },
                {
                    "id": f"admission-para-color-belongs-{slug}",
                    "prompt": f"which color belongs to {person} {obj}\nanswer:",
                    "target": f" {fact['color']}.",
                },
                {
                    "id": f"admission-para-color-ask-{slug}",
                    "prompt": f"ask: color for {person} {obj}\nanswer:",
                    "target": f" {fact['color']}.",
                },
                {
                    "id": f"admission-para-owner-belongs-{slug}",
                    "prompt": f"which person has {obj}\nanswer:",
                    "target": f" {person}.",
                },
                {
                    "id": f"admission-para-owner-ask-{slug}",
                    "prompt": f"ask: owner for {obj}\nanswer:",
                    "target": f" {person}.",
                },
                {
                    "id": f"admission-para-status-tag-{slug}",
                    "prompt": f"training data: {person} {obj}\nanswer:",
                    "target": " yes.",
                },
            ]
        )
    return records
