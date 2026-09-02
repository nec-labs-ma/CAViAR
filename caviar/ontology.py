"""Jurisdiction-agnostic traffic-rule violation ontology + deterministic mapper.

Maps each free-text Violation-Identification answer to ONE primary rule family
using an ordered (priority) keyword lexicon. Fully reproducible; no LLM in the loop.
"""

from __future__ import annotations

import argparse
import sys

# ORDER = matching priority (first match wins).
ONTOLOGY = [
    ("SG", "Signal / sign violation", [
        "red light", "traffic light", "traffic signal", "stop sign",
        "ran the", "ran a red", "running the light", "running a red",
        "disobey", "ignored the signal", "ignored the light"]),
    ("RW", "Failure to yield / right-of-way", [
        "yield", "right of way", "right-of-way", "give way", "giving way",
        "priority", "main road", "side road", "right of passage",
        "failed to give", "stop and look", "before entering the intersection",
        "before entering an intersection"]),
    ("FD", "Unsafe following distance / rear-end", [
        "following distance", "safe distance", "safe following", "too close",
        "tailgat", "stopping distance", "brake in time", "rear-end",
        "rear end", "distance from the vehicle", "keep a safe distance",
        "keeping a safe distance", "maintain a safe distance",
        "safe braking distance"]),
    ("CT", "Loss of vehicle control", [
        "lose control", "lost control", "loss of control", "uncontrolled",
        "skid", "malfunction", "tilted", "control of the vehicle",
        "control of their vehicle", "spun", "rolled over"]),
    ("LC", "Improper lane change / merging", [
        "lane change", "change lane", "changing lane", "change lanes",
        "changed lane", "merg", "cut in", "cut off", "cutting",
        "lane discipline", "improper lane", "weav"]),
    ("OT", "Improper overtaking / passing", [
        "overtak", "overtook", "overtaking"]),
    ("TU", "Improper turn / U-turn / reversing", [
        "u-turn", "u turn", "turning", "while turning", "made a turn",
        "left turn", "right turn", "revers", "backing up", "back up"]),
    ("SP", "Unsafe speed / reckless driving", [
        "speed", "too fast", "reckless", "excessive", "overspeed",
        "slow down", "high velocity", "drove fast"]),
    ("ST", "Sudden stop / improper stopping or parking", [
        "suddenly stop", "sudden stop", "abrupt", "stopped without",
        "sudden brak", "hard brak", "emergency brak", "parking", "parked",
        "stopped in the middle", "stopping in the"]),
    ("PD", "Pedestrian / non-motorized / illegal crossing", [
        "pedestrian", "bicycle", "cyclist", "non-motor", "nonmotor",
        "crossed the road", "crossing the road", "jaywalk", "crosswalk",
        "illegally cross", "cross the road"]),
    ("AT", "Inattentive / improper observation", [
        "distract", "attention", "inattentive", "observe", "observation",
        "failed to notice", "lookout", "look out", "careless",
        "confirm the safety", "confirm safety", "check the rear",
        "checking the rear", "aware of", "vigilan", "pay attention"]),
]

CODE2NAME = {c: n for c, n, _ in ONTOLOGY}

UNSPEC_TOKENS = {"n/a", "na", "none", "unknown", "not applicable", "-", ""}

NEG_GUARDS = {
    "SG": [
        "without traffic", "no traffic light", "no traffic signal",
        "without a traffic", "no signal", "absence of traffic",
    ],
}


def norm(s: str) -> str:
    return (s or "").strip().lower()


def is_unspecified(a: str) -> bool:
    n = norm(a)
    if n in UNSPEC_TOKENS:
        return True
    if len(n) < 25:
        return True
    return False


def fires(code: str, kws: list[str], n: str) -> bool:
    if not any(k in n for k in kws):
        return False
    for bad in NEG_GUARDS.get(code, []):
        if bad in n:
            return False
    return True


def all_matches(a: str) -> list[str]:
    """Return family codes whose lexicon fires (for overlap diagnostics)."""
    n = norm(a)
    return [code for code, _, kws in ONTOLOGY if fires(code, kws, n)]


def classify(a: str) -> tuple[str, str]:
    """Return (code, bucket) where bucket in {family, OTHER, UNSPEC}."""
    if is_unspecified(a):
        return ("UNSPEC", "UNSPEC")
    n = norm(a)
    for code, _, kws in ONTOLOGY:
        if fires(code, kws, n):
            return (code, "family")
    return ("OTHER", "OTHER")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map violation text to a CAViAR rule family")
    parser.add_argument("--text", type=str, help="Single free-text violation answer")
    parser.add_argument("--file", type=str, help="Text file with one answer per line")
    args = parser.parse_args(argv)

    texts: list[str] = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    if not texts:
        parser.print_help()
        return 1

    for t in texts:
        code, bucket = classify(t)
        name = CODE2NAME.get(code, bucket)
        matches = all_matches(t)
        print(f"{code}\t{name}\tmatches={matches}\ttext={t[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
