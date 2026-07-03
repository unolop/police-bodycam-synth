"""Combinatorial prompt generator for bodycam scenarios."""

import random
from typing import Iterator

from . import templates as T


def _pick(pool: list, rng: random.Random):
    return rng.choice(pool)


def generate_poi_prompt(rng: random.Random) -> dict:
    """Generate a single Scenario 1 (Person of Interest) prompt."""
    slots = {
        "style_prefix": T.STYLE_PREFIX,
        "location": _pick(T.LOCATIONS, rng),
        "time_of_day": _pick(T.TIME_OF_DAY, rng),
        "weather": _pick(T.WEATHER, rng),
        "crowd_density": _pick(T.CROWD_DENSITY, rng),
        "subject_description": _pick(T.SUBJECT_DESCRIPTIONS, rng),
        "subject_distance": _pick(T.SUBJECT_DISTANCES, rng),
        "subject_action": _pick(T.SUBJECT_ACTIONS, rng),
        "additional_objects": _pick(T.ADDITIONAL_OBJECTS, rng),
        "lighting_detail": _pick(T.LIGHTING_DETAILS, rng),
    }
    prompt = T.TEMPLATE_POI.format(**slots)
    # Clean up artifacts from empty slots
    prompt = prompt.replace(". .", ".").replace("  ", " ").strip()
    return {
        "scenario": 1,
        "prompt": prompt,
        "negative_prompt": T.NEGATIVE_PROMPT,
        "slots": {k: v for k, v in slots.items() if k != "style_prefix"},
    }


def generate_missing_prompt(rng: random.Random) -> dict:
    """Generate a single Scenario 2 (Missing Person) prompt."""
    target_idx = rng.randint(0, len(T.TARGET_TYPES) - 1)
    desc_idx = target_idx if target_idx < len(T.TARGET_DESCRIPTIONS) else rng.randint(0, len(T.TARGET_DESCRIPTIONS) - 1)

    slots = {
        "style_prefix": T.STYLE_PREFIX,
        "location": _pick(T.LOCATIONS, rng),
        "time_of_day": _pick(T.TIME_OF_DAY, rng),
        "weather": _pick(T.WEATHER, rng),
        "target_type": T.TARGET_TYPES[target_idx],
        "target_description": T.TARGET_DESCRIPTIONS[desc_idx],
        "distance": _pick(T.DISTANCES, rng),
        "target_action": _pick(T.TARGET_ACTIONS, rng),
        "bystanders": _pick(T.BYSTANDERS, rng),
        "lighting_detail": _pick(T.LIGHTING_DETAILS, rng),
    }
    prompt = T.TEMPLATE_MISSING.format(**slots)
    prompt = " ".join(prompt.split())
    return {
        "scenario": 2,
        "prompt": prompt,
        "negative_prompt": T.NEGATIVE_PROMPT,
        "slots": {k: v for k, v in slots.items() if k != "style_prefix"},
    }


def generate_danger_prompt(rng: random.Random) -> dict:
    """Generate a single Scenario 3 (Dangerous Object / Threat) prompt.

    Randomly picks between held-object and abandoned-object sub-scenarios
    with appropriate templates, actions, and objects for each.
    """
    # 60% held object, 40% abandoned object
    is_held = rng.random() < 0.6

    if is_held:
        template = T.TEMPLATE_DANGER_HELD
        threat_action = _pick(T.THREAT_ACTIONS_HELD, rng)
        threat_object = _pick(T.THREAT_OBJECTS_HELD, rng)
    else:
        template = T.TEMPLATE_DANGER_ABANDONED
        threat_action = _pick(T.THREAT_ACTIONS_ABANDONED, rng)
        threat_object = _pick(T.THREAT_OBJECTS_ABANDONED, rng)

    slots = {
        "style_prefix": T.STYLE_PREFIX,
        "location": _pick(T.LOCATIONS, rng),
        "time_of_day": _pick(T.TIME_OF_DAY, rng),
        "weather": _pick(T.WEATHER, rng),
        "threat_subject": _pick(T.THREAT_SUBJECTS, rng),
        "threat_action": threat_action,
        "threat_object": threat_object,
        "bystanders": _pick(T.BYSTANDERS, rng),
        "lighting_detail": _pick(T.LIGHTING_DETAILS, rng),
    }
    prompt = template.format(**slots)
    prompt = prompt.replace(". .", ".").replace("  ", " ").strip()
    return {
        "scenario": 3,
        "prompt": prompt,
        "negative_prompt": T.NEGATIVE_PROMPT,
        "slots": {k: v for k, v in slots.items() if k != "style_prefix"},
    }


_SCENARIO_FNS = {
    1: generate_poi_prompt,
    2: generate_missing_prompt,
    3: generate_danger_prompt,
}


def generate_prompts(scenario: int, num_prompts: int, seed: int = 42) -> Iterator[dict]:
    """Generate N prompts for a given scenario."""
    rng = random.Random(seed)
    gen_fn = _SCENARIO_FNS[scenario]

    for i in range(num_prompts):
        entry = gen_fn(rng)
        entry["prompt_id"] = f"s{scenario}_{i:06d}"
        yield entry
