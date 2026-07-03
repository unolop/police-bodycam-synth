"""Scene-sequential prompt generator for video-like frame sequences.

Generates groups of prompts that share the same scene context (location, people,
lighting) but vary slightly per frame (pose, position, action phase) to simulate
consecutive bodycam footage.

Each scene = N frames with:
- Same location, time_of_day, weather, lighting
- Same subject identity (appearance description)
- Slight per-frame variation in action/pose/position
- Consistent bystander context

Output structure:
  scene_001/frame_000, scene_001/frame_001, ..., scene_001/frame_009
  scene_002/frame_000, ...
"""

import random
from typing import Iterator

from . import templates as T


# Per-frame action progressions (ordered sequences that simulate movement)
POI_ACTION_SEQUENCES = [
    ["walking toward the officer",
     "slowing down near the officer",
     "walking through the crowd at close range",
     "pausing to look around",
     "turning slightly to the right",
     "continuing to walk past the officer",
     "pausing near the curb",
     "glancing back briefly",
     "moving through nearby pedestrians",
     "walking past the officer"],
    ["standing near a parked vehicle",
     "leaning against the vehicle",
     "looking at a phone",
     "putting the phone away",
     "looking around",
     "stepping away from the vehicle",
     "walking along the sidewalk nearby",
     "slowing down in front of the officer",
     "pausing near the officer",
     "walking past the officer"],
    ["standing beside a bench",
     "looking at a phone while standing beside the bench",
     "stepping forward from the bench",
     "adjusting clothing",
     "looking around",
     "starting to walk",
     "walking along the path nearby",
     "slowing down near the officer",
     "turning at the intersection",
     "walking past the officer"],
    ["standing at a corner looking at a phone",
     "pacing slowly",
     "talking on a phone while pacing",
     "stopping and looking around",
     "putting the phone in pocket",
     "walking toward the officer",
     "entering a building doorway",
     "pausing at the entrance",
     "looking back briefly",
     "going through the door"],
    ["walking through the crowd",
     "bumping into a pedestrian",
     "continuing to walk",
     "pausing near a shop front",
     "looking into the shop window",
     "walking ahead again",
     "pausing near the curb",
     "waiting at the curb",
     "turning back toward the officer",
     "continuing along the sidewalk nearby"],
]

MISSING_ACTION_SEQUENCES = [
    ["standing still and looking confused",
     "turning around slowly",
     "looking in different directions",
     "starting to walk slowly",
     "walking aimlessly",
     "stopping and looking lost",
     "sitting down on the ground",
     "sitting on the ground looking around",
     "starting to cry",
     "sitting on the ground crying"],
    ["wandering aimlessly",
     "walking slowly along the road edge",
     "stumbling slightly",
     "stopping and looking confused",
     "turning around",
     "walking in a different direction",
     "approaching a crosswalk",
     "trying to cross the street alone",
     "standing near a busy intersection",
     "looking around with confusion"],
    ["walking alone",
     "walking slowly and looking lost",
     "stopping near a bench",
     "sitting on a bench alone",
     "looking down at the ground",
     "standing up slowly",
     "walking in a circle",
     "stopping again",
     "looking around with confusion",
     "wandering aimlessly"],
]

DANGER_ACTION_SEQUENCES = [
    ["pacing back and forth near a building entrance",
     "reaching into a bag",
     "pulling something out of the bag",
     "holding something behind their back",
     "turning to face the street",
     "waving an object threateningly",
     "moving aggressively toward pedestrians",
     "shouting and gesturing",
     "continuing to advance",
     "people scattering away"],
    ["arguing loudly and gesturing",
     "getting more aggressive",
     "blocking a doorway and confronting someone",
     "pushing someone back",
     "reaching for something",
     "holding something up threateningly",
     "standing over a person on the ground",
     "turning to look around",
     "moving toward the street",
     "running away from the scene"],
    ["standing near a building entrance",
     "crouching near a parked car",
     "placing something on the ground",
     "stepping back from the object",
     "looking around nervously",
     "walking away quickly",
     "turning a corner",
     "walking briskly away",
     "glancing back at the object",
     "disappearing from view"],
]


def _pick(pool: list, rng: random.Random):
    return rng.choice(pool)


def _sample_action_sequence(sequences: list, frames_needed: int, rng: random.Random) -> list[str]:
    """Pick an action sequence and sample/interpolate to match frame count."""
    seq = _pick(sequences, rng)
    if len(seq) == frames_needed:
        return seq
    elif len(seq) > frames_needed:
        # Evenly sample from the sequence
        indices = [int(i * (len(seq) - 1) / (frames_needed - 1)) for i in range(frames_needed)]
        return [seq[i] for i in indices]
    else:
        # Repeat last actions to fill
        result = seq[:]
        while len(result) < frames_needed:
            result.append(seq[-1])
        return result[:frames_needed]


def generate_scene_prompts(
    scenario: int,
    num_scenes: int,
    frames_per_scene: int,
    seed: int = 42,
) -> Iterator[dict]:
    """Generate scene-sequential prompts.

    Yields prompt dicts with extra fields:
      - scene_id: str (e.g. "s1_scene_000001")
      - frame_index: int (0-based within scene)
      - scene_seed: int (shared seed for the scene)
    """
    rng = random.Random(seed)

    for scene_idx in range(num_scenes):
        scene_id = f"s{scenario}_scene_{scene_idx:06d}"
        scene_seed = rng.randint(0, 2**31)

        if scenario == 1:
            yield from _generate_poi_scene(rng, scene_id, scene_idx, scene_seed, frames_per_scene)
        elif scenario == 2:
            yield from _generate_missing_scene(rng, scene_id, scene_idx, scene_seed, frames_per_scene)
        elif scenario == 3:
            yield from _generate_danger_scene(rng, scene_id, scene_idx, scene_seed, frames_per_scene)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")


def _generate_poi_scene(
    rng: random.Random, scene_id: str, scene_idx: int, scene_seed: int, num_frames: int,
) -> Iterator[dict]:
    """Generate a POI scene sequence — same person, location, progressive action."""
    # Scene-level constants (shared across all frames)
    location = _pick(T.LOCATIONS, rng)
    time_of_day = _pick(T.TIME_OF_DAY, rng)
    weather = _pick(T.WEATHER, rng)
    crowd_density = _pick(T.CROWD_DENSITY, rng)
    subject_desc = _pick(T.SUBJECT_DESCRIPTIONS, rng)
    subject_distance = _pick(T.SUBJECT_DISTANCES, rng)
    additional_objects = _pick(T.ADDITIONAL_OBJECTS, rng)
    lighting = _pick(T.LIGHTING_DETAILS, rng)

    # Per-frame: progressive action sequence
    actions = _sample_action_sequence(POI_ACTION_SEQUENCES, num_frames, rng)

    for frame_idx, action in enumerate(actions):
        slots = {
            "style_prefix": T.STYLE_PREFIX,
            "location": location,
            "time_of_day": time_of_day,
            "weather": weather,
            "crowd_density": crowd_density,
            "subject_description": subject_desc,
            "subject_distance": subject_distance,
            "subject_action": action,
            "additional_objects": additional_objects,
            "lighting_detail": lighting,
        }
        prompt = T.TEMPLATE_POI.format(**slots)
        prompt = prompt.replace(". .", ".").replace("  ", " ").strip()

        yield {
            "scenario": 1,
            "prompt": prompt,
            "negative_prompt": T.NEGATIVE_PROMPT,
            "slots": {k: v for k, v in slots.items() if k != "style_prefix"},
            "prompt_id": f"{scene_id}_f{frame_idx:03d}",
            "scene_id": scene_id,
            "frame_index": frame_idx,
            "scene_seed": scene_seed,
        }


def _generate_missing_scene(
    rng: random.Random, scene_id: str, scene_idx: int, scene_seed: int, num_frames: int,
) -> Iterator[dict]:
    """Generate a missing person scene sequence."""
    location = _pick(T.LOCATIONS, rng)
    time_of_day = _pick(T.TIME_OF_DAY, rng)
    weather = _pick(T.WEATHER, rng)
    bystanders = _pick(T.BYSTANDERS, rng)
    lighting = _pick(T.LIGHTING_DETAILS, rng)

    target_idx = rng.randint(0, len(T.TARGET_TYPES) - 1)
    desc_idx = target_idx if target_idx < len(T.TARGET_DESCRIPTIONS) else rng.randint(0, len(T.TARGET_DESCRIPTIONS) - 1)
    target_desc = T.TARGET_DESCRIPTIONS[desc_idx]

    # Distance gets closer over frames (approaching the missing person)
    distances = [
        "about 20 meters", "about 15 meters", "about 15 meters",
        "about 10 meters", "about 10 meters", "about 10 meters",
        "about 5 meters", "about 5 meters", "about 5 meters", "about 3 meters",
    ]
    if num_frames <= len(distances):
        frame_distances = distances[:num_frames]
    else:
        frame_distances = distances + [distances[-1]] * (num_frames - len(distances))

    actions = _sample_action_sequence(MISSING_ACTION_SEQUENCES, num_frames, rng)

    for frame_idx, (action, distance) in enumerate(zip(actions, frame_distances)):
        slots = {
            "style_prefix": T.STYLE_PREFIX,
            "location": location,
            "time_of_day": time_of_day,
            "weather": weather,
            "target_type": T.TARGET_TYPES[target_idx],
            "target_description": target_desc,
            "distance": distance,
            "target_action": action,
            "bystanders": bystanders,
            "lighting_detail": lighting,
        }
        prompt = T.TEMPLATE_MISSING.format(**slots)
        prompt = " ".join(prompt.split())

        yield {
            "scenario": 2,
            "prompt": prompt,
            "negative_prompt": T.NEGATIVE_PROMPT,
            "slots": {k: v for k, v in slots.items() if k != "style_prefix"},
            "prompt_id": f"{scene_id}_f{frame_idx:03d}",
            "scene_id": scene_id,
            "frame_index": frame_idx,
            "scene_seed": scene_seed,
        }


def _generate_danger_scene(
    rng: random.Random, scene_id: str, scene_idx: int, scene_seed: int, num_frames: int,
) -> Iterator[dict]:
    """Generate a dangerous situation scene sequence."""
    location = _pick(T.LOCATIONS, rng)
    time_of_day = _pick(T.TIME_OF_DAY, rng)
    weather = _pick(T.WEATHER, rng)
    threat_subject = _pick(T.THREAT_SUBJECTS, rng)
    threat_object = _pick(T.THREAT_OBJECTS, rng)
    bystanders = _pick(T.BYSTANDERS, rng)
    lighting = _pick(T.LIGHTING_DETAILS, rng)

    actions = _sample_action_sequence(DANGER_ACTION_SEQUENCES, num_frames, rng)

    for frame_idx, action in enumerate(actions):
        slots = {
            "style_prefix": T.STYLE_PREFIX,
            "location": location,
            "time_of_day": time_of_day,
            "weather": weather,
            "threat_subject": threat_subject,
            "threat_action": action,
            "threat_object": threat_object,
            "bystanders": bystanders,
            "lighting_detail": lighting,
        }
        prompt = T.TEMPLATE_DANGER.format(**slots)
        prompt = prompt.replace(". .", ".").replace("  ", " ").strip()

        yield {
            "scenario": 3,
            "prompt": prompt,
            "negative_prompt": T.NEGATIVE_PROMPT,
            "slots": {k: v for k, v in slots.items() if k != "style_prefix"},
            "prompt_id": f"{scene_id}_f{frame_idx:03d}",
            "scene_id": scene_id,
            "frame_index": frame_idx,
            "scene_seed": scene_seed,
        }
