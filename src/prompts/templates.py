"""Bodycam prompt templates with variable slots.

Key design: describe what the CAMERA SEES, not the officer wearing it.
Mentioning 'police officer' as a visible subject triggers 3rd-person images.
Subject must FACE the camera — in real bodycam scenarios, the officer
approaches/confronts subjects who face toward the officer.
"""

STYLE_PREFIX = (
    "Raw bodycam footage frame, chest-mounted wide angle camera, "
    "flat colors, compressed dynamic range, slight barrel distortion"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, sketch, "
    "watermark, deformed, disfigured, bad anatomy, extra limbs, "
    "3d render, cgi, digital art, concept art, "
    "studio lighting, professional photography, DSLR, bokeh, shallow depth of field, "
    "portrait, selfie, close-up headshot, "
    "cinematic, movie still, color grading, dramatic lighting, "
    "police uniform, tactical gear, badge visible, "
    "bird's-eye view, drone shot, surveillance camera, CCTV, "
    "back of head, person facing away, person from behind, rear view"
)

# Scenario 1: Person of Interest Identification
TEMPLATE_POI = (
    "{style_prefix}. "
    "A {subject_description} facing the camera {subject_distance} ahead, "
    "{subject_action}. "
    "The person's face and upper body are clearly visible, front view. "
    "{location} at {time_of_day}, {weather}. "
    "{crowd_density} crowd. "
    "{additional_objects}. "
    "{lighting_detail}"
)

# Scenario 2: Missing Person Detection
TEMPLATE_MISSING = (
    "{style_prefix}. "
    "{target_description} facing toward the camera {distance} ahead, "
    "{target_action}. "
    "The person's face is clearly visible from the front. "
    "{location} at {time_of_day}, {weather}. "
    "{bystanders}. "
    "{lighting_detail}"
)

# Scenario 3: Dangerous Object / Threat Detection
# Split into two sub-templates: person with held object, or abandoned object in scene
TEMPLATE_DANGER_HELD = (
    "{style_prefix}. "
    "A {threat_subject} facing the camera about 3 meters ahead, "
    "{threat_action}. "
    "The person's face and hands are clearly visible, front view. "
    "{threat_object}. "
    "{location} at {time_of_day}, {weather}. "
    "{bystanders}. "
    "{lighting_detail}"
)

TEMPLATE_DANGER_ABANDONED = (
    "{style_prefix}. "
    "{threat_object}. "
    "A {threat_subject} is standing {threat_action} about 4 meters ahead, "
    "facing the camera. The person's face is clearly visible, front view. "
    "{location} at {time_of_day}, {weather}. "
    "{bystanders}. "
    "{lighting_detail}"
)

# Slot value pools
LOCATIONS = [
    "a neon-lit entertainment district with bars and restaurants",
    "a busy subway station platform with fluorescent lights",
    "an urban park with trees and benches",
    "a shopping mall entrance with glass doors",
    "a narrow residential alley with parked cars",
    "a convenience store parking lot",
    "a bus stop on a main road",
    "a school zone crosswalk area",
    "a crowded night market with food stalls and lanterns",
    "a public plaza with a fountain",
    "a dimly lit underpass tunnel",
    "a train station concourse",
    "a hospital entrance area",
    "a construction site perimeter",
    "a large parking garage with concrete pillars",
    "a riverside walking path",
    "a busy intersection with traffic lights",
    "an outdoor cafe terrace",
    "a traditional market street with vendors",
    "a commercial district sidewalk with shop fronts",
]

TIME_OF_DAY = [
    "daytime with bright sunlight",
    "dusk with orange sky",
    "nighttime with streetlights",
    "early morning with soft light",
    "overcast midday",
    "late afternoon with long shadows",
]

WEATHER = [
    "clear", "rainy with wet pavement reflections", "overcast",
    "foggy with reduced visibility", "drizzling lightly",
    "clear and cold with visible breath",
]

CROWD_DENSITY = [
    "A sparse", "A moderate", "A moderate", "A dense",
]

SUBJECT_DISTANCES = [
    "about 1 meter",
    "about 1.5 meters",
    "about 2 meters",
    "about 3 meters",
]

# Subject descriptions for POI scenario
SUBJECT_DESCRIPTIONS = [
    "man in a dark hoodie and jeans",
    "woman wearing a red jacket and carrying a large bag",
    "young man in a baseball cap and tracksuit",
    "middle-aged man in a business suit",
    "person in a black mask and dark clothing",
    "man with a shaved head wearing a leather jacket",
    "woman in a long coat with a scarf",
    "young person in streetwear with a backpack",
    "tall man in a gray coat",
    "person in work overalls and boots",
    "man wearing sunglasses and a beanie",
    "woman in athletic wear carrying a duffel bag",
]

SUBJECT_ACTIONS = [
    "looking directly at the camera with a neutral expression",
    "turning toward the camera while standing still",
    "looking up from a phone toward the camera",
    "facing forward with hands at sides",
    "gesturing while talking, facing the camera",
    "standing still and staring ahead",
    "walking toward the camera",
    "stopping and turning to face the camera",
    "looking to the side with face in three-quarter view",
    "standing with arms crossed, facing forward",
]

ADDITIONAL_OBJECTS = [
    "A parked patrol vehicle with flashing lights is in the background",
    "Several bicycles are locked to a rack nearby",
    "A delivery truck is double-parked on the street",
    "Street vendors with small carts line the sidewalk",
    "Traffic cones and construction barriers are nearby",
    "A motorcycle is parked on the sidewalk",
    "Shopping bags and litter are scattered on the ground",
    "A taxi is stopping to pick up passengers",
    "",  # no additional objects
]

# Missing person scenario slots
TARGET_TYPES = [
    "child aged 5-8",
    "child aged 8-12",
    "elderly man in his 70s",
    "elderly woman in her 80s",
    "elderly person with dementia",
    "teenage girl",
]

TARGET_DESCRIPTIONS = [
    "A very small 6-year-old child in a yellow raincoat, knee-height",
    "A small 10-year-old boy in a school uniform with a backpack",
    "A frail elderly man in his 70s wearing pajamas and slippers, white hair",
    "An elderly woman in her 80s with gray hair in a floral dress, stooped posture",
    "A confused elderly person in a hospital gown, thin and frail",
    "A 12-year-old girl in a pink jacket and jeans, small stature",
    "An elderly man with a walking cane and flat cap, wrinkled face",
    "A very small 5-year-old child in a blue t-shirt and shorts, toddler height",
    "An elderly woman hunched over a walking frame, gray hair",
    "A small 8-year-old boy in a red hoodie, child height",
]

TARGET_ACTIONS = [
    "looking up toward the camera with a lost expression",
    "sitting on the ground facing the camera and crying",
    "standing still facing forward looking confused",
    "looking toward the camera with a frightened expression",
    "standing alone facing the camera",
    "turning toward the camera with tearful eyes",
    "standing still near a bench facing forward",
    "looking directly ahead with a blank expression",
]

DISTANCES = [
    "about 2 meters", "about 3 meters", "about 4 meters",
    "about 5 meters", "about 6 meters",
]

BYSTANDERS = [
    "A few bystanders are nearby but not interacting",
    "No one else is nearby",
    "Several pedestrians walk past without noticing",
    "A couple stops to look concerned",
    "Other pedestrians are in the background",
]

LIGHTING_DETAILS = [
    "Warm artificial streetlights cast yellow tones",
    "Harsh fluorescent lighting from nearby shops",
    "Natural daylight with some shadows",
    "Dim ambient lighting with neon sign reflections",
    "Bright overhead lights illuminate the area",
    "Backlit scene with the sun behind the subjects",
    "",  # no extra lighting detail
]

# Scenario 3: Dangerous situation / threat detection slots
THREAT_SUBJECTS = [
    "man in a dark hoodie",
    "person in all black clothing",
    "man in a windbreaker jacket",
    "young man in a tracksuit",
    "person in a heavy winter coat",
    "man in a cap and sunglasses",
    "woman in a dark parka",
    "person in a gray hoodie and jeans",
]

# Actions for HELD objects — person interacting with/holding something
THREAT_ACTIONS_HELD = [
    "holding an object in their right hand",
    "gripping an object with both hands",
    "standing still with an object in hand",
    "walking toward the camera while holding an object",
    "standing with arms raised, holding an object",
    "gesturing with an object in one hand",
    "standing in an open stance with an object visible",
    "stepping forward with an object at their side",
]

# Actions for ABANDONED objects — person standing near suspicious item
THREAT_ACTIONS_ABANDONED = [
    "near a suspicious item on the ground",
    "a few steps away from an abandoned item",
    "looking around nervously near an item on the ground",
    "walking away from an item left on the sidewalk",
    "standing next to an unattended item",
    "near a conspicuous object on the pavement",
]

# Held objects — described factually as visible props, no intent language
THREAT_OBJECTS_HELD = [
    "A large kitchen knife is visible in their right hand",
    "A metal baseball bat is held in both hands",
    "A long metal pipe is gripped in both hands",
    "A broken glass bottle is visible in their hand",
    "A crowbar is held at their side",
    "A claw hammer is visible in their right hand",
    "A can of spray paint is in their hand",
    "A heavy chain hangs from their hand",
    "A wooden stick is held in their hands",
    "A box cutter is visible in their grip",
]

# Abandoned objects — items on the ground
THREAT_OBJECTS_ABANDONED = [
    "A large black duffel bag is sitting alone on the ground in the center of the frame",
    "A taped-up cardboard box is left near the wall",
    "A lone backpack is sitting in the middle of the sidewalk",
    "A suspicious wrapped package sits on the ground",
    "A plastic bag with wires sticking out is on the pavement",
    "Two unattended suitcases are left side by side on the ground",
]
