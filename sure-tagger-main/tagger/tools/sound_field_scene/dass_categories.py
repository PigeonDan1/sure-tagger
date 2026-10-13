"""Category mapping for DASS AudioSet labels (docs/DASS.md seven categories).

The deployed DASS checkpoint exposes 527 AudioSet display names. This module
maps each name to one of eight buckets used by the categorized evidence layer:

  music                ④ 音乐 (instruments, genres, vocal music)
  animal               ② 动物声音
  mechanical           ③ 物体/机械声音
  nature               ⑤ 自然声音
  formless             ⑥ 无明确声源的声音形态 (silence, hum, click, ...)
  channel_environment  ⑦ 声道/环境/背景属性 (reverberation, echo, noise, ...)
  human                ① 人类声音 (voice and bodily sounds; evidence only)
  other                fallback bucket (evidence only)

Classification is deterministic: exact-name overrides first, then substring
keywords scanned in the priority order music > human > animal > mechanical >
nature > channel_environment > formless; anything unmatched falls back to
``other``. The keyword table was iterated against the full id2label list of the
deployed checkpoint (models/DASS/saurabhati__DASS_medium_AudioSet_48.9,
2026-08-24) so that no real label lands in ``other``.
"""


# Public composition keys (docs/DASS.md ②-⑦, music first for the AED gate).
# Human and unclassified labels are evidence-only and never enter the public
# noise composition.
PUBLIC_COMPOSITION_CATEGORIES = (
    "music",
    "animal",
    "mechanical",
    "nature",
    "formless",
    "channel_environment",
)

EVIDENCE_CATEGORIES = PUBLIC_COMPOSITION_CATEGORIES + ("human", "other")

# Loose upper bound for the public audit: composition top-k is configurable,
# but the auditor rejects values beyond this cap.
NOISE_COMPOSITION_AUDIT_MAX_ITEMS = 10

# Exact-name overrides for names whose substrings would otherwise match an
# earlier (higher-priority) category.
_OVERRIDES = {
    "Beep, bleep": "formless",
    "Bird vocalization, bird call, bird song": "animal",
    "Boat, Water vehicle": "mechanical",
    "Burst, pop": "mechanical",
    "Buzzer": "mechanical",
    "Chirp tone": "formless",
    "Crackle": "nature",
    "Crunch": "formless",
    "Fire alarm": "mechanical",
    "Fire engine, fire truck (siren)": "mechanical",
    "Firecracker": "mechanical",
    "Fireworks": "mechanical",
    "Gunshot, gunfire": "mechanical",
    "Artillery fire": "mechanical",
    "Microwave oven": "mechanical",
    "Power windows, electric windows": "mechanical",
    "Reversing beeps": "mechanical",
    "Shuffling cards": "mechanical",
    "Ska": "music",
    "Speech synthesizer": "human",
    "Steam whistle": "mechanical",
    "Vehicle horn, car horn, honking": "mechanical",
    "Water tap, faucet": "mechanical",
    "Whimper (dog)": "animal",
    "Wind chime": "mechanical",
}

# Substring keywords per category, scanned in category priority order.
_KEYWORDS = {
    "music": [
        "music", "musical", "instrument", "song", "singing", "choir",
        "yodeling", "chant", "mantra", "rapping", "beatbox", "capella",
        "humming", "orchestra", "jazz", "classical", "opera", "guitar",
        "strum", "banjo", "sitar", "mandolin", "zither", "ukulele",
        "keyboard (musical)", "piano", "organ", "synthesizer", "sampler",
        "harpsichord", "percussion", "drum", "rimshot", "timpani", "tabla",
        "cymbal", "hi-hat", "wood block", "tambourine", "maraca", "gong",
        "tubular", "mallet", "marimba", "xylophone", "glockenspiel",
        "vibraphone", "steelpan", "brass", "french horn", "trumpet",
        "trombone", "bowed", "string section", "violin", "fiddle",
        "pizzicato", "cello", "double bass", "wind instrument", "woodwind",
        "flute", "saxophone", "clarinet", "harp", "tuning fork",
        "harmonica", "accordion", "bagpipes", "didgeridoo", "shofar",
        "theremin", "singing bowl", "performance technique", "effects unit",
        "chorus effect", "hip hop",
        "rock", "heavy metal", "punk", "grunge", "progressive",
        "rhythm and blues", "soul", "reggae", "country", "swing",
        "bluegrass", "funk", "folk", "middle eastern", "techno", "dubstep",
        "electronica", "drum and bass", "ambient", "trance", "salsa",
        "flamenco", "blues", "new-age", "vocal music", "afrobeat", "gospel",
        "carnatic", "bollywood", "traditional", "independent", "lullaby",
        "soundtrack", "theme music", "dance music", "video game music",
        "disco",
    ],
    "human": [
        "speech", "speaking", "conversation", "narration", "monologue",
        "babbling", "shout", "bellow", "whoop", "yell", "battle cry",
        "children shouting", "screaming", "whispering", "laughter",
        "giggle", "snicker", "belly laugh", "chuckle", "chortle", "crying",
        "sobbing", "baby cry", "infant cry", "whimper", "wail", "moan",
        "sigh", "groan", "grunt", "whistling", "breathing", "wheeze",
        "snoring", "gasp", "pant", "snort", "cough", "throat clearing",
        "sneeze", "sniff", "run", "shuffle", "walk", "footsteps",
        "chewing", "mastication", "biting", "gargling", "stomach",
        "burping", "eructation", "hiccup", "fart", "hands",
        "finger snapping", "clapping", "heart", "heartbeat", "murmur",
        "cheering", "applause", "chatter", "crowd", "hubbub", "babble",
        "children playing", "children shouting",
    ],
    "animal": [
        "animal", "domestic animals", "pets", "dog", "bark", "yip", "howl",
        "bow-wow", "growling", "whimper (dog)", "cat", "purr", "meow",
        "hiss", "caterwaul", "livestock", "farm animals", "horse",
        "clip-clop", "neigh", "whinny", "cattle", "bovinae", "moo",
        "cowbell", "pig", "oink", "goat", "bleat", "sheep", "fowl",
        "chicken", "rooster", "cluck", "crowing", "cock-a-doodle", "turkey",
        "gobble", "duck", "quack", "goose", "honk", "wild animals",
        "roaring", "lion", "tiger", "roar", "bird", "chirp", "tweet",
        "squawk", "pigeon", "dove", "coo", "crow", "caw", "owl", "hoot",
        "flapping wings", "canidae", "wolves", "rodents", "rats", "mice",
        "mouse", "patter", "insect", "cricket", "mosquito", "housefly",
        "buzz", "bee", "wasp", "frog", "croak", "snake", "rattle",
        "whale", "vocalization",
    ],
    "mechanical": [
        "vehicle", "boat", "sailboat", "rowboat", "motorboat", "ship",
        "motor vehicle", "car", "toot", "car alarm", "skidding", "tire",
        "race car", "auto racing", "truck", "air brake", "air horn",
        "reversing", "ice cream truck", "bus", "emergency vehicle", "police",
        "ambulance", "motorcycle", "traffic", "rail", "train", "railroad",
        "subway", "metro", "underground", "aircraft", "jet", "propeller",
        "airscrew", "helicopter", "airplane", "fixed-wing", "bicycle",
        "skateboard", "engine", "drill", "lawn mower", "chainsaw",
        "knocking", "starting", "idling", "accelerating", "revving",
        "vroom", "door", "doorbell", "ding-dong", "sliding", "slam",
        "knock", "tap", "squeak", "cupboard", "drawer", "dishes",
        "cutlery", "silverware", "chopping", "frying", "microwave",
        "blender", "water tap", "faucet", "sink", "bathtub", "hair dryer",
        "toilet", "toothbrush", "vacuum", "zipper", "keys jangling",
        "coin", "scissors", "shaver", "razor", "typing", "typewriter",
        "keyboard", "writing", "alarm", "telephone", "ringtone", "dialing",
        "dtmf", "dial tone", "busy signal", "alarm clock", "siren", "buzzer",
        "smoke detector", "fire alarm", "foghorn", "whistle",
        "steam whistle", "bell", "chime", "change ringing", "campanology",
        "mechanisms", "ratchet", "pawl", "clock", "tick",
        "gears", "pulleys", "sewing machine", "fan", "air conditioning",
        "cash register", "printer", "camera", "reflex", "tools", "hammer",
        "jackhammer", "sawing", "filing", "sanding", "power tool",
        "explosion", "gunshot", "machine gun", "fusillade", "artillery",
        "cap gun", "fireworks", "firecracker", "burst", "eruption", "boom",
        "wood", "chop", "splinter", "crack", "glass", "chink", "clink",
        "shatter", "liquid", "splash", "splatter", "slosh", "squish",
        "drip", "pour", "trickle", "dribble", "gush", "fill", "spray",
        "pump", "stir", "boiling", "sonar", "arrow", "tuner", "basketball",
        "bouncing", "whip", "flap", "scratch", "scrape", "rub", "roll",
        "breaking", "crushing", "crumpling", "crinkling", "tearing",
        "clang", "rustle", "whir", "clatter", "sizzle", "clickety-clack",
        "television", "radio",
    ],
    "nature": [
        "wind", "rustling leaves", "thunderstorm", "thunder", "water",
        "rain", "raindrop", "stream", "waterfall", "ocean", "waves", "surf",
        "steam", "gurgling", "fire", "crackle",
    ],
    "channel_environment": [
        "inside", "outside", "small room", "large room", "hall",
        "public space", "urban", "rural", "reverberation", "echo", "noise",
        "environmental noise", "static", "mains hum", "distortion",
        "sidetone", "cacophony", "white noise", "pink noise",
        "field recording",
    ],
    "formless": [
        "whoosh", "swoosh", "swish", "thump", "thud", "thunk", "bang",
        "slap", "smack", "whack", "thwack", "smash", "crash", "beep",
        "bleep", "ping", "ding", "squeal", "creak", "clicking", "rumble",
        "plop", "jingle", "tinkle", "hum", "zing", "boing", "crunch",
        "silence", "sine wave", "harmonic", "chirp tone", "sound effect",
        "pulse", "throbbing", "vibration",
    ],
}

_CATEGORY_PRIORITY = (
    "music", "human", "animal", "mechanical", "nature",
    "channel_environment", "formless",
)


def classify_dass_label(display_name):
    """Return the category key for an AudioSet display name.

    Exact-name overrides win; otherwise keywords are scanned in category
    priority order. Unmatched names fall back to ``other``. Never raises for
    string input.
    """
    if not isinstance(display_name, str):
        raise ValueError("display_name must be a string")
    if not display_name:
        raise ValueError("display_name must be non-empty")
    if display_name in _OVERRIDES:
        return _OVERRIDES[display_name]
    lower = display_name.lower()
    for category in _CATEGORY_PRIORITY:
        for keyword in _KEYWORDS[category]:
            if keyword in lower:
                return category
    return "other"


def build_category_composition(
    labels, scores, threshold, top_k, music_present=None
):
    """Bucket score-ranked DASS labels into categories.

    Args:
        labels: list of ``{"index": int, "display_name": str}`` in score
            order (full 527-class vector).
        scores: parallel list of probabilities in ``[0, 1]``.
        threshold: minimum score for a label to enter its category.
        top_k: maximum labels per category.
        music_present: FireRed AED gate for the music category. ``True`` keeps
            DASS music labels, ``False`` empties the public music bucket, and
            ``None`` (AED unavailable) disables the gate.

    Returns:
        ``(public_composition, category_events)`` where
        ``public_composition`` maps each public category key to a score-ranked
        list of display names (music gated by ``music_present``), and
        ``category_events`` maps every evidence category (including ``human``
        and ``other``) to a list of ``{"index", "display_name", "score"}``.
    """
    if not isinstance(labels, list) or not isinstance(scores, list):
        raise ValueError("labels and scores must be lists")
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have equal length")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    threshold = float(threshold)
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be within [0, 1]")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if music_present is not None and not isinstance(music_present, bool):
        raise ValueError("music_present must be a boolean or None")

    category_events = {category: [] for category in EVIDENCE_CATEGORIES}
    for position, (label, raw_score) in enumerate(zip(labels, scores)):
        if not isinstance(label, dict):
            raise ValueError("label must be an object")
        index = label.get("index")
        display_name = label.get("display_name")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise ValueError("label indexes must match score order")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError("display names must be non-empty strings")
        score = float(raw_score)
        if score < 0 or score > 1 or score != score:
            raise ValueError("scores must be finite probabilities in [0, 1]")
        if score < threshold:
            continue
        category_events[classify_dass_label(display_name)].append(
            {
                "index": index,
                "display_name": display_name,
                "score": round(score, 6),
            }
        )

    for category in EVIDENCE_CATEGORIES:
        category_events[category].sort(
            key=lambda item: (-item["score"], item["index"])
        )
        category_events[category] = category_events[category][:top_k]

    public_composition = {}
    for category in PUBLIC_COMPOSITION_CATEGORIES:
        names = [
            item["display_name"] for item in category_events[category]
        ]
        if category == "music" and music_present is False:
            names = []
        public_composition[category] = names
    return public_composition, category_events
