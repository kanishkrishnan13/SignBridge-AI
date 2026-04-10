"""
NLP mapper: converts natural-language medical text to ISL sign tokens.
"""

import logging
import os
import json
import re

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KEYPOINTS_DIR = os.path.join(_BASE_DIR, "frontend", "assets", "keypoints")

# ---------------------------------------------------------------------------
# Vocabulary: medical / healthcare words → ISL sign token
# ---------------------------------------------------------------------------

SIGN_VOCAB: dict[str, str] = {
    # Distress / emergency
    "help": "HELP",
    "emergency": "EMERGENCY",
    "stop": "STOP",
    "call_doctor": "CALL_DOCTOR",
    "ambulance": "AMBULANCE",
    # Affirmatives / negatives
    "yes": "YES",
    "no": "NO",
    "ok": "OK",
    # Pain general
    "pain": "PAIN",
    "no_pain": "NO_PAIN",
    # Body parts
    "head": "HEAD",
    "headache": "HEADACHE",
    "chest": "CHEST",
    "chest_pain": "CHEST_PAIN",
    "stomach": "STOMACH",
    "stomach_pain": "STOMACH_PAIN",
    "back": "BACK",
    "hand": "HAND",
    "leg": "LEG",
    "heart": "HEART",
    "blood": "BLOOD",
    # Medical staff / places
    "doctor": "DOCTOR",
    "nurse": "NURSE",
    "hospital": "HOSPITAL",
    # Common symptoms
    "fever": "FEVER",
    "cold": "COLD",
    "cough": "COUGH",
    "vomit": "VOMIT",
    "dizzy": "DIZZY",
    "breathe": "BREATHE",
    "tired": "TIRED",
    # Measurements / procedures
    "pressure": "PRESSURE",
    "injection": "INJECTION",
    "surgery": "SURGERY",
    # Social / daily needs
    "water": "WATER",
    "medicine": "MEDICINE",
    "family": "FAMILY",
    "sleep": "SLEEP",
    "eat": "EAT",
    "drink": "DRINK",
    "toilet": "TOILET",
    # Conditions
    "allergic": "ALLERGIC",
    "diabetic": "DIABETIC",
    "pregnant": "PREGNANT",
    "broken": "BROKEN",
    "wound": "WOUND",
    # Emotions
    "happy": "HAPPY",
    "sad": "SAD",
    # Politeness
    "thank_you": "THANK_YOU",
    "thanks": "THANK_YOU",
    "thank": "THANK_YOU",
}

# Aliases / compound surface forms that should map before single-word lookup
_COMPOUND_MAP: dict[str, str] = {
    "chest pain": "CHEST_PAIN",
    "stomach pain": "STOMACH_PAIN",
    "no pain": "NO_PAIN",
    "call doctor": "CALL_DOCTOR",
    "thank you": "THANK_YOU",
}


def _normalize(text: str) -> str:
    """Lower-case and strip punctuation, preserving underscores (used in compound tokens)."""
    return re.sub(r"[^a-z0-9 _]", "", text.lower().strip())


def text_to_signs(text: str) -> list[str]:
    """
    Tokenise *text* and map each word (or compound) to an ISL sign token.

    Unknown words are either skipped (if very short) or spelled out as
    individual letter tokens (e.g. "abc" → ["LETTER_A", "LETTER_B", "LETTER_C"]).

    Returns
    -------
    list[str]  Ordered list of ISL sign tokens.
    """
    if not text or not text.strip():
        return []

    normalised = _normalize(text)
    tokens: list[str] = []

    # --- Pass 1: replace known compound phrases --------------------------
    for phrase, sign in sorted(_COMPOUND_MAP.items(), key=lambda kv: -len(kv[0])):
        normalised = normalised.replace(phrase, sign.lower())

    # --- Pass 2: word-by-word lookup -------------------------------------
    for word in normalised.split():
        # Already a resolved sign token (uppercase after normalise → lower)
        if word.upper() in SIGN_VOCAB.values():
            tokens.append(word.upper())
            continue

        sign = SIGN_VOCAB.get(word)
        if sign:
            tokens.append(sign)
        elif len(word) == 1 and word.isalpha():
            tokens.append(f"LETTER_{word.upper()}")
        elif len(word) <= 1:
            pass  # skip punctuation / single digits
        else:
            # Spell out unknown words character by character
            logger.debug("Unknown word '%s' — spelling out", word)
            for ch in word:
                if ch.isalpha():
                    tokens.append(f"LETTER_{ch.upper()}")

    return tokens


def get_gesture_sequence(sign_token: str) -> dict | None:
    """
    Return the keypoint JSON for *sign_token*, or None if not found.

    Looks for: frontend/assets/keypoints/<SIGN_TOKEN>.json
    Only alphanumeric characters and underscores are allowed in the token
    to prevent path traversal.
    """
    # Validate token: allow only A-Z, 0-9, underscore
    if not re.match(r"^[A-Za-z0-9_]+$", sign_token):
        logger.warning("Rejected sign token with unsafe characters: %r", sign_token)
        return None

    filename = f"{sign_token}.json"
    keypoints_abs = os.path.abspath(_KEYPOINTS_DIR)
    filepath = os.path.abspath(os.path.join(keypoints_abs, filename))

    # Ensure resolved path stays within the keypoints directory
    if not filepath.startswith(keypoints_abs + os.sep):
        logger.warning("Path traversal attempt for sign token: %r", sign_token)
        return None

    if not os.path.exists(filepath):
        logger.debug("Keypoint file not found: %s", filepath)
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read keypoint file %s: %s", filepath, exc)
        return None
