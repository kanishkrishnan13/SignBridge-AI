import os
import json
import re
import warnings


class NLPMapper:
    """Map free-text sentences to ISL sign tokens and their keypoint sequences."""

    def __init__(self):
        self.sign_map: dict[str, str] = {
            # Core medical signs
            'help': 'help',
            'pain': 'pain',
            'hurt': 'pain',
            'hurts': 'pain',
            'ache': 'pain',
            'headache': 'headache',
            'head': 'head',
            'head pain': 'headache',
            'head ache': 'headache',
            'stomach': 'stomach',
            'stomach ache': 'stomach_pain',
            'stomach pain': 'stomach_pain',
            'tummy': 'stomach',
            'tummy ache': 'stomach_pain',
            'abdomen': 'stomach',
            'abdominal pain': 'stomach_pain',
            'chest': 'chest',
            'chest pain': 'chest_pain',
            'chest ache': 'chest_pain',
            'emergency': 'emergency',
            'urgent': 'emergency',
            'stop': 'stop',
            'doctor': 'call_doctor',
            'call doctor': 'call_doctor',
            'get doctor': 'call_doctor',
            'need doctor': 'call_doctor',
            'no pain': 'no_pain',
            'fine': 'no_pain',
            'okay': 'yes',
            'ok': 'yes',
            'back': 'back',
            'back pain': 'back',
            'hand': 'hand',
            'arm': 'hand',
            'leg': 'leg',
            'foot': 'leg',
            'feet': 'leg',
            'yes': 'yes',
            'yeah': 'yes',
            'yep': 'yes',
            'no': 'no',
            'nope': 'no',
            'nah': 'no',
            'thank you': 'thank_you',
            'thanks': 'thank_you',
            'thank': 'thank_you',
            'water': 'water',
            'drink': 'water',
            'thirsty': 'water',
            'medicine': 'medicine',
            'medication': 'medicine',
            'drugs': 'medicine',
            'pill': 'medicine',
            'pills': 'medicine',
        }

        # Prefer the keypoints next to the frontend assets; fall back to a
        # sibling directory of backend/ so the path survives different CWDs.
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        self.keypoints_dir = os.path.normpath(
            os.path.join(backend_dir, '..', 'frontend', 'assets', 'keypoints')
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_text_to_tokens(self, text: str) -> list[str]:
        """
        Tokenise *text* into ISL sign tokens.

        Multi-word phrases are matched greedily (longest match first) before
        falling back to individual words.
        """
        normalised = text.lower().strip()
        normalised = re.sub(r"[^\w\s]", '', normalised)
        words = normalised.split()

        tokens: list[str] = []
        i = 0
        while i < len(words):
            matched = False
            # Try longest phrase first (up to 3 words)
            for length in range(min(3, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i + length])
                if phrase in self.sign_map:
                    tokens.append(self.sign_map[phrase])
                    i += length
                    matched = True
                    break
            if not matched:
                # Unknown word — skip silently
                i += 1

        return tokens

    def get_gesture_sequence(self, token: str) -> list | None:
        """
        Load the keypoint sequence for *token* from
        ``keypoints_dir/<token>.json``.

        Returns the parsed JSON data (expected: list of keypoint frames),
        or None if the file does not exist or cannot be parsed.
        """
        json_path = os.path.join(self.keypoints_dir, f"{token}.json")
        if not os.path.isfile(json_path):
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"Could not load keypoint file '{json_path}': {exc}")
            return None

    def map_sentence(self, sentence: str) -> dict:
        """
        Full pipeline: sentence → ISL tokens → keypoint sequences.

        Returns::

            {
                'tokens':    ['help', 'pain', ...],
                'sequences': {'help': [...], 'pain': None, ...},
            }
        """
        tokens = self.map_text_to_tokens(sentence)
        sequences = {token: self.get_gesture_sequence(token) for token in tokens}
        return {'tokens': tokens, 'sequences': sequences}
