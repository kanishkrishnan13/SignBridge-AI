import re
import logging

logger = logging.getLogger(__name__)

KNOWN_SIGNS = frozenset([
    'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
    'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
    'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
    'thank_you', 'water', 'medicine',
])


class NLPMapper:
    """Maps natural-language text to ISL sign token sequences."""

    WORD_TO_SIGN = {
        # Pain and body symptoms
        'pain': 'pain', 'hurt': 'pain', 'ache': 'pain', 'hurting': 'pain',
        'headache': 'headache', 'head': 'head', 'migraine': 'headache',
        'stomach': 'stomach', 'belly': 'stomach', 'abdomen': 'stomach',
        'stomachache': 'stomach_pain',
        'chest': 'chest', 'heart': 'chest',
        'back': 'back', 'spine': 'back',
        'hand': 'hand', 'arm': 'hand',
        'leg': 'leg', 'foot': 'leg', 'knee': 'leg',
        # Medical actions
        'help': 'help', 'assist': 'help', 'need': 'help',
        'emergency': 'emergency', 'urgent': 'emergency', 'critical': 'emergency',
        'stop': 'stop', 'halt': 'stop', 'wait': 'stop',
        'call': 'call_doctor', 'doctor': 'call_doctor', 'nurse': 'call_doctor',
        # Responses
        'yes': 'yes', 'yeah': 'yes', 'ok': 'yes', 'okay': 'yes', 'sure': 'yes',
        'no': 'no', 'not': 'no', 'never': 'no', 'none': 'no',
        'better': 'no_pain', 'fine': 'no_pain',
        # Basic needs
        'water': 'water', 'drink': 'water', 'thirsty': 'water',
        'medicine': 'medicine', 'medication': 'medicine', 'drug': 'medicine',
        'pill': 'medicine', 'tablet': 'medicine',
        # Social
        'thank': 'thank_you', 'thanks': 'thank_you', 'grateful': 'thank_you',
    }

    def __init__(self):
        logger.info("NLP Mapper initialized")

    def map_to_signs(self, text):
        """Map text to an ISL sign token sequence."""
        text_lower = text.lower().strip()

        signs = self._check_phrases(text_lower)

        if not signs:
            words = re.findall(r'\b\w+\b', text_lower)
            seen = set()
            signs = []
            for word in words:
                sign = self.WORD_TO_SIGN.get(word)
                if sign and sign not in seen:
                    seen.add(sign)
                    signs.append(sign)

        signs = [s for s in signs if s in KNOWN_SIGNS]

        if not signs:
            signs = ['help']

        return {
            'text': text,
            'signs': signs,
            'count': len(signs),
            'sequence': signs,
        }

    def _check_phrases(self, text):
        """Return sign tokens matched by multi-word phrase patterns."""
        signs = []

        # Special case: "I have <body_part> pain"
        m = re.search(r'\bi(?:\s+am)?\s+having?\s+(\w+)\s+pain\b', text)
        if m:
            body = m.group(1)
            candidate = body + '_pain'
            signs.append(candidate if candidate in KNOWN_SIGNS else 'pain')

        for pattern, result in [
            (r'\bcall.*doctor\b',  ['call_doctor']),
            (r'\bno\s+pain\b',     ['no_pain']),
            (r'\bneed.*help\b',    ['help']),
            (r'\bthank\s+you\b',   ['thank_you']),
        ]:
            if re.search(pattern, text):
                for token in result:
                    if token not in signs:
                        signs.append(token)

        return signs

    def get_sign_vocabulary(self):
        """Return the list of all mappable sign tokens."""
        return sorted(KNOWN_SIGNS)
