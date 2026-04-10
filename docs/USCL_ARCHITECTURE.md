# Module 3 — Universal Sign Concept Layer (USCL) Architecture

## Overview

The **Universal Sign Concept Layer (USCL)** is a language-neutral semantic bridge that maps signs from one sign language (e.g. ISL) to a shared concept identifier, then re-maps that concept to an equivalent sign in any target sign language (ASL, BSL, JSL, and more).

This decouples the SignBridge AI system from any single sign language and makes it trivially extensible: adding a new sign language requires only an additional mapping dictionary, not changes to the pipeline.

---

## Table of Contents

1. [Design Goals](#design-goals)
2. [USCL Concept Structure](#uscl-concept-structure)
3. [JSON Schema](#json-schema)
4. [Full USCL Entry Example](#full-uscl-entry-example)
5. [Mapping Dictionary Structure](#mapping-dictionary-structure)
6. [Conversion Pipeline](#conversion-pipeline)
7. [Worked Example: "pain"](#worked-example-pain)
8. [Storage Format](#storage-format)
9. [Future Implementation Roadmap](#future-implementation-roadmap)

---

## Design Goals

| Goal | Description |
|------|-------------|
| **Language-neutral** | Concepts are language-agnostic identifiers, not tied to any single sign vocabulary |
| **Extensible** | New sign languages are added as mapping columns, not schema changes |
| **Reversible** | The pipeline supports both ISL→ASL and ASL→ISL translations |
| **Medical-first** | Initial vocabulary focuses on healthcare communication |
| **Graceful degradation** | Missing mappings fall back to finger-spelling or a `null` sequence |

---

## USCL Concept Structure

Each USCL entry represents one **concept** — an abstract meaning independent of language.

```
USCL Entry
├── id              : Unique stable identifier  (e.g. "USCL_ID_001")
├── concept         : English gloss / canonical name  (e.g. "pain")
├── category        : Semantic category  (e.g. "symptom", "body_part", "emergency")
├── description     : Plain-English definition used for NLP disambiguation
├── isl_signs       : List of ISL sign tokens that express this concept
├── asl_signs       : List of ASL sign tokens
├── bsl_signs       : List of BSL sign tokens
├── jsl_signs       : List of JSL (Japanese Sign Language) sign tokens
└── synonyms        : Alternative English words / phrases that map to this concept
```

The `*_signs` arrays support **one-to-many** mappings because a single concept may require a sequence of signs in a given language (e.g. "chest pain" in BSL may be two signs: CHEST + HURT).

---

## JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "USCLEntry",
  "type": "object",
  "required": ["id", "concept", "category", "description", "isl_signs", "asl_signs", "bsl_signs", "jsl_signs"],
  "properties": {
    "id": {
      "type": "string",
      "pattern": "^USCL_ID_\\d{3,}$",
      "description": "Stable unique identifier for this concept"
    },
    "concept": {
      "type": "string",
      "description": "Canonical English gloss (lowercase)"
    },
    "category": {
      "type": "string",
      "enum": [
        "symptom", "body_part", "emergency", "emotion",
        "medical_staff", "procedure", "daily_need",
        "affirmative", "greeting", "condition"
      ]
    },
    "description": {
      "type": "string",
      "description": "Plain-English definition for NLP disambiguation"
    },
    "isl_signs": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Ordered list of ISL sign tokens"
    },
    "asl_signs": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Ordered list of ASL sign tokens"
    },
    "bsl_signs": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Ordered list of BSL sign tokens"
    },
    "jsl_signs": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Ordered list of JSL sign tokens"
    },
    "synonyms": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Alternative surface forms (English) that resolve to this concept",
      "default": []
    },
    "notes": {
      "type": "string",
      "description": "Optional: regional variation notes or usage caveats"
    }
  },
  "additionalProperties": false
}
```

---

## Full USCL Entry Example

```json
{
  "id": "USCL_ID_001",
  "concept": "pain",
  "category": "symptom",
  "description": "A feeling of physical discomfort or suffering in a body part. Used as a standalone complaint or combined with a body-part sign.",
  "isl_signs": ["PAIN"],
  "asl_signs": ["ASL_PAIN_23"],
  "bsl_signs": ["BSL_HURT_17"],
  "jsl_signs": ["JSL_ITAMI_04"],
  "synonyms": ["hurt", "ache", "sore", "suffering", "discomfort"],
  "notes": "In ASL, PAIN is often produced with repeated movement toward the painful area. Combine with a body-part sign for specificity, e.g. ASL_PAIN_23 + ASL_CHEST_07."
}
```

Additional entries in the same vocabulary file:

```json
[
  {
    "id": "USCL_ID_002",
    "concept": "chest pain",
    "category": "symptom",
    "description": "Pain or pressure felt in the chest; may indicate cardiac event.",
    "isl_signs": ["CHEST_PAIN"],
    "asl_signs": ["ASL_CHEST_07", "ASL_PAIN_23"],
    "bsl_signs": ["BSL_CHEST_03", "BSL_HURT_17"],
    "jsl_signs": ["JSL_MUNE_01", "JSL_ITAMI_04"],
    "synonyms": ["chest ache", "heart pain", "chest pressure"],
    "notes": "High-priority medical concept. If detected, system should also flag EMERGENCY."
  },
  {
    "id": "USCL_ID_003",
    "concept": "help",
    "category": "emergency",
    "description": "Request for assistance; urgency context varies.",
    "isl_signs": ["HELP"],
    "asl_signs": ["ASL_HELP_01"],
    "bsl_signs": ["BSL_HELP_01"],
    "jsl_signs": ["JSL_TASUKETE_01"],
    "synonyms": ["assist", "assistance", "support"],
    "notes": null
  },
  {
    "id": "USCL_ID_004",
    "concept": "fever",
    "category": "symptom",
    "description": "Elevated body temperature, typically above 38 °C (100.4 °F).",
    "isl_signs": ["FEVER"],
    "asl_signs": ["ASL_FEVER_09"],
    "bsl_signs": ["BSL_TEMPERATURE_HIGH_05"],
    "jsl_signs": ["JSL_NETSU_02"],
    "synonyms": ["high temperature", "pyrexia", "temperature"],
    "notes": null
  },
  {
    "id": "USCL_ID_005",
    "concept": "doctor",
    "category": "medical_staff",
    "description": "A qualified medical physician.",
    "isl_signs": ["DOCTOR"],
    "asl_signs": ["ASL_DOCTOR_11"],
    "bsl_signs": ["BSL_DOCTOR_06"],
    "jsl_signs": ["JSL_ISHA_03"],
    "synonyms": ["physician", "GP", "medic"],
    "notes": null
  }
]
```

---

## Mapping Dictionary Structure

The mapping dictionary (`model/uscl_map.json`) provides fast O(1) look-up from any sign token or synonym to a USCL entry.

```json
{
  "version": "1.0.0",
  "generated": "2024-01-01T00:00:00Z",
  "index": {
    "PAIN":            "USCL_ID_001",
    "hurt":            "USCL_ID_001",
    "ache":            "USCL_ID_001",
    "CHEST_PAIN":      "USCL_ID_002",
    "chest pain":      "USCL_ID_002",
    "ASL_PAIN_23":     "USCL_ID_001",
    "BSL_HURT_17":     "USCL_ID_001",
    "JSL_ITAMI_04":    "USCL_ID_001",
    "HELP":            "USCL_ID_003",
    "ASL_HELP_01":     "USCL_ID_003",
    "FEVER":           "USCL_ID_004",
    "DOCTOR":          "USCL_ID_005"
  },
  "entries": {
    "USCL_ID_001": { "...": "full entry object as above" },
    "USCL_ID_002": { "...": "..." }
  }
}
```

**Key design decisions:**

- Both ISL *tokens* (`"PAIN"`) and surface-form *synonyms* (`"ache"`) are indexed.
- Sign tokens from all languages are indexed, enabling reverse translation.
- `entries` stores the full USCL object, keyed by ID, for constant-time retrieval.

---

## Conversion Pipeline

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                  USCL Conversion Pipeline                       │
 └─────────────────────────────────────────────────────────────────┘

  User input
  (spoken / typed text  OR  ISL webcam sign)
        │
        ▼
  ┌─────────────┐
  │  NLP Mapper │  ← text_to_signs() in backend/nlp_mapper.py
  │  or         │    detects known words/phrases, returns ISL tokens
  │  Classifier │  ← GestureDetector in backend/gesture_detector.py
  └──────┬──────┘
         │  ISL token(s):  e.g.  ["CHEST_PAIN", "DIZZY"]
         ▼
  ┌──────────────────┐
  │  USCL Resolver   │  look up each ISL token in uscl_map.json["index"]
  │  (Module 3)      │  → USCL IDs:  ["USCL_ID_002", "USCL_ID_XXX"]
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────────┐
  │  Target Language     │  read <target>_signs from each USCL entry
  │  Sign Selector       │  e.g.  asl_signs, bsl_signs, jsl_signs
  └──────┬───────────────┘
         │  Target sign tokens:  e.g.  ["ASL_CHEST_07", "ASL_PAIN_23"]
         ▼
  ┌────────────────────┐
  │  Keypoint Lookup   │  load frontend/assets/keypoints/<token>.json
  │  / Avatar Engine   │  render 3-D avatar animation sequence
  └────────────────────┘
         │
         ▼
  Output:  animated avatar + optional TTS voice-over
```

---

## Worked Example: "pain"

**Input text:** `"I have pain in my chest"`

### Step 1 — NLP Mapper

```python
text_to_signs("I have pain in my chest")
# → ["PAIN", "CHEST"]
```

The compound phrase is not matched here; single tokens are returned.

### Step 2 — USCL Resolver

```python
uscl_map["index"]["PAIN"]   # → "USCL_ID_001"
uscl_map["index"]["CHEST"]  # → "USCL_ID_008"  (body-part entry)
```

### Step 3 — Target Language Sign Selector (target = ASL)

```python
uscl_map["entries"]["USCL_ID_001"]["asl_signs"]  # → ["ASL_PAIN_23"]
uscl_map["entries"]["USCL_ID_008"]["asl_signs"]  # → ["ASL_CHEST_07"]
```

Combined ASL sequence: `["ASL_CHEST_07", "ASL_PAIN_23"]`

### Step 4 — Keypoint Lookup

```
GET /api/avatar/sequence/ASL_CHEST_07  →  keypoint animation JSON
GET /api/avatar/sequence/ASL_PAIN_23   →  keypoint animation JSON
```

### Step 5 — Avatar Playback

The avatar renders `ASL_CHEST_07` followed immediately by `ASL_PAIN_23`, producing a fluent cross-language sign sequence.

**Summary table:**

| Stage | Value |
|-------|-------|
| Input text | `"pain"` |
| ISL token | `PAIN` |
| USCL ID | `USCL_ID_001` |
| ASL token | `ASL_PAIN_23` |
| BSL token | `BSL_HURT_17` |
| JSL token | `JSL_ITAMI_04` |

---

## Storage Format

| File / Directory | Purpose |
|-----------------|---------|
| `model/uscl_vocabulary.json` | Master list of all USCL entries (JSON array) |
| `model/uscl_map.json` | Pre-built index mapping tokens → USCL IDs (generated by build script) |
| `frontend/assets/keypoints/<TOKEN>.json` | Per-sign keypoint animation data (all languages) |
| `scripts/build_uscl_index.py` | Script that generates `uscl_map.json` from `uscl_vocabulary.json` |

---

## Future Implementation Roadmap

### Phase 1 — Foundation (current)

- [x] ISL ↔ USCL concept mapping (healthcare vocabulary, ~60 signs)
- [x] ISL → ASL translation via USCL bridge
- [x] JSON schema definition and vocabulary file
- [ ] `scripts/build_uscl_index.py` — automated index builder
- [ ] Unit tests for USCL resolver

### Phase 2 — Expanded Language Support

- [ ] BSL mapping: complete 60-sign healthcare vocabulary
- [ ] JSL mapping: complete 60-sign healthcare vocabulary
- [ ] Auslan (Australian SL) mapping
- [ ] Automatic synonym expansion using WordNet

### Phase 3 — Bidirectional Translation

- [ ] ASL → USCL → ISL reverse pipeline
- [ ] BSL → USCL → ASL cross-translation
- [ ] Confidence scores on concept mappings (some concepts have imperfect equivalents)

### Phase 4 — Community & Extensibility

- [ ] Web-based USCL editor / vocabulary management UI
- [ ] Community submission workflow for new sign mappings
- [ ] Versioned vocabulary releases (semantic versioning on `uscl_vocabulary.json`)
- [ ] Regional dialect support (e.g. IS variants across Indian states)

### Phase 5 — AI-Assisted Mapping

- [ ] Embedding-based concept similarity for approximate matching
- [ ] Automatic mapping suggestion from sign video corpus
- [ ] Confidence-weighted fallback chain: exact match → approximate → finger-spell
