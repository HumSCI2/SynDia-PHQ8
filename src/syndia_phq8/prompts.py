"""Prompt templates used by the public reference implementation."""

GENERATION_SYSTEM = """You generate fictional therapist-patient dialogues for research.
Use only facts supported by the supplied synthetic profile. Do not state a PHQ-8 score,
diagnosis, or unsupported crisis. Return JSON only with key 'conversations'. Its value must
be a list of dialogue lists; every turn must have role 'therapist' or 'patient' and text.
The examples are synthetic and must not introduce names, locations, employers, or dates."""

QUALITY_SYSTEM = """You are evaluating a set of synthetic therapist-patient dialogues.
Rate each requested dimension from 1 (poor) to 5 (excellent), using only the supplied profile
and dialogues. Return JSON only. Do not infer a clinical diagnosis."""

ASSESSMENT_SYSTEM = """Reconstruct PHQ-8 labels from the supplied dialogue for research.
Return JSON only with depressed (boolean), total (integer 0-24), severity (string), and items
(a list of eight integers from 0 to 3 in PHQ-8 order). This is label reconstruction, not a
clinical diagnosis."""
