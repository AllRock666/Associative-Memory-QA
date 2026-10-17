"""
Format and Intent Detector for Adaptive QA
==========================================
Infers user-requested examination mark tiers, comparison structures,
worked examples, and revision notes from natural language queries.
"""

import re
from typing import Tuple, Optional
from core.format_guidance import FORMAT_CATALOG, AnswerFormatGuidance


class FormatDetector:
    """Classifies user queries to identify requested format, mark tiers, and structure."""

    MARK_PATTERNS = [
        (r'\b(?:10|ten)\s*[-_\s]*marks?\b|\b(?:as\s+(?:a\s+)?10\s*[-_\s]*mark)\b|\bfor\s+10\s+marks\b', "10_mark"),
        (r'\b(?:8|eight)\s*[-_\s]*marks?\b|\b(?:as\s+(?:a\s+)?8\s*[-_\s]*mark)\b|\bfor\s+8\s+marks\b', "8_mark"),
        (r'\b(?:5|five)\s*[-_\s]*marks?\b|\b(?:as\s+(?:a\s+)?5\s*[-_\s]*mark)\b|\bfor\s+5\s+marks\b', "5_mark"),
        (r'\b(?:2|two)\s*[-_\s]*marks?\b|\b(?:as\s+(?:a\s+)?2\s*[-_\s]*mark)\b|\bfor\s+2\s+marks\b', "2_mark"),
    ]

    GENRE_PATTERNS = [
        (r'\b(?:compare|comparison|versus|vs\.?|difference\s+between|contrast)\b', "comparison"),
        (r'\b(?:worked\s+example|numerical\s+example|concrete\s+example|walk\s+through\s+an?\s+example)\b', "worked_example"),
        (r'\b(?:step\s+by\s+step|chronological|lifecycle\s+steps|how\s+to\s+execute)\b', "step_by_step"),
        (r'\b(?:revision\s+notes|study\s+notes|summary\s+notes|cheat\s*sheet|structured\s+notes)\b', "structured_notes"),
        (r'\b(?:executive\s+summary|executive\s+briefing|briefing\s+doc)\b', "executive"),
        (r'\b(?:briefly|in\s+brief|in\s+one\s+sentence|in\s+2\s+sentences|concise|short\s+answer)\b', "concise"),
        (r'\b(?:in\s+detail|detailed|comprehensive|exhaustive|deep\s+dive|in-depth)\b', "detailed"),
    ]

    @classmethod
    def detect_format(cls, query: str, explicit_preference: Optional[str] = None) -> Tuple[str, Optional[AnswerFormatGuidance]]:
        """
        Detects requested answer format.
        Explicit preference takes precedence if provided and valid.
        Returns: (format_id, guidance_object_or_none)
        """
        # 1. Explicit preference check
        if explicit_preference and explicit_preference.strip():
            pref = explicit_preference.lower().strip()
            # Normalize common aliases
            aliases = {
                "2": "2_mark", "2mark": "2_mark", "2-mark": "2_mark", "2 marks": "2_mark",
                "5": "5_mark", "5mark": "5_mark", "5-mark": "5_mark", "5 marks": "5_mark",
                "8": "8_mark", "8mark": "8_mark", "8-mark": "8_mark", "8 marks": "8_mark",
                "10": "10_mark", "10mark": "10_mark", "10-mark": "10_mark", "10 marks": "10_mark",
                "table": "comparison", "compare": "comparison",
                "notes": "structured_notes", "cheat_sheet": "structured_notes",
                "example": "worked_example"
            }
            normalized_pref = aliases.get(pref, pref)
            if normalized_pref in FORMAT_CATALOG:
                return normalized_pref, FORMAT_CATALOG[normalized_pref]
            if normalized_pref in ("concise", "executive", "detailed", "conversational"):
                return normalized_pref, None

        q_lower = query.lower().strip()

        # 2. Check for explicit marks specification (e.g. "8-mark answer", "for 5 marks")
        for pattern, fid in cls.MARK_PATTERNS:
            if re.search(pattern, q_lower):
                return fid, FORMAT_CATALOG.get(fid)

        # 3. Check for specific structural genres (comparison, worked example, notes)
        for pattern, fid in cls.GENRE_PATTERNS:
            if re.search(pattern, q_lower):
                guidance = FORMAT_CATALOG.get(fid)
                return fid, guidance

        # 4. Default fallback: Assess complexity
        q_words = q_lower.split()
        if len(q_words) >= 6 or any(w in q_lower for w in ["explain", "describe", "architecture", "mechanism", "overview"]):
            return "conversational", None
        return "conversational", None
