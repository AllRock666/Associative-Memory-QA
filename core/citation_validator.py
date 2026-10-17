"""
Citation Validator Module
=========================
Re-exports the robust CitationValidator from core.generative_synthesizer
to satisfy modular imports and external interfaces.
"""

from core.generative_synthesizer import CitationValidator

__all__ = ["CitationValidator"]
