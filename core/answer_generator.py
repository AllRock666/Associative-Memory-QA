"""
Answer Generation Module
=========================
Synthesizes answers from retrieved chunks using:
  - Extractive highlighting
  - Template-based generation
  - Confidence scoring
  - Source attribution
"""

import re
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from core.hopfield_torch import RetrievalResult
from core.query_handler import ProcessedQuery

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Complete answer with supporting evidence and metadata."""
    query: str
    answer: str
    confidence: float  # [0, 1]
    evidence_chunks: List[RetrievalResult]
    highlighted_passages: List[Dict]  # [{text, chunk_id, relevance}]
    sources: List[str]
    answer_type: str  # extractive | abstractive | hybrid | no_answer
    generation_stats: Dict = field(default_factory=dict)


class AnswerGenerator:
    """
    Generates answers from retrieved document chunks.

    Approach: Extractive + Template-based synthesis without requiring
    an external LLM — works offline with pure NLP.

    For LLM-enhanced answers, see LLMAnswerGenerator below.
    """

    # Sentence boundary patterns
    SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Highlight markers
    HIGHLIGHT_START = "**"
    HIGHLIGHT_END = "**"

    def __init__(
        self,
        max_answer_length: int = 16000,
        min_confidence_threshold: float = 0.1,
        top_sentences: int = 10,
    ):
        self.max_answer_length = max_answer_length
        self.min_confidence_threshold = min_confidence_threshold
        self.top_sentences = top_sentences

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split chunk text into individual sentences."""
        sentences = self.SENT_SPLIT.split(text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _score_sentence(
        self,
        sentence: str,
        keywords: List[str],
        query: str,
    ) -> float:
        """Score a sentence's relevance to the query."""
        sentence_lower = sentence.lower()
        query_lower = query.lower()
        score = 0.0

        # Keyword coverage
        kw_hits = sum(1 for kw in keywords if kw.lower() in sentence_lower)
        score += (kw_hits / max(len(keywords), 1)) * 0.5

        # N-gram overlap with query
        query_words = set(query_lower.split())
        sent_words = set(sentence_lower.split())
        overlap = len(query_words & sent_words) / max(len(query_words), 1)
        score += overlap * 0.3

        # Prefer sentences with numbers/facts
        if re.search(r'\d+', sentence):
            score += 0.1

        # Length preference (not too short, not too long)
        word_count = len(sentence.split())
        if 10 <= word_count <= 50:
            score += 0.1
        elif word_count < 5:
            score -= 0.2

        return max(0.0, score)

    def _extract_best_sentences(
        self,
        chunks: List[RetrievalResult],
        keywords: List[str],
        query: str,
    ) -> List[Tuple[str, float, str]]:
        """
        Extract and score the best sentences from retrieved chunks.

        Returns:
            List of (sentence, score, chunk_id) tuples, sorted by score.
        """
        scored_sentences = []

        for chunk in chunks:
            sentences = self._split_into_sentences(chunk.text)
            chunk_weight = chunk.similarity_score * (1.0 / chunk.rank)

            for sent in sentences:
                sent_score = self._score_sentence(sent, keywords, query)
                final_score = sent_score * chunk_weight
                scored_sentences.append((sent, final_score, chunk.chunk_id, chunk.source))

        # Sort by score descending
        scored_sentences.sort(key=lambda x: -x[1])
        return scored_sentences

    def _compute_confidence(
        self,
        top_chunks: List[RetrievalResult],
        answer_sentences: List[Tuple],
    ) -> float:
        """
        Compute overall answer confidence.

        Factors:
        - Similarity scores of top chunks
        - Number of supporting chunks
        - Sentence score distribution
        """
        if not top_chunks:
            return 0.0

        # Weighted average of top-k chunk similarities
        sims = [c.similarity_score for c in top_chunks[:3]]
        avg_sim = np.mean(sims) if sims else 0.0

        # Bonus for multiple supporting sources
        unique_sources = len(set(c.source for c in top_chunks))
        source_bonus = min(unique_sources * 0.05, 0.15)

        # Sentence quality
        if answer_sentences:
            avg_sent_score = np.mean([s[1] for s in answer_sentences[:3]])
            sent_bonus = min(avg_sent_score, 0.2)
        else:
            sent_bonus = 0.0

        confidence = (avg_sim * 0.65) + source_bonus + sent_bonus
        return min(float(confidence), 1.0)

    def _deduplicate_sentences(
        self,
        sentences: List[Tuple[str, float, str, str]],
    ) -> List[Tuple[str, float, str, str]]:
        """Remove near-duplicate sentences using prefix fingerprinting."""
        seen = set()
        unique = []
        for item in sentences:
            sent = item[0]
            # Normalize: lowercase, strip punctuation, take first 60 chars as key
            key = re.sub(r'[^a-z0-9\s]', '', sent.lower())[:60].strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _rephrase_sentence(self, sent: str) -> str:
        """
        Light rephrasing: strip trailing citations, normalize whitespace,
        ensure sentence ends with punctuation.
        """
        # Remove bracket citations like [1], (2020), etc.
        sent = re.sub(r'\s*[\(\[]\s*\d{1,4}\s*[\)\]]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()
        if sent and sent[-1] not in '.!?':
            sent += '.'
        # Capitalize first letter
        if sent:
            sent = sent[0].upper() + sent[1:]
        return sent

    def _build_summary_paragraph(
        self,
        selected: List[Tuple[str, float, str, str]],
        query_type: str,
    ) -> str:
        """
        Build the 2–4 line direct answer paragraph.
        Rephrases and merges the top 2–3 sentences coherently.
        """
        # Sort by chunk position for narrative coherence
        ordered = sorted(selected[:3], key=lambda x: x[2])
        parts = [self._rephrase_sentence(s[0]) for s in ordered]

        if query_type == "procedural":
            # Stitch with transitional phrasing
            if len(parts) >= 2:
                return f"{parts[0]} {parts[1]}"
            return parts[0] if parts else ""
        elif query_type == "comparative":
            if len(parts) >= 2:
                return f"{parts[0]} In contrast, {parts[1][0].lower() + parts[1][1:]}"
            return parts[0] if parts else ""
        else:
            return " ".join(parts[:2])

    def _build_key_points(
        self,
        selected: List[Tuple[str, float, str, str]],
        query_type: str,
        keywords: List[str],
    ) -> List[str]:
        """
        Extract 3–5 distinct key points from the retrieved sentences.
        Each bullet is a standalone, rephrased insight.
        """
        # Use sentences 1 onwards (sentence 0 already in summary)
        candidates = selected[1:8]
        points = []

        for sent, score, cid, source in candidates:
            rephrased = self._rephrase_sentence(sent)

            # Skip if too short or too similar to existing points
            if len(rephrased.split()) < 6:
                continue
            dup = False
            for existing in points:
                overlap = set(rephrased.lower().split()) & set(existing.lower().split())
                if len(overlap) / max(len(rephrased.split()), 1) > 0.6:
                    dup = True
                    break
            if not dup:
                points.append(rephrased)

            if len(points) >= 5:
                break

        # Trim to 3–5
        return points[:5] if len(points) >= 3 else points

    def _build_example(
        self,
        selected: List[Tuple[str, float, str, str]],
        query_type: str,
    ) -> Optional[str]:
        """
        Find a sentence that reads like an example, formula, or definition
        and return it as an illustrative note. Returns None if none found.
        """
        example_signals = [
            r'\bfor example\b', r'\bfor instance\b', r'\bsuch as\b',
            r'\be\.g\.', r'\bi\.e\.', r'=', r'→', r':\s',
            r'\bstep\s+\d', r'\d+\.\s', r'\bformula\b', r'\bequation\b',
        ]
        pattern = re.compile('|'.join(example_signals), re.IGNORECASE)

        for sent, score, cid, source in selected:
            if pattern.search(sent) and len(sent.split()) > 8:
                return self._rephrase_sentence(sent)
        return None

    def _build_answer_text(
        self,
        sentences: List[Tuple[str, float, str, str]],
        query_type: str,
        keywords: List[str] = None,
    ) -> str:
        """
        Build a structured, clean answer following strict formatting rules:

        1. Direct answer paragraph (2–4 lines)
        2. Key Points section (3–5 bullet points)
        3. Optional example / clarification
        """
        keywords = keywords or []

        if not sentences:
            return "No relevant information was found for this question."

        # Deduplicate
        sentences = self._deduplicate_sentences(sentences)

        # Select top candidates
        seen_content: set = set()
        selected: List[Tuple] = []

        for item in sentences[:self.top_sentences * 3]:
            sent = item[0]
            key = sent[:50].lower()
            if key in seen_content:
                continue
            seen_content.add(key)
            selected.append(item)
            if len(selected) >= self.top_sentences * 2:
                break

        if not selected:
            return "No relevant information was found for this question."

        # ── 1. Direct answer paragraph ──────────────────────────
        summary = self._build_summary_paragraph(selected, query_type)

        # ── 2. Key points ────────────────────────────────────────
        key_points = self._build_key_points(selected, query_type, keywords)

        # ── 3. Optional example ──────────────────────────────────
        example = self._build_example(selected, query_type)

        # ── Assemble final answer ────────────────────────────────
        lines = []

        if summary:
            lines.append(summary)

        if key_points:
            lines.append("\n**Key Points:**")
            for pt in key_points:
                lines.append(f"- {pt}")

        if example and query_type in ("factual", "conceptual", "procedural"):
            lines.append(f"\n**Example / Note:** {example}")

        answer = "\n".join(lines).strip()
        return answer

    def _build_highlighted_passages(
        self,
        top_chunks: List[RetrievalResult],
        keywords: List[str],
    ) -> List[Dict]:
        """Build passage highlights with keyword marking."""
        passages = []
        for chunk in top_chunks[:4]:
            text = chunk.text

            # Find keyword positions
            highlighted_words = []
            for kw in keywords[:8]:
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                highlighted_words.extend([m.start() for m in pattern.finditer(text)])

            # Keep full passage text flexible for display
            display_text = text

            passages.append({
                "text": display_text,
                "full_text": text,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "similarity": chunk.similarity_score,
                "attention": chunk.hopfield_attention,
                "rank": chunk.rank,
                "keyword_hits": len(highlighted_words),
                "keywords_found": [
                    kw for kw in keywords
                    if kw.lower() in text.lower()
                ],
            })

        return passages

    def generate(
        self,
        processed_query: ProcessedQuery,
        retrieved_chunks: List[RetrievalResult],
    ) -> GeneratedAnswer:
        """
        Generate a complete answer from retrieved chunks.

        Args:
            processed_query: Processed query object.
            retrieved_chunks: List of retrieved chunks from Hopfield network.

        Returns:
            GeneratedAnswer with text, confidence, evidence, and metadata.
        """
        query = processed_query.cleaned
        keywords = processed_query.keywords
        query_type = processed_query.query_type

        # Handle empty retrieval
        if not retrieved_chunks:
            return GeneratedAnswer(
                query=query,
                answer="No relevant documents found. Please ensure documents are loaded and try rephrasing your query.",
                confidence=0.0,
                evidence_chunks=[],
                highlighted_passages=[],
                sources=[],
                answer_type="no_answer",
                generation_stats={"n_chunks": 0, "n_sentences": 0},
            )

        # Filter low-confidence chunks
        filtered_chunks = [
            c for c in retrieved_chunks
            if c.similarity_score >= self.min_confidence_threshold
        ]

        if not filtered_chunks:
            filtered_chunks = retrieved_chunks[:2]  # fallback

        # Extract best sentences
        scored_sentences = self._extract_best_sentences(
            filtered_chunks, keywords, query
        )

        # Build answer
        answer_text = self._build_answer_text(scored_sentences, query_type, keywords)

        # Compute confidence
        confidence = self._compute_confidence(filtered_chunks, scored_sentences)

        # Build highlighted passages
        highlights = self._build_highlighted_passages(filtered_chunks, keywords)

        # Unique sources
        sources = list(dict.fromkeys(c.source for c in filtered_chunks))

        # Determine answer type
        if confidence > 0.6:
            answer_type = "extractive"
        elif confidence > 0.3:
            answer_type = "hybrid"
        else:
            answer_type = "low_confidence"

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            confidence=confidence,
            evidence_chunks=filtered_chunks,
            highlighted_passages=highlights,
            sources=sources,
            answer_type=answer_type,
            generation_stats={
                "n_chunks": len(filtered_chunks),
                "n_sentences_considered": len(scored_sentences),
                "n_sentences_used": min(self.top_sentences, len(scored_sentences)),
                "query_type": query_type,
                "n_keywords": len(keywords),
            },
        )


class LLMAnswerGenerator(AnswerGenerator):
    """
    Enhanced answer generator that uses an LLM for abstractive synthesis.
    Falls back to extractive if LLM is unavailable.
    """

    def __init__(
        self,
        llm_provider: str = "local",  # "local" | "openai" | "anthropic"
        model_name: str = "gpt2",
        max_context_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self._llm = None

    def _build_context(self, chunks: List[RetrievalResult]) -> str:
        """Build context string from top chunks."""
        context_parts = []
        total_chars = 0
        char_limit = self.max_context_tokens * 4  # rough token-to-char ratio

        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Passage {i} from '{chunk.source}']\n{chunk.text}"
            if total_chars + len(chunk_text) > char_limit:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str, query_type: str) -> str:
        """Construct the LLM prompt."""
        type_instruction = {
            "factual": "Provide a concise, accurate factual answer.",
            "procedural": "Provide step-by-step instructions.",
            "comparative": "Compare and contrast the relevant aspects clearly.",
            "conceptual": "Explain the concept clearly with relevant details.",
        }.get(query_type, "Answer the question based on the provided passages.")

        return f"""You are a helpful assistant answering questions based on provided document passages.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTION: {type_instruction} Use only information from the provided passages. If the passages don't contain sufficient information, say so clearly.

ANSWER:"""

    def generate(
        self,
        processed_query: ProcessedQuery,
        retrieved_chunks: List[RetrievalResult],
    ) -> GeneratedAnswer:
        """Generate answer with LLM if available, else fall back to extractive."""
        # Try LLM generation
        if self.llm_provider != "local":
            try:
                return self._llm_generate(processed_query, retrieved_chunks)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}. Falling back to extractive.")

        # Fallback to extractive
        result = super().generate(processed_query, retrieved_chunks)
        return result

    def _llm_generate(
        self,
        processed_query: ProcessedQuery,
        retrieved_chunks: List[RetrievalResult],
    ) -> GeneratedAnswer:
        """Generate using external LLM API."""
        context = self._build_context(retrieved_chunks[:5])
        prompt = self._build_prompt(
            processed_query.cleaned, context, processed_query.query_type
        )

        # Placeholder: in production, call your LLM API here
        # response = call_llm_api(prompt, self.model_name)
        raise NotImplementedError("External LLM integration not configured.")
from typing import List, Dict, Any, Tuple


class ExtractiveQAAnswerGenerator(AnswerGenerator):
    """
    End-to-End Extractive QA generator using a pre-trained QA model.
    """
    def __init__(self, model_name='deepset/roberta-base-squad2', **kwargs):
        super().__init__(**kwargs)
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            import torch
        except ImportError:
            raise ImportError("Please install transformers and torch to use ExtractiveQAAnswerGenerator")
            
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def generate(self, processed_query: ProcessedQuery, retrieved_chunks: List[RetrievalResult]) -> GeneratedAnswer:
        import torch
        
        if not retrieved_chunks:
            return GeneratedAnswer(
                query=processed_query.original,
                answer="I could not find any relevant information to answer your question.",
                confidence=0.0,
                evidence_chunks=[],
                highlighted_passages=[],
                sources=[],
                answer_type="extractive_span",
                generation_stats={"model": self.model_name}
            )

        question = processed_query.original
        
        # Proper chunking to prevent truncation loss
        best_score = float('-inf')
        best_start = 0
        best_end = 0
        best_input_ids = None
        best_chunk = retrieved_chunks[0]
        
        for chunk in retrieved_chunks:
            context = chunk.text if hasattr(chunk, 'text') else chunk['text']
            
            inputs = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                truncation="only_second",
                max_length=512,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
            
            # evaluate each window
            for i in range(len(inputs["input_ids"])):
                input_ids = inputs["input_ids"][i].unsqueeze(0)
                attention_mask = inputs["attention_mask"][i].unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    start_logits = outputs.start_logits[0]
                    end_logits = outputs.end_logits[0]
                    
                # Exclude question tokens from answer space
                sequence_ids = inputs.sequence_ids(i)
                context_mask = [1 if s == 1 else 0 for s in sequence_ids]
                
                # Mask out logits outside of context
                for idx, mask_val in enumerate(context_mask):
                    if mask_val == 0:
                        start_logits[idx] = -10000
                        end_logits[idx] = -10000
                
                start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
                end_probs = torch.nn.functional.softmax(end_logits, dim=-1)
                
                start_idx = torch.argmax(start_logits)
                end_idx = torch.argmax(end_logits)
                
                score = (start_probs[start_idx] * end_probs[end_idx]).item()
                if score > best_score and start_idx <= end_idx:
                    best_score = score
                    best_start = start_idx
                    best_end = end_idx
                    best_input_ids = input_ids[0]
                    best_chunk = chunk

        if best_score > float('-inf') and best_start > 0 and best_start <= best_end:
            answer_tokens = best_input_ids[best_start:best_end+1]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            conf = min(best_score, 1.0)
        else:
            first_text = retrieved_chunks[0].text if hasattr(retrieved_chunks[0], 'text') else retrieved_chunks[0]['text']
            answer_text = "Could not extract a specific answer span. Context suggests: " + first_text[:200] + "..."
            conf = 0.3
            
        return GeneratedAnswer(
            query=question,
            answer=answer_text,
            confidence=conf,
            evidence_chunks=retrieved_chunks,
            highlighted_passages=self._build_highlighted_passages(retrieved_chunks, processed_query.keywords),
            sources=list(dict.fromkeys(c.source if hasattr(c, 'source') else c.get('metadata', {}).get('source', 'Unknown') for c in retrieved_chunks)),
            answer_type="extractive_span",
            generation_stats={"model": self.model_name}
        )
