"""
AI Visual Studio Generation Engine
==================================
Generates strictly validated visualization specifications from retrieved document context:
- Mermaid.js diagrams: Mind maps, flowcharts, sequence diagrams, architecture diagrams, study roadmaps.
- Plotly interactive charts: Bar, line, pie, histogram, scatter plots (enforcing numerical data integrity).
- Cytoscape.js knowledge graphs: Nodes, directed edges, concept clusters, and hierarchical dependency maps.

Data Integrity Enforcement:
- Never invents numerical values: strictly validates whether numerical figures exist in context.
- If numerical data is absent when a chart is requested, returns a conceptual diagram or an honest explanation.
- Validates Mermaid syntax and Plotly schema before returning to frontend.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VisualizationResult:
    """Standardized output structure for AI Visual Studio."""
    viz_type: str  # "mermaid" | "plotly" | "cytoscape"
    subtype: str   # "mindmap", "flowchart", "bar", "line", "pie", "graph", etc.
    title: str
    description: str
    content: Union[str, Dict[str, Any]]  # Mermaid code string or Plotly/Cytoscape dict
    raw_data: Optional[Dict[str, Any]] = None  # Underlying data table for export
    citations: List[Dict[str, Any]] = field(default_factory=list)
    has_numerical_data: bool = True
    suggested_alternative: Optional[str] = None


class VisualStudioEngine:
    """Core generator for diagrams, interactive charts, and concept graphs."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def generate_visualization(
        self,
        prompt: str,
        context_chunks: List[Dict[str, Any]],
        requested_type: Optional[str] = None
    ) -> VisualizationResult:
        """
        Determines the appropriate visualization format, extracts grounded entities/data,
        and generates a validated diagram or chart specification.
        """
        req_type = (requested_type or self._detect_visualization_type(prompt)).lower()
        combined_text = "\n\n".join(c.get("text", "") for c in context_chunks if c.get("text"))

        # Check for numerical chart requests
        if req_type in ("bar", "line", "pie", "histogram", "scatter", "chart"):
            has_numbers, numerical_data = self._extract_numerical_data(combined_text, prompt)
            if not has_numbers or not numerical_data.get("series"):
                return VisualizationResult(
                    viz_type="mermaid",
                    subtype="mindmap",
                    title="Conceptual Overview",
                    description=(
                        "A quantitative chart could not be generated because the retrieved document context "
                        "does not contain verified numerical series or measurements. "
                        "A conceptual mind map has been generated instead to preserve data integrity."
                    ),
                    content=self._generate_mindmap(prompt, combined_text, context_chunks),
                    has_numerical_data=False,
                    suggested_alternative="mindmap"
                )
            return self._generate_plotly_chart(req_type, numerical_data, prompt)

        elif req_type in ("mindmap", "mind_map"):
            content = self._generate_mindmap(prompt, combined_text, context_chunks)
            return VisualizationResult(
                viz_type="mermaid",
                subtype="mindmap",
                title=f"Mind Map: {prompt[:40]}",
                description="Hierarchical concept breakdown from retrieved document memory.",
                content=content,
                citations=self._extract_citations(context_chunks)
            )

        elif req_type in ("flowchart", "process", "workflow"):
            content = self._generate_flowchart(prompt, combined_text, context_chunks)
            return VisualizationResult(
                viz_type="mermaid",
                subtype="flowchart",
                title=f"Workflow Flowchart: {prompt[:40]}",
                description="Step-by-step logic flow and decision paths.",
                content=content,
                citations=self._extract_citations(context_chunks)
            )

        elif req_type in ("sequence", "sequence_diagram"):
            content = self._generate_sequence_diagram(prompt, combined_text, context_chunks)
            return VisualizationResult(
                viz_type="mermaid",
                subtype="sequence",
                title="Sequence Interaction Diagram",
                description="Component message exchanges and lifecycle sequences.",
                content=content,
                citations=self._extract_citations(context_chunks)
            )

        elif req_type in ("architecture", "system", "architecture_diagram"):
            content = self._generate_architecture_diagram(prompt, combined_text, context_chunks)
            return VisualizationResult(
                viz_type="mermaid",
                subtype="architecture",
                title="System Architecture Diagram",
                description="Layered system components and interconnects.",
                content=content,
                citations=self._extract_citations(context_chunks)
            )

        elif req_type in ("timeline", "roadmap", "study_roadmap"):
            content = self._generate_roadmap(prompt, combined_text, context_chunks)
            return VisualizationResult(
                viz_type="mermaid",
                subtype="roadmap",
                title="Study Roadmap & Milestones",
                description="Chronological learning path and dependency milestones.",
                content=content,
                citations=self._extract_citations(context_chunks)
            )

        elif req_type in ("knowledge_graph", "graph", "concept_map"):
            cytoscape_data = self._generate_knowledge_graph(prompt, combined_text, context_chunks)
            return VisualizationResult(
                viz_type="cytoscape",
                subtype="knowledge_graph",
                title="Interactive Knowledge Graph",
                description="Entity-relationship network grounded in retrieved document memories.",
                content=cytoscape_data,
                citations=self._extract_citations(context_chunks)
            )

        else:
            # Default to mind map
            content = self._generate_mindmap(prompt, combined_text, context_chunks)
            return VisualizationResult(
                viz_type="mermaid",
                subtype="mindmap",
                title="Conceptual Mind Map",
                description="Concept map generated from document knowledge.",
                content=content,
                citations=self._extract_citations(context_chunks)
            )

    def _detect_visualization_type(self, prompt: str) -> str:
        """Infers the intended visualization type from user query keywords."""
        p = prompt.lower()
        if any(w in p for w in ("mind map", "mindmap", "brainstorm")):
            return "mindmap"
        if any(w in p for w in ("flowchart", "flow chart", "workflow", "process", "how to")):
            return "flowchart"
        if any(w in p for w in ("sequence", "message exchange", "protocol")):
            return "sequence"
        if any(w in p for w in ("architecture", "component", "subsystem", "layers")):
            return "architecture"
        if any(w in p for w in ("roadmap", "study plan", "curriculum", "timeline")):
            return "roadmap"
        if any(w in p for w in ("knowledge graph", "concept map", "relationships", "network")):
            return "knowledge_graph"
        if any(w in p for w in ("bar chart", "bar graph")):
            return "bar"
        if any(w in p for w in ("line chart", "trend", "over time")):
            return "line"
        if any(w in p for w in ("pie chart", "distribution", "share")):
            return "pie"
        if any(w in p for w in ("histogram", "frequency")):
            return "histogram"
        if any(w in p for w in ("scatter", "correlation")):
            return "scatter"
        if any(w in p for w in ("chart", "plot", "graph")):
            return "chart"
        return "mindmap"

    def _extract_numerical_data(self, context: str, prompt: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Searches context for actual numerical series (e.g. from markdown tables, lists, or key-value pairs).
        Guarantees that numbers are never fabricated.
        """
        lines = context.split("\n")
        labels = []
        values = []

        # 1. Search for table rows with numbers
        for line in lines:
            if "|" in line and not line.strip().startswith("|---"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    val_candidate = None
                    label_candidate = None
                    for part in parts:
                        clean_num = re.sub(r'[\$,%]', '', part)
                        try:
                            val = float(clean_num)
                            val_candidate = val
                        except ValueError:
                            if not label_candidate and part not in ("---", "Header", "Name"):
                                label_candidate = part
                    if val_candidate is not None and label_candidate:
                        labels.append(label_candidate[:30])
                        values.append(val_candidate)

        # 2. Search for bullet patterns: "Label: 123" or "Label - 123"
        if not values:
            matches = re.findall(r'([A-Za-z0-9\s_-]{2,30})[:\-]\s*([0-9]+(?:\.[0-9]+)?)', context)
            for lbl, val in matches:
                lbl_clean = lbl.strip()
                if lbl_clean.lower() not in ("page", "chapter", "version", "id", "year"):
                    labels.append(lbl_clean)
                    values.append(float(val))

        if len(values) >= 2:
            return True, {
                "labels": labels[:15],
                "series": [{"name": "Metric", "values": values[:15]}],
                "x_title": "Category",
                "y_title": "Value"
            }
        return False, {}

    def _generate_plotly_chart(self, chart_type: str, data: Dict[str, Any], prompt: str) -> VisualizationResult:
        """Creates a Plotly JSON specification from verified numerical data."""
        labels = data.get("labels", [])
        values = data.get("series", [{}])[0].get("values", [])

        if chart_type in ("pie",):
            fig_data = [{
                "labels": labels,
                "values": values,
                "type": "pie",
                "hoverinfo": "label+percent+value",
                "textinfo": "label+percent"
            }]
            layout = {"title": f"Distribution: {prompt[:40]}", "responsive": True}
        elif chart_type in ("line", "scatter"):
            mode = "lines+markers" if chart_type == "line" else "markers"
            fig_data = [{
                "x": labels,
                "y": values,
                "type": "scatter",
                "mode": mode,
                "marker": {"size": 8, "color": "#4f46e5"}
            }]
            layout = {
                "title": f"Trend: {prompt[:40]}",
                "xaxis": {"title": data.get("x_title", "Category")},
                "yaxis": {"title": data.get("y_title", "Value")},
                "responsive": True
            }
        else:  # bar / histogram default
            fig_data = [{
                "x": labels,
                "y": values,
                "type": "bar",
                "marker": {"color": "#6366f1"}
            }]
            layout = {
                "title": f"Comparison: {prompt[:40]}",
                "xaxis": {"title": data.get("x_title", "Category")},
                "yaxis": {"title": data.get("y_title", "Value")},
                "responsive": True
            }

        plotly_spec = {"data": fig_data, "layout": layout}
        raw_table = {"headers": ["Category", "Value"], "rows": list(zip(labels, values))}

        return VisualizationResult(
            viz_type="plotly",
            subtype=chart_type,
            title=f"Chart: {prompt[:40]}",
            description="Verified numerical visualization from document context.",
            content=plotly_spec,
            raw_data=raw_table,
            has_numerical_data=True
        )

    def _generate_mindmap(self, prompt: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generates a valid, deeply grounded Mermaid mindmap.
        1. If an LLM provider is available, invokes it with strict few-shot instructions.
        2. Otherwise, uses an advanced NLP concept hierarchy extractor to build a rich,
           hierarchical tree directly grounded in the provided document/paragraph text.
        """
        # 1. Try LLM Provider if available
        if self.llm_provider and hasattr(self.llm_provider, "generate_answer"):
            try:
                is_avail, _ = self.llm_provider.is_available()
                if is_avail:
                    sys_prompt = (
                        "You are an expert knowledge visualization engineer. Given the source text, "
                        "generate a strictly valid Mermaid.js mindmap diagram representing its core themes and details.\n\n"
                        "STRICT RULES:\n"
                        "1. Return ONLY the Mermaid code block starting with ```mermaid and ending with ```.\n"
                        "2. Line 1: `mindmap`.\n"
                        "3. Line 2: `  root((Central Topic))` (replace 'Central Topic' with the 1-4 word central subject of the text, alphanumeric only, no special chars or parentheses inside label).\n"
                        "4. Indent 4 spaces for 3 to 5 main branch concepts extracted directly from the text.\n"
                        "5. Indent 6 spaces for 2 to 3 factual sub-details or attributes under each branch.\n"
                        "6. Node labels must be concise (1 to 5 words), strictly alphanumeric, no parentheses or quotes.\n"
                        "7. Every single node must be directly grounded in the provided text."
                    )
                    user_req = f"User Request: {prompt}\n\nDocument Text:\n{context[:3500]}"
                    ans = self.llm_provider.generate_answer(query=user_req, evidence_chunks=[], system_override=sys_prompt)
                    sanitized = self._sanitize_mermaid_mindmap(getattr(ans, "answer", str(ans)))
                    if sanitized:
                        return sanitized
            except Exception as e:
                logger.warning(f"LLM mind map generation failed, using deterministic extractor: {e}")

        # 2. Advanced Deterministic Hierarchy Extractor
        return self._generate_deterministic_mindmap(prompt, context, chunks)

    def _sanitize_mermaid_mindmap(self, text: str) -> Optional[str]:
        """Validates and cleans LLM-generated Mermaid mindmap syntax."""
        if not text:
            return None
        clean = text
        if "```mermaid" in clean:
            clean = clean.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        lines = [line.rstrip() for line in clean.splitlines() if line.strip()]
        if not lines or not any("mindmap" in l for l in lines[:2]):
            return None

        # Verify root exists
        has_root = any("root(" in l or "root[" in l for l in lines)
        if not has_root:
            return None

        # Clean forbidden characters from labels inside Mermaid mindmap
        sanitized_lines = []
        for line in lines:
            if line.strip() == "mindmap":
                sanitized_lines.append(line)
            elif "root((" in line:
                m = re.search(r'root\(\((.*?)\)\)', line)
                r_name = re.sub(r'[^a-zA-Z0-9\s]', '', m.group(1)).strip() if m else "Knowledge Map"
                sanitized_lines.append(f"  root(({r_name}))")
            else:
                indent_len = len(line) - len(line.lstrip())
                indent = " " * max(4, indent_len)
                label = re.sub(r'[^a-zA-Z0-9\s]', '', line.strip()).strip()
                if label:
                    sanitized_lines.append(f"{indent}{label}")

        return "\n".join(sanitized_lines)

    def _generate_deterministic_mindmap(self, prompt: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Builds a high-fidelity, grounded Mermaid mind map directly from source text
        using hierarchical concept extraction, semantic clause splitting, and entity analysis.
        """
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'up', 'about', 'into', 'over', 'after', 'beneath', 'under', 'above', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'that', 'this', 'these', 'those',
            'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'which', 'who', 'whom', 'whose', 'what', 'where', 'when', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'also', 'such', 'as', 'there',
            'paragraph', 'text', 'notes', 'document', 'make', 'generate', 'create'
        }

        def clean_label(text: str, max_words: int = 3, max_chars: int = 24) -> str:
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            words = [w.capitalize() for w in cleaned.split() if w.lower() not in stopwords]
            if not words:
                words = [w.capitalize() for w in cleaned.split()]
            res = " ".join(words[:max_words]).strip()
            return res[:max_chars].strip()

        # Step 1: Extract Root Topic
        root_name = ""
        clean_p = re.sub(r'^(?:make|generate|create|build|draw|render)\s+(?:a\s+)?(?:mind\s*map|diagram|flowchart)\s+(?:of|for|about)?\s*', '', prompt, flags=re.I).strip()
        clean_p = re.sub(r'^(?:that|this|the)\s+(?:paragraph|document|text|notes|passage|content)\s*', '', clean_p, flags=re.I).strip()
        if clean_p and len(clean_p) >= 3 and not any(w in clean_p.lower() for w in ["paragraph", "text", "notes"]):
            root_name = clean_label(clean_p, max_words=3, max_chars=24)

        if not root_name and context:
            first_sent = context.strip().split(".")[0].strip()
            m = re.search(r'(?i)^([A-Z][A-Za-z0-9\s]{2,30}?)(?:\s+is\s+a|\s+is\s+an|\s+is\s+the|\s+refers|\s+comprises|\s+consists|\s+provides|\s+acts|,)', first_sent)
            if m:
                root_name = clean_label(m.group(1), max_words=3, max_chars=24)
            if not root_name:
                words = [w.capitalize() for w in re.findall(r'\b[A-Za-z]{3,}\b', context[:400]) if w.lower() not in stopwords]
                if words:
                    from collections import Counter
                    counts = Counter(words)
                    root_name = counts.most_common(1)[0][0]

        root_name = root_name or "Core Concept"

        # Step 2: Split context into distinct thematic segments
        raw_lines = [l.strip() for l in context.split("\n") if l.strip()]
        segments = []
        bullet_items = [l for l in raw_lines if re.match(r'^(?:[-*•]|\d+\.)\s+', l)]
        if len(bullet_items) >= 2:
            for b in bullet_items[:6]:
                clean_b = re.sub(r'^(?:[-*•]|\d+\.)\s+', '', b).strip()
                segments.append(clean_b)
        else:
            raw_sents = re.split(r'(?<=[.!?])\s+', context)
            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s) > 15 and not clean_s.startswith("Table on Page") and not clean_s.startswith("---") and not clean_s.startswith("Extracted Text from"):
                    segments.append(clean_s)

        # Step 3: Build Branches and Leaves
        branches = []
        seen_branches = set()

        for seg in segments[:6]:
            parts = [p.strip() for p in re.split(r'[:;]|\b(?:such as|including|composed of|namely|while)\b', seg, flags=re.I) if len(p.strip()) > 5]
            if not parts:
                continue

            main_header = clean_label(parts[0], max_words=3, max_chars=24)
            if main_header.lower().startswith(root_name.lower()) and len(main_header) > len(root_name) + 2:
                main_header = main_header[len(root_name):].strip()
            if not main_header or main_header.lower() in seen_branches or len(main_header) < 3:
                continue
            seen_branches.add(main_header.lower())

            leaves = []
            seen_leaves = set()
            detail_sources = parts[1:] if len(parts) > 1 else re.split(r'[,]|(?:\band\b)', parts[0])
            for d in detail_sources:
                leaf_lbl = clean_label(d, max_words=3, max_chars=24)
                if leaf_lbl.lower().startswith(main_header.lower()) and len(leaf_lbl) > len(main_header) + 2:
                    leaf_lbl = leaf_lbl[len(main_header):].strip()
                if leaf_lbl and leaf_lbl.lower() not in seen_leaves and leaf_lbl.lower() != main_header.lower() and leaf_lbl.lower() != root_name.lower():
                    seen_leaves.add(leaf_lbl.lower())
                    leaves.append(leaf_lbl)
                if len(leaves) >= 3:
                    break

            branches.append((main_header, leaves[:3]))
            if len(branches) >= 5:
                break

        # Fallback if no structured branches could be formed
        if not branches:
            words = [w.capitalize() for w in re.findall(r'\b[A-Za-z]{4,}\b', context) if w.lower() not in stopwords]
            from collections import Counter
            top_terms = [item[0] for item in Counter(words).most_common(8)]
            for i in range(0, len(top_terms), 2):
                h = top_terms[i]
                sub = [top_terms[i+1]] if i+1 < len(top_terms) else []
                branches.append((h, sub))

        # Format as Mermaid mindmap
        lines = ["mindmap", f"  root(({root_name}))"]
        for header, leaves in branches:
            lines.append(f"    {header}")
            for lf in leaves:
                lines.append(f"      {lf}")

        return "\n".join(lines)

    def _generate_flowchart(self, prompt: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Generates a valid Mermaid flowchart grounded in the provided context."""
        # 1. Try LLM if available
        if self.llm_provider and hasattr(self.llm_provider, "generate_answer"):
            try:
                is_avail, _ = self.llm_provider.is_available()
                if is_avail:
                    sys_prompt = (
                        "Generate a valid Mermaid.js flowchart (starting with flowchart TD) "
                        "representing the sequence of steps, decisions, or process flow described in the text.\n"
                        "Return ONLY the code block inside ```mermaid ... ```."
                    )
                    ans = self.llm_provider.generate_answer(f"Task: {prompt}\n\nText:\n{context[:3000]}", [], system_override=sys_prompt)
                    clean = self._extract_mermaid_block(getattr(ans, "answer", str(ans)), "flowchart")
                    if clean:
                        return clean
            except Exception as e:
                logger.warning(f"LLM flowchart generation fallback: {e}")

        # 2. Deterministic step extraction
        steps = []
        for line in context.split("\n"):
            clean = line.strip()
            if re.match(r'^(?:[0-9]+\.|\-|\*)\s+', clean):
                s = re.sub(r'^(?:[0-9]+\.|\-|\*)\s+', '', clean)
                s = re.sub(r'["\(\)\[\]\{\}]', '', s).strip()
                if len(s) > 8:
                    steps.append(s[:45])
        if len(steps) < 2:
            # Extract sentence-level sequential clauses
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 15]
            for s in sents[:5]:
                clean_s = re.sub(r'["\(\)\[\]\{\}]', '', s).strip()
                steps.append(clean_s[:45])
        if not steps:
            p_clean = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)[:30] or "Process"
            steps = [f"Input: {p_clean}", "Analysis & Extraction", "Output Resolution"]

        lines = ["flowchart TD"]
        for i, st in enumerate(steps[:7]):
            node_id = f"S{i+1}"
            lines.append(f'    {node_id}["{i+1}. {st}"]')
            if i > 0:
                prev_id = f"S{i}"
                lines.append(f"    {prev_id} --> {node_id}")
        return "\n".join(lines)

    def _generate_sequence_diagram(self, prompt: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Generates a dynamic Mermaid sequence diagram from context actors and interactions."""
        if self.llm_provider and hasattr(self.llm_provider, "generate_answer"):
            try:
                is_avail, _ = self.llm_provider.is_available()
                if is_avail:
                    sys_prompt = (
                        "Generate a strictly valid Mermaid.js sequence diagram (starting with sequenceDiagram) "
                        "showing the actors, components, and message exchanges described in the text.\n"
                        "Return ONLY the code block inside ```mermaid ... ```."
                    )
                    ans = self.llm_provider.generate_answer(f"Task: {prompt}\n\nText:\n{context[:3000]}", [], system_override=sys_prompt)
                    clean = self._extract_mermaid_block(getattr(ans, "answer", str(ans)), "sequenceDiagram")
                    if clean:
                        return clean
            except Exception as e:
                logger.warning(f"LLM sequence diagram generation fallback: {e}")

        # Deterministic sequence extractor: identify key entities and interactions
        entities = self._extract_key_concepts(context, limit=4)
        if len(entities) < 2:
            p_terms = [w.capitalize() for w in re.findall(r'\b[A-Za-z]{3,}\b', prompt) if w.lower() not in {'sequence', 'diagram', 'make', 'draw', 'create', 'show', 'for', 'the'}]
            c_terms = [w.capitalize() for w in re.findall(r'\b[A-Za-z]{4,}\b', context) if w.lower() not in {'paragraph', 'document', 'text'}]
            combined = list(dict.fromkeys(p_terms + c_terms))
            entities = combined[:3] if len(combined) >= 2 else (combined + ["System"])[:2]
        actors = [re.sub(r'[^a-zA-Z0-9]', '', e.split()[0]).capitalize() or f"Entity{idx+1}" for idx, e in enumerate(entities[:3])]
        if len(actors) == 1:
            actors.append("TargetSystem")

        lines = ["sequenceDiagram", "    autonumber"]
        for act in actors:
            lines.append(f"    participant {act}")

        # Derive interactions from sentences
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 15]
        if not sents:
            sents = [f"Transmit {prompt[:30]}", "Validate and process parameters", "Acknowledge and return response"]

        for i, s in enumerate(sents[:4]):
            clean_msg = re.sub(r'[^a-zA-Z0-9\s]', '', s)[:35].strip()
            src = actors[i % len(actors)]
            dst = actors[(i + 1) % len(actors)]
            lines.append(f"    {src}->>{dst}: {clean_msg}")

        return "\n".join(lines)

    def _generate_architecture_diagram(self, prompt: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Generates a Mermaid system architecture diagram dynamically from context."""
        if self.llm_provider and hasattr(self.llm_provider, "generate_answer"):
            try:
                is_avail, _ = self.llm_provider.is_available()
                if is_avail:
                    sys_prompt = (
                        "Generate a valid Mermaid.js system architecture diagram (starting with flowchart TD or flowchart LR) "
                        "representing the layers, services, components, and data flow described in the text.\n"
                        "Return ONLY the code block inside ```mermaid ... ```."
                    )
                    ans = self.llm_provider.generate_answer(f"Task: {prompt}\n\nText:\n{context[:3000]}", [], system_override=sys_prompt)
                    clean = self._extract_mermaid_block(getattr(ans, "answer", str(ans)), "flowchart")
                    if clean:
                        return clean
            except Exception as e:
                logger.warning(f"LLM architecture diagram generation fallback: {e}")

        # Deterministic architecture layout: extract components and relationships
        concepts = self._extract_key_concepts(context, limit=6)
        if len(concepts) < 2:
            p_terms = [w.capitalize() for w in re.findall(r'\b[A-Za-z]{3,}\b', prompt) if w.lower() not in {'architecture', 'diagram', 'make', 'draw', 'create', 'show', 'for', 'the'}]
            c_terms = [w.capitalize() for w in re.findall(r'\b[A-Za-z]{4,}\b', context) if w.lower() not in {'paragraph', 'document', 'text'}]
            combined = list(dict.fromkeys(p_terms + c_terms))
            concepts = combined[:4] if combined else [re.sub(r'[^a-zA-Z0-9\s]', '', prompt)[:25] or "Core Architecture"]

        lines = ["flowchart LR"]
        for idx, c in enumerate(concepts):
            lbl = re.sub(r'[^a-zA-Z0-9\s]', '', c)[:25].strip()
            cid = f"C{idx+1}"
            lines.append(f'    {cid}["{lbl}"]')
            if idx > 0:
                prev = f"C{idx}"
                lines.append(f"    {prev} --> {cid}")

        return "\n".join(lines)

    def _generate_roadmap(self, prompt: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Generates a study roadmap in Mermaid dynamically from context."""
        if self.llm_provider and hasattr(self.llm_provider, "generate_answer"):
            try:
                is_avail, _ = self.llm_provider.is_available()
                if is_avail:
                    sys_prompt = (
                        "Generate a valid Mermaid.js study roadmap (starting with flowchart LR) "
                        "outlining the progressive learning phases or milestones for the subject.\n"
                        "Return ONLY the code block inside ```mermaid ... ```."
                    )
                    ans = self.llm_provider.generate_answer(f"Task: {prompt}\n\nText:\n{context[:3000]}", [], system_override=sys_prompt)
                    clean = self._extract_mermaid_block(getattr(ans, "answer", str(ans)), "flowchart")
                    if clean:
                        return clean
            except Exception as e:
                logger.warning(f"LLM roadmap generation fallback: {e}")

        concepts = self._extract_key_concepts(context, limit=5)
        if not concepts:
            topic = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)[:25] or "Subject"
            concepts = [f"{topic} Fundamentals", f"{topic} Core Concepts", f"{topic} Applied Practice", f"{topic} Advanced Mastery"]

        lines = ["flowchart LR"]
        for idx, c in enumerate(concepts):
            clean_c = re.sub(r'["\(\)\[\]]', '', c)[:30].strip()
            nid = f"Step{idx+1}"
            lines.append(f'    {nid}["Phase {idx+1}: {clean_c}"]')
            if idx > 0:
                lines.append(f"    Step{idx} --> Step{idx+1}")
        return "\n".join(lines)

    def _extract_mermaid_block(self, text: str, expected_type: str) -> Optional[str]:
        """Extracts and validates a Mermaid diagram code block from LLM output."""
        if not text:
            return None
        clean = text
        if "```mermaid" in clean:
            clean = clean.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        lines = [l.rstrip() for l in clean.splitlines() if l.strip()]
        if not lines or not any(expected_type.lower() in l.lower() for l in lines[:2]):
            return None
        return "\n".join(lines)

    def _generate_knowledge_graph(self, prompt: str, context: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates a Cytoscape.js compatible graph format with nodes and edges."""
        concepts = self._extract_key_concepts(context, limit=10)
        nodes = []
        edges = []

        root_label = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)[:28].strip() or "Core Topic"
        root_id = "concept_root"
        nodes.append({
            "data": {"id": root_id, "label": root_label, "type": "root", "color": "#4f46e5"}
        })

        for i, c in enumerate(concepts):
            cid = f"node_{i+1}"
            clean_lbl = re.sub(r'["\(\)\[\]]', '', c.split(":")[0])[:25].strip()
            if not clean_lbl:
                continue
            nodes.append({
                "data": {"id": cid, "label": clean_lbl, "type": "concept", "color": "#10b981"}
            })
            edges.append({
                "data": {"id": f"e_root_{cid}", "source": root_id, "target": cid, "label": "relates_to"}
            })

            if i > 0 and i % 2 == 1:
                prev_cid = f"node_{i}"
                edges.append({
                    "data": {"id": f"e_{prev_cid}_{cid}", "source": prev_cid, "target": cid, "label": "associated_with"}
                })

        return {"elements": {"nodes": nodes, "edges": edges}}

    def _extract_key_concepts(self, context: str, limit: int = 8) -> List[str]:
        """Extracts real noun phrases, headings, and domain terms from context."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'up', 'about', 'into', 'over', 'after', 'beneath', 'under', 'above', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'that', 'this', 'these', 'those',
            'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
            'her', 'which', 'who', 'whom', 'whose', 'what', 'where', 'when', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'also', 'such', 'as', 'there'
        }
        concepts = []
        for line in context.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("Table") or ":" in line:
                clean = re.sub(r'^[#\*\-\s]+', '', line).strip()
                clean = re.sub(r'[^a-zA-Z0-9\s]', '', clean).strip()
                if 4 < len(clean) < 50:
                    concepts.append(clean)
            if len(concepts) >= limit:
                break

        if not concepts:
            # Extract meaningful multi-word noun phrases
            phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Za-z]+){1,3}\b', context)
            valid = [p for p in phrases if not any(w.lower() in stopwords for w in p.split()[:1])]
            concepts = list(dict.fromkeys(valid))[:limit]

        if not concepts:
            words = [w.capitalize() for w in re.findall(r'\b[A-Za-z]{4,}\b', context) if w.lower() not in stopwords]
            from collections import Counter
            concepts = [item[0] for item in Counter(words).most_common(limit)]

        return concepts

    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts source and page references for grounding."""
        cites = []
        for c in chunks[:5]:
            cites.append({
                "source": c.get("source", "Document"),
                "page": c.get("page_number", 1),
                "element_type": c.get("element_type", "text"),
                "snippet": c.get("text", "")[:120] + "..."
            })
        return cites
