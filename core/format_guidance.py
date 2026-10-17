"""
Answer-Format Guidance and Examination Rubric Knowledge Base
============================================================
Provides reusable, structured answer-format guidance for:
- 2-mark examination answers
- 5-mark examination answers
- 8-mark examination answers
- 10-mark examination answers
- Comparison and contrast tables
- Step-by-step technical explanations
- Worked examples
- Structured revision notes
- Brief factual and executive briefings
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class AnswerFormatGuidance:
    """Reusable specification for structuring answers to meet user or exam expectations."""
    format_id: str
    display_name: str
    target_marks: Optional[int]
    expected_word_range: str
    target_components: List[str]
    structural_outline: List[str]
    system_instruction_prompt: str
    rubric_guidelines: List[str]

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def mark_weight(self) -> Optional[int]:
        return self.target_marks

    @property
    def description(self) -> str:
        return self.expected_word_range

    @property
    def outline(self) -> List[str]:
        return self.structural_outline


FORMAT_CATALOG: Dict[str, AnswerFormatGuidance] = {
    "2_mark": AnswerFormatGuidance(
        format_id="2_mark",
        display_name="2-Mark Exam Answer",
        target_marks=2,
        expected_word_range="50-100 words (2-4 concise sentences or 2 distinct points)",
        target_components=[
            "Direct core definition or main thesis statement",
            "Exactly 2 distinct key technical points, functions, or characteristics"
        ],
        structural_outline=[
            "1. Definitional Sentence: Precise definition of the requested concept [E#].",
            "2. Core Points: Exactly two bullet points highlighting primary function and characteristics [E#]."
        ],
        system_instruction_prompt=(
            "FORMAT AS A PRECISE 2-MARK EXAMINATION ANSWER:\n"
            "1. Sentence 1 MUST provide a clear, unambiguous textbook definition or direct answer citing [E#].\n"
            "2. Follow with exactly TWO concise bullet points highlighting key functions, components, or characteristics.\n"
            "3. Keep strictly factual and concise (~50-100 words). Do not include unnecessary padding or conversational preamble."
        ),
        rubric_guidelines=[
            "1 mark awarded for accurate definition/concept statement.",
            "1 mark awarded for two distinct, relevant technical points or mechanisms."
        ]
    ),
    "5_mark": AnswerFormatGuidance(
        format_id="5_mark",
        display_name="5-Mark Exam Answer",
        target_marks=5,
        expected_word_range="200-350 words (Definition + 3-5 structured explanatory points + example)",
        target_components=[
            "Clear introductory definition and scope",
            "3-5 structured technical points covering architecture, mechanisms, or taxonomy",
            "Concrete practical example or deployment context"
        ],
        structural_outline=[
            "### 1. Definition & Core Purpose\nClear, direct explanation of the entity/process [E#].",
            "### 2. Key Architecture & Mechanisms\n3 to 5 structured points with bold headers detailing operational mechanisms [E#].",
            "### 3. Application Example & Significance\nBrief real-world example or operational significance [E#]."
        ],
        system_instruction_prompt=(
            "FORMAT AS A COMPREHENSIVE 5-MARK EXAMINATION ANSWER:\n"
            "1. Lead with a strong, self-contained definition/overview paragraph (~2-3 sentences) citing [E#].\n"
            "2. Provide a structured section with 3 to 5 bulleted points with bold lead-ins detailing primary architecture, daemons, or operational mechanisms.\n"
            "3. Conclude with a brief practical example or deployment use case.\n"
            "4. Target length: ~200-350 words. Ensure every factual claim has inline citations [E#]."
        ),
        rubric_guidelines=[
            "1 mark for precise conceptual definition.",
            "3 marks for explaining core mechanisms/components thoroughly.",
            "1 mark for real-world example, workload context, or operational application."
        ]
    ),
    "8_mark": AnswerFormatGuidance(
        format_id="8_mark",
        display_name="8-Mark Exam Answer",
        target_marks=8,
        expected_word_range="500-800 words (Multi-section structured technical examination paper)",
        target_components=[
            "Answer Outline",
            "Conceptual Definition & Problem Addressed",
            "Architectural Blueprint & Subsystems",
            "Step-by-Step Execution Workflow",
            "Practical Application & Workload Diversity",
            "Operating Guarantees, Trade-offs & Limitations"
        ],
        structural_outline=[
            "> **Answer Outline**: Definition | Architecture & Components | Workflow Mechanics | Real-World Application | Guarantees & Trade-offs",
            "### 1. Conceptual Foundation & Core Definition\nDirect, rigorous definition and the fundamental engineering problem it solves [E#].",
            "### 2. Architecture & Subsystem Responsibilities\nEnumeration and deep explanation of individual components, daemons, and coordination mechanisms [E#].",
            "### 3. Operational Mechanics & Execution Workflow\nStep-by-step sequence of how tasks, resources, or packets flow through the system [E#].",
            "### 4. Practical Implementation & Workload Support\nReal-world deployment scenarios, supported frameworks, or industry use cases [E#].",
            "### 5. Technical Guarantees, Trade-offs & Limitations\nFault tolerance, isolation guarantees, bottleneck avoidance, and known system trade-offs [E#]."
        ],
        system_instruction_prompt=(
            "FORMAT AS A RIGOROUS 8-MARK TECHNICAL EXAMINATION ANSWER:\n"
            "1. Begin with an explicit '> **Answer Outline**:' block listing the 5 major topics addressed.\n"
            "2. Section 1 (Foundation): Definitive conceptual definition and foundational motivation [E#].\n"
            "3. Section 2 (Architecture): Detail core subsystems and daemon responsibilities with bold names and bullet points [E#].\n"
            "4. Section 3 (Operational Mechanics): Explain the execution sequence and workflow step-by-step [E#].\n"
            "5. Section 4 (Practical Context): Explain real-world deployment, workloads, and system integration [E#].\n"
            "6. Section 5 (Guarantees & Trade-offs): Detail fault tolerance, performance bounds, isolation, and trade-offs [E#].\n"
            "7. Target depth: ~500-800 words with thorough evidence grounding across all sections."
        ),
        rubric_guidelines=[
            "1.5 marks for foundational definition and engineering motivation.",
            "2.5 marks for component decomposition and subsystem interactions.",
            "2.0 marks for operational execution workflow and mechanics.",
            "1.0 mark for applied deployment context and workload support.",
            "1.0 mark for guarantees, fault tolerance, and limitations/trade-offs."
        ]
    ),
    "10_mark": AnswerFormatGuidance(
        format_id="10_mark",
        display_name="10-Mark Exam Answer",
        target_marks=10,
        expected_word_range="750-1100 words (Exhaustive university/industry technical examination answer)",
        target_components=[
            "Answer Outline & Taxonomy",
            "Formal Definition & Theoretical Foundations",
            "Exhaustive Architectural Breakdown",
            "Lifecycle Sequence & Dynamic Inter-Process Coordination",
            "Comparative Evaluation Matrix (Markdown Table)",
            "Scalability, Security & Fault Tolerance Guarantees",
            "Critical Analysis, Trade-offs & Concluding Assessment"
        ],
        structural_outline=[
            "> **Comprehensive Answer Outline**: 1. Foundations | 2. Architecture | 3. Execution Lifecycle | 4. Comparative Matrix | 5. Fault Tolerance & Guarantees | 6. Critical Analysis",
            "### 1. Formal Definition & Theoretical Foundations\nComprehensive conceptual definition, design philosophy, and historical motivation [E#].",
            "### 2. Exhaustive Architectural Decomposition\nDetailed breakdown of all daemons, agents, controllers, and their communication protocols [E#].",
            "### 3. Dynamic Execution Lifecycle & Workflow Sequence\nChronological end-to-end execution path from query/job submission to completion [E#].",
            "### 4. Comparative Architectural Dimensions & Trade-off Matrix\nMarkdown table comparing key parameters, alternatives, and operational characteristics [E#].",
            "### 5. High Availability, Fault Tolerance & Performance Guarantees\nHeartbeats, container isolation, failover mechanics, and SLA enforcement [E#].",
            "### 6. Critical Engineering Trade-offs & Strategic Conclusion\nSummary of architectural compromises, scalability bottlenecks, and final technical synthesis [E#]."
        ],
        system_instruction_prompt=(
            "FORMAT AS AN EXHAUSTIVE 10-MARK UNIVERSITY / PROFESSIONAL EXAMINATION ANSWER:\n"
            "1. Begin with an explicit '> **Comprehensive Answer Outline**:' summarizing the sections.\n"
            "2. Section 1 (Foundations): Formal technical definition, design philosophy, and core objectives [E#].\n"
            "3. Section 2 (Architecture): Exhaustive breakdown of every daemon, agent, and scheduler responsibility [E#].\n"
            "4. Section 3 (Lifecycle): Ordered execution workflow detailing state transitions and protocol messages [E#].\n"
            "5. Section 4 (Comparative Table): A clean Markdown table comparing architectural dimensions, paradigms, or alternatives [E#].\n"
            "6. Section 5 (Guarantees): Rigorous analysis of high availability, failover, isolation, and SLA mechanisms [E#].\n"
            "7. Section 6 (Conclusion): Summary of engineering trade-offs, scaling limits, and final assessment [E#].\n"
            "8. Target length: ~750-1100 words. Maintain high technical density and exhaustive inline citations."
        ),
        rubric_guidelines=[
            "2.0 marks for formal definition, design philosophy, and core objectives.",
            "3.0 marks for architectural decomposition and daemon responsibilities.",
            "2.0 marks for execution workflow and inter-process lifecycle sequence.",
            "1.5 marks for structured comparative analysis table and trade-offs.",
            "1.5 marks for high availability, fault tolerance guarantees, and concluding assessment."
        ]
    ),
    "comparison": AnswerFormatGuidance(
        format_id="comparison",
        display_name="Comparison & Contrast Table",
        target_marks=None,
        expected_word_range="300-600 words (Markdown Table + Analytical Takeaway)",
        target_components=[
            "Executive contrast summary",
            "Multi-dimensional Markdown Comparison Table",
            "Key architectural differentiators",
            "Deployment recommendation / synthesis"
        ],
        structural_outline=[
            "### Comparative Overview\nDirect 1-2 sentence executive contrast between the entities [E#].",
            "### Dimension-by-Dimension Comparison\n| Evaluation Dimension | Entity A | Entity B | Evidence & Citations |",
            "### Critical Architectural Differentiators\nDetailed analytical paragraphs examining trade-offs and operational contexts [E#].",
            "### Selection Criteria & Conclusion\nActionable guidance on when to choose each system [E#]."
        ],
        system_instruction_prompt=(
            "FORMAT AS A STRUCTURED COMPARISON AND CONTRAST ANALYSIS:\n"
            "1. Lead with a concise comparative summary stating the fundamental difference between the entities [E#].\n"
            "2. Provide a clean Markdown comparison table with columns: | Architectural Dimension | <Entity A> | <Entity B> | Citations |.\n"
            "3. Follow with dedicated paragraphs explaining the operational trade-offs and when to deploy each entity [E#]."
        ),
        rubric_guidelines=[
            "Explicit comparative criteria.",
            "Symmetrical coverage of both entities.",
            "Evidence citations on all contrasting claims."
        ]
    ),
    "worked_example": AnswerFormatGuidance(
        format_id="worked_example",
        display_name="Worked Technical Example",
        target_marks=None,
        expected_word_range="350-650 words (Scenario setup, execution steps, output validation)",
        target_components=[
            "Problem Scenario Setup",
            "Step-by-Step Numerical/Procedural Walkthrough",
            "System State Transitions",
            "Validation & Outcome Analysis"
        ],
        structural_outline=[
            "### 1. Concrete Scenario Setup\nParameters, initial conditions, and system configuration [E#].",
            "### 2. Execution Walkthrough\nStep 1 -> Step 2 -> Step 3 tracing data/token/packet propagation [E#].",
            "### 3. Resulting System State & Verification\nFinal state, mathematical/computational outcome, and verification checks [E#]."
        ],
        system_instruction_prompt=(
            "FORMAT AS A STEP-BY-STEP WORKED TECHNICAL EXAMPLE:\n"
            "1. State a concrete, realistic technical scenario with explicit parameters [E#].\n"
            "2. Walk through the procedural execution step-by-step showing data transitions and system calculations [E#].\n"
            "3. Conclude with the resulting state and verification criteria."
        ),
        rubric_guidelines=[
            "Realistic parameters and inputs.",
            "Clear step-by-step state tracking.",
            "Clear result validation."
        ]
    ),
    "structured_notes": AnswerFormatGuidance(
        format_id="structured_notes",
        display_name="Structured Revision Notes",
        target_marks=None,
        expected_word_range="300-600 words (Hierarchical high-yield revision bullet points)",
        target_components=[
            "Core Definition & Acronyms",
            "Key Architectural Components",
            "Operational Workflows",
            "Key Performance Metrics & Formulas",
            "High-Yield Exam Takeaways"
        ],
        structural_outline=[
            "### Key Definitions & Acronyms\n- **Term**: Definition [E#].",
            "### Architecture & Component Cheat-Sheet\n- **Component**: Responsibility [E#].",
            "### High-Yield Exam Takeaways\n- Quick bullet points summarizing common exam pitfalls and essentials [E#]."
        ],
        system_instruction_prompt=(
            "FORMAT AS STRUCTURED, HIGH-YIELD REVISION NOTES:\n"
            "1. Provide a bulleted 'Key Definitions & Acronyms' section [E#].\n"
            "2. Provide a 'Core Architecture & Daemons' cheat-sheet with bold keys [E#].\n"
            "3. Include a 'High-Yield Exam Takeaways' bullet list summarizing critical principles and common trade-offs [E#]."
        ),
        rubric_guidelines=[
            "High informational density.",
            "Logical hierarchical categorization.",
            "Fast readability."
        ]
    )
}


class FormatGuidanceCatalog:
    """Manages the catalog of answer formats and persists custom rubrics."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "data/format_guidance.json")
        self._catalog = dict(FORMAT_CATALOG)
        self._load_custom_guidance()

    def _load_custom_guidance(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("formats", []):
                        g = AnswerFormatGuidance(**item)
                        self._catalog[g.format_id] = g
            except Exception as e:
                pass

    def get_guidance(self, format_id: str) -> Optional[AnswerFormatGuidance]:
        return self._catalog.get(format_id.lower().strip())

    def list_formats(self) -> List[Dict[str, Any]]:
        return [
            {
                "format_id": g.format_id,
                "display_name": g.display_name,
                "target_marks": g.target_marks,
                "expected_word_range": g.expected_word_range,
                "components": g.target_components
            }
            for g in self._catalog.values()
        ]


# Singleton default catalog instance
default_format_catalog = FormatGuidanceCatalog()
