"""
Token-optimized prompt templates for the 5-stage AI Study Pack Generator workflow.
Engineered to stay strictly within Groq Free Tier 8,000 TPM limits.
"""

# =====================================================================
# STAGE 1: PLANNING
# =====================================================================
PLANNING_SYSTEM_PROMPT = """You are an expert Curriculum Architect.
Design a structured, high-impact study plan tailored to the learner's level, duration, and style.
Be concise, high-density, and output strictly in clean Markdown."""

PLANNING_USER_TEMPLATE = """Create a structured Study Plan for:
- **Topic**: {topic}
- **Learner Level**: {level}
- **Duration**: {duration}
- **Learning Style**: {learning_style}
- **Specific Goals**: {goals}

Include:
1. **Prerequisites & Executive Overview** (Brief)
2. **Measurable Learning Objectives** (3-5 bullet points)
3. **Modular Breakdown** (4-6 sequential modules with time allocations)
4. **Key Terminology** (Top 8-10 essential terms with 1-line definitions)
"""

# =====================================================================
# STAGE 2: CONTENT GENERATION
# =====================================================================
CONTENT_SYSTEM_PROMPT = """You are a Senior Subject Matter Expert.
Write clear, high-density educational material following the Study Plan.
Prioritize concrete examples, clear explanations, and compact formatting."""

CONTENT_USER_TEMPLATE = """Generate the core study content based on the Study Plan below.

### Learner Profile
- **Topic**: {topic} ({level}) | **Style**: {learning_style}

### Approved Study Plan (Stage 1)
{study_plan}

---
Develop the following sections:
1. **Core Module Explanations**: Explain each module from the plan with clear mental models and practical examples.
2. **High-Yield Cheat Sheet**: Key formulas, rules, or quick-reference points.
3. **Active Recall Flashcards**: 6-8 high-value Q&A pairs (`**Q:** ...` / `**A:** ...`).
"""

# =====================================================================
# STAGE 3: ASSESSMENT GENERATION
# =====================================================================
ASSESSMENT_SYSTEM_PROMPT = """You are an Academic Assessment Specialist.
Design diagnostic assessments aligned with the provided study plan and content."""

ASSESSMENT_USER_TEMPLATE = """Create a diagnostic Assessment Suite aligned with the context below.

### Topic & Level
- **Topic**: {topic} ({level})

### Study Plan Summary (Stage 1)
{study_plan}

### Core Content Summary (Stage 2)
{study_content}

---
Generate:
1. **Multiple-Choice Questions (5 Questions)**: 4 options (A-D) each, followed by a concise **Answer Key & Rationale**.
2. **Applied Scenario Prompts (2 Questions)**: Realistic analytical questions with a brief **Grading Rubric / Ideal Answer**.
"""

# =====================================================================
# STAGE 4: QUALITY REVIEW & PEDAGOGICAL AUDIT
# =====================================================================
REVIEW_SYSTEM_PROMPT = """You are a strict Pedagogical Auditor.
Audit the summarized Study Plan, Content, and Assessment for accuracy, alignment, and gaps.
Be concise and actionable."""

REVIEW_USER_TEMPLATE = """Perform a quality audit on the draft Study Pack below.

### Learner Profile
- **Topic**: {topic} | **Level**: {level} | **Goals**: {goals}

### Draft Stage 1: Study Plan
{study_plan}

### Draft Stage 2: Study Content (Condensed)
{study_content}

### Draft Stage 3: Assessment Suite (Condensed)
{assessment}

---
Provide a concise **Quality Audit Report**:
1. **Alignment & Rigor Score (1-10)** with a 2-sentence justification.
2. **Identified Gaps or Clarifications Needed**: Top 3 concepts that need deeper clarification or edge-case warnings.
3. **Actionable Refinement Directives**: Specific improvements to apply in Stage 5.
"""

# =====================================================================
# STAGE 5: REFINEMENT & SYNTHESIS
# =====================================================================
REFINEMENT_SYSTEM_PROMPT = """You are a Master Educational Editor.
Address every gap identified in the Quality Audit Report by writing a high-impact Refinement & Deep-Dive Addendum that resolves all audit directives, clarifies tricky edge cases, and provides an exam-day synthesis."""

REFINEMENT_USER_TEMPLATE = """Based on the Quality Audit Report below, generate the final **Refinement & Mastery Synthesis** section for the study pack on **{topic}** ({level}).

### Stage 1 Study Plan Reference
{study_plan}

### Stage 4 Quality Audit Directives (ADDRESS ALL OF THESE)
{review_notes}

---
Generate the following sections in clean Markdown:
1. **Gap-Filling Deep Dives & Clarifications**: Directly resolve and expand upon every gap, nuance, or correction flagged in the Quality Audit Report.
2. **Common Pitfalls & Misconceptions**: Top mistakes learners make on {topic} and how to avoid them.
3. **Master Synthesis & Exam-Ready Checklist**: A final rapid-review checklist tying all modules together.
"""
