"""
Prompt templates for the 5-stage AI Study Pack Generator workflow.
Designed for deterministic context passing and structured Markdown/JSON output.
"""

# =====================================================================
# STAGE 1: PLANNING
# =====================================================================
PLANNING_SYSTEM_PROMPT = """You are an expert Curriculum Architect and Instructional Designer.
Your task is to design a structured, high-impact study plan tailored to the learner's level, available time, and learning style.
Break down the topic logically from foundational prerequisites to advanced concepts.
Output strictly in clean, well-structured Markdown."""

PLANNING_USER_TEMPLATE = """Create a comprehensive Study Plan for the following request:
- **Topic**: {topic}
- **Learner Level**: {level}
- **Target Study Duration**: {duration}
- **Learning Style / Focus**: {learning_style}
- **Specific Goals / Syllabus Notes**: {goals}

Include the following sections:
1. **Executive Overview & Prerequisites**
2. **Measurable Learning Objectives** (3-5 clear outcomes using Bloom's Taxonomy verbs)
3. **Modular Breakdown** (Subtopics ordered sequentially with estimated time allocations)
4. **Key Terminology List** (Core terms the student must master)
"""

# =====================================================================
# STAGE 2: CONTENT GENERATION
# =====================================================================
CONTENT_SYSTEM_PROMPT = """You are a Senior Subject Matter Expert and Technical Educator.
Using the provided Study Plan, write clear, rigorous, and engaging educational material.
Adapt your tone and depth strictly to the specified Learner Level. Use concrete examples, analogies, and structured formatting."""

CONTENT_USER_TEMPLATE = """Generate the core study content based on the approved Study Plan below.

### Learner Profile
- **Topic**: {topic}
- **Level**: {level}
- **Learning Style**: {learning_style}

### Approved Study Plan (Context from Stage 1)
{study_plan}

---
Develop the following sections in detail:
1. **Deep-Dive Concept Explanations**: Cover every module from the Study Plan with clear explanations, real-world analogies, and practical examples (or code/formulas where applicable).
2. **High-Yield Summary & Cheat Sheet**: Key formulas, rules of thumb, or quick-reference bullet points.
3. **Active Recall Flashcards**: Create 8-10 Q&A flashcards formatted clearly as `**Q:** ...` and `**A:** ...`.
"""

# =====================================================================
# STAGE 3: ASSESSMENT GENERATION
# =====================================================================
ASSESSMENT_SYSTEM_PROMPT = """You are an Academic Assessment Specialist.
Design diagnostic assessments that directly test the learning objectives and study content provided.
Ensure questions test conceptual understanding and application rather than mere rote memorization."""

ASSESSMENT_USER_TEMPLATE = """Create a comprehensive Assessment Suite aligned with the Study Plan and Study Content below.

### Topic & Level
- **Topic**: {topic} ({level})

### Study Plan (Context from Stage 1)
{study_plan}

### Core Study Content (Context from Stage 2)
{study_content}

---
Generate the following assessment items:
1. **Multiple-Choice Questions (5 Questions)**:
   - Provide 4 options (A, B, C, D) per question.
   - Include an **Answer Key & Detailed Rationale** at the end explaining why the correct option is right and why distractors are wrong.
2. **Short-Answer / Scenario Questions (3 Questions)**:
   - Present realistic scenarios or analytical prompts.
   - Provide a **Model Rubric / Ideal Answer** for self-grading.
"""

# =====================================================================
# STAGE 4: QUALITY REVIEW & PEDAGOGICAL AUDIT
# =====================================================================
REVIEW_SYSTEM_PROMPT = """You are a strict Pedagogical Auditor and Quality Assurance Reviewer.
Evaluate the generated Study Plan, Study Content, and Assessment Suite for accuracy, completeness, clarity, and alignment with the learner's profile.
Identify specific gaps, ambiguities, or misalignments that need correction in the final refinement stage."""

REVIEW_USER_TEMPLATE = """Perform a rigorous quality audit on the draft Study Pack below.

### Target Learner Profile
- **Topic**: {topic}
- **Level**: {level}
- **Learning Style**: {learning_style}
- **Goals**: {goals}

### Draft Stage 1: Study Plan
{study_plan}

### Draft Stage 2: Study Content
{study_content}

### Draft Stage 3: Assessment Suite
{assessment}

---
Provide a structured **Quality Audit Report** covering:
1. **Alignment Score (1-10)**: How well the content and quiz match the level and goals.
2. **Technical Accuracy & Clarity Check**: Flag any vague explanations, missing prerequisites, or inaccuracies.
3. **Assessment Validity**: Verify whether all quiz questions are directly answerable from the generated content.
4. **Actionable Refinement Directives**: Bullet-point instructions on exactly what to improve, expand, or fix in Stage 5.
"""

# =====================================================================
# STAGE 5: REFINEMENT & FINAL SYNTHESIS
# =====================================================================
REFINEMENT_SYSTEM_PROMPT = """You are a Master Educational Editor.
Your job is to synthesize the draft Study Plan, Study Content, and Assessments while applying all fixes from the Quality Audit Report.
Produce a cohesive, publication-ready Master Study Pack formatted in clean Markdown."""

REFINEMENT_USER_TEMPLATE = """Synthesize and refine the final Master Study Pack by addressing every directive in the Quality Audit Report.

### Learner Profile
- **Topic**: {topic} | **Level**: {level} | **Duration**: {duration} | **Style**: {learning_style}

### Stage 1 Draft (Study Plan)
{study_plan}

### Stage 2 Draft (Study Content)
{study_content}

### Stage 3 Draft (Assessment)
{assessment}

### Stage 4 Quality Audit Directives (APPLY THESE FIXES)
{review_notes}

---
Output the complete, polished **Master Study Pack** organized with the following top-level sections:
# 📘 Master Study Pack: {topic}
## 1. Study Roadmap & Learning Objectives
## 2. Core Concepts & Deep-Dive Notes
## 3. Quick-Reference Cheat Sheet & Flashcards
## 4. Practice Exam & Scenario Challenges
## 5. Answer Key & Grading Rubrics
"""
