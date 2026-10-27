"""
Multi-stage AI workflow engine with TPM Token Budgeting, 413 Auto-Shrink Recovery,
and Inter-Stage Rate-Limit Protection.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional
from groq import Groq, APIError, RateLimitError, APIConnectionError, BadRequestError

import prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """State container passed sequentially across all 5 pipeline stages."""
    topic: str
    level: str
    duration: str
    learning_style: str
    goals: str
    model_name: str = "openai/gpt-oss-120b"
    temperature: float = 0.4
    inter_stage_delay: float = 4.0  # Pauses briefly between stages to let TPM window recover

    # Stage Outputs
    study_plan: str = ""
    study_content: str = ""
    assessment: str = ""
    review_notes: str = ""
    refinement_addendum: str = ""
    final_study_pack: str = ""

    # Telemetry & Error Tracking
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: list = field(default_factory=list)
    completed_stages: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "topic": self.topic,
                "level": self.level,
                "duration": self.duration,
                "learning_style": self.learning_style,
                "goals": self.goals,
                "model_name": self.model_name,
                "stage_timings_sec": self.stage_timings,
                "completed_stages": self.completed_stages,
            },
            "stages": {
                "1_planning": self.study_plan,
                "2_content": self.study_content,
                "3_assessment": self.assessment,
                "4_review": self.review_notes,
                "5_refinement_addendum": self.refinement_addendum,
                "5_master_study_pack": self.final_study_pack,
            },
            "errors": self.errors,
        }


class WorkflowError(Exception):
    """Custom exception raised when a workflow stage fails after all retries."""
    pass


class StudyPackWorkflow:
    """Orchestrates the 5-stage AI Study Pack generation pipeline with strict TPM budgeting."""

    def __init__(self, api_key: str, max_retries: int = 4, base_delay: float = 5.0):
        if not api_key or not api_key.strip():
            raise ValueError("A valid Groq API Key is required to initialize the workflow.")
        self.client = Groq(api_key=api_key.strip())
        self.max_retries = max_retries
        self.base_delay = base_delay

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough heuristic: ~1 token per 3.8 characters in English/Markdown."""
        return max(1, int(len(text) / 3.8))

    @staticmethod
    def _compress_context(text: str, max_chars: int = 3500) -> str:
        """
        Compresses long stage outputs so cumulative context never blows past Groq's 8,000 TPM limit.
        Preserves the beginning (headings/structure) and end (conclusions/summaries).
        """
        if not text or len(text) <= max_chars:
            return text
        half = max_chars // 2
        return (
            text[:half]
            + "\n\n... [Intermediate section condensed to fit token budget] ...\n\n"
            + text[-half:]
        )

    def _call_llm_with_retry(
        self,
        stage_name: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int = 1800,
    ) -> str:
        """
        Executes an LLM API call with automatic 413 payload shrinking and 429 TPM rate-limit backoff.
        Ensures total requested tokens (input + max_tokens) stays strictly under 7,000 tokens.
        """
        last_exception = None
        current_user_prompt = user_prompt
        current_max_tokens = max_tokens

        # Proactive Token Guard: keep (input + max_tokens) safely below the 8,000 TPM limit
        SAFE_TPM_CEILING = 6500
        est_input = self._estimate_tokens(system_prompt) + self._estimate_tokens(current_user_prompt)
        if est_input + current_max_tokens > SAFE_TPM_CEILING:
            available_for_output = max(800, SAFE_TPM_CEILING - est_input)
            if available_for_output < current_max_tokens:
                logger.info(
                    f"[{stage_name}] Adjusting max_tokens from {current_max_tokens} -> {available_for_output} "
                    f"(est. input: {est_input} tokens) to stay under 8,000 TPM ceiling."
                )
                current_max_tokens = available_for_output

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"[{stage_name}] Executing LLM call (Attempt {attempt}/{self.max_retries}, "
                    f"max_tokens={current_max_tokens})..."
                )
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": current_user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError(f"Empty response received from model during {stage_name}.")
                return content.strip()

            except (RateLimitError, APIError, BadRequestError, APIConnectionError) as e:
                last_exception = e
                err_str = str(e).lower()

                # Detect 413 Request Too Large or TPM limit exceeded
                if "413" in err_str or "too large" in err_str or "tokens per minute" in err_str or "rate_limit" in err_str:
                    # Shrink prompt context and output token budget by 30% so the next retry succeeds
                    new_char_limit = int(len(current_user_prompt) * 0.68)
                    current_user_prompt = self._compress_context(current_user_prompt, max_chars=max(1500, new_char_limit))
                    current_max_tokens = max(700, int(current_max_tokens * 0.75))
                    wait_time = self.base_delay * attempt + 6.0  # Give the 60s TPM window time to replenish
                    logger.warning(
                        f"[{stage_name}] Token/Rate limit hit (413/429). Shrunk prompt to {len(current_user_prompt)} chars, "
                        f"max_tokens to {current_max_tokens}. Waiting {wait_time:.1f}s before retry..."
                    )
                    time.sleep(wait_time)
                else:
                    wait_time = self.base_delay * (2 ** (attempt - 1))
                    logger.warning(f"[{stage_name}] API error ({e}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)

            except Exception as e:
                last_exception = e
                logger.error(f"[{stage_name}] Unexpected error: {e}")
                break

        error_msg = f"Stage '{stage_name}' failed after {self.max_retries} attempts. Details: {str(last_exception)}"
        raise WorkflowError(error_msg)

    def run_stage_1_planning(self, ctx: WorkflowContext) -> str:
        start = time.time()
        user_prompt = prompts.PLANNING_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            duration=ctx.duration,
            learning_style=ctx.learning_style,
            goals=ctx.goals or "Comprehensive foundational mastery",
        )
        ctx.study_plan = self._call_llm_with_retry(
            stage_name="1. Planning",
            system_prompt=prompts.PLANNING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=ctx.temperature,
            max_tokens=1500,
        )
        ctx.stage_timings["1_planning"] = round(time.time() - start, 2)
        ctx.completed_stages.append("1_planning")
        return ctx.study_plan

    def run_stage_2_content(self, ctx: WorkflowContext) -> str:
        start = time.time()
        user_prompt = prompts.CONTENT_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            learning_style=ctx.learning_style,
            study_plan=self._compress_context(ctx.study_plan, max_chars=3500),
        )
        ctx.study_content = self._call_llm_with_retry(
            stage_name="2. Content Generation",
            system_prompt=prompts.CONTENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=ctx.temperature,
            max_tokens=2200,
        )
        ctx.stage_timings["2_content"] = round(time.time() - start, 2)
        ctx.completed_stages.append("2_content")
        return ctx.study_content

    def run_stage_3_assessment(self, ctx: WorkflowContext) -> str:
        start = time.time()
        user_prompt = prompts.ASSESSMENT_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            study_plan=self._compress_context(ctx.study_plan, max_chars=2200),
            study_content=self._compress_context(ctx.study_content, max_chars=3500),
        )
        ctx.assessment = self._call_llm_with_retry(
            stage_name="3. Assessment",
            system_prompt=prompts.ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=ctx.temperature,
            max_tokens=1800,
        )
        ctx.stage_timings["3_assessment"] = round(time.time() - start, 2)
        ctx.completed_stages.append("3_assessment")
        return ctx.assessment

    def run_stage_4_review(self, ctx: WorkflowContext) -> str:
        start = time.time()
        # Compress Stage 1, 2, and 3 so total Stage 4 request stays well under 8,000 tokens
        user_prompt = prompts.REVIEW_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            learning_style=ctx.learning_style,
            goals=ctx.goals or "General mastery",
            study_plan=self._compress_context(ctx.study_plan, max_chars=2000),
            study_content=self._compress_context(ctx.study_content, max_chars=2800),
            assessment=self._compress_context(ctx.assessment, max_chars=2000),
        )
        ctx.review_notes = self._call_llm_with_retry(
            stage_name="4. Quality Review",
            system_prompt=prompts.REVIEW_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=0.2,
            max_tokens=1200,
        )
        ctx.stage_timings["4_review"] = round(time.time() - start, 2)
        ctx.completed_stages.append("4_review")
        return ctx.review_notes

    def run_stage_5_refinement(self, ctx: WorkflowContext) -> str:
        start = time.time()
        user_prompt = prompts.REFINEMENT_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            study_plan=self._compress_context(ctx.study_plan, max_chars=2200),
            review_notes=self._compress_context(ctx.review_notes, max_chars=2500),
        )
        ctx.refinement_addendum = self._call_llm_with_retry(
            stage_name="5. Refinement",
            system_prompt=prompts.REFINEMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=ctx.temperature,
            max_tokens=1800,
        )

        # Assemble the complete Master Study Pack without losing any detail from Stages 1-3
        ctx.final_study_pack = (
            f"# 📘 Master Study Pack: {ctx.topic}\n"
            f"**Level:** {ctx.level} | **Duration:** {ctx.duration} | **Style:** {ctx.learning_style}\n\n"
            f"---\n\n"
            f"## 🗺️ Part 1: Curriculum & Study Plan\n\n{ctx.study_plan}\n\n"
            f"---\n\n"
            f"## 📖 Part 2: Core Study Notes, Cheat Sheet & Flashcards\n\n{ctx.study_content}\n\n"
            f"---\n\n"
            f"## ✨ Part 3: Audited Deep-Dives & Edge-Case Clarifications\n\n{ctx.refinement_addendum}\n\n"
            f"---\n\n"
            f"## 📝 Part 4: Diagnostic Assessment & Answer Rubrics\n\n{ctx.assessment}\n"
        )

        ctx.stage_timings["5_refinement"] = round(time.time() - start, 2)
        ctx.completed_stages.append("5_refinement")
        return ctx.final_study_pack

    def execute_full_pipeline(
        self,
        ctx: WorkflowContext,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> WorkflowContext:
        """
        Runs all 5 stages sequentially with TPM rate-limit pacing between stages.
        """
        stages = [
            (1, "Stage 1/5: Designing Curriculum & Study Plan...", self.run_stage_1_planning),
            (2, "Stage 2/5: Generating Deep-Dive Study Content & Flashcards...", self.run_stage_2_content),
            (3, "Stage 3/5: Building Diagnostic Quizzes & Rubrics...", self.run_stage_3_assessment),
            (4, "Stage 4/5: Performing Pedagogical Audit & Quality Review...", self.run_stage_4_review),
            (5, "Stage 5/5: Synthesizing & Refining Final Study Pack...", self.run_stage_5_refinement),
        ]

        for idx, (step_num, status_text, stage_fn) in enumerate(stages):
            try:
                if progress_callback:
                    progress_callback(step_num, status_text)
                stage_fn(ctx)

                # Pause briefly between stages so rolling 60s TPM window doesn't saturate
                if idx < len(stages) - 1 and ctx.inter_stage_delay > 0:
                    time.sleep(ctx.inter_stage_delay)

            except WorkflowError as e:
                ctx.errors.append(str(e))
                logger.error(f"Pipeline halted at step {step_num}: {e}")
                break
            except Exception as e:
                unexpected_msg = f"Unexpected failure at step {step_num}: {str(e)}"
                ctx.errors.append(unexpected_msg)
                logger.error(unexpected_msg)
                break

        return ctx
