"""
Multi-stage AI workflow engine with context passing, telemetry, and resilient error handling.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional
from groq import Groq, APIError, RateLimitError, APIConnectionError

import prompts

# Configure logging
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
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.4

    # Stage Outputs
    study_plan: str = ""
    study_content: str = ""
    assessment: str = ""
    review_notes: str = ""
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
                "5_refined_pack": self.final_study_pack,
            },
            "errors": self.errors,
        }


class WorkflowError(Exception):
    """Custom exception raised when a workflow stage fails after all retries."""
    pass


class StudyPackWorkflow:
    """Orchestrates the 5-stage AI Study Pack generation pipeline."""

    def __init__(self, api_key: str, max_retries: int = 3, base_delay: float = 2.0):
        if not api_key or not api_key.strip():
            raise ValueError("A valid Groq API Key is required to initialize the workflow.")
        self.client = Groq(api_key=api_key.strip())
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _call_llm_with_retry(
        self,
        stage_name: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> str:
        """Executes an LLM API call with exponential backoff and error handling."""
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[{stage_name}] Executing LLM call (Attempt {attempt}/{self.max_retries})...")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError(f"Empty response received from model during {stage_name}.")
                return content.strip()

            except RateLimitError as e:
                last_exception = e
                wait_time = self.base_delay * (2 ** (attempt - 1)) + 1.5
                logger.warning(f"[{stage_name}] Rate limit hit. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

            except (APIConnectionError, APIError) as e:
                last_exception = e
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
            study_plan=ctx.study_plan,
        )
        ctx.study_content = self._call_llm_with_retry(
            stage_name="2. Content Generation",
            system_prompt=prompts.CONTENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=ctx.temperature,
        )
        ctx.stage_timings["2_content"] = round(time.time() - start, 2)
        ctx.completed_stages.append("2_content")
        return ctx.study_content

    def run_stage_3_assessment(self, ctx: WorkflowContext) -> str:
        start = time.time()
        user_prompt = prompts.ASSESSMENT_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            study_plan=ctx.study_plan,
            study_content=ctx.study_content,
        )
        ctx.assessment = self._call_llm_with_retry(
            stage_name="3. Assessment",
            system_prompt=prompts.ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=ctx.temperature,
        )
        ctx.stage_timings["3_assessment"] = round(time.time() - start, 2)
        ctx.completed_stages.append("3_assessment")
        return ctx.assessment

    def run_stage_4_review(self, ctx: WorkflowContext) -> str:
        start = time.time()
        user_prompt = prompts.REVIEW_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            learning_style=ctx.learning_style,
            goals=ctx.goals or "General mastery",
            study_plan=ctx.study_plan,
            study_content=ctx.study_content,
            assessment=ctx.assessment,
        )
        ctx.review_notes = self._call_llm_with_retry(
            stage_name="4. Quality Review",
            system_prompt=prompts.REVIEW_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=0.2,  # Lower temperature for stricter analytical critique
        )
        ctx.stage_timings["4_review"] = round(time.time() - start, 2)
        ctx.completed_stages.append("4_review")
        return ctx.review_notes

    def run_stage_5_refinement(self, ctx: WorkflowContext) -> str:
        start = time.time()
        user_prompt = prompts.REFINEMENT_USER_TEMPLATE.format(
            topic=ctx.topic,
            level=ctx.level,
            duration=ctx.duration,
            learning_style=ctx.learning_style,
            study_plan=ctx.study_plan,
            study_content=ctx.study_content,
            assessment=ctx.assessment,
            review_notes=ctx.review_notes,
        )
        ctx.final_study_pack = self._call_llm_with_retry(
            stage_name="5. Refinement",
            system_prompt=prompts.REFINEMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ctx.model_name,
            temperature=ctx.temperature,
            max_tokens=6000,
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
        Runs all 5 stages sequentially, passing context forward and reporting progress.
        Gracefully captures partial progress if an unrecoverable error occurs mid-pipeline.
        """
        stages = [
            (1, "Stage 1/5: Designing Curriculum & Study Plan...", self.run_stage_1_planning),
            (2, "Stage 2/5: Generating Deep-Dive Study Content & Flashcards...", self.run_stage_2_content),
            (3, "Stage 3/5: Building Diagnostic Quizzes & Rubrics...", self.run_stage_3_assessment),
            (4, "Stage 4/5: Performing Pedagogical Audit & Quality Review...", self.run_stage_4_review),
            (5, "Stage 5/5: Synthesizing & Refining Final Study Pack...", self.run_stage_5_refinement),
        ]

        for step_num, status_text, stage_fn in stages:
            try:
                if progress_callback:
                    progress_callback(step_num, status_text)
                stage_fn(ctx)
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
