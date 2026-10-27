"""
Streamlit Frontend for the Multi-Stage AI Study Pack Generator.
"""

import os
import io
import json
import zipfile
import streamlit as st
from dotenv import load_dotenv

from workflow import WorkflowContext, StudyPackWorkflow
import prompts

# Load local environment variables if present
load_dotenv()

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def create_project_zip() -> bytes:
    """Packages the project source files into a downloadable ZIP archive for GitHub/Streamlit deployment."""
    buffer = io.BytesIO()
    files_to_pack = ["app.py", "workflow.py", "prompts.py", "requirements.txt"]

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in files_to_pack:
            if os.path.exists(filename):
                zf.write(filename, arcname=filename)
        # Add a helpful README.md inside the zip
        readme_content = (
            "# AI Study Pack Generator\n\n"
            "Multi-stage AI workflow decomposing study pack creation into Planning, "
            "Content Generation, Assessment, Quality Review, and Refinement.\n\n"
            "## Quick Start\n"
            "1. `pip install -r requirements.txt`\n"
            "2. `streamlit run app.py`\n"
        )
        zf.writestr("README.md", readme_content)

    buffer.seek(0)
    return buffer.getvalue()


def main():
    # -----------------------------------------------------------------
    # SIDEBAR: API Configuration & Source Code Downloads
    # -----------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Workflow Settings")

        # Resolve API Key from Streamlit Secrets, Environment, or User Input
        default_key = os.getenv("GROQ_API_KEY", "")
        if not default_key and "GROQ_API_KEY" in st.secrets:
            default_key = st.secrets["GROQ_API_KEY"]

        api_key = st.text_input(
            "Groq API Key",
            value=default_key,
            type="password",
            help="Get your free API key from console.groq.com",
        )

        model_name = st.selectbox(
            "LLM Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ],
            index=0,
            help="llama-3.3-70b-versatile is recommended for complex multi-stage reasoning.",
        )

        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Lower values keep outputs strictly factual and structured.",
        )

        st.divider()
        st.subheader("📦 Download Project Files")
        st.caption("Download the deployment-ready files for your GitHub repository:")

        try:
            zip_bytes = create_project_zip()
            st.download_button(
                label="⬇️ Download All Source Files (.zip)",
                data=zip_bytes,
                file_name="ai_study_pack_generator_source.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"Zip bundler note: {e}")

    # -----------------------------------------------------------------
    # MAIN HEADER
    # -----------------------------------------------------------------
    st.title("🎓 Multi-Stage AI Study Pack Generator")
    st.markdown(
        "Decomposes personalized study material creation into a **5-stage AI pipeline**: "
        "**Planning → Content Generation → Assessment → Quality Audit → Refinement**."
    )

    # -----------------------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------------------
    with st.form("study_pack_form"):
        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "📌 Study Topic or Subject",
                placeholder="e.g., Network Security & Cryptography, Python Data Structures...",
            )
            level = st.selectbox(
                "🎯 Learner Proficiency Level",
                options=[
                    "Beginner (Foundational & Intuitive)",
                    "Intermediate (Undergraduate / Practical Application)",
                    "Advanced (Graduate / Deep Technical & Architectural)",
                ],
                index=1,
            )

        with col2:
            duration = st.selectbox(
                "⏱️ Target Study Duration",
                options=[
                    "1 Hour Crash Course",
                    "3 Hours Focused Session",
                    "1 Week Sprint (1-2 hrs/day)",
                    "2 Weeks Comprehensive Module",
                ],
                index=1,
            )
            learning_style = st.selectbox(
                "🧠 Preferred Learning Style",
                options=[
                    "Practical & Example-Driven (Code/Real-World Scenarios)",
                    "Visual & Structured (Tables, Frameworks & Mental Models)",
                    "Exam-Focused (High-Yield Summaries & Active Recall)",
                    "First-Principles & Theoretical Depth",
                ],
                index=0,
            )

        goals = st.text_area(
            "📝 Specific Syllabus Topics or Learning Goals (Optional)",
            placeholder="e.g., Focus on AES vs RSA encryption, TLS handshake steps, and common vulnerabilities...",
            height=80,
        )

        submit_btn = st.form_submit_button(
            "🚀 Run 5-Stage AI Workflow", use_container_width=True
        )

    # -----------------------------------------------------------------
    # WORKFLOW EXECUTION
    # -----------------------------------------------------------------
    if submit_btn:
        if not api_key:
            st.error("🔑 Please enter your Groq API Key in the sidebar to run the workflow.")
            st.stop()
        if not topic.strip():
            st.warning("⚠️ Please enter a valid Study Topic before starting.")
            st.stop()

        ctx = WorkflowContext(
            topic=topic.strip(),
            level=level,
            duration=duration,
            learning_style=learning_style,
            goals=goals.strip(),
            model_name=model_name,
            temperature=temperature,
        )

        workflow = StudyPackWorkflow(api_key=api_key)
        progress_bar = st.progress(0)

        with st.status("Initializing Multi-Stage AI Pipeline...", expanded=True) as status_box:
            def update_ui(step: int, message: str):
                status_box.update(label=message, state="running")
                st.write(f"⏳ **{message}**")
                progress_bar.progress(int((step - 1) * 20))

            ctx = workflow.execute_full_pipeline(ctx, progress_callback=update_ui)

            if ctx.errors:
                status_box.update(
                    label="⚠️ Pipeline encountered an error", state="error", expanded=True
                )
                for err in ctx.errors:
                    st.error(err)
            else:
                progress_bar.progress(100)
                total_time = sum(ctx.stage_timings.values())
                status_box.update(
                    label=f"✅ Study Pack Generated Successfully in {total_time:.1f}s!",
                    state="complete",
                    expanded=False,
                )

        # Save result in session state so downloads persist across reruns
        st.session_state["workflow_ctx"] = ctx

    # -----------------------------------------------------------------
    # DISPLAY RESULTS & STAGE ARTIFACTS
    # -----------------------------------------------------------------
    if "workflow_ctx" in st.session_state:
        ctx: WorkflowContext = st.session_state["workflow_ctx"]

        st.divider()
        # Telemetry Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("1. Planning", f"{ctx.stage_timings.get('1_planning', 0)}s")
        m2.metric("2. Content", f"{ctx.stage_timings.get('2_content', 0)}s")
        m3.metric("3. Assessment", f"{ctx.stage_timings.get('3_assessment', 0)}s")
        m4.metric("4. QA Review", f"{ctx.stage_timings.get('4_review', 0)}s")
        m5.metric("5. Refinement", f"{ctx.stage_timings.get('5_refinement', 0)}s")

        # Tabs for inspecting the final output and every intermediate stage
        tab_final, tab_plan, tab_content, tab_quiz, tab_review = st.tabs([
            "📘 Final Refined Study Pack",
            "🗺️ Stage 1: Study Plan",
            "📖 Stage 2: Draft Content",
            "📝 Stage 3: Draft Assessment",
            "🔍 Stage 4: QA Audit Report",
        ])

        with tab_final:
            if ctx.final_study_pack:
                st.markdown(ctx.final_study_pack)
            else:
                st.info("Final refinement stage did not complete. Check earlier stage tabs for partial outputs.")

        with tab_plan:
            st.markdown(ctx.study_plan or "*No output generated.*")

        with tab_content:
            st.markdown(ctx.study_content or "*No output generated.*")

        with tab_quiz:
            st.markdown(ctx.assessment or "*No output generated.*")

        with tab_review:
            st.markdown(ctx.review_notes or "*No output generated.*")

        # -------------------------------------------------------------
        # EXPORT / DOWNLOAD STUDY PACK
        # -------------------------------------------------------------
        st.divider()
        st.subheader("📥 Export Generated Study Pack")
        dl_col1, dl_col2 = st.columns(2)

        safe_topic = "".join(c if c.isalnum() else "_" for c in ctx.topic).lower()

        with dl_col1:
            export_md = ctx.final_study_pack or ctx.study_content or ctx.study_plan
            st.download_button(
                label="📄 Download Final Study Pack (.md)",
                data=export_md,
                file_name=f"study_pack_{safe_topic}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with dl_col2:
            export_json = json.dumps(ctx.to_dict(), indent=2)
            st.download_button(
                label="🗂️ Download Full Pipeline Trace (.json)",
                data=export_json,
                file_name=f"workflow_trace_{safe_topic}.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
