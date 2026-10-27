import logging
import os

import streamlit as st
from dotenv import load_dotenv

import guardrails
import interview
import metrics
import voice

load_dotenv()
logging.basicConfig(level=logging.INFO)

MAX_ANSWER_CHARS = 3000  # caps prompt size and cost

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤")
st.title("🎤 AI Interview Coach")
st.caption("AutoGen team: interviewer, evaluator and reporter")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to the .env file.")
    st.stop()

state = st.session_state
state.setdefault("config", None)  # role, level, total questions
state.setdefault("turns", [])  # question, answer, score, strengths, improvement
state.setdefault("report", None)
state.setdefault("spoken", None)  # last question read aloud
state.setdefault("tracker", metrics.Tracker())  # latency and cost of this session

with st.sidebar:
    st.header("Interview setup")
    role = st.text_input("Role", "Data Analyst", max_chars=80)
    level = st.selectbox("Level", ["Junior", "Mid-level", "Senior"], index=1)
    total = st.slider("Number of questions", 3, 8, 5)
    voice_on = st.toggle("Voice mode", value=True, help="Hear questions and answer by microphone")
    if st.button("Start interview", type="primary", use_container_width=True) and role.strip():
        state.tracker = metrics.Tracker()
        with st.spinner("Preparing your first question..."):
            try:
                usage = metrics.Usage()
                with state.tracker.track("first question", usage):
                    question = interview.first_question(role.strip(), level, usage)
            except Exception:
                logging.exception("Could not start the interview")
                st.error("Could not start the interview. Please try again.")
                st.stop()
        state.config = {"role": role.strip(), "level": level, "total": total}
        state.turns = [{"question": question}]
        state.report = None
        state.spoken = None
        st.rerun()  # refresh the sidebar numbers

    scores = [t["score"] for t in state.turns if "score" in t]
    if scores:
        st.divider()
        st.subheader("Eval score")
        st.metric("Average", f"{sum(scores) / len(scores):.1f} / 10", help=f"Scores so far: {scores}")
    if state.tracker.calls:
        st.divider()
        st.subheader("Latency and cost")
        st.metric("Estimated cost", f"${state.tracker.total_cost:.4f}")
        st.metric("Time waiting on AI", f"{state.tracker.total_seconds:.1f}s")
        for name, (count, avg, cost) in state.tracker.summary().items():
            st.caption(f"{name}: {count}x · avg {avg:.1f}s · ${cost:.4f}")

if not state.config:
    st.info("Choose a role in the sidebar and click **Start interview**.")
    st.stop()

# Show the conversation so far
latest_scored = max((i for i, t in enumerate(state.turns) if "score" in t), default=-1)
for i, turn in enumerate(state.turns):
    st.chat_message("assistant").write(turn["question"])
    if "answer" in turn:
        st.chat_message("user").write(turn["answer"])
    if "score" in turn:
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.markdown(f"✅ **Good:** {turn['strengths']}")
            st.markdown(f"🔧 **Fix:** {turn['improvement']}")
            # the newest model answer is open so the candidate can learn from it right away
            with st.expander("💡 A strong answer would be", expanded=i == latest_scored):
                st.write(turn["model_answer"])
            st.caption(f"Score: {turn['score']}/10")

if state.report:
    st.divider()
    st.markdown(state.report)
    st.download_button("Download report", state.report, "interview_report.md")
    st.stop()

current = state.turns[-1]["question"]
if voice_on:
    try:
        # autoplay only once per question, so reruns don't repeat it
        first_play = state.spoken != current
        if first_play:
            with state.tracker.track("text-to-speech", chars=len(current)):
                audio_bytes = voice.speak(current)
        else:
            audio_bytes = voice.speak(current)  # cached, free
        st.audio(audio_bytes, format="audio/mp3", autoplay=first_play)
        state.spoken = current
    except Exception:
        logging.exception("Text-to-speech failed")
        st.warning("Could not play the question aloud.")

answer = st.chat_input("Type your answer", max_chars=MAX_ANSWER_CHARS)
if voice_on and not answer:
    # the key changes every turn so the recorder starts empty for the next question
    recording = st.audio_input("Or record your answer", key=f"mic_{len(state.turns)}")
    if recording:
        with st.spinner("Transcribing..."):
            try:
                with state.tracker.track("speech-to-text", audio=recording):
                    answer = voice.transcribe(recording)[:MAX_ANSWER_CHARS]
            except Exception:
                logging.exception("Transcription failed")
                st.error("Could not transcribe the recording. Please try again.")
if answer and answer.strip():
    rejection = guardrails.check_input(answer)
    if rejection:
        st.warning(rejection)
        st.stop()
    turn = state.turns[-1]
    turn["answer"] = answer.strip()
    st.chat_message("user").write(turn["answer"])
    cfg = state.config
    is_last = len(state.turns) == cfg["total"]
    with st.spinner("Evaluating your answer..."):
        try:
            usage = metrics.Usage()
            with state.tracker.track("evaluate + next", usage):
                evaluation, reply = interview.review(cfg["role"], cfg["level"], state.turns, is_last, usage)
        except Exception:
            logging.exception("Review failed")
            del turn["answer"]  # let the candidate resubmit
            st.error("Something went wrong. Please send your answer again.")
            st.stop()
    turn.update(evaluation.model_dump())
    if is_last:
        state.report = reply
    else:
        state.turns.append({"question": reply})
    st.rerun()
