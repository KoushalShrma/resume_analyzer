import streamlit as st
from pypdf import PdfReader
import docx
import re
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Resume Analyzer")
st.write(
    "Upload a **resume** and a **job description** to evaluate skill matching using an AI agent."
)

# --------------------------------------------------
# GROQ CLIENT (API KEY FROM STREAMLIT SECRETS)
# --------------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------------------------------
# FILE TEXT EXTRACTION
# --------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_docx(uploaded_file):
    document = docx.Document(uploaded_file)
    return " ".join([para.text for para in document.paragraphs])


def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload PDF or DOCX.")
        return ""

# --------------------------------------------------
# GROQ CHAT MODEL SELECTION (SAFE)
# --------------------------------------------------
def get_chat_model():
    preferred_models = [
        "llama3-8b-8192",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]

    available_models = [m.id for m in client.models.list().data]

    for model in preferred_models:
        if model in available_models:
            return model

    raise RuntimeError("No supported Groq chat model available.")

# --------------------------------------------------
# SKILL NORMALIZATION (CRITICAL FIX)
# --------------------------------------------------
def normalize_skills(skills_text):
    skills_text = skills_text.lower()
    skills_text = skills_text.replace("(", "").replace(")", "")

    raw_skills = skills_text.split(",")

    cleaned_skills = set()
    for skill in raw_skills:
        skill = skill.strip()
        if skill:
            cleaned_skills.add(skill)

    return list(cleaned_skills)

# --------------------------------------------------
# LLM: SKILL EXTRACTION FROM JOB DESCRIPTION
# --------------------------------------------------
def extract_skills_from_job_description(job_description):
    job_description = job_description[:6000]  # safety trim
    model_name = get_chat_model()

    prompt = (
        "You are an ATS system.\n"
        "Extract ONLY important technical skills and tools from the job description below.\n"
        "Return ONLY a comma-separated list.\n\n"
        f"Job Description:\n{job_description}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You extract skills for resume screening."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=256,
    )

    skills_text = response.choices[0].message.content
    return normalize_skills(skills_text)

# --------------------------------------------------
# RESUME ANALYSIS LOGIC (DETERMINISTIC)
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9 ]", " ", text)


def analyze_resume(resume_text, required_skills):
    resume_text = clean_text(resume_text)

    matched_skills = []
    missing_skills = []

    for skill in required_skills:
        if skill in resume_text:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
    return matched_skills, missing_skills, score

# --------------------------------------------------
# UI: FILE UPLOADS
# --------------------------------------------------
resume_file = st.file_uploader(
    "📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"]
)

jd_file = st.file_uploader(
    "📄 Upload Job Description (PDF or DOCX)", type=["pdf", "docx"]
)

# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------
if st.button("🔍 Analyze Resume"):

    if not resume_file or not jd_file:
        st.warning("Please upload both the resume and the job description.")
    else:
        with st.spinner("Running AI analysis..."):
            resume_text = extract_text(resume_file)
            job_description = extract_text(jd_file)

            required_skills = extract_skills_from_job_description(job_description)
            matched, missing, score = analyze_resume(resume_text, required_skills)

        st.subheader("📌 Required Skills")
        st.write(required_skills)

        st.subheader("✅ Matched Skills")
        st.write(matched)

        st.subheader("❌ Missing Skills")
        st.write(missing)

        st.subheader("📈 Resume Match Score")
        st.metric("Match Percentage", f"{score:.2f}%")

        if score >= 70:
            st.success("🟢 Strong Match")
        elif score >= 40:
            st.warning("🟡 Moderate Match")
        else:
            st.error("🔴 Weak Match")
