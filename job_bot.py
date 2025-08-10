import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import fitz  # PyMuPDF

# ---------------------- DARK MODE TOGGLE ----------------------
# Theme toggle in sidebar
theme = st.sidebar.radio("🌓 Choose Theme", ["Light", "Dark"])

# Apply CSS for dark mode
def set_dark_mode():
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #0e1117;
                color: #f5f6fa;
            }
            .css-18e3th9 {
                background-color: #0e1117;
            }
            .css-1d391kg {
                color: #f5f6fa;
            }
            .stTextInput>div>div>input {
                background-color: #2c2f33;
                color: #f5f6fa;
            }
            .stSlider {
                color: #f5f6fa;
            }
            .stButton>button {
                background-color: #2c2f33;
                color: #f5f6fa;
                border-color: #7289da;
            }
        </style>
        """, unsafe_allow_html=True)

if theme == "Dark":
    set_dark_mode()

# ---------------------- MODEL & DATASET ----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
df = pd.read_csv("jobs_dataset.csv")

# ---------------------- UI LAYOUT ----------------------
st.title("🤖 Advanced Job Role Recommender")
st.write("Upload your resume or enter your technical skills to get personalized job recommendations.")

resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
user_skills = st.text_input("💡 Or Enter Your Technical Skills (e.g., Python, SQL, ML):")

industries = ["All"] + sorted(df['Industry'].dropna().unique())
selected_industry = st.selectbox("🏭 Select Industry", industries)

experience_levels = ["All"] + sorted(df['Experience_Level'].dropna().unique())
selected_experience = st.selectbox("🧠 Experience Level", experience_levels)

job_types = ["All"] + sorted(df['Job_Type'].dropna().unique())
selected_job_type = st.selectbox("🏢 Job Type", job_types)

locations = ["All"] + sorted(df['Location'].dropna().unique())
selected_location = st.selectbox("🌍 Preferred Location", locations)

try:
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    min_salary = int(df['Salary'].min())
    max_salary = int(df['Salary'].max())
    selected_salary = st.slider("💰 Minimum Salary", min_salary, max_salary, min_salary)
except:
    st.warning("⚠️ Salary column must be numeric for slider to work.")

# ---------------------- PDF RESUME EXTRACTOR ----------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---------------------- INPUT HANDLING ----------------------
input_text = ""
if resume_file:
    with st.spinner("🔍 Extracting text from resume..."):
        extracted_text = extract_text_from_pdf(resume_file)
        st.success("✅ Resume text extracted.")
        st.text_area("📝 Extracted Resume Text", extracted_text, height=200)
        input_text = extracted_text
elif user_skills:
    input_text = user_skills

# ---------------------- SESSION STATES ----------------------
if "bookmarked_jobs" not in st.session_state:
    st.session_state.bookmarked_jobs = []

# ---------------------- RECOMMENDATION ENGINE ----------------------
if input_text:
    filtered_df = df.copy()

    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]
    if selected_experience != "All":
        filtered_df = filtered_df[filtered_df['Experience_Level'] == selected_experience]
    if selected_job_type != "All":
        filtered_df = filtered_df[filtered_df['Job_Type'] == selected_job_type]
    if selected_location != "All":
        filtered_df = filtered_df[filtered_df['Location'] == selected_location]

    filtered_df = filtered_df[filtered_df['Salary'] >= selected_salary]

    if filtered_df.empty:
        st.warning("❌ No jobs match the selected criteria. Try adjusting filters.")
    else:
        user_embedding = model.encode(input_text, convert_to_tensor=True)
        job_embeddings = model.encode(filtered_df['Skills'].tolist(), convert_to_tensor=True)
        scores = util.pytorch_cos_sim(user_embedding, job_embeddings)[0]
        filtered_df['Score'] = scores.cpu().numpy()
        top_jobs = filtered_df.sort_values(by="Score", ascending=False).head(3)

        st.subheader("📌 Top Job Recommendations")
        for idx, row in top_jobs.iterrows():
            st.markdown(f"### ✅ {row['Job_Title']}")
            st.write(f"**Industry:** {row['Industry']}")
            st.write(f"**Experience Level:** {row['Experience_Level']}")
            st.write(f"**Job Type:** {row['Job_Type']}")
            st.write(f"**Location:** {row['Location']}")
            st.write(f"**Salary:** {row['Salary']} PKR")
            st.write(f"**Skills Required:** {row['Skills']}")
            st.write(f"**Match Confidence:** `{row['Score']:.2f}`")

            matched_skills = [s.strip().lower() for s in row['Skills'].split(',') if s.strip().lower() in input_text.lower()]
            if matched_skills:
                st.info(f"💡 Matching Skills: {', '.join(matched_skills)}")
            else:
                st.warning("🔍 No direct skill overlap, but semantic match is high.")

            # Word cloud
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(row['Skills'])
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # ⭐ Feedback system
            rating = st.slider(f"⭐ Rate this job: {row['Job_Title']}", 1, 5, 3, key=f"rating_{idx}")
            if st.button(f"🔖 Bookmark {row['Job_Title']}", key=f"bookmark_{idx}"):
                st.success(f"✅ {row['Job_Title']} bookmarked!")
                st.session_state.bookmarked_jobs.append({
                    "Job_Title": row['Job_Title'],
                    "Industry": row['Industry'],
                    "Location": row['Location'],
                    "Rating": rating
                })

# ---------------------- SIDEBAR CHATBOT ----------------------
st.sidebar.title("💬 JobBot Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_prompt = st.sidebar.chat_input("Tell me your skills...")

if user_prompt:
    st.session_state.chat_history.append(("🧑", user_prompt))
    with st.spinner("🤖 Thinking..."):
        user_embedding = model.encode(user_prompt, convert_to_tensor=True)
        job_embeddings = model.encode(df['Skills'].tolist(), convert_to_tensor=True)
        scores = util.pytorch_cos_sim(user_embedding, job_embeddings)[0]
        df['Score'] = scores.cpu().numpy()
        top_jobs = df.sort_values(by="Score", ascending=False).head(3)

        bot_response = "Here are some jobs you might like:\n\n"
        for idx, row in top_jobs.iterrows():
            bot_response += f"✅ **{row['Job_Title']}** ({row['Industry']})\n"
            bot_response += f"Required Skills: {row['Skills']}\n\n"

        st.session_state.chat_history.append(("🤖", bot_response))

# Display chat
for sender, message in st.session_state.chat_history:
    st.sidebar.markdown(f"**{sender}**: {message}")

# ---------------------- BOOKMARKED JOBS ----------------------
if st.session_state.bookmarked_jobs:
    st.subheader("📌 Your Bookmarked Jobs")
    for job in st.session_state.bookmarked_jobs:
        st.markdown(f"**🔖 {job['Job_Title']}** | {job['Industry']} | {job['Location']} | ⭐ {job['Rating']}/5")
