import os
import html
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

st.set_page_config(
    page_title="AskSutube",
    page_icon="🎬",
    layout="wide",
)

# Inject a Gen Z inspired glassmorphism theme
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');
        :root {
            --bg: #07111f;
            --panel: rgba(8, 20, 38, 0.82);
            --panel-strong: rgba(10, 24, 46, 0.96);
            --border: rgba(156, 181, 214, 0.14);
            --text: #f5f7fb;
            --muted: #9fb0ca;
            --accent: #7cf0cf;
            --accent-2: #ff8f5a;
            --accent-3: #89b8ff;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
        }
        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--text);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(124, 240, 207, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 143, 90, 0.18), transparent 24%),
                radial-gradient(circle at 50% 80%, rgba(137, 184, 255, 0.14), transparent 26%),
                linear-gradient(160deg, #04101d 0%, #07111f 44%, #0b1930 100%);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stToolbar"] {
            right: 1rem;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(145deg, rgba(11, 26, 49, 0.92), rgba(9, 18, 31, 0.84)),
                linear-gradient(135deg, rgba(124, 240, 207, 0.08), rgba(255, 143, 90, 0.08));
            border: 1px solid var(--border);
            border-radius: 30px;
            box-shadow: var(--shadow);
            padding: 2.25rem;
            margin-bottom: 1.25rem;
        }
        .hero-shell::before {
            content: "";
            position: absolute;
            inset: auto -4rem -4rem auto;
            width: 16rem;
            height: 16rem;
            background: radial-gradient(circle, rgba(124, 240, 207, 0.18), transparent 70%);
            filter: blur(14px);
        }
        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background: rgba(124, 240, 207, 0.08);
            border: 1px solid rgba(124, 240, 207, 0.22);
            color: #dffbf2;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            font-family: 'Sora', sans-serif;
            font-size: clamp(2.4rem, 4vw, 4.8rem);
            line-height: 0.95;
            font-weight: 800;
            max-width: 11ch;
            letter-spacing: -0.05em;
            margin: 1rem 0 0.9rem;
        }
        .hero-gradient {
            background: linear-gradient(135deg, #ffffff 0%, #c3ffef 45%, #ffccaa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-sub {
            max-width: 52rem;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.7;
            margin-bottom: 1.25rem;
        }
        .stats-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .stat-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.75rem 0.95rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #d7e4f7;
            font-size: 0.92rem;
        }
        .panel-card {
            background: linear-gradient(180deg, rgba(10, 22, 40, 0.9), rgba(8, 18, 34, 0.96));
            border: 1px solid var(--border);
            border-radius: 26px;
            box-shadow: var(--shadow);
            padding: 1.1rem;
        }
        .side-note {
            height: 100%;
            border-radius: 22px;
            background:
                linear-gradient(180deg, rgba(124, 240, 207, 0.12), rgba(124, 240, 207, 0.02)),
                rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(124, 240, 207, 0.14);
            padding: 1.2rem;
        }
        .side-title {
            font-family: 'Sora', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .side-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.65;
            margin-bottom: 1rem;
        }
        .side-list {
            margin: 0;
            padding: 0;
            list-style: none;
            color: #e5eefc;
        }
        .side-list li {
            margin-bottom: 0.7rem;
            padding-left: 1rem;
            position: relative;
        }
        .side-list li::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0.55rem;
            width: 0.38rem;
            height: 0.38rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
        }
        .stTextInput label p {
            font-size: 0.9rem;
            font-weight: 700;
            color: #dce6f6;
        }
        .stTextInput > div > div > input {
            min-height: 3.35rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text);
            padding: 0 1rem;
            transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease;
        }
        .stTextInput > div > div > input::placeholder {
            color: #8ea3c5;
        }
        .stTextInput > div > div > input:focus {
            border-color: rgba(124, 240, 207, 0.68);
            box-shadow: 0 0 0 4px rgba(124, 240, 207, 0.12);
            background: rgba(255, 255, 255, 0.07);
        }
        .stButton > button {
            width: 100%;
            min-height: 3.35rem;
            margin-top: 1.9rem;
            border: 0;
            border-radius: 18px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #07111f;
            font-family: 'Sora', sans-serif;
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            box-shadow: 0 18px 40px rgba(255, 143, 90, 0.2);
            transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            filter: saturate(1.05);
            box-shadow: 0 22px 44px rgba(124, 240, 207, 0.16);
        }
        .stButton > button:focus {
            box-shadow: 0 0 0 4px rgba(124, 240, 207, 0.16);
        }
        [data-testid="stSpinner"] > div {
            border-top-color: var(--accent);
        }
        .answer-card {
            margin-top: 1.25rem;
            border-radius: 26px;
            padding: 1.3rem 1.4rem;
            background: linear-gradient(180deg, rgba(10, 22, 40, 0.92), rgba(6, 17, 31, 0.98));
            border: 1px solid rgba(137, 184, 255, 0.18);
            box-shadow: var(--shadow);
        }
        .answer-label {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #9ec6ff;
            margin-bottom: 0.7rem;
        }
        .answer-copy {
            color: #ecf3ff;
            line-height: 1.75;
            font-size: 1rem;
        }
        .helper-copy {
            color: var(--muted);
            font-size: 0.93rem;
            margin-top: 0.55rem;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 1rem;
            }
            .hero-shell {
                padding: 1.35rem;
                border-radius: 24px;
            }
            .panel-card {
                padding: 0.85rem;
            }
            .stButton > button {
                margin-top: 0.8rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()

# Function to extract video ID from YouTube URL
def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed_url.query).get("v", [None])[0]
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

@st.cache_data(show_spinner=False)
def get_transcript(video_id: str):
    try:
        api = YouTubeTranscriptApi()
        try:
            transcript_items = api.fetch(video_id, languages=["en"])
            return " ".join(item.text for item in transcript_items)
        except Exception:
            transcript_list = api.list(video_id)
            transcript = next(iter(transcript_list), None)
            if transcript is None:
                return "Error: No captions found for this video."

        if transcript.language_code != "en" and transcript.is_translatable:
            transcript = transcript.translate("en")

        transcript_items = transcript.fetch()
        return " ".join(item.text for item in transcript_items)

    except TranscriptsDisabled:
        return "No captions available for this video."
    except Exception as e:
        return f"Error: {e}"


# Function to create vector store
@st.cache_resource(show_spinner=False)
def create_vector_store(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Function to generate response
def generate_response(vector_store, question: str):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer in English only.
          Answer ONLY from the provided transcript context.
          The transcript context may be in a non-English language.
          Understand it and answer the user's question in English.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    answer = main_chain.invoke(question)
    return answer

with st.container():
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-kicker">AskSutube • Video Q&A</div>
            <div class="hero-title">
                Turn any <span class="hero-gradient">YouTube transcript</span> into instant answers.
            </div>
            <div class="hero-sub">
                Drop a link, ask a sharp question, and get a clean English answer grounded in the video transcript.
                Fast enough for skimming, strong enough for real research.
            </div>
            <div class="stats-row">
                <div class="stat-chip">Transcript-aware retrieval</div>
                <div class="stat-chip">English output from any supported caption language</div>
                <div class="stat-chip">Built for quick follow-up questions</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

with st.container():
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.7, 0.9], gap="large")
    with col1:
        youtube_url = st.text_input("YouTube URL", placeholder="https://youtu.be/xyz or https://youtube.com/watch?v=xyz")
        question = st.text_input("Your question", placeholder="e.g., What are the 3 main takeaways?")
        st.markdown('<div class="helper-copy">Tip: precise questions usually produce tighter answers.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            <div class="side-note">
                <div class="side-title">How it works</div>
                <div class="side-copy">
                    The app fetches captions, searches the most relevant transcript chunks, and answers from that context only.
                </div>
                <ul class="side-list">
                    <li>Works with English and supported translated captions</li>
                    <li>Optimized for repeat questions on the same video</li>
                    <li>Best for summary, explanation, and topic lookup</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        go = st.button("Get Answer", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if go:
    if youtube_url and question:
        with st.spinner("Processing..."):
            video_id = extract_video_id(youtube_url)
            if video_id:
                transcript = get_transcript(video_id)
                if not transcript.startswith("Error:") and not transcript.startswith("No captions"):
                    vector_store = create_vector_store(transcript)
                    answer = generate_response(vector_store, question)
                    st.markdown(
                        f"""
                        <section class="answer-card">
                            <div class="answer-label">Answer</div>
                            <div class="answer-copy">{html.escape(answer)}</div>
                        </section>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(transcript)
            else:
                st.error("Invalid YouTube URL.")
    else:
        st.warning("Please fill in all fields.")
