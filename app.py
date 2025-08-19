import os
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Hardcode API Key ---
# Set the OpenAI API key from the one provided in the notebook
os.environ["OPENAI_API_KEY"] = "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"

# --- Helper Functions ---

def extract_video_id(url: str) -> str:
    """Extracts the video ID from a YouTube URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed_url.query).get("v", [None])[0]
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

def get_available_transcripts(video_id: str):
    """Fetches and displays available transcript languages for a video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        st.write("Available transcript languages:")
        available_langs = {t.language: t.language_code for t in transcript_list}
        st.json(available_langs)
        return True
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return False
    except Exception as e:
        st.error(f"Could not retrieve transcripts: {e}")
        return False

def fetch_transcript_text(video_id: str, languages: list[str]) -> str:
    """Fetches and concatenates transcript text for selected languages."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return " ".join([item['text'] for item in transcript])
    except Exception as e:
        st.error(f"Error fetching transcript for languages {languages}. Please check the language codes.")
        return None

# --- RAG Pipeline ---

def get_rag_chain(vector_store, embedding_model_name):
    """Creates and returns a RAG chain."""
    # The LLM for generation will be OpenAI, as setting up a local HuggingFace LLM is complex.
    # The API key is already set in the environment.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          Context: {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Transcript Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Session State Initialization ---
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    embedding_model_name = st.selectbox(
        "Choose Embedding Model",
        ("OpenAI", "HuggingFace")
    )
    if embedding_model_name == "HuggingFace":
        st.info("Using HuggingFace for embeddings, but OpenAI's GPT-4o-mini for generation.")

    st.markdown("---")
    st.markdown(
        "Built by [Jules](https://github.com/jules-agent)"
    )

# --- Main Content ---
st.title("🎬 YouTube Transcript Assistant")
st.markdown(
    "Paste a YouTube video URL below, choose your language, and fetch the transcript."
)

# --- Input Section ---
st.header("1. Fetch Transcript")
youtube_url = st.text_input("YouTube URL", key="youtube_url_input")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
    else:
        if get_available_transcripts(video_id):
            lang_codes_input = st.text_input(
                "Enter language code(s) (e.g., 'en', or 'hi,en')",
                "en"
            )

            if st.button("Fetch Transcript", type="primary"):
                if not lang_codes_input:
                    st.warning("Please enter at least one language code.")
                else:
                    lang_codes = [code.strip() for code in lang_codes_input.split(',')]
                    with st.spinner("Fetching transcript and building vector store..."):
                        transcript_text = fetch_transcript_text(video_id, lang_codes)

                        if transcript_text:
                            st.session_state.transcript = transcript_text

                            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            chunks = splitter.create_documents([transcript_text])

                            try:
                                if embedding_model_name == "OpenAI":
                                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                                else: # HuggingFace
                                    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-4B")

                                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                                st.success("Transcript fetched and processed successfully!")
                                # Clear previous chat history on new transcript
                                st.session_state.messages = []
                            except Exception as e:
                                st.error(f"Failed to create embeddings: {e}")

# --- Transcript Display ---
if st.session_state.transcript:
    st.header("2. View and Download Transcript")
    with st.expander("View Transcript"):
        st.text_area("", st.session_state.transcript, height=250)

    st.download_button(
        label="Download Transcript",
        data=st.session_state.transcript.encode('utf-8'),
        file_name="transcript.txt",
        mime="text/plain",
    )

# --- Q&A Section ---
st.header("3. Ask a Question")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.vector_store:
    if prompt := st.chat_input("Ask a question about the transcript..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating answer..."):
            rag_chain = get_rag_chain(st.session_state.vector_store, embedding_model_name)
            if rag_chain:
                answer = rag_chain.invoke(prompt)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Could not create RAG chain.")
else:
    st.info("Fetch a transcript to enable the Q&A section.")
