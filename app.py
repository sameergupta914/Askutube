import os
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    """Fetches available transcript languages for a video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return {t.language: t.language_code for t in transcript_list}
    except TranscriptsDisabled:
        return None
    except Exception as e:
        st.error(f"Error fetching transcript list: {e}")
        return None

def fetch_transcript_text(video_id: str, languages: list[str]) -> str:
    """Fetches and concatenates transcript text for selected languages."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return " ".join([item['text'] for item in transcript])
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

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


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Embedding model selection
    embedding_model_name = st.selectbox(
        "Choose Embedding Model",
        ("OpenAI", "HuggingFace")
    )

    # API key input
    api_key = st.text_input(
        "Enter your API Key",
        type="password",
        help="Get your API key from OpenAI or HuggingFace."
    )

    st.markdown("---")
    st.markdown(
        "Built by [Jules](https://github.com/jules-agent)"
    )

# --- Main Content ---
st.title("🎬 YouTube Transcript Assistant")
st.markdown(
    "Paste a YouTube video URL below, fetch its transcript, and start asking questions!"
)

# --- Input Section ---
st.header("1. Enter YouTube URL")
youtube_url = st.text_input("YouTube URL", key="youtube_url_input")

video_id = None
if youtube_url:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid URL.")

if video_id:
    available_transcripts = get_available_transcripts(video_id)
    if available_transcripts:
        selected_languages = st.multiselect(
            "Select transcript language(s)",
            options=list(available_transcripts.keys()),
            format_func=lambda lang: f"{lang} ({available_transcripts[lang]})"
        )

        if st.button("Fetch Transcript", type="primary"):
            if not selected_languages:
                st.warning("Please select at least one language.")
            else:
                with st.spinner("Fetching transcript and building vector store..."):
                    transcript_text = fetch_transcript_text(video_id, [available_transcripts[lang] for lang in selected_languages])

                    if transcript_text:
                        st.session_state.transcript = transcript_text

                        # Split text into chunks
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = splitter.create_documents([transcript_text])

                        # Create embeddings and vector store
                        try:
                            if embedding_model_name == "OpenAI":
                                if not api_key:
                                    st.error("Please enter your OpenAI API key in the sidebar.")
                                else:
                                    os.environ["OPENAI_API_KEY"] = api_key
                                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                                    st.success("Transcript fetched and processed successfully!")

                            elif embedding_model_name == "HuggingFace":
                                # Note: HuggingFace embeddings run locally and may be slow.
                                # A more robust implementation might use a dedicated sentence-transformer server.
                                embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-4B")
                                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                                st.success("Transcript fetched and processed successfully!")

                        except Exception as e:
                            st.error(f"Failed to create embeddings: {e}")

    else:
        st.error("No transcripts available for this video or an error occurred.")


# --- Transcript Display ---
st.header("2. Transcript")
if st.session_state.transcript:
    with st.expander("View Transcript"):
        st.text_area("", st.session_state.transcript, height=250)

    st.download_button(
        label="Download Transcript",
        data=st.session_state.transcript,
        file_name="transcript.txt",
        mime="text/plain",
    )
else:
    st.info("Transcript will be displayed here after fetching.")


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- RAG Pipeline ---

def get_rag_chain(vector_store, api_key, embedding_model_name):
    """Creates and returns a RAG chain."""
    if embedding_model_name == "OpenAI":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
    else:
        # Using a HuggingFace model for generation would require more setup
        # (e.g., a local model or an API). For now, we'll default to a
        # message indicating this is not fully implemented.
        # As a fallback, we can use OpenAI's LLM for generation,
        # even with HuggingFace embeddings.
        st.warning("HuggingFace LLM for generation is not implemented. Falling back to OpenAI's gpt-4o-mini for the chat part.")
        if not api_key:
            st.error("An OpenAI API key is still required for the generation step.")
            return None
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

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


# --- Q&A Section ---
st.header("3. Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
            rag_chain = get_rag_chain(st.session_state.vector_store, api_key, embedding_model_name)
            if rag_chain:
                answer = rag_chain.invoke(prompt)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Could not create RAG chain. Please check your API key.")

else:
    st.info("Fetch a transcript to enable the Q&A section.")
