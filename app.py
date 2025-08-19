import os
import streamlit as st
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Set the OpenAI API key
# For local development, create a .env file and add your OPENAI_API_KEY.
# For Streamlit sharing, use st.secrets["OPENAI_API_KEY"].
from dotenv import load_dotenv
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set your OPENAI_API_KEY environment variable or add it to your .env file.")
    st.stop()

def extract_video_id(url: str) -> str:
    """Extracts video ID from YouTube URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        qs = parse_qs(parsed_url.query)
        return qs.get("v", [None])[0]
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

def get_transcript(video_id: str, language: str) -> str:
    """Fetches transcript for a given video ID and language."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        transcript = " ".join(d['text'] for d in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except Exception as e:
        # Fallback to English if the selected language is not available
        if language != 'en':
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                transcript = " ".join(d['text'] for d in transcript_list)
                st.warning(f"Falling back to English transcript as '{language}' is not available.")
                return transcript
            except Exception as en_e:
                return f"Error: Could not fetch transcript in '{language}' or English. Details: {en_e}"
        return f"Error: {e}"

@st.cache_data
def create_rag_chain(youtube_url, language):
    """Creates and caches the RAG chain for a given video URL and language."""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None, "Invalid YouTube URL."

    transcript = get_transcript(video_id, language)
    if transcript.startswith("Error:"):
        return None, transcript

    # 1. Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 2. Embedding Generation and Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. Prompt and LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 4. RAG Chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | StrOutputParser()
    return main_chain, None


def main():
    st.title("YouTube Video Question Answering")

    youtube_url = st.text_input("Enter YouTube Video URL:")
    language = st.selectbox("Select Language:", ("en", "hi"))
    question = st.text_input("Ask a question about the video:")
    submit_button = st.button("Get Answer")

    if submit_button:
        if youtube_url and question:
            with st.spinner("Finding answer..."):
                chain, error_message = create_rag_chain(youtube_url, language)

                if error_message:
                    st.error(error_message)
                else:
                    try:
                        answer = chain.invoke(question)
                        st.success("Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred while generating the answer: {e}")
        else:
            st.warning("Please provide a YouTube URL and a question.")

if __name__ == "__main__":
    main()
