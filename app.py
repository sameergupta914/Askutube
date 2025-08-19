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
from dotenv import load_dotenv

load_dotenv()

# Function to extract video ID from YouTube URL
def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed_url.query).get("v", [None])[0]
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

# Function to get transcript

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def get_transcript(video_id: str, language: str):
    try:
        # fetch() is the correct method in your version
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=[language, 'en'])

        # fetched.snippets returns list of FetchedTranscriptSnippet objects
        transcript = " ".join(s.text for s in fetched.snippets)
        return transcript

    except TranscriptsDisabled:
        return "No captions available for this video."
    except Exception as e:
        return f"Error: {e}"


# Function to create vector store
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
          Answer ONLY from the provided transcript context.
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

st.title("YouTube RAG Model")

youtube_url = st.text_input("Enter YouTube URL:")
language = st.text_input("Enter transcript language (e.g., 'en', 'hi'):")
question = st.text_input("Ask a question about the video:")

if st.button("Get Answer"):
    if youtube_url and language and question:
        with st.spinner("Processing..."):
            video_id = extract_video_id(youtube_url)
            if video_id:
                transcript = get_transcript(video_id, language)
                if not transcript.startswith("Error:") and not transcript.startswith("No captions"):
                    vector_store = create_vector_store(transcript)
                    answer = generate_response(vector_store, question)
                    st.write("Answer:", answer)
                else:
                    st.error(transcript)
            else:
                st.error("Invalid YouTube URL.")
    else:
        st.warning("Please fill in all fields.")
