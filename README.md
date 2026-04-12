# AskSutube

AskSutube is a Streamlit app that lets users paste a YouTube link, ask a question about the video, and get an English answer grounded in the video's transcript.

Instead of sending the full video to an LLM directly, the app uses a retrieval pipeline:
- fetch the transcript
- split it into chunks
- embed the chunks with OpenAI embeddings
- store them in FAISS
- retrieve the most relevant chunks for the question
- generate an answer only from that retrieved context

This keeps answers more focused, faster, and better tied to the source material.

Live app: `https://asksutube.streamlit.app/`

## What The Project Does

AskSutube solves a simple problem: long YouTube videos are hard to search manually.

With this app, a user can:
- paste a YouTube URL
- ask a natural-language question
- get an answer based on the transcript, not a generic guess

Typical use cases:
- summarize a lecture or podcast episode
- find a specific concept explained in a long video
- extract key takeaways from interviews, tutorials, or talks
- ask follow-up questions on the same video

## Current Features

- YouTube transcript-based question answering
- Retrieval-Augmented Generation using LangChain
- English answers even when the original transcript is non-English
- Automatic transcript fetching with fallback behavior
- FAISS vector search for transcript chunk retrieval
- Streamlit frontend with a custom responsive UI
- Caching for transcript fetches and vector store creation
- Error handling for invalid URLs and missing captions

## How It Works

### 1. URL parsing

The app extracts the YouTube video ID from:
- `https://www.youtube.com/watch?v=...`
- `https://youtu.be/...`

### 2. Transcript retrieval

The app uses `youtube-transcript-api` to fetch captions.

Current behavior:
- it first tries to fetch English captions directly
- if English captions are not available, it looks for another available transcript
- if that transcript is translatable, it asks YouTube for an English-translated transcript

If captions are disabled or unavailable, the app returns an error message to the user.

### 3. Chunking

The transcript is split with `RecursiveCharacterTextSplitter` into overlapping chunks so retrieval remains relevant even for long videos.

Current settings:
- `chunk_size=1000`
- `chunk_overlap=200`

### 4. Embeddings and vector search

Each chunk is converted into embeddings using:
- `text-embedding-3-small`

Those embeddings are stored in a FAISS vector index for semantic similarity search.

### 5. Answer generation

When the user asks a question:
- the app retrieves the most relevant transcript chunks
- injects them into a prompt
- sends the prompt to `gpt-4o-mini`
- returns an English answer

The prompt explicitly instructs the model to answer only from transcript context and say it does not know when the transcript is insufficient.

## Tech Stack

### Frontend

- Streamlit
- custom CSS for layout, typography, and visual styling

### Backend

- Python
- `youtube-transcript-api`
- LangChain core + community packages

### LLM and embeddings

- OpenAI `gpt-4o-mini` for answer generation
- OpenAI `text-embedding-3-small` for embeddings

### Vector store

- FAISS

### Utilities

- `python-dotenv`
- `urllib.parse`

## Project Structure

```text
AskSutube/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .env
`-- rag_using_langchain.ipynb
```

### Main files

- `app.py`
  Main Streamlit application, transcript retrieval logic, vector creation, and answer generation.

- `requirements.txt`
  Python dependencies required to run the project.

- `rag_using_langchain.ipynb`
  Notebook version / experimentation space for the RAG workflow.

- `.env`
  Stores the OpenAI API key locally.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/sameergupta914/Askutube.git
cd Askutube
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

After that, Streamlit will open the app in your browser.

## How To Use

1. Open the app.
2. Paste a valid YouTube URL.
3. Enter a question about the video.
4. Click `Get Answer`.
5. Read the generated answer in English.

Examples:
- `What are the main takeaways from this video?`
- `What does the speaker say about startup growth?`
- `Summarize the section about AI agents.`
- `What are the pros and cons mentioned?`

## Example Request Flow

```text
User enters URL + question
        |
        v
Extract video ID
        |
        v
Fetch transcript
        |
        v
Split transcript into chunks
        |
        v
Embed chunks with OpenAI
        |
        v
Store/search with FAISS
        |
        v
Retrieve relevant chunks
        |
        v
Generate final answer with GPT-4o-mini
```

## Caching And Performance

The app currently uses Streamlit caching to reduce repeated work:

- `@st.cache_data` for transcript retrieval
- `@st.cache_resource` for vector store creation

This helps a lot when users ask multiple questions about the same video.

## Known Limitations

- The app depends on transcript availability. If a video has no captions, it cannot answer questions.
- Some YouTube videos may block transcript retrieval.
- Transcript quality depends on the original captions. Auto-generated captions can be noisy.
- Answers are only as good as the retrieved transcript chunks.
- Very long transcripts still require embedding and retrieval work on first load.
- The app does not currently show source chunk citations in the UI.

## Error Cases The App Handles

- invalid YouTube URL
- missing captions
- captions disabled
- generic transcript fetch failures

These are surfaced to the user through Streamlit error messages.

## Deploy to Streamlit Cloud

This project is ready to deploy on [Streamlit Community Cloud](https://streamlit.io/cloud).

### Steps to Deploy

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Create a new app**:
   - Click "New app"
   - Select your repository: `sameergupta914/Askutube`
   - Branch: `main`
   - App file: `app.py`

4. **Add your OpenAI API key**:
   - In the Streamlit dashboard, click on your app
   - Go to "Secrets" (or "Settings" → "Secrets")
   - Add the following:
     ```toml
     OPENAI_API_KEY = "your_actual_openai_api_key_here"
     ```
   - Click "Save"

5. **Deploy**:
   - Click "Deploy!"
   - Your app will be live at `https://asksutube.streamlit.app/`

### Local Testing with Secrets

For local development, create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

## Why This Project Is Useful

AskSutube is a practical example of a lightweight RAG system applied to real content that people already consume every day.

It demonstrates:
- transcript ingestion
- semantic chunking
- vector search
- context-grounded LLM answering
- a usable end-user interface built with Streamlit

So the project is useful both as:
- a product users can directly use
- a portfolio project showing applied LLM engineering

## Possible Future Improvements

- show transcript snippets used for each answer
- support chat history across multiple follow-up questions
- support transcript fallback via speech-to-text when captions are unavailable
- add source timestamps from transcript chunks
- let users choose answer style such as concise, detailed, or bullet summary
- add sample questions and better empty states
- persist transcript indexes for repeated videos

## Security Notes

- Keep `OPENAI_API_KEY` in `.env`, not in source code.
- Do not commit `.env` to Git.
- Rotate the API key if it is ever exposed.

## Contributing

Contributions are welcome.

Useful contribution areas:
- improving retrieval quality
- adding timestamp-aware citations
- improving latency for first-time video processing
- enhancing the frontend UX
- expanding test coverage

## License

MIT License.
