🎬 AskSutube – AI Q&A for YouTube Videos

AskSutube is an AI-powered web app that allows users to ask questions about any YouTube video and receive accurate, context-aware answers.
It leverages Retrieval-Augmented Generation (RAG) with LangChain, OpenAI embeddings, and FAISS to retrieve relevant transcript chunks and generate precise answers with GPT-4o-mini.
The app is deployed on Streamlit with a modern UI and robust error handling for a smooth user experience.

🚀 Features

✅ Ask Questions About Any YouTube Video using video transcripts

✅ Retrieval-Augmented Generation (RAG) pipeline for context-grounded answers

✅ Multilingual transcript support (e.g., en, hi) with automatic fallback

✅ Modern Streamlit UI with transcript preview, Q&A box, and context viewer

✅ Error handling for invalid URLs, missing captions, or restricted videos

✅ Secure key management with .env + python-dotenv + .gitignore

🛠 Tech Stack

Frontend / UI: Streamlit (modern dark theme, glassmorphism style)

Backend / Pipeline: Python, LangChain, youtube-transcript-api

LLM: OpenAI GPT-4o-mini (generation)

Embeddings: OpenAI text-embedding-3-small

Vector Store: FAISS (semantic similarity search, k=4)

Utilities: RecursiveCharacterTextSplitter, PromptTemplate, RunnableParallel, StrOutputParser

Secret Management: python-dotenv, .gitignore

⚙️ How It Works

Extracts video ID from YouTube URL

Fetches transcript with youtube-transcript-api (.fetch)

Splits transcript into chunks (RecursiveCharacterTextSplitter)

Converts chunks into embeddings (OpenAI) and stores in FAISS

Retrieves top-k relevant chunks for each question

Uses GPT-4o-mini to generate answers only from retrieved context

📦 Installation

Clone the repo:

git clone https://github.com/sameergupta914/Askutube.git
cd Askutube


Create a virtual environment & install dependencies:

pip install -r requirements.txt


Add your OpenAI API key in .env:

OPENAI_API_KEY=your_api_key_here


Run the app:

streamlit run app.py

📸 Screenshots

(Add screenshots of your app here for better presentation)

🔒 Security

API keys are stored in .env and never exposed in code.

.gitignore ensures .env is not pushed to GitHub.

🚧 Limitations

YouTube may block transcript requests from cloud IPs (Streamlit Cloud, AWS, etc.).

Some videos do not provide captions (especially music or copyrighted content).

For robustness, a Whisper speech-to-text fallback can be added in the future.

🤝 Contributing

Pull requests and feature suggestions are welcome!

📜 License

MIT License – free to use and modify.
