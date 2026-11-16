# PDF RAG Chatbot

A Streamlit chatbot that uses Groq for responses and retrieves context from local PDFs using LangChain, Hugging Face embeddings, and Chroma.

## Configuration

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

Configure the values in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
USER_NAME=user
KNOWLEDGE_BASE_DIR=knowledge_base
```

`KNOWLEDGE_BASE_DIR` can be a path relative to the project directory or an absolute path.

## Add PDFs

By default, the app reads PDFs from the `knowledge_base/` directory. Replace the provided PDF with your own selectable-text PDF, or add one or more PDFs to that directory. If you use a different location, set `KNOWLEDGE_BASE_DIR` in `.env` to that directory's relative or absolute path.


## Run

Install the project dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run chatbot.py
```

On the first run, the app downloads the embedding model and creates the local Chroma index in `vector_store/`. Later runs reuse the saved index.

The index does not automatically update when PDFs change. To rebuild it, stop the app, delete `vector_store/`, make sure the PDFs are present, and start the app again.