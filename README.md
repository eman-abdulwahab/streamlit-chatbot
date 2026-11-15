# Generative AI and LangChain Starter

This is my starter experimental project for learning and working with generative AI and LangChain. It is a small Streamlit chatbot that uses LangChain to connect to a Groq-hosted language model. I plan to use it as a sandbox for trying ideas and building my understanding of these tools.

## Getting Started

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:

```bash
streamlit run chatbot.py
```