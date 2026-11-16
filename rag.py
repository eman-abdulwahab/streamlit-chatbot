from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


load_dotenv()
project_dir = Path(__file__).resolve().parent


def get_knowledge_dir() -> Path:
    path = Path(os.getenv("KNOWLEDGE_BASE_DIR") or "knowledge_base").expanduser()
    if not path.is_absolute():
        path = project_dir / path

    return path.resolve()

def load_pdf_chunks():
    knowledge_dir = get_knowledge_dir()
    pdf_paths = sorted(knowledge_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {knowledge_dir}. Add a text-based PDF and try again."
        )

    pages = []
    for pdf_path in pdf_paths:
        for page in PyPDFLoader(str(pdf_path)).load():
            if not page.page_content.strip():
                continue

            page.metadata["source"] = pdf_path.name
            page.metadata["page_number"] = int(page.metadata.get("page", 0)) + 1
            pages.append(page)

    if not pages:
        raise ValueError(
            "No extractable text found in the PDFs. Scanned PDFs require OCR, "
            "which this version does not support."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(pages)

def build_or_load_vector_store() -> Chroma:
    vector_dir = project_dir / "vector_store"
    ready_marker = vector_dir / ".index_ready"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if ready_marker.is_file():
        return Chroma(
            collection_name="pdf_knowledge",
            embedding_function=embeddings,
            persist_directory=str(vector_dir),
        )

    chunks = load_pdf_chunks()

    # This directory is reserved for the generated index.
    if vector_dir.exists():
        shutil.rmtree(vector_dir)

    vector_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_knowledge",
        persist_directory=str(vector_dir),
    )

    ready_marker.touch()
    return vector_store





def create_rag_chain(vector_store: Chroma, llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer using the supplied PDF context when it is relevant. "
            "Cite document-supported claims using the source and page details "
            "shown in the context. If the context does not answer the question, "
            "say so and clearly label any answer based on general knowledge.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{input}"),
    ])

    document_prompt = PromptTemplate.from_template(
        "Source: {source}, page {page_number}\n{page_content}"
    )

    answer_chain = create_stuff_documents_chain(
        llm,
        prompt,
        document_prompt=document_prompt,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return create_retrieval_chain(retriever, answer_chain)