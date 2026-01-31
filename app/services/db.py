from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from app.core.config import settings


# Initialize Embedding Model
embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=settings.GOOGLE_API_KEY,
)

# Initialize Vector Store (will clear after reload)
vector_store = Chroma(
    collection_name="user_documents",
    embedding_function=embedding_function,
)

def ingest_file(file_path: str, session_id: str):
    """
    Loads a PDF, splits it into chunks, and saves it to ChromaDB
    with the specific session_id in the metadata.
    """
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # 3. Tag with session_id
    for split in splits:
        split.metadata["session_id"] = session_id

    # 4. Save to DB
    vector_store.add_documents(splits)
    print(f"--- INGESTION: Saved {len(splits)} chunks for session {session_id} ---")

def get_retriever(session_id: str):
    """
    Returns a retriever that strictly filters documents by session_id.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": {"session_id": session_id}
        }
    )