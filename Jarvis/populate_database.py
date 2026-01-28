import argparse
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"
DATA_PATH = "data"
DEFAULT_TTL_DAYS = 30

# Supported code file extensions and their languages
CODE_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
}

# Code extensions without language-specific splitters (use default text splitter)
CODE_EXTENSIONS_NO_LANG = {".css", ".json", ".yaml", ".yml", ".xml", ".sql", ".sh", ".bash", ".ps1", ".bat"}

# Text-based extensions (not code, but text files)
TEXT_EXTENSIONS = {".txt", ".log", ".ini", ".cfg", ".conf", ".env", ".gitignore", ".dockerignore"}


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument(
        "--path",
        type=str,
        default=DATA_PATH,
        help="PDF file or directory to ingest (defaults to ./data)",
    )
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    db = get_db()
    prune_stale_chunks(db, ttl_days=DEFAULT_TTL_DAYS)

    # Create (or update) the data store.
    documents = load_documents(args.path)
    chunks = split_documents(documents)
    add_to_chroma(db, chunks)


def get_db() -> Chroma:
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


def load_documents(target_path: str):
    target = Path(target_path)

    if target.is_dir():
        all_documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            str(target), glob="**/*.pdf", loader_cls=PDFPlumberLoader, show_progress=True
        )
        try:
            all_documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Warning: Could not load PDFs: {e}")
        
        # Load code and text files
        all_code_extensions = list(CODE_EXTENSIONS.keys()) + list(CODE_EXTENSIONS_NO_LANG)
        code_and_text_extensions = all_code_extensions + list(TEXT_EXTENSIONS)
        for ext in code_and_text_extensions:
            pattern = f"**/*{ext}"
            try:
                text_loader = DirectoryLoader(
                    str(target), 
                    glob=pattern, 
                    loader_cls=TextLoader,
                    loader_kwargs={"autodetect_encoding": True},
                    show_progress=True
                )
                all_documents.extend(text_loader.load())
            except Exception as e:
                print(f"Warning: Could not load {ext} files: {e}")
        
        return all_documents
    
    if target.is_file():
        suffix = target.suffix.lower()
        
        if suffix == ".pdf":
            return PDFPlumberLoader(str(target)).load()
        elif suffix in CODE_EXTENSIONS or suffix in CODE_EXTENSIONS_NO_LANG or suffix in TEXT_EXTENSIONS:
            return TextLoader(str(target), autodetect_encoding=True).load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    raise FileNotFoundError(f"No file or directory found at: {target}")


def get_splitter_for_document(doc: Document):
    """Get the appropriate text splitter based on document source."""
    source = doc.metadata.get("source", "")
    suffix = Path(source).suffix.lower()
    
    if suffix in CODE_EXTENSIONS:
        language = CODE_EXTENSIONS[suffix]
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1000,
            chunk_overlap=100,
        )
    
    # Default splitter for PDFs and text files
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(documents: list[Document]):
    all_chunks = []
    
    # Group documents by their file type for efficient splitting
    for doc in documents:
        splitter = get_splitter_for_document(doc)
        chunks = splitter.split_documents([doc])
        
        # Add file type metadata to chunks
        source = doc.metadata.get("source", "")
        suffix = Path(source).suffix.lower()
        file_type = "code" if (suffix in CODE_EXTENSIONS or suffix in CODE_EXTENSIONS_NO_LANG) else "pdf" if suffix == ".pdf" else "text"
        
        for chunk in chunks:
            chunk.metadata["file_type"] = file_type
            if suffix in CODE_EXTENSIONS or suffix in CODE_EXTENSIONS_NO_LANG:
                chunk.metadata["language"] = suffix[1:]  # Remove the dot
        
        all_chunks.extend(chunks)
    
    return all_chunks


def add_to_chroma(db: Chroma, chunks: list[Document], session_id: str | None = "global"):
    """Persist document chunks with session scoping.

    session_id is stored on every chunk so queries can be filtered per chat session.
    When omitted, chunks are tagged as "global" and will only be retrieved by
    session-less queries.
    """

    now = datetime.now(timezone.utc).isoformat()
    session_key = session_id or "global"

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    for chunk in chunks_with_ids:
        chunk.metadata.setdefault("created_at", now)
        chunk.metadata["last_accessed"] = now
        chunk.metadata["session_id"] = session_key

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2" for PDFs
    # Or "data/script.py:0:2" for code files (page=0 for code)
    # Source : Page/Section Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        # For code files, use 0 as page (they don't have pages)
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def prune_stale_chunks(db: Chroma, ttl_days: int = DEFAULT_TTL_DAYS):
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    items = db.get(include=["metadatas"])

    stale_ids = []
    for doc_id, metadata in zip(items.get("ids", []), items.get("metadatas", [])):
        last_accessed_raw = metadata.get("last_accessed") if metadata else None
        try:
            last_accessed = (
                datetime.fromisoformat(last_accessed_raw)
                if last_accessed_raw
                else None
            )
        except (TypeError, ValueError):
            last_accessed = None

        if last_accessed is None or last_accessed < cutoff:
            stale_ids.append(doc_id)

    if stale_ids:
        print(f"🧹 Pruning stale chunks: {len(stale_ids)} (TTL {ttl_days}d)")
        db.delete(ids=stale_ids)


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
