import os
import io
import logging
import requests
from typing import Optional, Tuple, List
import chromadb
try:
    from agno.schema import Document
except Exception:
    # Fallback lightweight Document shim for environments without agno.schema
    class Document:
        def __init__(self, id: str, text: str, metadata: Optional[dict] = None):
            self.id = id
            self.text = text
            self.metadata = metadata or {}

        def to_dict(self):
            return {"id": self.id, "text": self.text, "metadata": self.metadata}


class ChromaDbAdapter:
    def __init__(self, name: str, client, embedder=None):
        self.name = name
        self.client = client
        self.embedder = embedder

    def create_collection(self):
        try:
            self.client.create_collection(name=self.name)
        except Exception as e:
            logging.getLogger(__name__).debug(f"create_collection: {e}")

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.name)
        except Exception as e:
            logging.getLogger(__name__).debug(f"delete_collection: {e}")

    def add(self, chunks):
        def _norm(chunks_list):
            out = []
            for idx, c in enumerate(chunks_list):
                if isinstance(c, dict):
                    text = c.get('text') or c.get('content') or c.get('chunk') or c.get('document') or ''
                    id_ = c.get('id') or c.get('chunk_id') or f"chunk_{idx}"
                    meta = c.get('meta') or c.get('metadata') or {}
                    out.append({'id': id_, 'text': text, 'meta': meta})
                elif isinstance(c, str):
                    out.append({'id': f"chunk_{idx}", 'text': c, 'meta': {}})
                else:
                    out.append({'id': f"chunk_{idx}", 'text': str(c), 'meta': {}})
            return out

        normalized = _norm(chunks)
        ids = [c['id'] for c in normalized]
        documents = [c['text'] for c in normalized]
        metadatas = [c['meta'] for c in normalized]

        embeddings = None
        if self.embedder:
            try:
                embeddings = self.embedder.get_embedding(documents)
            except Exception:
                logging.getLogger(__name__).debug("Adapter embedding failed")

        try:
            try:
                coll = self.client.get_collection(name=self.name)
            except Exception:
                coll = self.client.create_collection(name=self.name)
            coll.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Adapter failed to add to collection: {e}")
            return False

    def create(self) -> None:
        try:
            self.client.create_collection(name=self.name)
        except Exception as e:
            logging.getLogger(__name__).debug(f"create (adapter): {e}")

    def exists(self) -> bool:
        try:
            self.client.get_collection(name=self.name)
            return True
        except Exception:
            return False

    def content_hash_exists(self, content_hash: str) -> bool:
        return False

    def search(self, query: str, limit: int = 5, filters: Optional[dict] = None):
        try:
            coll = self.client.get_collection(name=self.name)
            res = coll.query(query_texts=[query], n_results=limit)

            docs_out: List[Document] = []
            # Chroma typically returns dicts with nested lists for multi-query responses
            if isinstance(res, dict):
                documents = res.get("documents")
                metadatas = res.get("metadatas")
                ids = res.get("ids")

                # normalize to inner lists (first query)
                if isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list):
                    docs_list = documents[0]
                    metas_list = metadatas[0] if metadatas and isinstance(metadatas, list) and len(metadatas) > 0 else None
                    ids_list = ids[0] if ids and isinstance(ids, list) and len(ids) > 0 else None
                else:
                    docs_list = documents if documents is not None else []
                    metas_list = metadatas
                    ids_list = ids

                for i, doc_text in enumerate(docs_list if docs_list is not None else []):
                    text = doc_text if isinstance(doc_text, str) else str(doc_text)
                    meta = (metas_list[i] if metas_list and i < len(metas_list) else {}) if metas_list is not None else {}
                    id_ = (ids_list[i] if ids_list and i < len(ids_list) else f"{self.name}_doc_{i}") if ids_list is not None else f"{self.name}_doc_{i}"
                    docs_out.append(Document(id=id_, text=text, metadata=meta))
                return docs_out

            # Fallback: if Chroma returned a list of items
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        text = item.get('document') or item.get('text') or str(item)
                        id_ = item.get('id') or item.get('ids') or f"{self.name}_doc"
                        meta = item.get('metadata') or item.get('meta') or {}
                        docs_out.append(Document(id=id_, text=text, metadata=meta))
                return docs_out
            return []
        except Exception:
            logging.getLogger(__name__).debug("Adapter search failed")
            return []

    async def async_search(self, query: str, limit: int = 5, filters: Optional[dict] = None):
        """Async wrapper around `search` for runtimes that call async_search on adapters.

        This uses asyncio.to_thread when available to run the blocking search in a thread.
        """
        try:
            try:
                import asyncio
                return await asyncio.to_thread(self.search, query, limit, filters)
            except Exception:
                # Fallback: run synchronously if asyncio.to_thread not available
                return self.search(query, limit, filters)
        except Exception:
            logging.getLogger(__name__).debug("Adapter async_search failed")
            return []


def setup_vector_db(name: str = "apple_docs", path: str = "./chroma_db", embedder=None, logger: Optional[logging.Logger] = None) -> Tuple[chromadb.PersistentClient, ChromaDbAdapter, bool, Optional[int]]:
    """Create or retrieve a chroma PersistentClient and adapter.

    Returns: (chroma_client, adapter, recreate_flag, collection_size)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    chroma_client = chromadb.PersistentClient(path=path)

    RECREATE_COLLECTION = os.getenv("RECREATE_COLLECTION", "false").lower() in ("1", "true", "yes")
    if RECREATE_COLLECTION:
        logger.info("RECREATE_COLLECTION=true — deleting + recreating '%s' collection." % name)
        try:
            chroma_client.delete_collection(name=name)
            logger.info(f"Deleted existing collection '{name}'.")
        except Exception as ex:
            logger.debug(f"Collection delete: {ex}")
        try:
            chroma_client.create_collection(name=name)
            logger.info(f"Created collection '{name}' on chroma client.")
        except Exception as ex:
            logger.debug(f"Collection creation returned: {ex} (it may already exist)")
    else:
        try:
            chroma_client.get_collection(name=name)
            logger.info(f"Collection '{name}' already exists — leaving intact.")
        except Exception:
            try:
                chroma_client.create_collection(name=name)
                logger.info(f"Created collection '{name}' on chroma client.")
            except Exception as ex:
                logger.debug(f"Collection creation returned: {ex}")

    # Determine collection size if possible
    collection_size = None
    try:
        coll = chroma_client.get_collection(name=name)
        if hasattr(coll, "count") and callable(getattr(coll, "count")):
            try:
                collection_size = coll.count()
            except Exception:
                collection_size = None
        if collection_size is None and hasattr(coll, "get") and callable(getattr(coll, "get")):
            try:
                res = coll.get(include=["ids"]) if True else coll.get()
                ids = res.get("ids") if isinstance(res, dict) else None
                if ids is not None:
                    collection_size = len(ids)
            except Exception:
                collection_size = None
    except Exception:
        collection_size = None

    adapter = ChromaDbAdapter(name=name, client=chroma_client, embedder=embedder)

    return chroma_client, adapter, RECREATE_COLLECTION, collection_size


def ingest_pdf_to_collection(
    pdf_url: str,
    adapter: ChromaDbAdapter,
    embedder,
    chroma_client: chromadb.PersistentClient,
    pdf_local_path: Optional[str] = None,
    ingest_mode: str = "auto",
    skip_ingest: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """Download/parse a PDF, chunk, embed, and add to the provided adapter/collection.

    This function mirrors the previous inline ingestion behavior but is kept
    separate so callers can choose when to run it.
    Returns True if chunks were added, False otherwise.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if skip_ingest:
        logger.info("skip_ingest=True — skipping PDF ingest.")
        return False

    pdf_content = None
    if pdf_local_path:
        try:
            logger.info(f"Loading PDF from local path: {pdf_local_path}")
            with open(pdf_local_path, "rb") as f:
                pdf_content = io.BytesIO(f.read())
        except Exception as e:
            logger.warning(f"Failed to read local PDF '{pdf_local_path}': {e}")
    else:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempting to download PDF (attempt {attempt}/{max_retries})...")
                response = requests.get(pdf_url, timeout=15)
                response.raise_for_status()
                pdf_content = io.BytesIO(response.content)
                break
            except requests.RequestException as e:
                logger.warning(f"PDF download attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    logger.error("All PDF download attempts failed; skipping ingestion.")

    if pdf_content is None:
        logger.warning("No PDF content available; skipping ingestion.")
        return False

    # Partition using unstructured if available, otherwise fallback to pypdf
    use_pypdf_only = ingest_mode in ("pypdf", "pypdf-only")
    elements = []
    if not use_pypdf_only:
        try:
            from unstructured.partition.auto import partition
            logger.info("Partitioning PDF with 'unstructured'...")
            elements = partition(file=pdf_content, content_type="application/pdf")
        except Exception as e:
            logger.warning(f"unstructured.partition failed: {e}")
            logger.info("Falling back to pypdf text extraction...")
            use_pypdf_only = True

    if use_pypdf_only:
        try:
            from pypdf import PdfReader
        except Exception:
            logger.error("pypdf required for pypdf-only mode but is not installed")
            return False
        pdf_content.seek(0)
        reader = PdfReader(pdf_content)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        elements = pages

    # Normalize to plain strings
    def _to_plain_text(obj) -> str:
        try:
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, bytes):
                try:
                    return obj.decode("utf-8", errors="ignore")
                except Exception:
                    return str(obj)
            if isinstance(obj, (list, tuple, set)):
                parts = [_to_plain_text(x) for x in obj]
                return " ".join(p for p in parts if p)
            if isinstance(obj, dict):
                for key in ("text", "content", "data", "children"):
                    if key in obj:
                        return _to_plain_text(obj[key])
                return " ".join(_to_plain_text(v) for v in obj.values())
            if hasattr(obj, "get_text") and callable(getattr(obj, "get_text")):
                try:
                    return _to_plain_text(obj.get_text())
                except Exception:
                    pass
            if hasattr(obj, "text"):
                try:
                    return _to_plain_text(getattr(obj, "text"))
                except Exception:
                    pass
            return str(obj)
        except Exception:
            return ""

    elements = [_to_plain_text(el) for el in elements]

    # Chunking (simple fallback implementation)
    try:
        from chonkie import SentenceChunker
        try:
            chonker = SentenceChunker(chunk_size=512, chunk_overlap=50)
        except TypeError:
            try:
                chonker = SentenceChunker(512, 50)
            except Exception:
                chonker = None
    except Exception:
        chonker = None

    def simple_chunk_docs(docs: List[str], chunk_size: int = 512, overlap: int = 50) -> List[dict]:
        chunks_out = []
        for doc_id, doc in enumerate(docs):
            words = doc.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks_out.append({
                    "id": f"doc{doc_id}_chunk{start}",
                    "text": chunk_text,
                    "meta": {"source_doc": doc_id}
                })
                start = end - overlap if end - overlap > start else end
        return chunks_out

    if chonker is not None:
        tried = False
        last_exc = None
        try_signatures = [("docs",), (None,)]
        for sig in try_signatures:
            try:
                if sig[0] == "docs":
                    chunks = chonker.chunk(docs=elements)
                else:
                    chunks = chonker.chunk(elements)
                tried = True
                break
            except Exception as e:
                last_exc = e
        if not tried:
            logger.warning(f"SentenceChunker.chunk failed ({last_exc}); falling back to simple chunker.")
            chunks = simple_chunk_docs(elements, chunk_size=512, overlap=50)
    else:
        chunks = simple_chunk_docs(elements, chunk_size=512, overlap=50)

    # Normalize and add to adapter/collection
    def _normalize_chunks(chunks_list):
        out = []
        for idx, c in enumerate(chunks_list):
            if isinstance(c, dict):
                text = c.get('text') or c.get('content') or c.get('chunk') or c.get('document') or ''
                id_ = c.get('id') or c.get('chunk_id') or f"chunk_{idx}"
                meta = c.get('meta') or c.get('metadata') or {}
                out.append({'id': id_, 'text': text, 'meta': meta})
            elif isinstance(c, str):
                out.append({'id': f"chunk_{idx}", 'text': c, 'meta': {}})
            else:
                out.append({'id': f"chunk_{idx}", 'text': str(c), 'meta': {}})
        return out

    normalized = _normalize_chunks(chunks)

    # Compute embeddings
    try:
        documents = [c['text'] for c in normalized]
        embeddings = embedder.get_embedding(documents)
    except Exception as e:
        logger.warning(f"Embedder bulk call failed: {e}; computing one-by-one.")
        embeddings = []
        for d in documents:
            embeddings.append(embedder.get_embedding(d))

    # Use adapter.add if available
    added = False
    try:
        if hasattr(adapter, 'add'):
            adapter.add(normalized)
            added = True
    except Exception as e:
        logger.warning(f"adapter.add failed: {e}")

    if not added:
        try:
            try:
                collection = chroma_client.get_collection(name=adapter.name)
            except Exception:
                collection = chroma_client.create_collection(name=adapter.name)
            ids = [c['id'] for c in normalized]
            documents = [c['text'] for c in normalized]
            metadatas = [c['meta'] for c in normalized]
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            added = True
        except Exception as e:
            logger.error(f"Failed to add chunks to chroma client collection: {e}")

    return added
