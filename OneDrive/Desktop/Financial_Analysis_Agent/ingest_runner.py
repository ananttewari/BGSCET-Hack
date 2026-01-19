import os
import logging
from textwrap import dedent

# Lightweight embedding class copied from my_os to avoid importing the whole module
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.dimensions = 384
        logging.getLogger(__name__).info("Embedding model initialized")

    def get_embedding(self, text):
        if isinstance(text, str):
            return self.model.encode(text).tolist()
        return self.model.encode(text).tolist()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    from vector_db import setup_vector_db, ingest_pdf_to_collection

    pdf_url = os.getenv('PDF_URL', 'https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2024.pdf')
    ingest_mode = os.getenv('PDF_INGEST_MODE', 'pypdf')
    pdf_local = os.getenv('PDF_LOCAL_PATH', None)
    skip_ingest = os.getenv('SKIP_INGEST', 'false').lower() in ('1','true','yes')

    logger.info('Initializing embedder and vector DB...')
    embedder = EmbeddingModel()
    client, adapter, recreated, size = setup_vector_db(name='apple_docs', path='./chroma_db', embedder=embedder, logger=logger)
    logger.info(f'Collection size: {size}')

    FORCE_INGEST = os.getenv('FORCE_INGEST', 'false').lower() in ('1','true','yes')

    # Auto-skip ingestion when collection already has items unless forced or recreated
    if size and isinstance(size, int) and size > 0 and not FORCE_INGEST and not recreated:
        logger.info(f"Collection 'apple_docs' already has {size} items; skipping ingest (set FORCE_INGEST=1 to override).")
        return

    if skip_ingest:
        logger.info('SKIP_INGEST is true; not running ingestion.')
        return

    logger.info('Running ingestion...')
    ok = ingest_pdf_to_collection(
        pdf_url=pdf_url,
        adapter=adapter,
        embedder=embedder,
        chroma_client=client,
        pdf_local_path=pdf_local,
        ingest_mode=ingest_mode,
        skip_ingest=False,
        logger=logger,
    )

    logger.info('Ingestion result: %s', ok)


if __name__ == '__main__':
    main()
