"""Motor de búsqueda con LlamaIndex: trocea, indexa, guarda y recupera."""
from functools import lru_cache

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_engine import MODEL, Fragment

LABEL = "LlamaIndex"


@lru_cache(maxsize=1)
def embedding():
    # El modelo E5 espera un prefijo distinto para las preguntas y para los textos.
    return HuggingFaceEmbedding(
        model_name=MODEL, device="cpu", max_length=512,
        query_instruction="query: ", text_instruction="passage: ",
        normalize=True,
    )


def build(pages):
    """Indexa las páginas de un tema. Devuelve el índice y el número de fragmentos."""
    documents = [Document(
        text=page["text"], metadata=page["metadata"],
        excluded_embed_metadata_keys=["pagina", "documento"],
        excluded_llm_metadata_keys=["documento"],
    ) for page in pages]
    nodes = SentenceSplitter(chunk_size=180, chunk_overlap=30).get_nodes_from_documents(documents)
    return VectorStoreIndex(nodes, embed_model=embedding()), len(nodes)


def save(store, folder):
    store.storage_context.persist(persist_dir=str(folder))


def load(folder):
    context = StorageContext.from_defaults(persist_dir=str(folder))
    return load_index_from_storage(context, embed_model=embedding())


def search(store, question, limit):
    hits = store.as_retriever(similarity_top_k=limit).retrieve(question)
    return [Fragment(hit.node.text, dict(hit.node.metadata), hit.score) for hit in hits]
