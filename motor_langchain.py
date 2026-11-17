"""Motor de búsqueda con LangChain: trocea, indexa, guarda, recupera y redacta con una cadena LCEL."""
import json
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_engine as rag
from rag_engine import MODEL, Fragment

LABEL = "LangChain"
FILE = "vectores.json"
PROMPT = ChatPromptTemplate.from_messages([("system", "{sistema}"), ("human", "{entrada}")])


@lru_cache(maxsize=1)
def embedding():
    # El modelo E5 espera un prefijo distinto para las preguntas y para los textos.
    return HuggingFaceEmbeddings(
        model_name=MODEL, model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
    )


@lru_cache(maxsize=1)
def splitter():
    # Mismo tamaño que en el otro motor (180 tokens, 30 de solapamiento), contado con el
    # tokenizador del propio modelo de embeddings.
    from transformers import AutoTokenizer
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(MODEL), chunk_size=180, chunk_overlap=30)


def build(pages):
    """Indexa las páginas de un tema. Devuelve el almacén de vectores y el número de fragmentos."""
    documents = [Document(page_content=page["text"], metadata=page["metadata"]) for page in pages]
    chunks = splitter().split_documents(documents)
    store = InMemoryVectorStore(embedding())
    store.add_documents(chunks)
    return store, len(chunks)


def save(store, folder):
    store.dump(str(folder / FILE))


def load(folder):
    return InMemoryVectorStore.load(str(folder / FILE), embedding())


def search(store, question, limit):
    # InMemoryVectorStore puntúa con similitud coseno, igual que el otro motor.
    return [Fragment(document.page_content, dict(document.metadata), float(score))
            for document, score in store.similarity_search_with_score(question, k=limit)]


def chat_model(model, provider):
    from langchain_openai import ChatOpenAI
    if provider == "api":
        key, base = rag.api_settings()
    else:
        key, base = "ollama", rag.OLLAMA + "/v1"  # Ollama también habla el protocolo de OpenAI
    return ChatOpenAI(model=model, api_key=key, base_url=base, temperature=0, max_tokens=500,
                      timeout=120, max_retries=0,
                      model_kwargs={"response_format": {"type": "json_object"}})


def generate(question, hits, model, provider="ollama"):
    """Cadena prompt | modelo | parser. La respuesta pasa por la misma validación de citas."""
    from openai import APIConnectionError
    chain = PROMPT | chat_model(model, provider) | JsonOutputParser()
    try:
        payload = chain.invoke({"sistema": rag.SYSTEM, "entrada": json.dumps(
            {"pregunta": question, "fragmentos": rag.fragments_payload(hits)}, ensure_ascii=False)})
    except APIConnectionError:
        if provider == "api":
            raise RuntimeError("no se puede conectar con el servicio de la API. Revisa la dirección en el archivo .env.") from None
        raise RuntimeError("no hay respuesta de Ollama en este equipo. Ábrelo y vuelve a intentarlo, o elige la API en la barra lateral.") from None
    return rag.validate_answer(payload, len(hits))
