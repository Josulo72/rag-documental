"""Interfaz del RAG documental. La lógica está en rag_engine.py."""
import json
import time

import streamlit as st

import rag_engine as rag

rag.load_env()

st.set_page_config(page_title="RAG documental | Jorge Rollón", page_icon="📚", layout="wide")
st.title("RAG documental")
st.caption("Jorge Ángel Rollón Rafael · Python · LlamaIndex o LangChain · Ollama o API")
st.info("Selecciona un tema, consulta tus documentos y comprueba las fuentes. La primera indexación descarga el modelo de búsqueda.")

with st.sidebar:
    st.header("Base documental")
    files = st.file_uploader("PDF con texto", type=["pdf"], accept_multiple_files=True)
    topics = [st.text_input(f"Tema de {file.name}", value="General", key=f"topic_{i}_{file.name}")
              for i, file in enumerate(files)]
    create = st.button("Crear base con estos PDF", disabled=not files)
    demo = st.button("Cargar documentos de ejemplo")
    st.caption("Crear o cargar una base sustituye la base activa. Los ejemplos son ficticios.")
    motors = {"LlamaIndex": "llamaindex", "LangChain": "langchain"}
    motor = motors[st.radio("Motor de búsqueda", list(motors), horizontal=True)]
    st.caption("Se aplica al crear la base. Los dos usan el mismo modelo de embeddings.")
    st.divider()
    providers = {"Ollama (local)": "ollama", "API compatible con OpenAI": "api"}
    provider = providers[st.radio("Quién redacta la respuesta", list(providers))]
    if provider == "api":
        model = st.text_input("Modelo de la API", rag.API_MODEL)
        st.caption("La clave se lee de OPENAI_API_KEY (archivo .env); no se escribe aquí. La pregunta y los fragmentos recuperados se envían a ese servicio.")
    else:
        model = st.text_input("Modelo instalado en Ollama", "qwen2.5:1.5b")
        st.caption("La búsqueda y la respuesta se hacen en este equipo.")
    threshold = st.slider("Similitud mínima", 0.0, 1.0, rag.DEFAULT_THRESHOLD, 0.01)
    st.caption("Umbral experimental. Debe calibrarse con tus documentos. No representa una probabilidad de acierto.")

if create or demo:
    try:
        with st.spinner("Preparando documentos e índice local..."):
            pages, warnings = (rag.demo_pages(), []) if demo else rag.pdf_pages(files, topics)
            bar = st.progress(0)
            st.session_state.base = rag.build_base(pages, motor, progress=bar.progress)
            bar.empty()
            st.session_state.pop("result", None)
            st.session_state.pop("evaluation", None)
        st.success("Base creada y guardada en la carpeta datos.")
        for warning in warnings:
            st.warning(warning)
    except Exception as error:
        st.error(f"No se pudo crear la base: {error}. La base anterior sigue activa.")

if "base" not in st.session_state and (rag.DATA / "manifest.json").exists():
    try:
        with st.spinner("Abriendo la base guardada..."):
            st.session_state.base = rag.restore_base()
    except Exception as error:
        st.error(f"No se pudo abrir la base: {error}")

if "base" not in st.session_state:
    st.write("Sube tus PDF o pulsa Cargar documentos de ejemplo para comenzar.")
    st.stop()

stores, manifest = st.session_state.base
a, b, c = st.columns(3)
a.metric("Temas", len(stores))
b.metric("Páginas con texto", manifest["paginas"])
c.metric("Fragmentos", manifest["fragmentos"])
st.caption(f"Base activa creada con {rag.get_motor(manifest['motor']).LABEL}.")
topic = st.selectbox("Buscar exclusivamente en este tema", sorted(stores))
with st.form("question"):
    question = st.text_area("Pregunta", max_chars=1500, placeholder="¿Qué plazo tengo para devolver un producto?")
    submitted = st.form_submit_button("Consultar")

if submitted:
    question = question.strip()
    if not question:
        st.warning("Escribe una pregunta.")
    else:
        start = time.perf_counter()
        try:
            with st.spinner("Buscando fuentes y preparando la respuesta..."):
                answer, cited, hits = rag.ask(stores, manifest, topic, question, threshold, model, provider)
            st.session_state.result = {
                "pregunta": question, "tema": topic, "respuesta": answer,
                "referencias_utilizadas": cited, "segundos": round(time.perf_counter() - start, 2),
                "fuentes": [{"id": i, **hit.metadata, "similitud": round(hit.score, 4),
                             "texto": hit.text} for i, hit in enumerate(hits, 1)]
            }
        except Exception as error:
            st.session_state.pop("result", None)
            st.error(f"No se pudo generar una respuesta: {error}")
            st.info("Con Ollama: comprueba que está abierto y que el modelo está instalado. Con API: revisa la clave del archivo .env y el nombre del modelo.")

result = st.session_state.get("result")
if result:
    st.subheader("Resultado de la última consulta")
    st.caption(f"Tema: {result['tema']} | Pregunta: {result['pregunta']}")
    st.write(result["respuesta"])
    st.caption(f"Tiempo total: {result['segundos']} s. Las fuentes permiten revisar la respuesta; no garantizan su corrección.")
    for source in result["fuentes"]:
        used = " · citada por el modelo" if source["id"] in result["referencias_utilizadas"] else ""
        with st.expander(f"[{source['id']}] {source['archivo']} · página {source['pagina']}{used}"):
            st.caption(f"Similitud: {source['similitud']}")
            st.text(source["texto"])
    st.download_button("Descargar consulta y evidencias", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="consulta_rag.json", mime="application/json")

with st.expander("Evaluar la recuperación", expanded=bool(st.session_state.get("evaluation"))):
    st.write("Carga un JSON con casos: tema, pregunta y archivo_esperado. Usa null para preguntas que deberían rechazarse. Si añades contiene, el fragmento recuperado tiene que incluir esa frase.")
    cases_file = st.file_uploader("Casos de evaluación", type=["json"])
    if st.button("Evaluar"):
        try:
            if cases_file:
                cases = json.loads(cases_file.getvalue())
            elif all(name in stores for name in ("Compras", "Personal")):
                cases = rag.DEMO_CASES
            else:
                raise ValueError("Carga la base de ejemplo o aporta tus propios casos.")
            st.session_state.evaluation = rag.evaluate(stores, manifest, cases, threshold)
        except Exception as error:
            st.session_state.pop("evaluation", None)
            st.error(f"Evaluación no completada: {error}")
    if st.session_state.get("evaluation"):
        rows = st.session_state.evaluation
        st.dataframe([{**row, "esperado": row["esperado"] or "(sin respuesta)"} for row in rows], use_container_width=True)
        positives = [row for row in rows if row["esperado"] is not None]
        negatives = [row for row in rows if row["esperado"] is None]
        if positives:
            st.write(f"Acierto de recuperación @3: {sum(row['cumple'] for row in positives)}/{len(positives)}")
        if negatives:
            st.write(f"Rechazo de preguntas sin respuesta: {sum(row['cumple'] for row in negatives)}/{len(negatives)}")
        st.caption("Estas pruebas evalúan recuperación y aislamiento. No evalúan la veracidad de las respuestas generadas.")
        st.download_button("Descargar evaluación", json.dumps(rows, ensure_ascii=False, indent=2),
                           "evaluacion_rag.json", "application/json")
