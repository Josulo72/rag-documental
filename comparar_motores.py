"""Compara los dos motores con las mismas preguntas sobre los documentos de ejemplo.

Uso:  python comparar_motores.py

Indexa los cuatro textos de ejemplo con LlamaIndex y con LangChain en una carpeta
temporal y enseña, para cada pregunta, qué documento queda primero y con qué similitud.
No llama a ningún modelo generativo.
"""
import sys
import tempfile
from pathlib import Path

import rag_engine as rag

QUESTIONS = [
    ("Compras", "devoluciones", "¿Cuántos días tengo para devolver una compra?"),
    ("Compras", "envios", "¿Cuánto tarda un envío nacional?"),
    ("Personal", "vacaciones", "¿Cuántos días de vacaciones hay?"),
    ("Personal", "gastos", "¿Cuándo se reembolsan los gastos de viaje?"),
    ("Compras", "devoluciones", "¿Qué plazo tengo para devolver un producto?"),
    ("Compras", "envios", "¿Cómo sigo mi pedido?"),
    ("Personal", "vacaciones", "¿Con cuánta antelación pido las vacaciones?"),
    ("Compras", None, "¿Cuál es la masa de Júpiter?"),
    ("Compras", None, "¿Quién ganó el mundial de 2010?"),
    ("Compras", None, "¿Cómo se hace una tortilla de patatas?"),
    ("Personal", None, "¿Cuál es el horario de la oficina?"),
    ("Personal", None, "¿Puedo teletrabajar los viernes?"),
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    bases = {}
    with tempfile.TemporaryDirectory() as folder:
        for name in rag.MOTORS:
            bases[name] = rag.build_base(rag.demo_pages(), name, Path(folder) / name)
    names = list(bases)
    print("Fragmentos: " + ", ".join(f"{name} {bases[name][1]['fragmentos']}" for name in names))
    print(f"{'pregunta':46} {'esperado':13} " + " ".join(f"{name:>22}" for name in names))
    for topic, expected, question in QUESTIONS:
        cells = []
        for name in names:
            stores, manifest = bases[name]
            best = rag.retrieve(stores, manifest, topic, question, 1)[0]
            found = best.metadata["archivo"].removeprefix("EJEMPLO_").removesuffix(".txt")
            cells.append(f"{found:>15} {best.score:.3f}")
        print(f"{question[:46]:46} {expected or '(ninguno)':13} " + " ".join(f"{cell:>22}" for cell in cells))
    for name in names:
        stores, manifest = bases[name]
        rows = rag.evaluate(stores, manifest, rag.DEMO_CASES, 0.85)
        hits = sum(row["cumple"] for row in rows if row["esperado"])
        rejected = sum(row["cumple"] for row in rows if not row["esperado"])
        print(f"{name}: evaluación de ejemplo con umbral 0,85 -> aciertos {hits}/4, rechazos {rejected}/1")


if __name__ == "__main__":
    main()
