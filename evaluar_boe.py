"""Evalúa la búsqueda de los dos motores con dos leyes reales del BOE.

Uso:  python evaluar_boe.py

Descarga, si no están ya, los textos consolidados de la Constitución Española y del
Estatuto de los Trabajadores a la carpeta datos_boe, los indexa con cada motor y les
pasa los casos de ejemplos/casos_boe.json. Cada caso con respuesta trae la frase
literal de la ley que la contiene: un caso se da por bueno si esa frase está en alguno
de los fragmentos recuperados. No llama a ningún modelo generativo.

Guarda el detalle en docs/evaluacion_boe.json.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import requests

import rag_engine as rag

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "datos_boe"
SOURCES = {
    "Constitución": "https://www.boe.es/buscar/pdf/1978/BOE-A-1978-31229-consolidado.pdf",
    "Estatuto de los Trabajadores": "https://www.boe.es/buscar/pdf/2015/BOE-A-2015-11430-consolidado.pdf",
}
THRESHOLDS = [0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]


class LocalFile:
    def __init__(self, path):
        self.name, self.path = path.name, path
    def getvalue(self):
        return self.path.read_bytes()


def download():
    FOLDER.mkdir(exist_ok=True)
    files = {}
    for topic, url in SOURCES.items():
        path = FOLDER / url.rsplit("/", 1)[1]
        if not path.exists():
            print("Descargando", url)
            response = requests.get(url, timeout=120, headers={"User-Agent": "rag-documental"})
            response.raise_for_status()
            path.write_bytes(response.content)
        files[topic] = path
    return files


def rank_of(hits, phrase):
    """Posición (desde 1) del primer fragmento que contiene la frase, o None."""
    wanted = rag.normalise(phrase)
    return next((i for i, hit in enumerate(hits, 1) if wanted in rag.normalise(hit.text)), None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    files = download()
    cases = json.loads((ROOT / "ejemplos" / "casos_boe.json").read_text(encoding="utf-8"))
    pages, warnings = rag.pdf_pages([LocalFile(path) for path in files.values()], list(files))
    text = {topic: rag.normalise(" ".join(p["text"] for p in pages if p["metadata"]["tema"] == topic)) for topic in files}
    positives = [case for case in cases if case["archivo_esperado"]]
    traps = [case for case in cases if not case["archivo_esperado"]]
    for case in positives:  # la frase esperada tiene que estar de verdad en la ley
        if rag.normalise(case["contiene"]) not in text[case["tema"]]:
            sys.exit(f"La frase {case['contiene']!r} no está en el PDF de {case['tema']}.")
    print(f"Páginas con texto: {len(pages)} | avisos: {len(warnings)} | casos: {len(positives)} con respuesta, {len(traps)} trampa")

    report = {"paginas": len(pages), "casos_con_respuesta": len(positives), "casos_trampa": len(traps), "motores": {}}
    with tempfile.TemporaryDirectory() as folder:
        for name in rag.MOTORS:
            start = time.perf_counter()
            stores, manifest = rag.build_base(pages, name, Path(folder) / name)
            seconds = round(time.perf_counter() - start, 1)
            rows = []
            for case in cases:
                hits = rag.retrieve(stores, manifest, case["tema"], case["pregunta"], 5)
                row = {"tema": case["tema"], "pregunta": case["pregunta"], "contiene": case.get("contiene"),
                       "similitudes": [round(hit.score, 4) for hit in hits]}
                if case["archivo_esperado"]:
                    row["posicion"] = rank_of(hits, case["contiene"])
                    row["similitud_del_acierto"] = round(hits[row["posicion"] - 1].score, 4) if row["posicion"] else None
                rows.append(row)
            good = [row for row in rows if row["contiene"]]
            bad = [row for row in rows if not row["contiene"]]
            top = lambda k: sum(1 for row in good if row["posicion"] and row["posicion"] <= k)
            sweep = []
            for threshold in THRESHOLDS:
                sweep.append({
                    "umbral": threshold,
                    "aciertos_en_3": sum(1 for row in good if row["posicion"] and row["posicion"] <= 3
                                         and row["similitud_del_acierto"] >= threshold),
                    "trampas_rechazadas": sum(1 for row in bad if row["similitudes"][0] < threshold),
                })
            report["motores"][name] = {
                "fragmentos": manifest["fragmentos"], "segundos_indexando": seconds,
                "acierto_en_1": top(1), "acierto_en_3": top(3), "acierto_en_5": top(5),
                "por_umbral": sweep, "casos": rows,
            }
            print(f"\n== {name}: {manifest['fragmentos']} fragmentos, {seconds} s indexando")
            print(f"   frase correcta en el 1.º: {top(1)}/{len(good)} | entre los 3 primeros: {top(3)}/{len(good)} | entre los 5: {top(5)}/{len(good)}")
            print("   similitud del fragmento correcto: de "
                  f"{min(r['similitud_del_acierto'] for r in good if r['posicion']):.3f} a "
                  f"{max(r['similitud_del_acierto'] for r in good if r['posicion']):.3f}")
            print(f"   mejor similitud de las preguntas trampa: de {min(r['similitudes'][0] for r in bad):.3f} a {max(r['similitudes'][0] for r in bad):.3f}")
            for line in sweep:
                print(f"   umbral {line['umbral']:.2f}: aciertos entre los 3 primeros {line['aciertos_en_3']}/{len(good)}, trampas rechazadas {line['trampas_rechazadas']}/{len(bad)}")
            missed = [row["pregunta"] for row in good if not row["posicion"] or row["posicion"] > 3]
            if missed:
                print("   no aparece entre los 3 primeros:", " | ".join(missed))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "evaluacion_boe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nDetalle guardado en docs/evaluacion_boe.json")


if __name__ == "__main__":
    main()
