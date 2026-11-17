"""Evalúa las respuestas redactadas, no solo la búsqueda, con las dos leyes del BOE.

Uso:  python evaluar_respuestas_boe.py MODELO [MOTOR UMBRAL]

Necesita la clave de la API en el archivo .env. Para cada motor indexa las dos leyes,
hace las 36 preguntas de ejemplos/casos_boe.json con el umbral por defecto y guarda
lo que contesta el modelo en docs/respuestas_boe.json.

La columna "auto" es solo una ayuda para revisar: mira si la respuesta contiene el dato
esperado. Decidir si una respuesta está bien hay que hacerlo leyéndola.
"""
import json
import re
import sys
import time
from pathlib import Path

import rag_engine as rag
from evaluar_boe import LocalFile, download

ROOT = Path(__file__).resolve().parent
BASES = ROOT / "datos_boe" / "indices"

# Dato que debería aparecer en una respuesta correcta, como expresión regular.
KEYS = {
    "¿Qué colores tiene la bandera de España?": r"roja.*amarilla",
    "¿Cuál es la capital del Estado?": r"madrid",
    "¿Cuál es la lengua oficial del Estado?": r"castellano",
    "¿A qué edad se alcanza la mayoría de edad?": r"dieciocho|18",
    "¿Cuánto puede durar como máximo una detención preventiva?": r"setenta y dos|72",
    "¿Es obligatoria la enseñanza básica?": r"obligatoria",
    "¿Cuántos diputados componen el Congreso?": r"300.*400|trescientos.*cuatrocientos",
    "¿Cuántos senadores se eligen en cada provincia?": r"cuatro|\b4\b",
    "¿Quién puede proponer una moción de censura?": r"décima parte",
    "¿Cuántos miembros tiene el Tribunal Constitucional?": r"\b12\b|doce",
    "¿Cuántas firmas hacen falta para una iniciativa legislativa popular?": r"500[. ]?000|quinientas mil",
    "¿Está permitida la pena de muerte?": r"abolida",
    "¿Tiene el Rey responsabilidad por sus actos?": r"inviolable|no está sujet",
    "¿Quién declara el estado de alarma?": r"gobierno",
    "¿Hay una religión oficial del Estado?": r"ninguna confesión|no hay|no existe|no tiene",
    "¿A partir de qué edad se puede trabajar?": r"dieciséis|16",
    "¿Cuánto puede durar el periodo de prueba?": r"seis meses",
    "¿Cuál es la jornada máxima semanal?": r"cuarenta|40",
    "¿Cuántas horas de descanso debe haber entre una jornada y la siguiente?": r"doce|\b12\b",
    "¿Cuánto dura el descanso semanal mínimo?": r"día y medio",
    "¿Cuántos días de permiso corresponden por matrimonio?": r"quince|\b15\b",
    "¿Cuántos días de vacaciones tengo al año como mínimo?": r"treinta|\b30\b",
    "¿Cuántas horas extraordinarias se pueden hacer al año?": r"ochenta|\b80\b",
    "¿Qué horario se considera trabajo nocturno?": r"diez de la noche|22",
    "¿Qué indemnización corresponde por un despido improcedente?": r"treinta y tres|\b33\b",
    "¿Cuántas pagas extraordinarias hay al año?": r"\bdos\b|\b2\b",
    "¿Qué interés se aplica si la empresa se retrasa en pagar el salario?": r"diez por ciento|10 ?%",
    "¿Cuánto descanso hay dentro de la jornada si dura más de seis horas?": r"quince minutos|15 minutos",
    "¿A partir de cuántos trabajadores hay comité de empresa?": r"cincuenta|\b50\b",
    "¿Cuál es el tope de la indemnización por despido improcedente?": r"veinticuatro|\b24\b",
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model = sys.argv[1]
    # Con MOTOR y UMBRAL se prueba un solo motor con otro umbral y se guarda en un archivo aparte.
    motors = [sys.argv[2]] if len(sys.argv) > 3 else list(rag.MOTORS)
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else rag.DEFAULT_THRESHOLD
    output = ROOT / "docs" / (f"respuestas_boe_{motors[0]}_{threshold:.2f}.json" if len(sys.argv) > 3 else "respuestas_boe.json")
    rag.load_env()
    files = download()
    cases = json.loads((ROOT / "ejemplos" / "casos_boe.json").read_text(encoding="utf-8"))
    report = {"modelo": model, "umbral": threshold, "motores": {}}
    for name in motors:
        folder = BASES / name
        if (folder / "manifest.json").exists():
            stores, manifest = rag.restore_base(folder)
        else:
            pages, _ = rag.pdf_pages([LocalFile(path) for path in files.values()], list(files))
            stores, manifest = rag.build_base(pages, name, folder)
        rows = []
        for case in cases:
            start = time.perf_counter()
            row = {"tema": case["tema"], "pregunta": case["pregunta"], "tiene_respuesta": bool(case["archivo_esperado"])}
            try:
                answer, cited, hits = rag.ask(stores, manifest, case["tema"], case["pregunta"],
                                              threshold, model, "api")
                row.update(respuesta=answer, citas=cited, fragmentos=len(hits),
                           paginas=[hit.metadata["pagina"] for hit in hits])
                if answer == rag.NO_ANSWER:
                    row["auto"] = "sin respuesta"
                elif not case["archivo_esperado"]:
                    row["auto"] = "contesta una trampa"
                else:
                    row["auto"] = "dato presente" if re.search(KEYS[case["pregunta"]], answer.lower()) else "revisar"
                    row["frase_en_lo_enviado"] = any(rag.normalise(case["contiene"]) in rag.normalise(hit.text) for hit in hits)
            except Exception as error:
                row.update(respuesta=None, auto="error", error=f"{type(error).__name__}: {error}"[:300])
            row["segundos"] = round(time.perf_counter() - start, 1)
            rows.append(row)
            print(f"[{name}] {row['auto']:20} {row['segundos']:5}s | {case['pregunta']}\n      -> {str(row.get('respuesta') or row.get('error'))[:300]}", flush=True)
        report["motores"][name] = rows
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in report["motores"].items():
        counts = {}
        for row in rows:
            counts[row["auto"]] = counts.get(row["auto"], 0) + 1
        print("==", name, counts)


if __name__ == "__main__":
    main()
