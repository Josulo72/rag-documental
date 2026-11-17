"""Pruebas de la lógica del RAG. Solo necesitan pypdf y requests: no cargan modelos ni llaman a ninguno."""
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import requests
from pypdf import PdfWriter

import rag_engine as rag
from rag_engine import Fragment


def hit(topic, score, name="a.pdf", text="texto"):
    return Fragment(text, {"tema": topic, "archivo": name, "pagina": 1}, score)


class Upload:
    def __init__(self, raw, name):
        self.raw, self.name = raw, name
    def getvalue(self):
        return self.raw


def pdf_bytes(password=None):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    if password:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def fake_motor():
    """Motor de mentira: la similitud es la proporción de palabras de la pregunta que están en el texto."""
    motor = types.SimpleNamespace(saved=[], loaded=[])
    def build(pages):
        return list(pages), len(pages)
    def save(store, folder):
        (Path(folder) / "paginas.json").write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        motor.saved.append(Path(folder))
    def load(folder):
        motor.loaded.append(Path(folder))
        return json.loads((Path(folder) / "paginas.json").read_text(encoding="utf-8"))
    def search(store, question, limit):
        words = set(question.lower().strip("¿?").split())
        scored = [Fragment(p["text"], p["metadata"], len(words & set(p["text"].lower().split())) / len(words))
                  for p in store]
        return sorted(scored, key=lambda f: f.score, reverse=True)[:limit]
    motor.build, motor.save, motor.load, motor.search = build, save, load, search
    return motor


class TopicsAndHits(unittest.TestCase):
    def test_topic_is_normalised(self):
        self.assertEqual(rag.clean_topic("  Compras  y   ventas "), "Compras y ventas")
    def test_empty_and_long_topics_rejected(self):
        for value in (" ", "a" * 81):
            with self.assertRaises(ValueError): rag.clean_topic(value)
    def test_threshold_topic_and_bad_scores(self):
        values = [hit("Compras", .9), hit("Personal", 1), hit("Compras", float("nan")), hit("Compras", None),
                  hit("Compras", .6), hit("Compras", float("inf"))]
        out = rag.accepted_hits(values, "Compras", .7)
        self.assertEqual([h.score for h in out], [.9])
    def test_threshold_is_inclusive(self):
        self.assertEqual(len(rag.accepted_hits([hit("Compras", .7)], "Compras", .7)), 1)
    def test_other_topic_returns_nothing(self):
        self.assertEqual(rag.accepted_hits([hit("Compras", .9)], "Ausente", 0), [])
    def test_hits_sorted_and_limited(self):
        out = rag.accepted_hits([hit("C", s) for s in (.5, .9, .7, .8, .6)], "C", 0, 3)
        self.assertEqual([h.score for h in out], [.9, .8, .7])


class Answers(unittest.TestCase):
    def test_sources_deduplicated_in_order(self):
        self.assertEqual(rag.validate_answer({"answer": "texto", "sources": [2, 1, 2]}, 2), ("texto", [2, 1]))
    def test_empty_answer_means_no_answer(self):
        self.assertEqual(rag.validate_answer({"answer": " ", "sources": []}, 2), (rag.NO_ANSWER, []))
    def test_bad_payloads_rejected(self):
        bad = [None, [], {"answer": "texto", "sources": []}, {"answer": "texto", "sources": [0]},
               {"answer": "texto", "sources": [3]}, {"answer": "texto", "sources": [True]},
               {"answer": 2, "sources": [1]}, {"answer": "texto"}]
        for payload in bad:
            with self.assertRaises(ValueError): rag.validate_answer(payload, 2)


class Documents(unittest.TestCase):
    def test_demo_pages(self):
        pages = rag.demo_pages()
        self.assertEqual(len(pages), 4)
        self.assertEqual({p["metadata"]["tema"] for p in pages}, {"Compras", "Personal"})
    def test_demo_cases_point_to_demo_files(self):
        names = {p["metadata"]["archivo"] for p in rag.demo_pages()}
        for case in rag.DEMO_CASES:
            self.assertTrue(case["archivo_esperado"] is None or case["archivo_esperado"] in names)
    def test_pdf_without_text_rejected(self):
        with self.assertRaises(ValueError): rag.pdf_pages([Upload(pdf_bytes(), "escaneado.pdf")], ["Compras"])
    def test_protected_pdf_rejected(self):
        with self.assertRaises(ValueError): rag.pdf_pages([Upload(pdf_bytes("clave"), "protegido.pdf")], ["Compras"])
    def test_big_pdf_rejected(self):
        with self.assertRaises(ValueError): rag.pdf_pages([Upload(b"0" * (10 * 1024 * 1024 + 1), "grande.pdf")], ["Compras"])


class Base(unittest.TestCase):
    def setUp(self):
        self.motor = fake_motor()
        self.patch = mock.patch.dict(rag.MOTORS, {"falso": self.motor})
        self.patch.start()
        self.folder = tempfile.TemporaryDirectory()
        self.data = Path(self.folder.name)
    def tearDown(self):
        self.patch.stop()
        self.folder.cleanup()
    def build(self):
        return rag.build_base(rag.demo_pages(), "falso", self.data)
    def test_one_index_per_topic(self):
        steps = []
        stores, manifest = rag.build_base(rag.demo_pages(), "falso", self.data, steps.append)
        self.assertEqual(sorted(stores), ["Compras", "Personal"])
        self.assertEqual((manifest["motor"], manifest["paginas"], manifest["fragmentos"]), ("falso", 4, 4))
        self.assertEqual(steps, [.5, 1.0])
        self.assertEqual(len(self.motor.saved), 2)
    def test_restore_uses_the_motor_that_built_it(self):
        self.build()
        stores, manifest = rag.restore_base(self.data)
        self.assertEqual(manifest["motor"], "falso")
        self.assertEqual(len(self.motor.loaded), 2)
        self.assertEqual(len(stores["Compras"]), 2)
    def test_old_manifest_without_motor_uses_default(self):
        self.build()
        path = self.data / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        del manifest["motor"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with mock.patch.dict(rag.MOTORS, {rag.DEFAULT_MOTOR: self.motor}):
            self.assertEqual(rag.restore_base(self.data)[1]["motor"], rag.DEFAULT_MOTOR)
    def test_unknown_motor_rejected(self):
        with self.assertRaises(ValueError): rag.build_base(rag.demo_pages(), "no-existe", self.data)
    def test_changed_model_rejected(self):
        self.build()
        with mock.patch.object(rag, "MODEL", "otro"):
            with self.assertRaises(ValueError): rag.restore_base(self.data)
    def test_index_path_cannot_leave_data_folder(self):
        self.build()
        path = self.data / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["temas"]["Compras"] = "../fuera"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError): rag.restore_base(self.data)
    def test_evaluation_rows(self):
        stores, manifest = self.build()
        cases = [{"tema": "Compras", "pregunta": "los envíos nacionales tardan", "archivo_esperado": "EJEMPLO_envios.txt"},
                 {"tema": "Compras", "pregunta": "masa de júpiter", "archivo_esperado": None},
                 {"tema": "Personal", "pregunta": "los envíos nacionales tardan", "archivo_esperado": "EJEMPLO_envios.txt"}]
        rows = rag.evaluate(stores, manifest, cases, .5)
        self.assertEqual([row["cumple"] for row in rows], [True, True, False])
        self.assertTrue(all(row["aislamiento"] for row in rows))
    def test_evaluation_rejects_bad_cases(self):
        stores, manifest = self.build()
        for cases in ([], {"tema": "Compras"}, [{"tema": "Compras"}],
                      [{"tema": "Ausente", "pregunta": "x", "archivo_esperado": None}],
                      [{"tema": "Compras", "pregunta": " ", "archivo_esperado": None}],
                      [{"tema": "Compras", "pregunta": "x", "archivo_esperado": 3}]):
            with self.assertRaises(ValueError): rag.evaluate(stores, manifest, cases, .5)
    def test_ask_below_threshold_does_not_call_model(self):
        stores, manifest = self.build()
        with mock.patch("rag_engine.requests.post") as post:
            out = rag.ask(stores, manifest, "Compras", "masa de júpiter", .5, "modelo", "api")
        self.assertEqual(out, (rag.NO_ANSWER, [], []))
        post.assert_not_called()
    def test_ask_sends_only_accepted_fragments(self):
        stores, manifest = self.build()
        reply = mock.Mock()
        reply.json.return_value = {"choices": [{"message": {"content": '{"answer":"De 48 a 72 horas.","sources":[1]}'}}]}
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "clave-de-prueba", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1/"}), \
             mock.patch("rag_engine.requests.post", return_value=reply) as post:
            answer, cited, hits = rag.ask(stores, manifest, "Compras", "los envíos nacionales tardan", .5, "modelo", "api")
        self.assertEqual((answer, cited), ("De 48 a 72 horas.", [1]))
        self.assertEqual([h.metadata["archivo"] for h in hits], ["EJEMPLO_envios.txt"])
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:9/v1/chat/completions")
        sent = json.loads(post.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertEqual([f["archivo"] for f in sent["fragmentos"]], ["EJEMPLO_envios.txt"])


class Generation(unittest.TestCase):
    hits = [hit("Compras", .9, "a.pdf", "uno"), hit("Compras", .8, "b.pdf", "dos")]
    def test_ollama_reply_is_validated(self):
        reply = mock.Mock()
        reply.json.return_value = {"message": {"content": '{"answer":"vale","sources":[2]}'}}
        with mock.patch("rag_engine.requests.post", return_value=reply) as post:
            self.assertEqual(rag.generate("pregunta", self.hits, "modelo"), ("vale", [2]))
        self.assertTrue(post.call_args.args[0].endswith("/api/chat"))
    def test_invented_source_is_rejected(self):
        reply = mock.Mock()
        reply.json.return_value = {"message": {"content": '{"answer":"vale","sources":[7]}'}}
        with mock.patch("rag_engine.requests.post", return_value=reply):
            with self.assertRaises(ValueError): rag.generate("pregunta", self.hits, "modelo")
    def test_ollama_closed_gives_clear_error(self):
        with mock.patch("rag_engine.requests.post", side_effect=requests.ConnectionError("rechazada")):
            with self.assertRaises(RuntimeError) as caught: rag.generate("pregunta", self.hits, "modelo")
        self.assertIn("no hay respuesta de Ollama", str(caught.exception))
    def test_api_without_key_does_not_call(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}), mock.patch("rag_engine.requests.post") as post:
            with self.assertRaises(ValueError): rag.generate("pregunta", self.hits, "modelo", "api")
            post.assert_not_called()


class Settings(unittest.TestCase):
    def test_env_file_does_not_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text('# nota\nPRUEBA_NUEVA="uno"\nPRUEBA_PUESTA=dos\nlinea suelta\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {"PRUEBA_PUESTA": "original"}):
                os.environ.pop("PRUEBA_NUEVA", None)
                rag.load_env(path)
                self.assertEqual(os.environ["PRUEBA_NUEVA"], "uno")
                self.assertEqual(os.environ["PRUEBA_PUESTA"], "original")
            rag.load_env(Path(folder) / "no-existe")
    def test_api_address_default_and_custom(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "clave-de-prueba", "OPENAI_BASE_URL": ""}):
            self.assertEqual(rag.api_settings(), ("clave-de-prueba", rag.API_URL))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "clave-de-prueba", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1/"}):
            self.assertEqual(rag.api_settings()[1], "http://127.0.0.1:9/v1")
    def test_engine_does_not_import_heavy_libraries(self):
        import sys
        self.assertFalse({"streamlit", "llama_index", "langchain_core", "torch"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
