"""Pruebas de los dos motores de verdad, con el modelo de embeddings real.

Necesitan la instalación completa (requirements.txt) y la primera vez descargan el
modelo. Por eso van aparte de tests.py y no se ejecutan en GitHub Actions.

Uso:  python tests_motores.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rag_engine as rag

THRESHOLD = rag.DEFAULT_THRESHOLD


class Motors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.bases = {name: rag.build_base(rag.demo_pages(), name, Path(cls.folder.name) / name)
                     for name in rag.MOTORS}
    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
    def each(self):
        for name, (stores, manifest) in self.bases.items():
            with self.subTest(motor=name):
                yield name, stores, manifest
    def test_manifest_records_the_motor(self):
        for name, stores, manifest in self.each():
            self.assertEqual(manifest["motor"], name)
            self.assertEqual(sorted(stores), ["Compras", "Personal"])
            self.assertEqual(manifest["fragmentos"], 4)
    def test_example_evaluation(self):
        for name, stores, manifest in self.each():
            rows = rag.evaluate(stores, manifest, rag.DEMO_CASES, THRESHOLD)
            self.assertTrue(all(row["cumple"] for row in rows), rows)
            self.assertTrue(all(row["aislamiento"] for row in rows))
    def test_fragments_have_text_metadata_and_cosine_score(self):
        for name, stores, manifest in self.each():
            hits = rag.retrieve(stores, manifest, "Compras", "¿Cuánto tarda un envío nacional?", 2)
            self.assertEqual(len(hits), 2)
            self.assertEqual(hits[0].metadata["archivo"], "EJEMPLO_envios.txt")
            self.assertEqual(hits[0].metadata["pagina"], 1)
            self.assertIn("48 a 72 horas", hits[0].text)
            self.assertTrue(0.85 < hits[0].score <= 1.0)
            self.assertGreaterEqual(hits[0].score, hits[1].score)
    def test_saved_index_gives_the_same_results(self):
        for name, stores, manifest in self.each():
            restored, restored_manifest = rag.restore_base(Path(self.folder.name) / name)
            self.assertEqual(restored_manifest["motor"], name)
            before = rag.retrieve(stores, manifest, "Personal", "¿Cuántos días de vacaciones hay?", 2)
            after = rag.retrieve(restored, restored_manifest, "Personal", "¿Cuántos días de vacaciones hay?", 2)
            self.assertEqual([h.metadata["archivo"] for h in before], [h.metadata["archivo"] for h in after])
            self.assertAlmostEqual(before[0].score, after[0].score, places=4)
    def test_both_motors_agree_on_the_example_cases(self):
        for case in rag.DEMO_CASES:
            if case["archivo_esperado"]:
                best = {name: rag.retrieve(stores, manifest, case["tema"], case["pregunta"], 1)[0]
                        for name, (stores, manifest) in self.bases.items()}
                self.assertEqual({hit.metadata["archivo"] for hit in best.values()}, {case["archivo_esperado"]})
                scores = [hit.score for hit in best.values()]
                self.assertLess(max(scores) - min(scores), 0.03)


class LangChainAnswer(unittest.TestCase):
    """La cadena de LangChain, con un modelo de mentira que devuelve lo que le digamos."""
    hits = [rag.Fragment("uno", {"archivo": "a.pdf", "pagina": 1, "tema": "C"}, .9),
            rag.Fragment("dos", {"archivo": "b.pdf", "pagina": 2, "tema": "C"}, .88)]
    def answer(self, content):
        import motor_langchain
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        with mock.patch.object(motor_langchain, "chat_model", return_value=FakeListChatModel(responses=[content])):
            return motor_langchain.generate("pregunta", self.hits, "modelo", "api")
    def test_valid_answer(self):
        self.assertEqual(self.answer('{"answer": "vale", "sources": [2, 2]}'), ("vale", [2]))
    def test_empty_answer_means_no_answer(self):
        self.assertEqual(self.answer('{"answer": "", "sources": []}'), (rag.NO_ANSWER, []))
    def test_invented_source_is_rejected(self):
        with self.assertRaises(ValueError): self.answer('{"answer": "vale", "sources": [7]}')
    def test_text_that_is_not_json_is_rejected(self):
        with self.assertRaises(Exception): self.answer("no soy JSON")


if __name__ == "__main__":
    unittest.main(verbosity=2)
