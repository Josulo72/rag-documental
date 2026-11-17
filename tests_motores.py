"""Pruebas de los dos motores de verdad, con el modelo de embeddings real.

Necesitan la instalación completa (requirements.txt) y la primera vez descargan el
modelo. Por eso van aparte de tests.py y no se ejecutan en GitHub Actions.

Uso:  python tests_motores.py
"""
import tempfile
import unittest
from pathlib import Path

import rag_engine as rag

THRESHOLD = 0.85


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
