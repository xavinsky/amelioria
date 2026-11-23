import unittest

import torch

from tests.outils import petit_modele
from commun import modeles


class Reponses(unittest.TestCase):
    def test_couper_reponse_s_arrete_a_la_question_suivante_et_a_la_ligne_answer(self):
        t = "2 + 2 = 4\nAnswer: 4\nQuestion: What is 3+3?\nAnswer: 6"
        self.assertEqual(modeles.couper_reponse(t), "2 + 2 = 4\nAnswer: 4")

    def test_couper_reponse_sans_answer_garde_le_texte(self):
        self.assertEqual(modeles.couper_reponse("pas de réponse"), "pas de réponse")


class DistillationTopK(unittest.TestCase):
    def setUp(self):
        import importlib.util, sys

        from tests.outils import RACINE

        spec = importlib.util.spec_from_file_location("cicatrisation_16", RACINE / "16-elagage" / "cicatrisation.py")
        self.cic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cic)

    def lot(self, model, k=8):
        ids = torch.randint(0, 64, (2, 12))
        with torch.no_grad():
            haut = torch.log_softmax(model(input_ids=ids).logits.float(), -1).topk(k, -1)
        return {"input_ids": ids, "top_idx": haut.indices.int(), "top_lp": haut.values.half()}

    def test_perte_nulle_quand_l_eleve_est_l_enseignant(self):
        model = petit_modele()
        perte = self.cic.TrainerTopK.compute_loss(None, model, self.lot(model))
        self.assertAlmostEqual(perte.item(), 0.0, places=3)

    def test_perte_positive_quand_l_eleve_differe(self):
        enseignant, eleve = petit_modele(graine=0), petit_modele(graine=1)
        perte = self.cic.TrainerTopK.compute_loss(None, eleve, self.lot(enseignant))
        self.assertGreater(perte.item(), 1e-4)   # petits modèles aléatoires : sorties presque uniformes, écart faible mais non nul

    def test_precalcul_ecrit_puis_relit_le_meme_contenu(self):
        import tempfile, pathlib

        from datasets import Dataset

        model = petit_modele()
        blocs = Dataset.from_dict({"input_ids": torch.randint(0, 64, (5, 12)).tolist()})
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "e.pt"
            a = self.cic.precalculer(model, blocs, 8, f, batch=2)
            b = self.cic.precalculer(None, blocs, 8, f)
        self.assertEqual(len(a), 5)
        self.assertTrue(torch.equal(a[3]["top_idx"], b[3]["top_idx"]))


if __name__ == "__main__":
    unittest.main()
