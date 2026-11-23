import unittest

import torch

from tests.outils import petit_modele
import elagage


class Masques(unittest.TestCase):
    def test_2_4_garde_deux_poids_par_groupe_de_quatre_les_plus_forts(self):
        score = torch.tensor([[4.0, 1.0, 3.0, 2.0, 1.0, 5.0, 2.0, 6.0]])
        m = elagage.masque(score, 0.5, "2:4")
        self.assertEqual(m.tolist(), [[True, False, True, False, False, True, False, True]])

    def test_sparsite_par_ligne(self):
        score = torch.rand(6, 40)
        m = elagage.masque(score, 0.5, None)
        self.assertTrue(torch.all(m.sum(1) == 20))

    def test_sparsite_nulle_garde_tout(self):
        self.assertTrue(elagage.masque(torch.rand(3, 8), 0.0, None).all())


class ModeleAleatoire(unittest.TestCase):
    def test_magnitude_atteint_la_sparsite_demandee(self):
        model = petit_modele()
        masques = elagage.elaguer_non_structure(model, "magnitude", 0.5)
        self.assertAlmostEqual(elagage.sparsite_reelle(model), 0.5, delta=0.02)
        self.assertEqual(len(masques), 4 * 7)

    def test_wanda_depend_des_normes_d_entree(self):
        a, b = petit_modele(), petit_modele()
        nom = next(n for n, _ in elagage.lineaires(a))
        normes_a = {n: torch.ones(m.in_features) for n, m in elagage.lineaires(a)}
        normes_b = {n: torch.arange(1, m.in_features + 1).float() for n, m in elagage.lineaires(b)}
        ma = elagage.elaguer_non_structure(a, "wanda", 0.5, normes_a)
        mb = elagage.elaguer_non_structure(b, "wanda", 0.5, normes_b)
        self.assertFalse(torch.equal(ma[nom], mb[nom]))

    def test_les_couches_lineaires_sont_les_memes_avec_un_adaptateur_lora(self):
        from peft import LoraConfig, get_peft_model

        model = petit_modele()
        avant = [n for n, _ in elagage.lineaires(model)]
        peft = get_peft_model(model, LoraConfig(r=4, target_modules="all-linear"))
        self.assertEqual(len(list(elagage.lineaires(peft))), len(avant))
        self.assertEqual(elagage.sparsite_reelle(peft), 0.0)

    def test_supprimer_couches(self):
        model = petit_modele(4)
        elagage.supprimer_couches(model, [1, 2])
        self.assertEqual(len(model.model.layers), 2)
        self.assertEqual(model.config.num_hidden_layers, 2)
        model(input_ids=torch.tensor([[1, 2, 3]]))

    def test_reduire_largeur(self):
        model = petit_modele()
        scores = {i: torch.rand(64) for i in range(4)}
        elagage.reduire_largeur(model, scores, 0.25)
        self.assertEqual(model.model.layers[0].mlp.gate_proj.out_features, 48)
        self.assertEqual(model.model.layers[0].mlp.down_proj.in_features, 48)
        model(input_ids=torch.tensor([[1, 2, 3]]))


if __name__ == "__main__":
    unittest.main()
