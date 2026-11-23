import unittest

import torch

import fusion


class Fusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = torch.randn(6, 8)
        self.a = self.base + 0.1 * torch.randn(6, 8)
        self.b = self.base + 0.1 * torch.randn(6, 8)

    def test_lineaire_normalise_les_poids(self):
        self.assertTrue(torch.allclose(fusion.lineaire([self.a, self.b], [2, 2]), (self.a + self.b) / 2))

    def test_slerp_aux_extremites(self):
        self.assertTrue(torch.allclose(fusion.slerp(self.a, self.b, 0.0), self.a, atol=1e-5))
        self.assertTrue(torch.allclose(fusion.slerp(self.a, self.b, 1.0), self.b, atol=1e-5))

    def test_arithmetique_de_taches_avec_poids_un_donne_la_somme_des_ecarts(self):
        r = fusion.arithmetique_taches(self.base, [self.a, self.b], [1, 1])
        self.assertTrue(torch.allclose(r, self.base + (self.a - self.base) + (self.b - self.base), atol=1e-6))

    def test_arithmetique_de_taches_sans_ecart_rend_la_base(self):
        self.assertTrue(torch.equal(fusion.arithmetique_taches(self.base, [self.base, self.base], [0.5, 0.5]), self.base))

    def test_ties_garde_la_forme_et_le_type(self):
        r = fusion.ties(self.base, [self.a, self.b], [1.0, 1.0], densite=0.5)
        self.assertEqual(r.shape, self.base.shape)
        self.assertEqual(r.dtype, self.base.dtype)

    def test_ties_a_densite_pleine_et_signes_d_accord_moyenne_les_ecarts(self):
        a, b = self.base + 1.0, self.base + 3.0
        r = fusion.ties(self.base, [a, b], [1.0, 1.0], densite=1.0)
        self.assertTrue(torch.allclose(r, self.base + 2.0, atol=1e-5))

    def test_ties_ecarte_l_avis_minoritaire_en_signe(self):
        a, b, c = self.base + 1.0, self.base + 1.0, self.base - 0.5
        r = fusion.ties(self.base, [a, b, c], [1.0, 1.0, 1.0], densite=1.0)
        self.assertTrue(torch.allclose(r, self.base + 1.0, atol=1e-5))

    def test_dare_lineaire_garde_la_forme(self):
        g = torch.Generator().manual_seed(0)
        self.assertEqual(fusion.dare_lineaire(self.base, [self.a, self.b], [0.5, 0.5], 0.5, g).shape, self.base.shape)


if __name__ == "__main__":
    unittest.main()
