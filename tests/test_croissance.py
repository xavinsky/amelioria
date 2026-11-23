import unittest

import torch

from tests.outils import petit_modele
import croissance


class Croissance(unittest.TestCase):
    def test_profondeur_donne_2_fois_n_moins_m_couches(self):
        model = croissance.profondeur(petit_modele(4), 1)
        self.assertEqual(len(model.model.layers), 6)
        self.assertEqual(model.config.num_hidden_layers, 6)
        model(input_ids=torch.tensor([[1, 2, 3]]))

    def test_expansion_de_blocs_ne_change_pas_la_sortie(self):
        model = petit_modele(4)
        x = torch.tensor([[1, 2, 3, 4]])
        avant = model(input_ids=x).logits
        model, indices = croissance.expansion_blocs(model, 2)
        self.assertEqual(len(model.model.layers), 6)
        self.assertEqual(indices, [2, 5])
        self.assertTrue(torch.allclose(model(input_ids=x).logits, avant, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
