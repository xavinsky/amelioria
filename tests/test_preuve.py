import unittest

from commun import evaluation, preuve


class Intervalles(unittest.TestCase):
    def test_wilson_a_cinquante_sur_cent(self):
        bas, haut = preuve.wilson(50, 100)
        self.assertAlmostEqual(bas, 0.4038, places=3)
        self.assertAlmostEqual(haut, 0.5962, places=3)

    def test_wilson_reste_dans_zero_un_aux_extremites(self):
        self.assertEqual(preuve.wilson(0, 10)[0], 0.0)
        self.assertEqual(preuve.wilson(10, 10)[1], 1.0)

    def test_wilson_sans_donnee_est_toute_la_plage(self):
        self.assertEqual(preuve.wilson(0, 0), [0.0, 1.0])

    def test_taux_garde_le_detail_par_question(self):
        t = preuve.taux([True, False, True, True])
        self.assertEqual(t["exactitude"], 0.75)
        self.assertEqual(t["succes"], [True, False, True, True])
        self.assertEqual(t["n"], 4)


class ComparaisonsAppariees(unittest.TestCase):
    def test_mcnemar_huit_gains_et_aucune_perte(self):
        self.assertAlmostEqual(preuve.mcnemar_exact([False] * 8, [True] * 8), 2 / 256, places=4)

    def test_mcnemar_sans_discordance(self):
        self.assertEqual(preuve.mcnemar_exact([True, False], [True, False]), 1.0)

    def test_mcnemar_est_symetrique(self):
        a, b = [True] * 3 + [False] * 7, [False] * 3 + [True] * 7
        self.assertEqual(preuve.mcnemar_exact(a, b), preuve.mcnemar_exact(b, a))

    def test_difference_nette_quand_tout_s_ameliore(self):
        r = preuve.difference_appariee([False] * 40, [True] * 40)
        self.assertEqual(r["difference"], 1.0)
        self.assertTrue(r["net"])

    def test_difference_non_nette_quand_rien_ne_change(self):
        r = preuve.difference_appariee([True, False] * 20, [True, False] * 20)
        self.assertEqual(r["difference"], 0.0)
        self.assertFalse(r["net"])

    def test_difference_est_reproductible_pour_une_graine(self):
        a, b = [True, False, False, True] * 10, [True, True, False, True] * 10
        self.assertEqual(preuve.difference_appariee(a, b, graine=3), preuve.difference_appariee(a, b, graine=3))

    def test_difference_refuse_des_listes_de_tailles_differentes(self):
        with self.assertRaises(ValueError):
            preuve.difference_appariee([True], [True, False])

    def test_ecart_moyen_net_sur_un_gain_constant(self):
        r = preuve.ecart_moyen_apparie([1.0, 2.0, 3.0, 4.0] * 5, [1.5, 2.5, 3.5, 4.5] * 5)
        self.assertAlmostEqual(r["difference"], 0.5)
        self.assertTrue(r["net"])


class Correction(unittest.TestCase):
    def test_nombre_lit_la_reponse_finale(self):
        self.assertTrue(evaluation.correct("2 + 3 = 5\nAnswer: 5", "5", "nombre"))
        self.assertFalse(evaluation.correct("Answer: 6", "5", "nombre"))

    def test_contient_ignore_la_casse(self):
        self.assertTrue(evaluation.correct("The capital is PARIS.", "paris", "contient"))

    def test_premiere_phrase_ignore_ce_qui_suit(self):
        self.assertTrue(evaluation.correct("Paris. Then Berlin is also a city.", "Paris", "premiere_phrase"))
        self.assertFalse(evaluation.correct("Rome. Paris is also a city.", "Paris", "premiere_phrase"))

    def test_mode_inconnu(self):
        with self.assertRaises(ValueError):
            evaluation.correct("x", "x", "inconnu")


if __name__ == "__main__":
    unittest.main()
