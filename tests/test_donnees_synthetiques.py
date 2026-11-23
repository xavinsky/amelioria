import importlib.util
import unittest

from tests.outils import RACINE

spec = importlib.util.spec_from_file_location("run_07", RACINE / "07-donnees-synthetiques" / "run.py")
run07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run07)


class JustesDistinctes(unittest.TestCase):
    SOLS = ["3 + 4 = 7\nAnswer: 7", "4 + 3 = 7\nAnswer: 7", "3 + 4 = 7\nAnswer: 7", "3 + 5 = 8\nAnswer: 8"]

    def test_toutes_les_justes_aux_calculs_distincts(self):
        r = run07.justes_distinctes(self.SOLS, "7", True)
        self.assertEqual(len(r), 2)   # 3+4 et 4+3 ; le doublon et la fausse sont écartés

    def test_une_seule_au_hasard_sinon(self):
        self.assertEqual(len(run07.justes_distinctes(self.SOLS, "7", False)), 1)

    def test_plafond_par_question(self):
        self.assertEqual(len(run07.justes_distinctes(self.SOLS, "7", True, plafond=1)), 1)

    def test_aucune_juste(self):
        self.assertEqual(run07.justes_distinctes(self.SOLS, "99", True), [])
        self.assertEqual(run07.justes_distinctes(self.SOLS, "99", False), [])

    def test_chemin_de_calcul_ignore_le_texte_autour(self):
        self.assertEqual(run07.chemin_de_calcul("So 1,200 + 3 = 1,203"), ("1200", "3", "1203"))


if __name__ == "__main__":
    unittest.main()
