import unittest

from tests.outils import RACINE  # noqa: F401  (chemins)
from commun import donnees, evaluation


class Entites(unittest.TestCase):
    def test_bio_regroupe_les_mots_d_une_entite(self):
        e = donnees.entites_bio(["Joe", "Biden", "visits", "New", "York", "."], ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O"])
        self.assertEqual(e, [("PER", "Joe Biden"), ("LOC", "New York")])

    def test_deux_entites_collees_de_meme_type_restent_separees(self):
        e = donnees.entites_bio(["Paris", "Berlin"], ["B-LOC", "B-LOC"])
        self.assertEqual(e, [("LOC", "Paris"), ("LOC", "Berlin")])

    def test_phrase_sans_entite(self):
        self.assertEqual(donnees.entites_bio(["a", "b"], ["O", "O"]), [])

    def test_aller_retour_ecriture_lecture(self):
        e = [("PER", "Joe Biden"), ("LOC", "New York")]
        self.assertEqual(donnees.lire_entites(donnees.liste_entites(e)), {("PER", "joe biden"), ("LOC", "new york")})

    def test_lecture_ignore_la_casse_l_ordre_et_le_bruit(self):
        self.assertEqual(donnees.lire_entites("loc: New York\nsome noise\nPER:  Joe Biden "), {("LOC", "new york"), ("PER", "joe biden")})

    def test_mode_entites_exige_l_ensemble_exact(self):
        cible = "PER: joe biden\nLOC: new york"
        self.assertTrue(evaluation.correct("LOC: New York\nPER: Joe Biden", cible, "entites"))
        self.assertFalse(evaluation.correct("PER: Joe Biden", cible, "entites"))
        self.assertFalse(evaluation.correct("PER: Joe Biden\nLOC: New York\nORG: UN", cible, "entites"))


if __name__ == "__main__":
    unittest.main()
