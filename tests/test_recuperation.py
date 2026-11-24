import importlib.util
import random
import unittest

from tests.outils import RACINE

spec = importlib.util.spec_from_file_location("recuperation_12", RACINE / "12-extension-contexte" / "recuperation.py")
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)


class JetonsParMot:
    """Tokenizer factice : un token par mot, les mots sont leurs propres identifiants."""

    def __call__(self, texte):
        return {"input_ids": texte.split()}


class Exemples(unittest.TestCase):
    POOL = ["lorem ipsum dolor sit amet consectetur " * 3] * 5

    def test_la_perte_ne_porte_que_sur_la_reponse(self):
        e = rec.exemple(JetonsParMot(), self.POOL, 200, 6, random.Random(0), True)
        cibles = [t for t, lab in zip(e["input_ids"], e["labels"]) if lab != -100]
        self.assertEqual(len(cibles), 1)   # un nombre, un seul mot ici
        self.assertTrue(cibles[0].isdigit())
        self.assertEqual(e["labels"][-1], e["input_ids"][-1])

    def test_la_reponse_est_le_code_de_la_phrase_cachee_du_nom_demande(self):
        for graine in range(5):
            mots = rec.exemple(JetonsParMot(), self.POOL, 200, 6, random.Random(graine), True)["input_ids"]
            reponse = mots[-1]
            self.assertEqual(sum(m.rstrip(".") == reponse for m in mots), 2)   # la phrase cachée et la réponse
            nom = next(n for n in rec.NOMS_ENTRAINEMENT if n in mots[-12:])
            cachee = next(i for i, m in enumerate(mots[:-1]) if m.rstrip(".") == reponse)
            self.assertIn(nom, mots[max(0, cachee - 8) : cachee])   # le nom interrogé précède son code

    def test_le_temoin_n_a_ni_phrase_cachee_ni_question(self):
        e = rec.exemple(JetonsParMot(), self.POOL, 200, 6, random.Random(0), False)
        self.assertFalse(any(m in rec.NOMS_ENTRAINEMENT for m in e["input_ids"]))
        self.assertEqual(e["labels"], e["input_ids"])

    def test_les_noms_d_entrainement_different_de_ceux_de_la_mesure(self):
        import sys

        sys.path.insert(0, str(RACINE / "12-extension-contexte"))
        texte = (RACINE / "12-extension-contexte" / "run.py").read_text()
        ligne = next(x for x in texte.splitlines() if x.startswith("NOMS = "))
        noms_mesure = eval(ligne.split("=", 1)[1])
        self.assertFalse(set(noms_mesure) & set(rec.NOMS_ENTRAINEMENT))

    def test_gabarits_d_entrainement_differents_de_celui_de_la_mesure(self):
        self.assertFalse(any("pass key" in g for g, _ in rec.GABARITS))


if __name__ == "__main__":
    unittest.main()
