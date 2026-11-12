"""Pages de méthode générées depuis le dépôt : pour chaque méthode, une page « Concept » et une page
« Exemple du dépôt », tirées du README.md de son dossier.

Source unique du texte : le README du dossier, dont toutes les sections ont les mêmes titres.
  Concept : citation d'introduction, Principe, Modèles concernés, Quand l'utiliser, Pièges, puis les
            références de la méthode dans docs/references.md (avec leurs liens) ;
  Exemple : Ce que fait l'exemple, Composants de l'exemple (schéma Mermaid), statut et commandes
            (exemples.yaml), Utilisation, Paramètres
            principaux, Grosse infra.
Propre au site, facultatif : NN-dossier/fiche.yaml (`id`, `aka` : autre nom, `schema` : schéma
d'intégration décrit comme pour build/schema.py).

Liens du README réécrits pour le site : un autre dossier de méthode mène à sa page Concept, une
variante à la page Exemple ; les autres chemins relatifs (fichiers du dépôt) deviennent du texte.
"""

import re
from pathlib import Path

import yaml

DEPOT = Path(__file__).resolve().parents[2]
REFERENCES = DEPOT / "docs" / "references.md"
CONCEPT = ["Principe", "Modèles concernés", "Quand l'utiliser", "Pièges"]
EXEMPLE = ["Utilisation", "Paramètres principaux", "Grosse infra"]
LIEN = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def fiche_yaml(dossier):
    """Métadonnées propres au site d'un dossier de méthode (fiche.yaml), ou dictionnaire vide."""
    chemin = DEPOT / dossier / "fiche.yaml"
    return yaml.safe_load(chemin.read_text(encoding="utf-8")) or {} if chemin.is_file() else {}


def ids_des_methodes(depot):
    """Numéro de méthode -> id de sa page Concept (fiche.yaml, sinon nom du dossier sans numéro)."""
    return {num: fiche_yaml(m["dossier"]).get("id") or m["dossier"][3:] for num, m in depot.methodes.items()}


def lire_readme(chemin):
    """README d'un dossier -> (introduction, {titre de section: texte})."""
    texte = chemin.read_text(encoding="utf-8")
    morceaux = re.split(r"^## +(.+?)\s*$", texte, flags=re.M)
    tete, sections = morceaux[0], {}
    for titre, corps in zip(morceaux[1::2], morceaux[2::2]):
        cle = re.sub(r"\s*\(`config\.yaml`\)", "", titre).strip()
        sections[cle] = corps.strip()
    intro = " ".join(ligne[1:].strip() for ligne in tete.splitlines() if ligne.startswith(">"))
    return intro, sections


def references_md(num):
    """Références d'une méthode dans docs/references.md, en liste Markdown avec liens."""
    texte = REFERENCES.read_text(encoding="utf-8")
    debut = re.search(rf"^## {num} — .*$", texte, re.M)
    if not debut:
        return ""
    fin = re.compile(r"^## ", re.M).search(texte, debut.end())
    lignes = []
    for ligne in texte[debut.end():fin.start() if fin else len(texte)].splitlines():
        cellules = [c.strip() for c in ligne.strip().strip("|").split("|")]
        if len(cellules) != 4 or cellules[0] in ("Technique", "---") or set(cellules[0]) <= {"-"}:
            continue
        technique, papier, lien, apport = cellules
        url = LIEN.search(lien)
        papier = f"[{papier}]({url.group(2)})" if url else papier
        lignes.append(f"- {technique} : {papier}. {apport}.")
    return "\n".join(lignes)


class Generateur:
    """Pages Concept et Exemple de chaque méthode, prêtes pour la boucle de build.py."""

    def __init__(self, depot):
        self.depot = depot
        self.ids = ids_des_methodes(depot)
        self.dossiers = {m["dossier"]: num for num, m in depot.methodes.items()}

    def liens(self, texte, dossier):
        """Réécrit les liens relatifs d'un README pour le site."""
        def remplacer(m):
            libelle, cible = m.group(1), m.group(2)
            if re.match(r"https?://", cible) or cible.startswith("#"):
                return m.group(0)
            autre = re.match(r"\.\./(\d\d-[^/#)]+)/?(.*)$", cible)
            if autre and autre.group(1) in self.dossiers:
                num = self.dossiers[autre.group(1)]
                suite = "-exemple" if autre.group(2).startswith("variantes") else ""
                return f"[{libelle}](#{self.ids[num]}{suite})"
            if cible.startswith("variantes"):
                return f"[{libelle}](#{self.ids[self.dossiers[dossier]]}-exemple)"
            return libelle
        return LIEN.sub(remplacer, texte)

    def sections(self, readme, titres, dossier):
        """Sections du README demandées, dans l'ordre, liens réécrits ; la ligne « Schéma : … » est retirée."""
        morceaux = []
        for titre in titres:
            if titre in readme:
                corps = "\n".join(li for li in readme[titre].splitlines() if not li.startswith("Schéma :"))
                morceaux.append(f"## {titre}\n\n{self.liens(corps.strip(), dossier)}")
        return "\n\n".join(morceaux)

    def entrees(self, groupe):
        """(chemin source, front matter, corps Markdown) des pages des méthodes d'une catégorie."""
        sortie = []
        for num, m in self.depot.methodes.items():
            if m["categorie"] != groupe:
                continue
            chemin = DEPOT / m["dossier"] / "README.md"
            intro, readme = lire_readme(chemin)
            extra = fiche_yaml(m["dossier"])
            ident = self.ids[num]
            refs = references_md(num)
            concept = (f"{self.liens(intro, m['dossier'])}\n\n{self.depot.en_bref(num, chemin)}\n"
                       + self.sections(readme, CONCEPT, m["dossier"])
                       + (f"\n\n## Références\n\n{refs}" if refs else "")
                       + f"\n\n## Dans le dépôt\n\nExemple, résultats mesurés et commandes : "
                       f"[{m['titre']} : exemple du dépôt](#{ident}-exemple).\n")
            exemple = (f"Ce que le dossier `{m['dossier']}/` fournit pour cette méthode. Principe et références : "
                       f"[{m['titre']}](#{ident}).\n\n"
                       + self.sections(readme, ["Ce que fait l'exemple", "Composants de l'exemple"], m["dossier"])
                       + "\n\n" + self.depot.essayer(num, chemin) + "\n\n"
                       + self.sections(readme, EXEMPLE, m["dossier"]))
            base = {"subgroup": f"{num} · {m['titre']}", "aka": extra.get("aka")}
            sortie.append((chemin, {**base, "title": m["titre"], "nav": "Concept", "id": ident,
                                    "schema": extra.get("schema"), "description": intro}, concept))
            sortie.append((chemin, {**base, "title": f"Exemple du dépôt : {m['titre']}", "nav": "Exemple du dépôt",
                                    "id": f"{ident}-exemple",
                                    "description": f"{m['titre']} : l'exemple exécutable du dépôt, son statut, "
                                                   "ses résultats mesurés et ses commandes."}, exemple))
        return sortie
