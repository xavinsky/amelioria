"""Données du dépôt pour le site AmeliorIA : catalogue des exemples (exemples.yaml à la racine du
dépôt) et arbre de décision en SVG.

Le site ne recopie aucun statut ni résultat : le build les lit ici et les écrit en Markdown, avec les
marqueurs de statut ([!DEMONTRE], [!NONPROUVE], [!NONTESTE]) que build.py rend en badges.

Arbre de décision (front matter `arbre:` d'une section) :
  racine: Que faut-il améliorer ?
  branches:
    - question: "Il ignore\\nmes données"        une colonne ; « \\n » coupe la ligne
      feuilles:                                  méthodes empilées sous la question
        - {si: "Ajouter des documents", m: 18}     critère de choix, numéro de méthode
        - {si: "Savoir à intégrer", m: 1, nom: "Pré-entraînement", apres: 18}
Les feuilles d'une colonne sont triées par le build du moins coûteux au plus coûteux :
coût = (niveau de calcul + 1) × niveau de temps, puis calcul, puis ordre d'écriture. `apres: NN` :
méthode de repli, à essayer seulement si la méthode NN de la même colonne ne suffit pas (elle doit
coûter au moins autant). `nom` : nom court facultatif. Chaque feuille prend la couleur de la catégorie de sa méthode, porte ses icônes
de calcul et de temps, et mène à sa fiche quand elle existe. `cas: id` sur une colonne : son en-tête
mène au graphe de décision du cas.
"""

import html
import re
from pathlib import Path

import yaml

DEPOT = Path(__file__).resolve().parents[2]
MARQUEURS = {"demontre": "[!DEMONTRE]", "non-prouve": "[!NONPROUVE]", "a-reexaminer": "[!AREEXAMINER]", "pret-a-tester": "[!PRETATESTER]", "non-teste": "[!NONTESTE]"}
# Largeur moyenne d'un caractère dans une case de l'arbre : nom de méthode (serif gras 11,5 px),
# critère (serif 10,5 px)
CHAR_FEUILLE, CHAR_CRITERE = 6.4, 5.6
# Même chose dans l'arbre général, écrit plus gros : nom (serif gras 15 px), critère (serif 13,5 px)
CHAR_NOM_ARBRE, CHAR_CRITERE_ARBRE = 8.1, 6.9
# Icônes de coût (www/icons/, Flaticon) : processeur pour le calcul, horloge pour le temps
ICONES = {"calcul": "icons/calcul.png", "duree": "icons/duree.png"}
TAILLE_ICONE = 13
PAS_ICONE = 14
# Pictogrammes de contrainte : accès au modèle en PNG (www/icons/, Flaticon), données à fournir en SVG
# inline (build/icones/, Lucide, licence ISC)
PICTOS = {"acces": {"ouvert": "icons/poids.png", "ferme": "icons/modele-ferme.png"},
          "donnees": {"texte": "file-text", "annotees": "tag", "preferences": "scale",
                      "verificateur": "circle-check"},
          "remelange": {True: "recycle"}}
DOSSIER_PICTOS = Path(__file__).resolve().parent / "icones"


def contenu_svg(nom):
    """Éléments intérieurs d'une icône Lucide (sans la balise <svg> qui les entoure)."""
    svg = (DOSSIER_PICTOS / f"{nom}.svg").read_text(encoding="utf-8")
    return re.search(r"<svg[^>]*>(.*)</svg>", svg, re.S).group(1).strip()  # écart horizontal entre deux icônes d'une même échelle


def esc(text):
    return html.escape(str(text), quote=True)


def cellule(text):
    """Texte sûr dans une cellule de tableau Markdown."""
    return str(text).replace("|", "\\|")


class Catalogue:
    """Catalogue des exemples, et fiches du site qui présentent chaque méthode."""

    def __init__(self):
        data = yaml.safe_load((DEPOT / "exemples.yaml").read_text(encoding="utf-8"))
        self.materiel = data["materiel_reference"]
        self.categories = {c["id"]: c for c in data["categories"]}
        self.methodes = {m["dossier"][:2]: m for m in data["methodes"]}
        self.exemples = data["exemples"]
        self.couts = data["couts"]
        self.contraintes = data["contraintes"]
        self.fiches = {}  # numéro de méthode -> id de la section qui la présente
        self.github = None  # URL du dépôt public (site.yml) : liens vers le code de chaque exemple

    def url_code(self, chemin):
        """URL GitHub d'un dossier ou d'un fichier du dépôt (branche main), ou None sans dépôt configuré."""
        if not self.github:
            return None
        chemin = chemin.strip("/")
        sorte = "tree" if (DEPOT / chemin).is_dir() else "blob"
        return f"{self.github.rstrip('/')}/{sorte}/main/{chemin}"

    def lien_code(self, chemin, libelle=None):
        """Lien Markdown vers le code d'un exemple sur GitHub (texte en code si aucun dépôt n'est configuré)."""
        libelle = libelle or f"`{chemin.strip('/')}/`"
        url = self.url_code(chemin)
        return f"[{libelle}]({url})" if url else libelle

    def methode(self, num, source=""):
        num = f"{int(num):02d}"
        if num not in self.methodes:
            raise ValueError(f"{source}: méthode {num} absente de exemples.yaml")
        return num, self.methodes[num]

    def ident(self, num):
        """Identifiant d'une méthode : code de sa catégorie suivi de son numéro (SIGN05)."""
        num, m = self.methode(num)
        return f"{self.categories[m['categorie']]['code']}{num}"

    def libelle_categorie(self, cat_id):
        """Code et titre d'une catégorie (« SIGN · Apprendre à partir d'un signal »)."""
        c = self.categories[cat_id]
        return f"{c['code']} · {c['titre']}"

    def lien(self, num):
        """Nom de la méthode, en lien vers sa fiche si elle existe."""
        num, m = self.methode(num)
        nom = f"{self.ident(num)} · {m['titre']}"
        return f"[{nom}](#{self.fiches[num]})" if num in self.fiches else nom

    def exemples_de(self, num):
        num, m = self.methode(num)
        return [e for e in self.exemples if e["chemin"].split("/")[0] == m["dossier"]]

    @staticmethod
    def durees(e):
        """Durées mesurées d'un exemple : test rapide et configuration complète."""
        test, complet = (e.get(k, "—") for k in ("duree_test", "duree_complet"))
        return f'<span class="duree">test : {test}</span><br><span class="duree">complet : {complet}</span>'

    @staticmethod
    def variante(exemple):
        return exemple["chemin"].split("/variantes/")[1] if "/variantes/" in exemple["chemin"] else ""

    # ----- Sections générées dans une fiche de méthode -----

    def en_bref(self, num, source):
        """Repères d'une méthode, placés en tête de sa fiche."""
        num, m = self.methode(num, source)
        cat = self.categories[m["categorie"]]
        return (f"- **Identifiant** : {self.ident(num)}\n- **Catégorie** : {cat['code']} · {cat['titre']}\n- **Regroupe** : {m['regroupe']}\n"
                f"- **Modèles concernés** : {m['modeles']}\n"
                f"- **Calcul** : {self.icones(num, 'calcul')}\n- **Temps** : {self.icones(num, 'duree')}\n"
                f"- **Modèle** : {self.contrainte(num, 'acces')}\n- **Données** : {self.contrainte(num, 'donnees')}\n"
                + (f"- **Contre l'oubli** : <span class=\"contrainte\">{self.picto('remelange', True)} "
                   f"{esc(self.contraintes['remelange'])}</span>\n" if m.get("remelange") else "")
                + 
                f"- **GPU local 12 Go** (exemple de code) : {m['local']}\n"
                f"- **Code** : {self.lien_code(m['dossier'])}\n")

    def avertissement(self, num, source):
        """Encart d'avertissement d'une page Exemple de code quand un de ses exemples locaux n'est pas démontré."""
        num, m = self.methode(num, source)
        exemples = self.exemples_de(num)
        nom = lambda e: f"variante `{self.variante(e)}`" if self.variante(e) else "exemple principal"
        demontres = [nom(e) for e in exemples if e["statut"] == "demontre"]
        autres = [nom(e) for e in exemples if e["statut"] != "demontre"]
        if not autres:
            return ""
        if not demontres:
            texte = ("**Pas encore démontré.** " + (
                "L'exemple de code de cette méthode s'exécute, mais son résultat ne prouve pas encore le gain "
                "de la méthode." if len(exemples) == 1 else
                f"Aucun des {len(exemples)} exemples de code de cette méthode ne démontre encore son gain : "
                "ils s'exécutent, mais leur résultat ne le prouve pas."))
        else:
            texte = (f"**Démontré en partie.** {len(demontres)} exemple{'s' if len(demontres) > 1 else ''} "
                     f"sur {len(exemples)} démontre{'nt' if len(demontres) > 1 else ''} le gain de la méthode "
                     f"({', '.join(demontres)}). "
                     + ("Les autres s'exécutent, mais leur résultat" if len(autres) > 1
                        else "L'autre s'exécute, mais son résultat")
                     + f" ne le prouve pas encore ({', '.join(autres)}).")
        if (DEPOT / m["dossier"] / "infra").is_dir():
            texte += " La version grosse infra n'a jamais été exécutée."
        return (f"> [!WARNING]\n> {texte} La raison figure dans le tableau « Essayer dans le dépôt » ; "
                "les pistes pour y arriver : [Travail en cours](#travail-en-cours).\n\n")

    def essayer(self, num, source):
        """Section « Essayer dans le dépôt » : exemples, statut, résultat mesuré, commandes."""
        num, m = self.methode(num, source)
        lignes = []
        for e in self.exemples_de(num):
            var = self.variante(e)
            nom = self.lien_code(e["chemin"], f"variante `{var}`" if var else "exemple principal")
            opt = f" VAR={var}" if var else ""
            reserve = f" — *{cellule(e['reserve'])}*" if e.get("reserve") else ""
            lignes.append(f"| {nom} | {MARQUEURS[e['statut']]}{self.essais(e)} | {cellule(e['resultat'])}{reserve} "
                          f"| {self.durees(e)} | `make test EX={self.ident(num)}{opt}` |")
        if (DEPOT / m["dossier"] / "infra").is_dir():
            lignes.append(f"| {MARQUEURS['non-teste']} version grosse infra | | jamais exécutée | "
                          f"| `{m['dossier']}/infra/` |")
        return ("## Essayer dans le dépôt\n\n"
                "| Exemple | Statut | Résultat mesuré | Durée | Commande |\n|---|---|---|---|---|\n"
                + "\n".join(lignes)
                + f"\n\nRésultats et durées mesurés sur {self.materiel} ; résultats de la configuration complète. "
                "`make test` lance un test rapide sur un petit modèle ; `make run` lance l'exemple complet ; "
                "`make verifier` contrôle d'abord les prérequis. Installation : [Installer et lancer](#installer).\n\n"
                f"```bash\nmake install EXTRAS=1\nmake test EX={self.ident(num)}\nmake run EX={self.ident(num)}\n```\n")

    def tableau_complet(self):
        """Tous les exemples, par catégorie."""
        blocs = []
        for cat in self.categories.values():
            lignes = []
            for num, m in self.methodes.items():
                if m["categorie"] != cat["id"]:
                    continue
                for e in self.exemples_de(num):
                    var = self.lien_code(e["chemin"], f"`{self.variante(e) or 'principal'}`")
                    reserve = f" — *{cellule(e['reserve'])}*" if e.get("reserve") else ""
                    lignes.append(f"| {self.lien(num)} | {var} | {MARQUEURS[e['statut']]} "
                                  f"| {cellule(e['resultat'])}{reserve} | {self.durees(e)} |")
            blocs.append(f"## {cat['code']} · {cat['titre']}\n\n{cat['resume']}\n\n"
                         "| Méthode | Exemple | Statut | Résultat mesuré | Durée |\n|---|---|---|---|---|\n"
                         + "\n".join(lignes))
        return "\n\n".join(blocs) + "\n"

    def comparatif(self, cat_id):
        """Tableau des méthodes d'une catégorie : coût, contraintes et statut des exemples de code."""
        if cat_id not in self.categories:
            raise ValueError(f"comparatif : catégorie « {cat_id} » absente de exemples.yaml")
        lignes = []
        for num, m in self.methodes.items():
            if m["categorie"] != cat_id:
                continue
            oubli = (f'{self.picto("remelange", True)} oui' if m.get("remelange") else "—")
            statuts = " ".join(MARQUEURS[e["statut"]] for e in self.exemples_de(num))
            cellules = [self.lien(num), self.icones(num, "calcul"), self.icones(num, "duree"),
                        self.contrainte(num, "donnees"), oubli, statuts]
            # Une ligne de tableau Markdown tient sur une ligne : les SVG inline sont mis à plat
            lignes.append("| " + " | ".join(" ".join(c.split()) for c in cellules) + " |")
        return ("| Méthode | Calcul | Temps | Données à fournir | Remélanger du général | Exemples de code |\n"
                "|---|---|---|---|---|---|\n" + "\n".join(lignes)
                + f"\n\nCalcul et temps : {self.couts['hypothese']}. Statuts des exemples : "
                "[Statut des exemples](#statut-des-exemples).\n")

    def travaux(self):
        """Page « Travail en cours » : exemples locaux démontrés, exemples locaux à démontrer (avec la piste),
        versions grosse infra en attente d'une infra cloud (matériel et durée estimée)."""
        demontres, a_faire = [], []
        for num, m in self.methodes.items():
            for e in self.exemples_de(num):
                var = self.variante(e)
                nom = self.lien(num) + (f" · `{var}`" if var else "") + f" · {self.lien_code(e['chemin'], 'code')}"
                ligne = f"| {nom} | {cellule(e.get('fait', ''))} | {cellule(e['resultat'])} |"
                if e["statut"] == "demontre":
                    demontres.append(ligne)
                else:
                    a_faire.append(ligne + f" {cellule(e.get('piste', ''))} |")
        cloud = [f"| {self.lien(num)} | {cellule(m['infra']['fait'])} | {m['infra']['materiel']} "
                 f"| {m['infra']['duree_estimee']} |"
                 for num, m in self.methodes.items() if m.get("infra")]
        return (f"## En local : démontrés ({len(demontres)})\n\n"
                "Le gain de la méthode est mesuré et net, sur le GPU local de 12 Go.\n\n"
                "| Exemple | Ce qu'il fait | Résultat mesuré |\n|---|---|---|\n" + "\n".join(demontres)
                + f"\n\n## En local : à démontrer ({len(a_faire)})\n\n"
                "Ces exemples s'exécutent sur le GPU local, mais leur résultat ne prouve pas encore le gain de la "
                "méthode (statut « prêt à tester » : le code de la démonstration est écrit, le calcul reste à lancer). La dernière colonne dit ce qui doit changer.\n\n"
                "| Exemple | Ce qu'il fait | Résultat mesuré | Piste d'amélioration |\n|---|---|---|---|\n"
                + "\n".join(a_faire)
                + f"\n\n## Sur cloud : en attente d'une infra cloud ({len(cloud)})\n\n"
                "Les versions grosse infra (dossier `infra/` de chaque méthode) demandent des GPU de centre de calcul. "
                "Leur code est fourni, mais aucune n'a été exécutée ; les durées sont des estimations.\n\n"
                "| Méthode | Ce que fait la version infra | Matériel visé | Durée estimée |\n|---|---|---|---|\n"
                + "\n".join(cloud) + "\n")

    @staticmethod
    def essais(e):
        """Nombre de façons de prouver la méthode déjà mesurées, pour un exemple non démontré."""
        n = e.get("tentatives")
        return f" · {n} tentative{'s' if n > 1 else ''}" if n and e["statut"] != "demontre" else ""

    def compteurs(self):
        """Chiffres du dépôt pour les pages qui les citent (`{{cle}}` dans le texte, front matter `compteurs: true`)."""
        n = {k: sum(e["statut"] == k for e in self.exemples) for k in MARQUEURS}
        return {"methodes": len(self.methodes), "categories": len(self.categories), "exemples": len(self.exemples),
                "demontres": n["demontre"], "non_prouves": n["non-prouve"] + n["pret-a-tester"] + n["a-reexaminer"], "a_reexaminer": n["a-reexaminer"], "prets_a_tester": n["pret-a-tester"],
                "infra": sum((DEPOT / m["dossier"] / "infra").is_dir() for m in self.methodes.values())}

    def bilan(self):
        """Phrase de bilan : nombre d'exemples par statut."""
        n = {k: sum(e["statut"] == k for e in self.exemples) for k in MARQUEURS}
        return (f"Bilan : {len(self.exemples)} exemples locaux, dont {n['demontre']} démontrés et "
                f"{n['non-prouve'] + n['pret-a-tester'] + n['a-reexaminer']} non prouvés (dont {n['pret-a-tester']} prêts à tester et {n['a-reexaminer']} à réexaminer) ; versions grosse infra non testées.")

    def verifier_menu(self, groupes):
        """Les groupes de l'onglet Méthodes sont les catégories du catalogue, dans le même ordre ; leur
        libellé (code et titre) vient du catalogue."""
        if list(groupes) != list(self.categories):
            raise ValueError(f"site.yml, onglet methodes : groupes {list(groupes)} différents des catégories "
                             f"de exemples.yaml {list(self.categories)}")
        for cat_id in groupes:
            groupes[cat_id] = self.libelle_categorie(cat_id)

    # ----- Coût : calcul et temps -----

    def icones(self, num, sorte, svg_xy=None, texte=True):
        """Trois icônes, dont `niveau` pleines : processeurs (calcul) ou horloges (temps).
        En HTML inline par défaut ; en groupe SVG positionné en (x, y) si `svg_xy` est donné."""
        _, m = self.methode(num)
        niveau = m["calcul"] if sorte == "calcul" else m["duree"]
        motifs = "".join(self.icone(sorte, k < niveau, k * PAS_ICONE) for k in range(3))
        titre = f"{'Calcul' if sorte == 'calcul' else 'Temps'} : {self.couts[sorte][niveau]} ({self.couts['hypothese']})"
        if svg_xy:
            x, y = svg_xy
            return f'<g transform="translate({x:g},{y:g})"><title>{esc(titre)}</title>{motifs}</g>'
        largeur = 2 * PAS_ICONE + TAILLE_ICONE
        libelle = f" {esc(self.couts[sorte][niveau])}" if texte else ""
        return (f'<span class="couts" title="{esc(titre)}"><svg viewBox="0 0 {largeur} {TAILLE_ICONE}" '
                f'width="{largeur}" height="{TAILLE_ICONE}" aria-hidden="true">{motifs}</svg>{libelle}</span>')

    @staticmethod
    def icone(sorte, pleine, dx=0):
        """Une icône (pleine ou atténuée), décalée de `dx`, dans un SVG."""
        etat = "on" if pleine else "off"
        return (f'<image class="ico-img {etat}" href="{ICONES[sorte]}" x="{dx}" y="0" '
                f'width="{TAILLE_ICONE}" height="{TAILLE_ICONE}"/>')

    # ----- Contraintes : accès au modèle, données à fournir -----

    def picto(self, sorte, valeur, svg_xy=None):
        """Pictogramme d'une contrainte (vide si la méthode ne demande aucune donnée)."""
        nom = PICTOS[sorte].get(valeur)
        if not nom:
            return ""
        titre = esc(self.contraintes[sorte] if sorte == "remelange" else self.contraintes[sorte][valeur])
        classe = f"picto picto-{valeur}"
        if nom.endswith(".png"):
            if svg_xy:
                x, y = svg_xy
                return (f'<image class="{classe} picto-img" href="{nom}" x="{x:g}" y="{y:g}" width="{TAILLE_ICONE}" '
                        f'height="{TAILLE_ICONE}"><title>{titre}</title></image>')
            return (f'<img class="{classe} picto-img" src="{nom}" width="{TAILLE_ICONE}" height="{TAILLE_ICONE}" '
                    'alt="">')
        attrs = ('viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                 'stroke-linecap="round" stroke-linejoin="round"')
        if svg_xy:
            x, y = svg_xy
            return (f'<svg class="{classe}" x="{x:g}" y="{y:g}" width="{TAILLE_ICONE}" height="{TAILLE_ICONE}" {attrs}>'
                    f"<title>{titre}</title>{contenu_svg(nom)}</svg>")
        return (f'<svg class="{classe}" width="{TAILLE_ICONE}" height="{TAILLE_ICONE}" {attrs} aria-hidden="true">'
                f"{contenu_svg(nom)}</svg>")

    def contrainte(self, num, sorte, texte=True):
        """Pictogramme et libellé (ou infobulle seule) d'une contrainte de la méthode, en HTML."""
        _, m = self.methode(num)
        valeur = m[sorte]
        libelle = esc(self.contraintes[sorte][valeur])
        picto = self.picto(sorte, valeur) or "—"
        if texte:
            return f'<span class="contrainte">{picto} {libelle}</span>'
        return f'<span class="contrainte" title="{libelle}">{picto}</span>'

    # ----- Arbre de décision -----

    def cout(self, num):
        """Coût d'une méthode pour le tri : (calcul + 1) × temps, puis calcul."""
        _, m = self.methode(num)
        return (m["calcul"] + 1) * m["duree"], m["calcul"]

    def feuilles_triees(self, branche, source):
        """Feuilles d'une colonne, de la moins coûteuse à la plus coûteuse, `apres` contrôlé."""
        feuilles = branche["feuilles"]
        nums = {self.methode(f["m"], source)[0] for f in feuilles}
        for f in feuilles:
            if "suite" in f:
                raise ValueError(f"{source}: arbre, clé `suite` remplacée par `apres: NN`")
            if "apres" in f:
                avant = self.methode(f["apres"], source)[0]
                if avant not in nums:
                    raise ValueError(f"{source}: arbre, `apres: {avant}` absent de la colonne de {f['m']}")
                if self.cout(f["m"]) < self.cout(avant):
                    raise ValueError(f"{source}: arbre, la méthode {f['m']} coûte moins que {avant}, "
                                     "dont elle est le repli")
        return sorted(feuilles, key=lambda f: self.cout(f["m"]))

    def arbre(self, spec, uid, source):
        """Arbre général, de haut en bas, sur deux bandes de trois besoins : la question racine, puis chaque
        besoin et ses méthodes de la moins coûteuse à la plus coûteuse. Renvoie le SVG et les sections
        visées par ses liens."""
        branches = spec["branches"]
        par_bande, col, gap, marge, tronc = 3, 300, 16, 4, 14
        bandes = [branches[k:k + par_bande] for k in range(0, len(branches), par_bande)]
        largeur = tronc + par_bande * col + (par_bande - 1) * gap + marge
        h_q, h_f, pas, ecart_bande = 52, 70, 84, 34
        cx = tronc + (largeur - tronc - marge) / 2
        marker = f"{uid}-fleche"
        cibles = []
        out = [f'<rect class="flow-box racine" x="{cx - 170:g}" y="6" width="340" height="44"/>',
               f'<text class="flow-label racine-label" x="{cx:g}" y="34" text-anchor="middle">{esc(spec["racine"])}</text>']
        y_bus = 72
        out.append(f'<path class="flow-line" d="M{cx:g},50 V{y_bus}"/>')
        for nb, bande in enumerate(bandes):
            centres = [tronc + i * (col + gap) + col / 2 for i in range(len(bande))]
            if nb == 0:
                out.append(f'<path class="flow-line" d="M{marge:g},{y_bus} H{centres[-1]:g}"/>')
            else:
                out.append(f'<path class="flow-line" d="M{marge:g},{y_bus_0} V{y_bus} H{centres[-1]:g}"/>')
            if nb == 0:
                y_bus_0 = y_bus
            y_q = y_bus + 18
            y_f = y_q + h_q + 18
            for i, b in enumerate(bande):
                x = tronc + i * (col + gap)
                out.append(f'<path class="flow-line" d="M{centres[i]:g},{y_bus} V{y_q - 1}" marker-end="url(#{marker})"/>')
                entete = [f'<rect class="flow-box branche" x="{x:g}" y="{y_q}" width="{col}" height="{h_q}"/>']
                titre = b["question"].replace("\n", " ")
                entete.append(f'<text class="flow-label branche-label" x="{centres[i]:g}" y="{y_q + h_q / 2 + 5:g}" '
                              f'text-anchor="middle">{esc(titre)}</text>')
                if b.get("cas"):
                    cibles.append(b["cas"])
                    out.append(f'<a href="#{b["cas"]}" class="leaf-link branche-link">{"".join(entete)}</a>')
                else:
                    out.append("".join(entete))
                epine = x + 8
                feuilles = self.feuilles_triees(b, source)
                out.append(f'<path class="flow-line" d="M{epine:g},{y_q + h_q} V{y_f + (len(feuilles) - 1) * pas + h_f / 2:g}"/>')
                for j, f in enumerate(feuilles):
                    out.append(self._case_arbre(f, x + 20, y_f + j * pas, col - 20, h_f, epine, marker, source, cibles))
            y_bus = y_f + max(len(b["feuilles"]) for b in bande) * pas - pas + h_f + ecart_bande
        hauteur = y_bus - ecart_bande + 6
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {largeur} {hauteur:g}" role="img" '
               f'class="arbre arbre-general" aria-label="{esc(spec["racine"])} Arbre de décision vers les méthodes">'
               f'<defs><marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
               'markerHeight="6" orient="auto-start-reverse"><path class="flow-arrow" d="M0,0 L10,5 L0,10 z"/>'
               "</marker></defs>" + "".join(out) + "</svg>")
        figure = f'<figure class="block-diagram arbre-figure">{svg}{self.legende_couts()}</figure>\n'
        return figure, cibles

    def _case_arbre(self, f, fx, fy, fw, h_f, epine, marker, source, cibles):
        """Une case de l'arbre général : critère, méthode, coûts et contraintes, lien vers la fiche."""
        num, m = self.methode(f["m"], source)
        nom = f"{self.ident(num)} · {f.get('nom', m['court'])}"
        for texte, largeur_car in ((nom, CHAR_NOM_ARBRE), (f["si"], CHAR_CRITERE_ARBRE)):
            if len(texte) * largeur_car > fw - 10:
                raise ValueError(f"{source}: arbre, « {texte} » trop long pour une case")
        fleche = (f'<path class="flow-line" d="M{epine:g},{fy + h_f / 2:g} H{fx - 1:g}" '
                  f'marker-end="url(#{marker})"/>')
        corps = (f'<rect class="flow-box leaf cat-{m["categorie"]}" x="{fx:g}" y="{fy}" width="{fw}" height="{h_f}"/>'
                 f'<text class="feuille-critere" x="{fx + 8:g}" y="{fy + 19}">{esc(f["si"])}</text>'
                 f'<text class="flow-label feuille-nom" x="{fx + 8:g}" y="{fy + 40}">{esc(nom)}</text>'
                 + f'<g transform="translate({fx + 8:g},{fy + 47}) scale(1.25)">'
                 + self.icones(num, "calcul", (0, 0)) + self.icones(num, "duree", (48, 0))
                 + self.picto("acces", m["acces"], (96, 0)) + self.picto("donnees", m["donnees"], (113, 0))
                 + self.picto("remelange", m.get("remelange", False), (130, 0)) + "</g>")
        if num in self.fiches:
            cibles.append(self.fiches[num])
            return fleche + f'<a href="#{self.fiches[num]}" class="leaf-link">{corps}</a>'
        return fleche + f'<g class="leaf-a-venir">{corps}</g>'

    def echelle(self, sorte, niveau):
        """Les trois icônes d'une échelle, `niveau` pleines, en SVG inline."""
        largeur = 2 * PAS_ICONE + TAILLE_ICONE
        motifs = "".join(self.icone(sorte, k < niveau, k * PAS_ICONE) for k in range(3))
        return (f'<svg viewBox="0 0 {largeur} {TAILLE_ICONE}" width="{largeur}" height="{TAILLE_ICONE}" '
                f'aria-hidden="true">{motifs}</svg>')

    def legende_couts(self):
        """Légende des arbres : petits tableaux côte à côte (catégorie, calcul, temps, modèle, données et oubli),
        qui passent à la ligne sur un écran étroit."""
        def bloc(*sections):
            lignes = []
            for titre, rangees in sections:
                lignes.append(f'<tr class="sous-titre"><th colspan="2">{titre}</th></tr>')
                lignes += [f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rangees]
            return f'<table class="legende-couts"><tbody>{"".join(lignes)}</tbody></table>'

        categories = ('<table class="legende-couts"><tbody><tr class="sous-titre"><th>Catégorie</th></tr>'
                      + "".join(f'<tr><td class="cat-cellule cat-{c["id"]}"><b>{esc(c["code"])}</b> {esc(c["titre"])}</td></tr>'
                                for c in self.categories.values()) + "</tbody></table>")
        blocs = [categories,
                 bloc(("Calcul", [(self.echelle("calcul", n), esc(v)) for n, v in self.couts["calcul"].items()])),
                 bloc(("Temps", [(self.echelle("duree", n), esc(v)) for n, v in self.couts["duree"].items()])),
                 bloc(("Modèle", [(self.picto("acces", v), esc(x)) for v, x in self.contraintes["acces"].items()])),
                 bloc(("Données à fournir", [(self.picto("donnees", v) or "—", esc(x))
                                             for v, x in self.contraintes["donnees"].items()]),
                      ("Contre l'oubli", [(self.picto("remelange", True), esc(self.contraintes["remelange"]))]))]
        return (f'<div class="legende-blocs">{"".join(blocs)}</div>'
                f'<p class="arbre-lecture credit">Calcul et temps : {esc(self.couts["hypothese"])}.</p>')

    # ----- Graphe de décision d'un cas -----

    def decision(self, spec, uid, source):
        """Graphe de décision d'un cas (front matter `decision:`) en SVG, et sections visées par ses liens.

        Un nœud est une question (`q`), une méthode (`m`, `nom` facultatif) ou une issue (`fin`) ;
        `choix` (libellé → nœud) donne ses suites, sur une question comme après une méthode
        (« ne suffit pas », « puis »). Disposition en arbre de haut en bas, calculée ici."""
        larg, ecart, h_rang, pas_rang = 172, 14, 50, 108
        marker = f"{uid}-fleche"
        cibles = []

        def largeur(n):
            enfants = list((n.get("choix") or {}).values())
            if not enfants:
                return larg
            return max(larg, sum(largeur(e) for e in enfants) + ecart * (len(enfants) - 1))

        def profondeur(n):
            return 1 + max((profondeur(e) for e in (n.get("choix") or {}).values()), default=0)

        out = []

        def dessiner(n, x0, rang):
            w = largeur(n)
            cx, y = x0 + w / 2, 8 + rang * pas_rang
            out.append(self._noeud(n, cx - larg / 2, y, larg, h_rang, source, cibles))
            enfants = list((n.get("choix") or {}).items())
            if not enfants:
                return
            bus = y + h_rang + 18
            x, centres = x0 + (w - (sum(largeur(e) for _, e in enfants) + ecart * (len(enfants) - 1))) / 2, []
            for label, e in enfants:
                we = largeur(e)
                centres.append((x + we / 2, label, e, x))
                x += we + ecart
            out.append(f'<path class="flow-line" d="M{cx:g},{y + h_rang} V{bus:g}"/>')
            if len(centres) > 1:
                out.append(f'<path class="flow-line" d="M{centres[0][0]:g},{bus:g} H{centres[-1][0]:g}"/>')
            for ccx, label, e, ex in centres:
                haut = 8 + (rang + 1) * pas_rang
                out.append(f'<path class="flow-line" d="M{ccx:g},{bus:g} V{haut - 1:g}" marker-end="url(#{marker})"/>')
                lignes = str(label).split("\n")
                lw = max(len(li) for li in lignes) * 6.2 + 14
                ch = 15 * len(lignes) + 3
                cy = bus + (haut - bus - ch) / 2
                out.append(f'<rect class="edge-chip" x="{ccx - lw / 2:g}" y="{cy:g}" width="{lw:g}" height="{ch}"/>')
                for k, li in enumerate(lignes):
                    out.append(f'<text class="edge-text" x="{ccx:g}" y="{cy + 13 + k * 15:g}">{esc(li)}</text>')
                dessiner(e, ex, rang + 1)

        w_total = largeur(spec)
        h_total = 8 + (profondeur(spec) - 1) * pas_rang + h_rang + 6
        dessiner(spec, 4, 0)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w_total + 8:g} {h_total:g}" '
               f'width="{w_total + 8:g}" role="img" '
               f'class="arbre decision" aria-label="Graphe de décision"><defs><marker id="{marker}" viewBox="0 0 10 10" '
               'refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
               '<path class="flow-arrow" d="M0,0 L10,5 L0,10 z"/></marker></defs>' + "".join(out) + "</svg>")
        figure = f'<figure class="block-diagram arbre-figure decision-figure">{svg}{self.legende_couts()}</figure>\n'
        return figure, cibles

    def _noeud(self, n, x, y, w, h, source, cibles):
        """Case d'un graphe de décision : question, méthode (avec coûts et contraintes) ou issue."""
        if "m" in n:
            num, m = self.methode(n["m"], source)
            nom = f"{self.ident(num)} · {n.get('nom', m['court'])}"
            if len(nom) * CHAR_FEUILLE > w - 8:
                raise ValueError(f"{source}: graphe, « {nom} » trop long pour une case")
            corps = (f'<rect class="flow-box leaf cat-{m["categorie"]}" x="{x:g}" y="{y:g}" width="{w}" height="{h}"/>'
                     f'<text class="flow-label feuille-nom" x="{x + 7:g}" y="{y + 19:g}">{esc(nom)}</text>'
                     + self.icones(num, "calcul", (x + 7, y + 29)) + self.icones(num, "duree", (x + 56, y + 29))
                     + self.picto("acces", m["acces"], (x + 108, y + 29))
                     + self.picto("donnees", m["donnees"], (x + 126, y + 29))
                     + self.picto("remelange", m.get("remelange", False), (x + 144, y + 29)))
            if num in self.fiches:
                cibles.append(self.fiches[num])
                return f'<a href="#{self.fiches[num]}" class="leaf-link">{corps}</a>'
            return f'<g class="leaf-a-venir">{corps}</g>'
        cle, classe = ("q", "branche") if "q" in n else ("fin", "issue")
        if cle not in n:
            raise ValueError(f"{source}: graphe, nœud sans `q`, `m` ni `fin` : {n}")
        lignes = str(n[cle]).split("\n")
        for li in lignes:
            if len(li) * CHAR_CRITERE > w - 10:
                raise ValueError(f"{source}: graphe, ligne « {li} » trop longue : couper par \\n")
        out = [f'<rect class="flow-box {classe}" x="{x:g}" y="{y:g}" width="{w}" height="{h}"/>']
        for k, li in enumerate(lignes):
            ty = y + h / 2 + 4 + (k - (len(lignes) - 1) / 2) * 14
            out.append(f'<text class="flow-label {classe}-label" x="{x + w / 2:g}" y="{ty:g}" text-anchor="middle">'
                       f"{esc(li)}</text>")
        return "".join(out)

