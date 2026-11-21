# AmeliorIA — site

Site statique qui présente les méthodes du dépôt : arbre de décision, une fiche par méthode, installation et statut des exemples, lexique. Moteur repris de memoria-gcp : contenu en Markdown dans `content/`, compilé par `build/build.py` en `www/data/content.js`, plus une page HTML statique par section pour le référencement.

```bash
make site                 # depuis la racine du dépôt : build, puis ouvrir site/www/index.html
```

| Élément | Rôle |
|---|---|
| `content/site.yml` | Onglets et groupes du menu ; les groupes de l'onglet Méthodes sont les catégories de `exemples.yaml`, dont le build reprend le libellé (code et titre) |
| `content/site.yml`, clé `github` | Dépôt public du code : onglet « GitHub » en fin de menu, et liens du site vers le dossier ou le fichier de chaque exemple (branche `main`). Les liens relatifs des README (fichiers, variantes, `commun/`) y mènent aussi |
| `content/site.yml`, clé `projets` | Autres projets IA de tekio.org (nom, URL, description), listés dans le menu latéral de l'Accueil |
| `content/accueil/` | Page d'accueil, premier onglet, ouverte par défaut. Ses chiffres (exemples démontrés, versions infra...) s'écrivent `{{demontres}}`, `{{exemples}}`, `{{non_prouves}}`, `{{infra}}`, `{{methodes}}`, `{{categories}}` avec `compteurs: true` : le build les tire de `exemples.yaml` |
| `content/<onglet>/<groupe>/<NN>-<slug>.md` | Une section ; `id` du front matter ou slug du nom de fichier |
| `content/lexique.md` | Un terme par titre `## Terme`, avec un lien vers la section qui le détaille |
| `build/build.py`, `build/schema.py` | Compilation, schémas d'intégration en SVG, contrôle des liens internes |
| `build/catalogue.py` | Données du dépôt : repères et exemples de chaque méthode, statuts, arbre de décision |
| `www/` | `index.html`, `engine.js`, `style.css` |
| `build/icones/` | Pictogrammes des données à fournir, de [Lucide](https://lucide.dev/), licence ISC (`LICENSE`), insérés en SVG inline par le build |
| `www/icons/` | Icônes d'accès au modèle `poids.png` et `modele-ferme.png` (Flaticon), et icônes de coût `calcul.png` (processeur, recoloré en vert `#23A055`) et `duree.png` (horloge, recolorée en bleu `#2F6FDB`), de [Flaticon](https://www.flaticon.com/) : licence gratuite avec attribution : titre, auteur et lien de chaque icône dans la page Refs (`content/refs/ressources/`) |

## Pages de méthode

Elles ne s'écrivent pas dans `content/` : `build/fiches.py` les génère depuis le dépôt, deux par méthode, rangées dans l'onglet Méthodes selon la catégorie de `exemples.yaml`.

Chaque catégorie a un code de 4 lettres lisible en anglais comme en français (`code` dans `exemples.yaml`) : DATA, SIGN, EDIT, FORM, EXEC. Une méthode est désignée partout par ce code suivi de son numéro (SIGN05), y compris dans `make test EX=SIGN05`.

En tête de chaque catégorie, une page écrite à la main, `content/methodes/<catégorie>/00-choisir.md`, explique le choix entre ses méthodes : graphe de décision (`decision:`), texte, puis tableau comparatif généré (`comparatif: <catégorie>` : coûts, données, remélange (replay), statut des exemples).

| Page | Contenu | Source |
|---|---|---|
| Concept (`#<id>`) | Introduction, repères (catégorie, coûts, contraintes), Principe, Modèles concernés, Quand l'utiliser, Pièges, Références | `NN-…/README.md` (citation d'introduction et sections du même nom), `exemples.yaml`, section de la méthode dans `docs/references.md` |
| Exemple de code (`#<id>-exemple`) | Ce que fait l'exemple, statut et résultats, commandes `make`, Utilisation, Paramètres principaux, Grosse infra | `NN-…/README.md`, `exemples.yaml` |

L'`id` est le nom du dossier sans son numéro (`peft`, `rag`), sauf s'il est fixé dans `NN-…/fiche.yaml`. Ce fichier facultatif, propre au site, donne aussi `aka` (autre nom de la méthode) et `schema` (schéma d'intégration, format de la docstring de `build/schema.py`).

Un README se lit donc aussi bien sur GitHub que sur le site : mêmes titres de section pour toutes les méthodes, une citation `> …` d'une phrase en introduction, pas d'historique. Ses liens vers un autre dossier de méthode deviennent des liens vers sa page Concept, ceux vers une variante mènent à la page Exemple ; les liens vers des fichiers du dépôt deviennent du texte.

## Statuts

`[!DEMONTRE]`, `[!NONPROUVE]`, `[!NONTESTE]` sont rendus en badges avec infobulle ; en tête d'une puce ou d'une ligne de tableau, `[!NONTESTE]` grise l'élément. Ils servent dans les sections générées et peuvent servir dans le texte.

## Arbre de décision

Front matter `arbre:` (format dans la docstring de `build/catalogue.py`) : une question par colonne, des méthodes empilées de la moins chère à la plus lourde. Chaque feuille prend la couleur de la catégorie de sa méthode et mène à sa fiche dès qu'elle existe.
