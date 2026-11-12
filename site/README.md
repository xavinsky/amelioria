# AmeliorIA — site

Site statique qui présente les méthodes du dépôt : arbre de décision, une fiche par méthode, installation et statut des exemples, lexique. Moteur repris de memoria-gcp : contenu en Markdown dans `content/`, compilé par `build/build.py` en `www/data/content.js`, plus une page HTML statique par section pour le référencement.

```bash
make site                 # depuis la racine du dépôt : build, puis ouvrir site/www/index.html
```

| Élément | Rôle |
|---|---|
| `content/site.yml` | Onglets et groupes du menu ; les groupes de l'onglet Méthodes sont les catégories de `exemples.yaml` |
| `content/<onglet>/<groupe>/<NN>-<slug>.md` | Une section ; `id` du front matter ou slug du nom de fichier |
| `content/lexique.md` | Un terme par titre `## Terme`, avec un lien vers la section qui le détaille |
| `build/build.py`, `build/schema.py` | Compilation, schémas d'intégration en SVG, contrôle des liens internes |
| `build/catalogue.py` | Données du dépôt : repères et exemples de chaque méthode, statuts, arbre de décision |
| `www/` | `index.html`, `engine.js`, `style.css` |
| `build/icones/` | Pictogrammes des données à fournir, de [Lucide](https://lucide.dev/), licence ISC (`LICENSE`), insérés en SVG inline par le build |
| `www/icons/` | Icônes d'accès au modèle `poids.png` et `modele-ferme.png` (Flaticon), et icônes de coût `calcul.png` (processeur, recoloré en vert `#23A055`) et `duree.png` (horloge, recolorée en bleu `#2F6FDB`), de [Flaticon](https://www.flaticon.com/) : licence gratuite avec attribution, créditée dans la page Refs (`content/refs/ressources/`) |

## Fiche de méthode

Front matter :

```yaml
title: PEFT (LoRA, QLoRA...)
id: peft
dossier: 03-peft            # méthode du dépôt présentée : repères et « Essayer dans le dépôt » générés
aka: Parameter-Efficient Fine-Tuning
schema: {...}               # schéma d'intégration, voir la docstring de build/schema.py
references:                 # libellé → URL, rendu en section « Références »
  "Hu et al., LoRA (2021)": https://arxiv.org/abs/2106.09685
```

Corps : un paragraphe d'introduction (il sert aussi de description pour le référencement), puis `## Principe`, `## Les variantes` si besoin, `## Quand l'utiliser` (avec un paragraphe **À éviter**), `## Pièges`. Les repères (catégorie, modèles, matériel), le statut des exemples, leurs résultats et les commandes viennent de `exemples.yaml` : ne pas les écrire à la main.

## Statuts

`[!DEMONTRE]`, `[!NONPROUVE]`, `[!NONTESTE]` sont rendus en badges avec infobulle ; en tête d'une puce ou d'une ligne de tableau, `[!NONTESTE]` grise l'élément. Ils servent dans les sections générées et peuvent servir dans le texte.

## Arbre de décision

Front matter `arbre:` (format dans la docstring de `build/catalogue.py`) : une question par colonne, des méthodes empilées de la moins chère à la plus lourde. Chaque feuille prend la couleur de la catégorie de sa méthode et mène à sa fiche dès qu'elle existe.
