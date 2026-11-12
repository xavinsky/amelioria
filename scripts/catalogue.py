"""Catalogue des exemples (`exemples.yaml`) : liste, résolution d'un exemple, vérification des prérequis.

    python3 scripts/catalogue.py liste
    python3 scripts/catalogue.py readme          # régénère la section des méthodes du README
    python3 scripts/catalogue.py chemin 03 [diffusion-lora]
    .venv/bin/python scripts/catalogue.py verifier 03 [variante] [--set cle=val ...] [--api URL] [--complet]

Utilisé par le Makefile ; ne dépend que de pyyaml (et de torch pour la vérification du GPU).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import urllib.request

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
MEMOIRE_REFERENCE_GIO = 11.5
COULEUR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
TEINTES = {"demontre": "32", "non-prouve": "33", "non-teste": "90", "erreur": "31", "titre": "1;34", "gris": "90"}
SOCLE = ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "sentence_transformers", "yaml"]


def teinte(texte: str, cle: str) -> str:
    return f"\033[{TEINTES[cle]}m{texte}\033[0m" if COULEUR else texte


def charger() -> dict:
    return yaml.safe_load((RACINE / "exemples.yaml").read_text(encoding="utf-8"))


def resoudre(cat: dict, ex: str, var: str = "") -> dict:
    """Trouve l'exemple désigné par un numéro (« 03 »), un nom de dossier ou un mot du nom, plus une variante."""
    ex = ex.strip().rstrip("/")
    if not ex:
        sortir("préciser l'exemple, ex. EX=03 (voir : make liste)")
    methodes = [m["dossier"] for m in cat["methodes"]]
    trouves = [d for d in methodes if d == ex or d.startswith(ex.zfill(2) + "-")] or [d for d in methodes if ex in d]
    if len(trouves) != 1:
        sortir(f"exemple « {ex} » {'ambigu' if trouves else 'inconnu'} (voir : make liste)")
    chemin = trouves[0] + (f"/variantes/{var.strip()}" if var.strip() else "")
    for e in cat["exemples"]:
        if e["chemin"] == chemin:
            return e
    variantes = [e["chemin"].split("/")[-1] for e in cat["exemples"] if e["chemin"].startswith(trouves[0] + "/")]
    sortir(f"variante « {var} » inconnue pour {trouves[0]} ; variantes : {', '.join(variantes) or 'aucune'}")


def sortir(message: str):
    print(teinte("erreur : ", "erreur") + message, file=sys.stderr)
    sys.exit(2)


def liste(cat: dict):
    statuts = cat["statuts"]
    print(f"Statuts : {teinte('● ' + statuts['demontre'], 'demontre')} : gain mesuré et net · "
          f"{teinte('● ' + statuts['non-prouve'], 'non-prouve')} : s'exécute, gain pas encore démontré · "
          f"{teinte('● ' + statuts['non-teste'], 'non-teste')} : jamais exécuté (versions infra/)")
    print(f"Mesures : configuration complète sur {cat['materiel_reference']}.\n")
    for c in cat["categories"]:
        print(teinte(c["titre"].upper(), "titre") + teinte(f"  {c['resume']}", "gris"))
        for m in (m for m in cat["methodes"] if m["categorie"] == c["id"]):
            num = m["dossier"][:2]
            print(f"  {num}  {m['titre']}")
            for e in (e for e in cat["exemples"] if e["chemin"].split("/")[0] == m["dossier"]):
                variante = e["chemin"].split("/variantes/")[1] if "/variantes/" in e["chemin"] else ""
                commande = f"make test EX={num}" + (f" VAR={variante}" if variante else "")
                etiquette = teinte(f"● {statuts[e['statut']]:<10}", e["statut"])
                print(f"      {etiquette} {commande:<44} {e['resultat']}")
                if e.get("reserve"):
                    print(" " * 63 + teinte(f"↳ {e['reserve']}", "gris"))
            if (RACINE / m["dossier"] / "infra").is_dir():
                print(f"      {teinte('● ' + statuts['non-teste'], 'non-teste'):<10}  "
                      + teinte(f"{m['dossier']}/infra/ (grosse infra)", "gris"))
        print()


PASTILLES = {"demontre": "🟢", "non-prouve": "🟡", "non-teste": "⚪"}
DEBUT, FIN = "<!-- methodes:debut (généré par make readme depuis exemples.yaml) -->", "<!-- methodes:fin -->"


def section_readme(cat: dict) -> str:
    """Tableaux des méthodes par catégorie, avec le statut de chaque exemple."""
    s = cat["statuts"]
    lignes = [DEBUT, "",
              f"Statut de chaque exemple : 🟢 {s['demontre']} (gain mesuré et net) · 🟡 {s['non-prouve']} (s'exécute, "
              f"gain pas encore démontré) · ⚪ {s['non-teste']} (versions `infra/`, jamais exécutées). Résultats et "
              "réserves : `make liste`.",
              "",
              "Colonne « Local 12 Go » : ✅ faisable sur un GPU grand public ; ⚠️ seulement sur de petits modèles ; "
              "☁️ les vrais cas d'usage demandent une grosse infra (exemple fourni dans `infra/`)."]
    for i, c in enumerate(cat["categories"]):
        lignes += ["", f"### {chr(65 + i)}. {c['titre']}", "", c["resume"], "",
                   "| # | Méthode | Regroupe | Modèles concernés | Local 12 Go | Exemples |", "|---|---|---|---|---|---|"]
        for m in (m for m in cat["methodes"] if m["categorie"] == c["id"]):
            exemples = [e for e in cat["exemples"] if e["chemin"].split("/")[0] == m["dossier"]]
            etats = " · ".join(PASTILLES[e["statut"]] + " " + (e["chemin"].split("/variantes/")[1]
                               if "/variantes/" in e["chemin"] else "principal") for e in exemples)
            lignes.append(f"| [{m['dossier'][:2]}]({m['dossier']}) | {m['titre']} | {m['regroupe']} | {m['modeles']} "
                          f"| {m['local']} | {etats} |")
    return "\n".join(lignes + ["", FIN])


def readme(cat: dict):
    chemin = RACINE / "README.md"
    texte = chemin.read_text(encoding="utf-8")
    if DEBUT not in texte or FIN not in texte:
        sortir(f"marqueurs absents de README.md : {DEBUT} … {FIN}")
    avant, reste = texte.split(DEBUT, 1)
    chemin.write_text(avant + section_readme(cat) + reste.split(FIN, 1)[1], encoding="utf-8")
    print("README.md : section des méthodes régénérée")


def extras_requis(prerequis: dict, sets: list[str]) -> list[str]:
    modules = list(prerequis.get("extras", []))
    for condition, mods in prerequis.get("extras_selon", {}).items():
        if condition in sets:
            modules += mods
    return modules


def verifier(cat: dict, e: dict, sets: list[str], api: str, complet: bool) -> int:
    pre = e.get("prerequis", {})
    erreurs, alertes = [], []

    manquants = [m for m in SOCLE if importlib.util.find_spec(m) is None]
    if manquants:
        erreurs.append(f"socle incomplet ({', '.join(manquants)}) : make install")

    for module in extras_requis(pre, sets):
        if module == "llmcompressor":
            if not (RACINE / ".venv-llmcompressor/bin/python").exists():
                erreurs.append("GPTQ/AWQ demandent le venv dédié : make install-gptq")
        elif importlib.util.find_spec(module) is None:
            erreurs.append(f"module {module} absent : make install EXTRAS=1")

    if "llamacpp" in pre.get("outils", []):
        cfg = yaml.safe_load((RACINE / e["chemin"] / "config.yaml").read_text(encoding="utf-8"))
        lcpp = pathlib.Path(next((s.split("=", 1)[1] for s in sets if s.startswith("dossier_llamacpp=")),
                                 cfg["dossier_llamacpp"])).expanduser()
        if not (lcpp / "convert_hf_to_gguf.py").exists() or not (lcpp / "build/bin/llama-quantize").exists():
            erreurs.append(f"llama.cpp compilé introuvable dans {lcpp} : make install-llamacpp "
                           "(ou SET=\"dossier_llamacpp=/chemin\")")
        if importlib.util.find_spec("gguf") is None:
            erreurs.append(f"module gguf absent : .venv/bin/pip install {lcpp}/gguf-py")

    if api:
        try:
            with urllib.request.urlopen(f"{api.rstrip('/')}/models", timeout=5) as r:
                noms = [m.get("id") for m in json.load(r).get("data", [])]
            print(f"[verifier] API {api} joignable ({len(noms)} modèles servis)")
        except Exception as exc:  # noqa: BLE001 - tout échec de connexion se traite pareil
            erreurs.append(f"API {api} injoignable ({exc.__class__.__name__}) : lancer le serveur (ex. ollama serve)")
    elif pre.get("api"):
        alertes.append("le modèle sera chargé localement ; API=http://localhost:11434/v1 MODELE=qwen3:8b pour passer par Ollama")

    gpu = pre.get("gpu", "conseille")
    if importlib.util.find_spec("torch") is not None and not (api and gpu != "requis"):
        import torch

        if torch.cuda.is_available():
            nom = torch.cuda.get_device_name(0)
            gio = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"[verifier] GPU : {nom}, {gio:.1f} Gio")
            if complet and gpu == "requis" and gio < MEMOIRE_REFERENCE_GIO:
                alertes.append(f"configuration complète calibrée pour {MEMOIRE_REFERENCE_GIO} Gio : risque de manque de "
                               "mémoire, essayer d'abord make test")
        elif gpu == "requis":
            erreurs.append("cet exemple demande un GPU CUDA (torch.cuda.is_available() est faux)")
        elif gpu == "conseille":
            alertes.append("pas de GPU CUDA : exécution sur CPU, lente")

    for a in alertes:
        print(teinte("[verifier] attention : ", "non-prouve") + a)
    for err in erreurs:
        print(teinte("[verifier] manque : ", "erreur") + err)
    if erreurs:
        return 1
    statut = cat["statuts"][e["statut"]]
    print(f"[verifier] {e['chemin']} : prêt — statut {teinte(statut, e['statut'])} ({e['resultat']})")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("commande", choices=["liste", "chemin", "verifier", "readme"])
    p.add_argument("ex", nargs="?", default="")
    p.add_argument("var", nargs="?", default="")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--api", default="")
    p.add_argument("--complet", action="store_true")
    a = p.parse_args()
    cat = charger()
    if a.commande == "liste":
        liste(cat)
    elif a.commande == "readme":
        readme(cat)
    elif a.commande == "chemin":
        print(resoudre(cat, a.ex, a.var)["chemin"])
    else:
        sys.exit(verifier(cat, resoudre(cat, a.ex, a.var), a.set, a.api, a.complet))


if __name__ == "__main__":
    main()
