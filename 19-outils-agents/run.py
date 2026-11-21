"""Outils et boucle agentique : le modèle appelle des fonctions (calculatrice, recherche) au lieu de tout faire « de tête ».

Boucle : le modèle répond ou demande un outil -> on exécute l'outil -> on lui renvoie le résultat -> jusqu'à la réponse finale.
Deux protocoles : chat template `tools` de transformers (Qwen, Llama 3.1+, Mistral, Hermes...) ou API compatible OpenAI.
Étapes : mesurer sans outils -> mesurer avec outils (exactitude, nombre d'appels), séparément pour chaque type de question :
calcul (calculatrice), fiche (recherche d'une fiche puis calcul) et questions faciles où aucun outil n'est utile (le modèle
ne doit pas appeler d'outil pour rien).
"""

import ast
import json
import operator
import pathlib
import random
import re
import sys
import urllib.request

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import config, donnees, evaluation, modeles, preuve, rapport

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculer(expression: str) -> str:
    """Évalue une expression arithmétique (nombres, + - * / // % ** et parenthèses).

    Args:
        expression: l'expression à calculer, par exemple "(347 * 29) + 1234"
    """
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("expression non autorisée")
    try:
        r = ev(ast.parse(expression, mode="eval").body)
        return str(int(r) if float(r).is_integer() else round(r, 6))
    except Exception as e:
        return f"erreur : {e}"


def fabriquer_recherche(entites):
    fiches = {e["nom"].lower(): donnees.fiche_entite(e) for e in entites}

    def chercher_fiche(nom: str) -> str:
        """Renvoie la fiche biographique d'un personnage à partir de son nom.

        Args:
            nom: nom complet du personnage, par exemple "Kavel Tormor"
        """
        cle = nom.lower().strip()
        return fiches.get(cle) or next((f for k, f in fiches.items() if cle in k or k in cle), "aucune fiche trouvée")

    return chercher_fiche


TYPES = ["calcul", "fiche", "calcul", "fiche", "facile"]   # 40 % calcul, 40 % fiche, 20 % sans outil utile


def questions(entites, n, graine):
    """Trois types : gros calcul (calculatrice), fait inconnu + calcul (fiche), calcul trivial (aucun outil utile)."""
    rng = random.Random(graine)
    res = []
    for i in range(n):
        type_ = TYPES[i % len(TYPES)]
        if type_ == "calcul":
            a, b, c = rng.randint(100, 999), rng.randint(11, 99), rng.randint(1000, 9999)
            res.append({"type": type_, "question": f"What is {a} * {b} + {c}?", "cible": str(a * b + c)})
        elif type_ == "fiche":
            e = rng.choice(entites)
            res.append({"type": type_, "question": f"How many years passed between the birth of {e['nom']} and the year 2024?",
                        "cible": str(2024 - int(e["annee"]))})
        else:
            a, b = rng.randint(2, 20), rng.randint(2, 20)
            res.append({"type": type_, "question": f"What is {a} + {b}?", "cible": str(a + b)})
    return res


def par_type(qs, sans, avec, appels):
    """Résultat de chaque type de question : sans et avec outils sur les mêmes questions, écart apparié, appels d'outils."""
    res = {}
    for t in dict.fromkeys(q["type"] for q in qs):
        idx = [i for i, q in enumerate(qs) if q["type"] == t]
        s = [evaluation.correct(sans[i], qs[i]["cible"], "nombre") for i in idx]
        a = [evaluation.correct(avec[i], qs[i]["cible"], "nombre") for i in idx]
        res[t] = {"sans_outils": preuve.taux(s), "avec_outils": preuve.taux(a), "avec_moins_sans": preuve.difference_appariee(s, a),
                  "appels_moyens": round(sum(appels[i] for i in idx) / len(idx), 2),
                  "part_avec_appel": round(sum(appels[i] > 0 for i in idx) / len(idx), 4)}
    return res


def executer_outil(outils, nom, arguments) -> str:
    """Un appel mal formé (outil inconnu, arguments invalides) renvoie l'erreur au modèle au lieu d'interrompre la boucle."""
    f = outils.get(nom)
    if f is None:
        return "outil inconnu"
    try:
        return f(**(arguments if isinstance(arguments, dict) else {}))
    except TypeError as e:
        return f"erreur d'arguments : {e}"


def boucle_locale(model, tok, question, outils, cfg):
    messages = [{"role": "system", "content": cfg.systeme}, {"role": "user", "content": question}]
    appels = 0
    for _ in range(cfg.max_tours):
        prompt = tok.apply_chat_template(messages, tools=list(outils.values()), tokenize=False, add_generation_prompt=True)
        sortie = modeles.generer(model, tok, [prompt], max_new_tokens=cfg.max_new_tokens, batch=1)[0]
        demandes = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", sortie, re.S)   # format Hermes / Qwen
        if not demandes:
            return sortie, appels
        tool_calls = []
        for d in demandes:
            try:
                appel = json.loads(d)
            except json.JSONDecodeError:
                continue
            tool_calls.append({"type": "function", "function": {"name": appel.get("name"), "arguments": appel.get("arguments", {})}})
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for tc in tool_calls:
            resultat = executer_outil(outils, tc["function"]["name"], tc["function"]["arguments"])
            messages.append({"role": "tool", "name": tc["function"]["name"], "content": resultat})
            appels += 1
    return sortie, appels


def boucle_api(cfg, question, outils):
    from transformers.utils import get_json_schema

    schemas = [get_json_schema(f) for f in outils.values()]
    messages = [{"role": "system", "content": cfg.systeme}, {"role": "user", "content": question}]
    appels = 0
    for _ in range(cfg.max_tours):
        corps = json.dumps({"model": cfg.modele.nom, "messages": messages, "tools": schemas, "temperature": 0}).encode()
        req = urllib.request.Request(f"{cfg.modele.api_base}/chat/completions", corps,
                                     {"Content-Type": "application/json", "Authorization": "Bearer x"})
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.load(r)["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return msg.get("content") or "", appels
        messages.append(msg)
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"] or "{}"
            args = json.loads(args) if isinstance(args, str) else args
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": executer_outil(outils, tc["function"]["name"], args)})
            appels += 1
    return "", appels


def main():
    cfg = config.charger(__file__)
    kb = donnees.faits_fictifs(cfg.donnees.n_entites, cfg.donnees.graine)
    qs = questions(kb["entites"], cfg.evaluation.n, cfg.graine)
    outils = {"calculer": calculer, "chercher_fiche": fabriquer_recherche(kb["entites"])}

    model = tok = None
    if not cfg.modele.get("api_base"):
        model, tok = modeles.charger_llm(cfg.modele.nom)
        sans = modeles.generer(model, tok, [modeles.prompt_texte(tok, q["question"], cfg.systeme_sans_outils) for q in qs],
                               max_new_tokens=cfg.max_new_tokens)
    else:
        from commun.generateur import Generateur

        g = Generateur.depuis_config(cfg.modele)
        sans = g.chat([[{"role": "system", "content": cfg.systeme_sans_outils}, {"role": "user", "content": q["question"]}] for q in qs],
                      max_new_tokens=cfg.max_new_tokens)

    avec, n_appels = [], []
    for q in qs:
        rep, k = boucle_api(cfg, q["question"], outils) if cfg.modele.get("api_base") else boucle_locale(model, tok, q["question"], outils, cfg)
        avec.append(rep)
        n_appels.append(k)

    score = lambda sorties: preuve.taux([evaluation.correct(s, q["cible"], "nombre") for s, q in zip(sorties, qs)])
    rapport.enregistrer(cfg, sans_outils={**score(sans), "exemple": sans[0].strip()[:200]},
                        avec_outils={**score(avec), "appels_outils_moyens": round(sum(n_appels) / len(qs), 2),
                                     "exemple": avec[0].strip()[:200]},
                        par_type=par_type(qs, sans, avec, n_appels))


if __name__ == "__main__":
    main()
