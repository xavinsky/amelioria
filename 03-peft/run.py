"""Fine-tuning efficace en paramètres (PEFT) : le modèle est gelé, on n'entraîne qu'un petit module ajouté.

Une seule recette pour toutes les variantes, seul l'adaptateur change (`--set peft.methode=...`) :
lora, qlora, dora, rslora, pissa, loha, lokr, ia3, prompt_tuning, prefix_tuning.
Étapes : évaluer -> entraîner l'adaptateur (SFT) -> réévaluer -> sauvegarder l'adaptateur (+ fusion optionnelle).
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import peft

from commun import config, donnees, entrainement, evaluation, modeles, rapport

FAMILLE_LORA = {"lora", "qlora", "dora", "rslora", "pissa"}


def config_peft(p, nom_modele):
    m = p.methode
    if m in FAMILLE_LORA:
        conf = entrainement.config_lora(p)
        conf.use_dora = m == "dora"
        conf.use_rslora = m == "rslora"
        if m == "pissa":
            conf.init_lora_weights = "pissa_niter_4"   # init par SVD des poids : converge plus vite
        return conf
    if m == "loha":
        return peft.LoHaConfig(r=p.r, alpha=p.alpha, target_modules=p.modules, task_type="CAUSAL_LM")
    if m == "lokr":
        return peft.LoKrConfig(r=p.r, alpha=p.alpha, target_modules=p.modules, task_type="CAUSAL_LM")
    if m == "ia3":
        return peft.IA3Config(task_type="CAUSAL_LM")   # modules cibles par défaut connus pour LLaMA, Qwen, Mistral...
    if m == "prompt_tuning":
        return peft.PromptTuningConfig(
            task_type="CAUSAL_LM", num_virtual_tokens=p.tokens_virtuels,
            prompt_tuning_init=peft.PromptTuningInit.TEXT, prompt_tuning_init_text=p.texte_init,
            tokenizer_name_or_path=nom_modele,
        )
    if m == "prefix_tuning":
        return peft.PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=p.tokens_virtuels)
    raise ValueError(f"méthode PEFT inconnue : {m}")


def main():
    cfg = config.charger(__file__)
    p = cfg.peft
    quantif = "4bit" if p.methode == "qlora" else None
    model, tok = modeles.charger_llm(cfg.modele.nom, quantification=quantif)
    if quantif:
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.entrainement.get("gradient_checkpointing", False))

    train = donnees.charger_qa(cfg.donnees, "train")
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    avant = evaluation.evaluer(model, tok, cfg, test)

    lr = cfg.entrainement.lr_par_methode.get(p.methode, cfg.entrainement.lr)
    cfg.entrainement.lr = lr
    if p.methode == "prefix_tuning":
        # Le gradient checkpointing retire le cache dans lequel le prefix tuning injecte ses clés/valeurs.
        cfg.entrainement.gradient_checkpointing = False
    trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, train), cfg.entrainement, cfg.sortie,
                               peft_config=config_peft(p, cfg.modele.nom), max_length=cfg.entrainement.max_length)
    modele_peft = trainer.model
    parametres = modeles.compter_parametres(modele_peft)
    print(f"[peft] {p.methode} : {parametres['entrainables']:,} paramètres entraînés ({parametres['pourcentage_entrainable']} %)")
    apres = evaluation.evaluer(modele_peft, tok, cfg, test)

    infos = {"parametres": parametres, "lr": lr}
    if cfg.get("sauvegarder", True):
        dossier = f"{cfg.sortie}/adaptateur-{p.methode}"
        trainer.save_model(dossier)
        infos["taille_adaptateur_mo"] = modeles.taille_dossier_mo(dossier)
        if p.get("fusionner") and p.methode in FAMILLE_LORA - {"qlora"}:
            # Intègre l'adaptateur dans les poids : plus de surcoût à l'inférence, modèle autonome.
            fusion = modele_peft.merge_and_unload()
            modeles.sauvegarder(fusion, tok, f"{cfg.sortie}/modele-fusionne")
    rapport.enregistrer(cfg, avant=avant, apres=apres, infos=infos)


if __name__ == "__main__":
    main()
