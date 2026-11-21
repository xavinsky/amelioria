# Schémas des 20 méthodes

Chaque schéma part du transformer décrit dans [transformers.md](transformers.md) et montre ce que la méthode ajoute, entraîne ou modifie. Le transformer y est simplifié en trois étages : embeddings, blocs (attention + MLP), tête de sortie.

## Légende

```mermaid
flowchart LR
    g["Inchangé ou gelé"]:::gele
    e["Entraîné"]:::entraine
    a["Ajouté par la méthode"]:::ajoute
    m["Modifié sans entraînement"]:::modifie
    ext["Hors du modèle : données, juge, outil, programme"]:::externe
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

## Où agit chaque méthode

Les 20 méthodes se rangent selon l'endroit du pipeline qu'elles changent : les données d'entraînement, la fonction de perte (loss), les poids (tous ou une partie), l'architecture, les poids sans entraînement, ou seulement l'entrée et la sortie du modèle à l'inférence.

```mermaid
flowchart TB
    subgraph donnees["1. Données d'entraînement"]
        direction LR
        d07["07 données synthétiques"] ~~~ d06s["06 distillation de séquences"]
    end
    subgraph perte["2. Objectif d'entraînement"]
        direction LR
        p01["01 prochain token<br/>sur le domaine"] ~~~ p02["02 réponse attendue<br/>(SFT)"] ~~~ p04["04 préférences"] ~~~ p05["05 récompense (RL)"] ~~~ p06["06 imiter un enseignant"] ~~~ p10["10 oublier"]
    end
    subgraph poids["3. Quels poids apprennent"]
        direction LR
        w03["03 adaptateur ajouté (PEFT)"] ~~~ w13["13 nouveaux embeddings"]
    end
    subgraph archi["4. Architecture"]
        direction LR
        a12["12 encodage des positions"] ~~~ a13["13 vocabulaire"] ~~~ a14["14 couches, experts"] ~~~ a16["16 élagage"]
    end
    subgraph sansentr["5. Poids modifiés sans entraînement"]
        direction LR
        s08["08 fusion"] ~~~ s09["09 édition d'un fait"] ~~~ s11["11 pilotage des activations"] ~~~ s15["15 quantification"]
    end
    subgraph inference["6. Inférence seulement"]
        direction LR
        i17["17 prompt"] ~~~ i18["18 contexte récupéré (RAG)"] ~~~ i19["19 outils"] ~~~ i20["20 tentatives + sélection"]
    end
    donnees --> perte --> poids --> archi --> sansentr --> inference
```

---

<a id="m01"></a>
## 01 — Pré-entraînement continu

Même objectif que le pré-entraînement initial (prédire le token suivant), sur du texte brut du domaine, mélangé à du texte général (replay) pour ne pas oublier le reste. Tous les poids apprennent.

```mermaid
flowchart LR
    dom["Corpus du domaine<br/>fiches de personnages fictifs,<br/>3 formulations par fait"]:::externe
    rep["Replay : texte général<br/>wikitext, 20 % des caractères,<br/>jamais répété"]:::ajoute
    blocs["Concaténation en blocs<br/>de 256 tokens"]:::externe
    subgraph modele["LLM de base : tous les poids apprennent"]
        emb["Embeddings"]:::entraine --> blk["Blocs × N"]:::entraine --> head["lm_head"]:::entraine
    end
    perte["Perte : prochain token<br/>sur tout le texte"]:::externe
    dom --> blocs
    rep --> blocs
    blocs --> emb
    head --> perte
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

### Variante `mlm-encodeur`

L'encodeur (RoBERTa) voit toute la phrase ; 15 % des tokens sont masqués et il doit les retrouver.

```mermaid
flowchart LR
    txt["« Kavel Tormor is a [MASK]<br/>born in [MASK]ia »"]:::ajoute
    subgraph enc["Encodeur (attention bidirectionnelle)"]
        e1["Embeddings"]:::entraine --> e2["Blocs × N"]:::entraine --> e3["Tête MLM"]:::entraine
    end
    perte["Perte sur les positions masquées<br/>« botanist », « Obbru »"]:::externe
    txt --> e1
    e3 --> perte
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m02"></a>
## 02 — Fine-tuning supervisé (SFT)

Des paires « entrée → sortie idéale », formatées avec le chat template. La perte ne porte que sur les tokens de la réponse : le modèle apprend à répondre, pas à recopier la question.

```mermaid
flowchart LR
    paire["Question + réponse attendue<br/>« What is (75 - 99) * 3? »<br/>« 75 - 99 = -24. -24 * 3 = -72. Answer: -72 »"]:::externe
    tpl["Chat template<br/>system / user / assistant"]:::gele
    subgraph modele["LLM instruct"]
        emb["Embeddings"]:::entraine --> blk["Blocs × N"]:::entraine --> head["lm_head"]:::entraine
    end
    masque["Masque de perte<br/>prompt ignoré, réponse comptée"]:::ajoute
    perte["Entropie croisée<br/>sur la réponse"]:::externe
    paire --> tpl --> emb
    head --> masque --> perte
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

### Variantes : autres types de modèles

Même principe, seules l'entrée, l'architecture et la tête changent.

```mermaid
flowchart TB
    subgraph cls["classification-encodeur : sentiment SST-2"]
        direction LR
        c0["Phrase"]:::externe --> c1["Encodeur ModernBERT"]:::entraine --> c2["Tête de classification<br/>ajoutée, 2 classes"]:::ajoute --> c3["Entropie croisée<br/>sur la classe"]:::externe
    end
    subgraph vis["vision : CIFAR-10"]
        direction LR
        v0["Image 224×224"]:::externe --> v1["Patchs 16×16 → tokens"]:::gele --> v2["Encodeur ViT"]:::entraine --> v3["Tête 10 classes<br/>ajoutée"]:::ajoute
    end
    subgraph s2s["seq2seq : T5"]
        direction LR
        t0["Question"]:::externe --> t1["Encodeur"]:::entraine --> t2["Décodeur<br/>+ attention croisée"]:::entraine --> t3["Réponse"]:::externe
    end
    subgraph aud["audio-whisper : transcription française"]
        direction LR
        a0["Audio 16 kHz"]:::externe --> a1["Spectrogramme log-Mel"]:::gele --> a2["Encodeur audio"]:::entraine --> a3["Décodeur texte<br/>langue et tâche imposées"]:::entraine --> a4["Transcription<br/>mesure : WER"]:::externe
    end
    cls ~~~ vis ~~~ s2s ~~~ aud
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m03"></a>
## 03 — PEFT (LoRA, QLoRA, DoRA, IA³, prompt et prefix tuning)

Le modèle est gelé. On ajoute un petit module entraînable, ici sur chaque couche linéaire des blocs. Mêmes données et même perte que le SFT (DATA02).

```mermaid
flowchart TB
    subgraph modele["LLM gelé"]
        emb["Embeddings"]:::gele --> blk
        subgraph blk["Chaque couche linéaire des blocs : q, k, v, o, gate, up, down"]
            entree["entrée x"]:::gele
            W["W (d × d)<br/>gelé ; en QLoRA : stocké en NF4 4 bits"]:::gele
            A["A (r × d)"]:::ajoute
            B["B (d × r), initialisée à 0"]:::ajoute
            somme(("+"))
            y["sortie : W·x + (α/r)·B·A·x"]:::gele
            entree --> W --> somme
            entree --> A --> B --> somme
            somme --> y
        end
        blk --> head["lm_head"]:::gele
    end
    remarque["Seuls A et B apprennent : ~1 % des paramètres,<br/>adaptateur de quelques dizaines de Mo"]:::entraine
    B -.-> remarque
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
```

Les autres méthodes (`--set peft.methode=...`) changent le module ajouté :

```mermaid
flowchart LR
    subgraph lora["Famille LoRA"]
        l1["dora : magnitude apprise +<br/>LoRA sur la direction de W"]:::ajoute
        l2["rslora : échelle α/√r"]:::ajoute
        l3["pissa : A, B initialisés<br/>par SVD de W"]:::ajoute
        l4["loha / lokr : produit de Hadamard<br/>ou de Kronecker"]:::ajoute
    end
    subgraph ia3["ia3"]
        i1["Vecteurs l_k, l_v, l_ff<br/>qui multiplient les activations"]:::ajoute
    end
    subgraph prompt["prompt_tuning"]
        p1["20 tokens virtuels appris"]:::ajoute --> p2["placés avant les<br/>embeddings du prompt"]:::gele
    end
    subgraph prefix["prefix_tuning"]
        f1["Clés et valeurs virtuelles apprises"]:::ajoute --> f2["injectées dans l'attention<br/>de chaque couche"]:::gele
    end
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
```

### Variante `diffusion-lora` (DreamBooth-LoRA)

Quelques photos d'un sujet et un identifiant rare (« sks ») dans le prompt. LoRA sur l'attention de l'UNet ; perte de débruitage.

```mermaid
flowchart LR
    img["5 photos du sujet"]:::externe --> vae["VAE : image → latents"]:::gele
    vae --> bruit["+ bruit ε au pas t<br/>(DDPMScheduler)"]:::externe
    txt["« a photo of sks dog »"]:::externe --> clip["Encodeur de texte CLIP"]:::gele
    subgraph unet["UNet gelé"]
        att["Attention : to_q, to_k, to_v, to_out"]:::gele
        lora["+ LoRA r = 8"]:::ajoute
        att --- lora
    end
    bruit --> unet
    clip -- "attention croisée" --> unet
    unet --> pred["bruit prédit ε̂"]:::gele --> perte["Perte : ‖ε̂ − ε‖²"]:::externe
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m04"></a>
## 04 — Optimisation des préférences (DPO et famille)

Pour chaque prompt, une réponse préférée (chosen) et une réponse rejetée (rejected). Le modèle entraîné (politique) doit augmenter l'écart de log-probabilité entre les deux, par rapport à un modèle de référence gelé. Pas de modèle de récompense, pas de génération pendant l'entraînement.

```mermaid
flowchart LR
    donnees["Prompt<br/>+ réponse préférée (juste)<br/>+ réponse rejetée (erreur de calcul)"]:::externe
    subgraph pol["Politique"]
        p1["LLM gelé"]:::gele
        p2["+ LoRA<br/>seul entraîné"]:::ajoute
        p1 --- p2
    end
    ref["Référence :<br/>le même LLM sans adaptateur"]:::gele
    lp["log π(préférée), log π(rejetée)"]:::entraine
    lr["log π_ref(préférée), log π_ref(rejetée)"]:::gele
    perte["Perte DPO<br/>−log σ(β · [(log π − log π_ref)préférée<br/>− (log π − log π_ref)rejetée])"]:::externe
    donnees --> pol --> lp --> perte
    donnees --> ref --> lr --> perte
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

Les autres algorithmes (`--set algo=...`) changent la perte : IPO (quadratique), rDPO (étiquettes bruitées), KTO (réponses non appariées, bonne ou mauvaise), ORPO, SimPO et CPO (sans référence).

### Variante `dpo-iteratif` (on-policy)

Le modèle génère lui-même les candidats, un juge les classe, et on recommence avec le modèle amélioré.

```mermaid
flowchart LR
    q["Questions du tour"]:::externe --> gen["Le modèle génère<br/>k = 8 réponses"]:::gele
    gen --> juge["Juge : vérificateur exact<br/>(ou modèle de récompense, LLM juge)"]:::externe
    juge --> paires["Meilleure = préférée<br/>pire = rejetée"]:::ajoute
    paires --> dpo["DPO"]:::entraine
    dpo --> suivant["Tour suivant : le modèle mis à jour<br/>génère les nouvelles paires"]:::gele
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m05"></a>
## 05 — RL en ligne (GRPO, DAPO, RLOO...)

Le modèle génère un groupe de réponses par prompt, une récompense les note, et on renforce celles qui font mieux que la moyenne du groupe. Les données sont produites par le modèle lui-même pendant l'entraînement.

```mermaid
flowchart LR
    prompts["Lot de prompts"]:::externe
    m["Politique : LLM<br/>(tous les poids)"]:::entraine
    groupe["G = 8 réponses par prompt<br/>(température 1)"]:::ajoute
    subgraph recomp["Récompense : recompense.source"]
        direction TB
        r1["regles (RLVR) : réponse exacte + format"]:::externe
        r2["modele (RLHF) : modèle de récompense"]:::externe
        r3["juge (RLAIF) : LLM qui note"]:::externe
    end
    av["Avantage = récompense<br/>− moyenne du groupe"]:::ajoute
    maj["Mise à jour de la politique<br/>policy gradient, ratio d'importance clippé<br/>(+ KL vers la référence si β > 0)"]:::entraine
    prompts --> m --> groupe --> recomp --> av --> maj
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

Les algorithmes (`--set algo=...`) diffèrent par la normalisation de l'avantage et le clipping : GRPO, Dr.GRPO, DAPO (clip asymétrique), GSPO (ratio par séquence), CISPO, RLOO (baseline leave-one-out).

### Variante `modele-recompense`

Le LLM reçoit une tête (head) scalaire ; il apprend à donner un score plus haut à la réponse préférée (Bradley-Terry). Le modèle obtenu sert de `recompense.source=modele`.

```mermaid
flowchart LR
    p["Prompt + préférée<br/>Prompt + rejetée"]:::externe
    subgraph rm["Modèle de récompense"]
        l["LLM"]:::entraine --> t["Tête scalaire<br/>d → 1"]:::ajoute
    end
    s["score(préférée), score(rejetée)"]:::entraine
    perte["Perte : −log σ(score préférée − score rejetée)"]:::externe
    p --> rm --> s --> perte
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m06"></a>
## 06 — Distillation

Un gros modèle enseignant (gelé) transmet son comportement à un petit modèle élève. Trois modes, du plus simple au plus fidèle.

```mermaid
flowchart LR
    q["Questions"]:::externe --> ens["Enseignant 1,5B<br/>gelé"]:::gele
    subgraph logits["mode logits"]
        direction LR
        e2["Distribution de l'enseignant<br/>sur chaque token de la réponse"]:::gele --> e4["Perte : α·KL(enseignant ‖ élève) à T<br/>+ (1−α)·entropie croisée"]:::externe
        e3["Élève 0,5B<br/>sur la même entrée"]:::entraine --> e4
    end
    subgraph sequence["mode sequence"]
        direction LR
        s1["L'enseignant rédige les réponses"]:::gele --> s2["Filtre : réponses justes"]:::externe --> s3["SFT de l'élève"]:::entraine
    end
    subgraph onpolicy["mode on_policy (GKD)"]
        direction LR
        o1["L'élève génère"]:::entraine --> o2["L'enseignant donne sa distribution<br/>sur ces mêmes tokens"]:::gele --> o3["Perte JSD généralisée"]:::externe
    end
    ens --> logits
    ens --> sequence
    ens --> onpolicy
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

### Variante `encodeur-classification`

L'enseignant annote des textes sans étiquette : l'élève apprend de ses probabilités de classes, en plus des quelques exemples étiquetés.

```mermaid
flowchart LR
    lab["2 000 phrases étiquetées"]:::externe
    unl["18 000 phrases sans étiquette<br/>(jeu de transfert)"]:::ajoute
    ens["Enseignant RoBERTa-base SST-2<br/>gelé, 125 M"]:::gele
    ele["Élève BERT-mini<br/>11 M"]:::entraine
    perte["Perte : α·KL(probabilités de l'enseignant à T)<br/>+ (1−α)·entropie croisée si étiquette"]:::externe
    lab --> ele
    unl --> ele
    lab --> ens
    unl --> ens
    ens --> perte
    ele --> perte
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m07"></a>
## 07 — Données synthétiques et auto-amélioration

Le modèle produit ses propres données d'entraînement ; un filtre ne garde que les bonnes ; SFT ; on recommence. Seules les données changent, la méthode d'entraînement est le SFT de 02.

```mermaid
flowchart LR
    q["Questions du tour"]:::externe
    gen["Le modèle génère<br/>k = 4 solutions par question"]:::gele
    subgraph strategie["strategie"]
        direction TB
        s1["rejet : solutions du modèle"]:::ajoute
        s2["self_instruct : nouvelles questions<br/>inventées, puis résolues"]:::ajoute
        s3["critique_revision : réponse → critique<br/>selon un principe → réécriture"]:::ajoute
    end
    filtre["Filtre : réponse vérifiée juste"]:::externe
    jeu["Jeu synthétique<br/>donnees-tour-N.jsonl"]:::ajoute
    sft["SFT"]:::entraine
    suivant["Tour suivant : le modèle amélioré<br/>devient le générateur"]:::gele
    q --> gen --> strategie --> filtre --> jeu --> sft --> suivant
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m08"></a>
## 08 — Fusion de modèles

Aucun entraînement. Les fine-tunes d'une même base diffèrent d'elle par un « vecteur de tâche » τ = θ_fine-tune − θ_base. On combine ces vecteurs tenseur par tenseur.

```mermaid
flowchart TB
    subgraph sources["Modèles d'une même base"]
        direction LR
        base["θ_base<br/>Qwen2.5-1.5B"]:::gele
        ft1["θ₁ : fine-tune 1"]:::gele
        ft2["θ₂ : fine-tune 2"]:::gele
    end
    subgraph vecteurs["Vecteurs de tâche"]
        direction LR
        tau1["τ₁ = θ₁ − θ_base"]:::ajoute
        tau2["τ₂ = θ₂ − θ_base"]:::ajoute
    end
    subgraph methode["methode : combinaison tenseur par tenseur"]
        direction LR
        t1["ties : garder les coordonnées fortes,<br/>élire un signe, moyenner les accords"]:::modifie
        t2["dare : abandon aléatoire<br/>+ remise à l'échelle 1/densité"]:::modifie
        t3["arithmetique_taches : Σ λᵢ·τᵢ"]:::modifie
    end
    fus["θ = θ_base + τ fusionné<br/>aucun entraînement"]:::modifie
    sources --> vecteurs --> methode --> fus
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m09"></a>
## 09 — Édition de connaissances (ROME)

Un fait « sujet → objet » est modifié par une mise à jour de rang 1 d'une seule matrice : `down_proj` du MLP d'une couche intermédiaire, qui agit comme une mémoire clé → valeur.

```mermaid
flowchart TB
    prompt["« The Eiffel Tower is located in the city of »"]:::externe
    subgraph modele["LLM"]
        emb["Embeddings"]:::gele --> av["Blocs 0 à l−1"]:::gele
        subgraph couche["Bloc l (20 % de la profondeur)"]
            att["Attention"]:::gele --> gu["gate_proj, up_proj"]:::gele --> k["k* = entrée de down_proj<br/>au dernier token du sujet"]:::ajoute --> W["down_proj : W"]:::modifie
        end
        av --> att
        W --> ap["Blocs l+1 à N"]:::gele --> head["lm_head → « Rome »"]:::gele
    end
    v["v* optimisé pour que le modèle<br/>prédise « Rome » (régularisé par KL)"]:::ajoute
    C["Covariance C des clés<br/>sur du texte générique"]:::externe
    maj["W ← W + (v* − W·k*)(C⁻¹k*)ᵀ / ((C⁻¹k*)ᵀk*)"]:::modifie
    prompt --> emb
    k --> maj
    v --> maj
    C --> maj
    maj --> W
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

`ft_contraint` (FT-L) fine-tune la même matrice avec une borne sur la variation des poids.

---

<a id="m10"></a>
## 10 — Désapprentissage

Deux ensembles : ce qu'il faut oublier et ce qu'il faut retenir. La perte monte sur le premier et reste basse sur le second.

```mermaid
flowchart LR
    impl["Implantation (SFT) :<br/>le modèle apprend 40 personnages"]:::entraine
    oubli["Ensemble à oublier<br/>4 personnages"]:::externe
    retenu["Ensemble à retenir<br/>36 personnages"]:::externe
    subgraph modele["Modèle"]
        m["LLM"]:::entraine
    end
    ref["Référence gelée<br/>(copie avant désapprentissage)"]:::gele
    po["Perte d'oubli : methode<br/>gradient_ascent : −NLL<br/>npo : −(2/β)·log σ(−β·log(π/π_ref))<br/>idk : SFT vers « je ne sais pas »"]:::ajoute
    pr["Perte de rétention : +NLL"]:::ajoute
    tot["Perte totale"]:::externe
    impl --> m
    oubli --> m --> po --> tot
    oubli --> ref --> po
    retenu --> m --> pr --> tot
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m11"></a>
## 11 — Pilotage par activations

Une direction du flux résiduel représente un concept (ici le sentiment). On l'extrait par différence de moyennes, puis on l'ajoute pendant la génération, ou on la retire des poids.

```mermaid
flowchart TB
    pos["Phrases positives"]:::externe --> hp["moyenne h⁺ à la couche l"]:::gele
    neg["Phrases négatives"]:::externe --> hn["moyenne h⁻ à la couche l"]:::gele
    d["Direction d = h⁺ − h⁻"]:::ajoute
    hp --> d
    hn --> d
    subgraph vecteur["mode vecteur : poids intacts"]
        v1["Blocs 0 à l"]:::gele --> v2["h ← h + α·d<br/>(crochet sur la sortie du bloc l)"]:::ajoute --> v3["Blocs suivants, lm_head"]:::gele
    end
    subgraph ablation["mode ablation : poids modifiés, sans entraînement"]
        a1["o_proj, down_proj, embeddings<br/>W ← (I − d̂·d̂ᵀ)·W"]:::modifie
    end
    d --> vecteur
    d --> ablation
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m12"></a>
## 12 — Extension de la fenêtre de contexte

La position est encodée par RoPE. On change la configuration de RoPE pour couvrir plus de positions, puis un court fine-tuning sur des textes longs apprend au modèle à s'en servir.

```mermaid
flowchart LR
    subgraph modele["LLM (contexte d'origine : 2 048 tokens)"]
        emb["Embeddings"]:::entraine
        subgraph attn["Attention de chaque bloc"]
            qk["q, k"]:::entraine --> rope["RoPE<br/>rope_parameters : yarn, facteur 4<br/>(ou linear, dynamic, abf)"]:::modifie
        end
        emb --> attn --> head["MLP, lm_head"]:::entraine
    end
    longs["Documents longs<br/>8 192 tokens"]:::ajoute
    test["Test de la clé cachée<br/>1 024 à 7 000 tokens"]:::externe
    longs --> emb
    head --> test
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m13"></a>
## 13 — Extension de vocabulaire

Des mots fréquents du domaine, découpés en plusieurs tokens, deviennent des tokens entiers. Leurs embeddings sont initialisés par la moyenne de leurs anciens sous-tokens, puis entraînés en deux phases.

```mermaid
flowchart TB
    tok["Tokenizer<br/>« ĠK | avel | mor » → « Kavelmor » (1 token)"]:::modifie
    subgraph modele["LLM"]
        subgraph emb["Embeddings (entrée et sortie liées)"]
            anciens["151 936 lignes d'origine"]:::gele
            nouveaux["+ 200 lignes<br/>init = moyenne des sous-tokens"]:::ajoute
        end
        blk["Blocs × N"]:::gele
        emb --> blk --> head["lm_head (même matrice)"]:::gele
    end
    p1["Phase 1 : seuls les embeddings apprennent<br/>(lr élevé)"]:::entraine
    p2["Phase 2 : tout le modèle<br/>(lr faible)"]:::entraine
    tok --> emb
    nouveaux -.-> p1 -.-> p2
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
```

---

<a id="m14"></a>
## 14 — Croissance d'architecture

On agrandit un modèle entraîné en réutilisant ses poids, puis on le « cicatrise » par un pré-entraînement continu.

```mermaid
flowchart LR
    origine["Modèle d'origine<br/>couches 0 … n−1"]:::gele
    subgraph variantes["croissance.methode"]
        direction TB
        subgraph profondeur["profondeur (SOLAR)"]
            direction LR
            pa["couches 0 … n−m−1"]:::entraine --> pb["copie des couches m … n−1"]:::ajoute
        end
        subgraph blocs["expansion_blocs (LLaMA Pro)"]
            direction LR
            b1["groupe de k couches<br/>gelé"]:::gele --> b2["copie avec o_proj = 0, down_proj = 0<br/>identité au départ, seule entraînée"]:::ajoute --> b3["groupe suivant<br/>gelé"]:::gele
        end
        subgraph moe["moe (upcycling)"]
            direction LR
            r["Routeur top-2<br/>ajouté"]:::ajoute --> e1["Experts 1 … E<br/>= copies du MLP"]:::ajoute
        end
        profondeur ~~~ blocs ~~~ moe
    end
    cic["Cicatrisation :<br/>prochain token sur texte général"]:::externe
    origine --> variantes --> cic
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m15"></a>
## 15 — Quantification

Les poids des couches linéaires sont stockés sur 8, 4 ou 3 bits, avec une échelle par groupe de poids. Le modèle est plus petit et le décodage, limité par la bande passante mémoire, peut être plus rapide.

```mermaid
flowchart LR
    w["Couche linéaire<br/>W en bf16"]:::gele
    subgraph famille["methode"]
        direction TB
        f1["bnb_nf4, torchao :<br/>arrondi par groupe au chargement"]:::modifie
        f2["gptq, awq : calibration sur 256 textes<br/>corrige les erreurs d'arrondi,<br/>protège les canaux importants"]:::modifie
        f3["qat : fausse quantification<br/>pendant un fine-tuning, puis conversion"]:::entraine
    end
    q["W_q : entiers 4 bits<br/>+ une échelle par groupe de 128 poids"]:::modifie
    reste["Embeddings et lm_head<br/>gardés en bf16"]:::gele
    w --> famille --> q
    q ~~~ reste
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

### Variante `gguf-llamacpp`

```mermaid
flowchart LR
    hf["Modèle Hugging Face"]:::gele --> f16["GGUF f16<br/>convert_hf_to_gguf.py"]:::gele
    cal["Texte de calibration"]:::externe --> imat["Matrice d'importance<br/>llama-imatrix"]:::ajoute
    f16 --> imat
    f16 --> quant["llama-quantize<br/>Q8_0, Q5_K_M, Q4_K_M, IQ3_M"]:::modifie
    imat --> quant
    quant --> run["llama.cpp, Ollama<br/>llama-perplexity"]:::externe
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m16"></a>
## 16 — Élagage et cicatrisation

On retire les poids, neurones ou couches qui contribuent le moins, puis on récupère la qualité par distillation depuis le modèle d'origine.

```mermaid
flowchart TB
    cal["Textes de calibration"]:::externe --> score["Score Wanda = |W_ij| · ‖X_j‖<br/>(ou magnitude, influence des couches,<br/>activation des neurones)"]:::ajoute
    subgraph modele["Modèle élagué (élève)"]
        subgraph lin["Couches linéaires"]
            w["W"]:::entraine --> mask["Masque booléen : 50 % des poids à zéro<br/>(ou 2 poids sur 4 ; ou couches, neurones retirés)"]:::modifie
        end
    end
    score --> mask
    orig["Modèle d'origine<br/>gelé (enseignant)"]:::gele
    texte["Texte général"]:::externe
    perte["Perte : KL(origine ‖ élagué)<br/>masque réappliqué après chaque pas"]:::externe
    texte --> modele --> perte
    texte --> orig --> perte
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m17"></a>
## 17 — Optimisation du prompt

Le modèle ne change pas. On cherche, sur un jeu de validation, la meilleure façon de le solliciter, puis on confirme sur un jeu de test.

```mermaid
flowchart TB
    subgraph prompt["Prompt construit"]
        direction LR
        sys["Instruction système<br/>(instruction, cot, optimisation)"]:::ajoute
        fs["Exemples question → réponse<br/>few_shot : au hasard<br/>few_shot_similaires : plus proches par embeddings"]:::ajoute
        q["Question"]:::externe
    end
    llm["LLM gelé<br/>(local ou API)"]:::gele
    dev["Exactitude<br/>sur la validation"]:::externe
    best["Meilleure stratégie<br/>confirmée sur le test"]:::externe
    prompt --> llm --> dev --> best
    subgraph opro["Boucle OPRO : stratégie optimisation"]
        direction LR
        hist["Instructions essayées<br/>+ leurs scores"]:::externe --> opt["LLM optimiseur :<br/>propose une instruction"]:::gele --> evalo["Évaluée sur la validation,<br/>ajoutée à l'historique"]:::externe
    end
    best ~~~ opro
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef entraine fill:#bfdbfe,stroke:#1d4ed8,color:#0b1b3f
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef modifie fill:#fecaca,stroke:#b91c1c,color:#450a0a
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m18"></a>
## 18 — RAG

Au moment de répondre, on cherche les passages pertinents et on les place dans le prompt. La connaissance est fournie dans le contexte au lieu d'être dans les poids.

```mermaid
flowchart LR
    docs["Base documentaire<br/>50 fiches + 2 000 paragraphes"]:::externe
    subgraph index["Index"]
        bm25["BM25 (mots)"]:::ajoute
        dense["Embeddings (sens)"]:::ajoute
    end
    q["Question"]:::externe
    rrf["Fusion des classements (RRF)<br/>top 20"]:::ajoute
    rr["Reranker cross-encoder<br/>top 3"]:::ajoute
    ctx["Prompt : consigne + 3 passages + question"]:::ajoute
    llm["LLM gelé"]:::gele
    rep["Réponse"]:::externe
    docs --> index
    q --> bm25
    q --> dense
    bm25 --> rrf
    dense --> rrf
    rrf --> rr --> ctx --> llm --> rep
    q --> ctx
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m19"></a>
## 19 — Outils et boucle agentique

Le modèle décrit ce qu'il veut faire (un appel de fonction), un programme l'exécute et lui renvoie le résultat, jusqu'à la réponse finale.

```mermaid
flowchart LR
    q["Question<br/>« How many years between the birth<br/>of Kavel Tormor and 2024? »"]:::externe
    tpl["Chat template + schémas JSON des outils"]:::ajoute
    llm["LLM gelé"]:::gele
    appel{"Appel d'outil ?"}
    exec["Programme : exécute<br/>chercher_fiche(nom) ou calculer(expression)"]:::externe
    res["Message « tool » : résultat"]:::ajoute
    fin["Réponse finale<br/>« ... Answer: N »"]:::externe
    q --> tpl --> llm --> appel
    appel -- "oui : balise tool_call" --> exec --> res --> llm
    appel -- "non" --> fin
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```

---

<a id="m20"></a>
## 20 — Calcul à l'inférence

Plus de calcul par question plutôt que plus d'entraînement : plusieurs réponses, puis une règle pour choisir.

```mermaid
flowchart LR
    q["Question"]:::externe
    llm["LLM gelé"]:::gele
    n["N = 16 réponses<br/>échantillonnées (température 0,7)"]:::ajoute
    subgraph selection["Sélection"]
        s1["vote_majoritaire :<br/>réponse finale la plus fréquente"]:::ajoute
        s2["best_of_n_recompense :<br/>meilleur score d'un modèle de récompense"]:::externe
        s3["best_of_n_vraisemblance :<br/>log-probabilité moyenne la plus haute"]:::ajoute
        s4["oracle_pass_a_N :<br/>au moins une juste (borne haute)"]:::externe
    end
    corr["auto_correction :<br/>le modèle relit et corrige sa réponse"]:::ajoute
    rep["Réponse retenue"]:::externe
    q --> llm --> n --> selection --> rep
    llm --> corr --> rep
    classDef gele fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef ajoute fill:#fed7aa,stroke:#c2410c,color:#431407
    classDef externe fill:#bbf7d0,stroke:#15803d,color:#052e16
```
