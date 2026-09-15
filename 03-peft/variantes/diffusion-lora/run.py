"""LoRA sur un modèle de DIFFUSION (Stable Diffusion) : apprendre un sujet ou un style à partir de quelques images.

Recette DreamBooth-LoRA : images du sujet + prompt contenant un identifiant rare ("sks"), perte de débruitage
standard, seuls les adaptateurs LoRA de l'UNet (couches d'attention) sont entraînés.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image

from commun import config, modeles, rapport


def images_entrainement(cfg) -> list[Image.Image]:
    d = cfg.donnees
    if d.source == "synthetique":
        # Motif reconnaissable (damier bicolore) : aucun téléchargement, sert au test rapide.
        rng = np.random.default_rng(cfg.graine)
        imgs = []
        for _ in range(d.n_images):
            case = rng.integers(4, 16)
            yy, xx = np.mgrid[: d.resolution, : d.resolution]
            masque = ((yy // case + xx // case) % 2).astype(bool)
            img = np.zeros((d.resolution, d.resolution, 3), np.uint8)
            img[masque], img[~masque] = (230, 40, 40), (30, 30, 200)
            imgs.append(Image.fromarray(img))
        return imgs
    from huggingface_hub import snapshot_download

    dossier = pathlib.Path(snapshot_download(d.depot, repo_type="dataset")) if d.source == "hub" else pathlib.Path(d.dossier)
    fichiers = sorted(f for f in dossier.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    return [Image.open(f).convert("RGB") for f in fichiers[: d.n_images]]


def main():
    cfg = config.charger(__file__)
    dev = modeles.appareil()
    dtype = modeles.dtype_calcul()
    pipe = StableDiffusionPipeline.from_pretrained(cfg.modele.nom, dtype=dtype, safety_checker=None).to(dev)
    unet, vae, text_encoder, tokenizer = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    bruit = DDPMScheduler.from_pretrained(cfg.modele.nom, subfolder="scheduler")
    for m in (unet, vae, text_encoder):
        m.requires_grad_(False)

    def generer(tag):
        g = torch.Generator(dev).manual_seed(cfg.graine)
        imgs = pipe([cfg.donnees.prompt] * cfg.evaluation.n_images, num_inference_steps=cfg.evaluation.pas, generator=g,
                    height=cfg.donnees.resolution, width=cfg.donnees.resolution).images
        for i, im in enumerate(imgs):
            im.save(f"{cfg.sortie}/{tag}-{i}.png")
        return imgs

    generer("avant")

    unet.add_adapter(LoraConfig(r=cfg.peft.r, lora_alpha=cfg.peft.alpha, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params_lora = [p for p in unet.parameters() if p.requires_grad]
    for p in params_lora:
        p.data = p.data.float()   # adaptateurs en fp32, reste du modèle en bf16
    optim = torch.optim.AdamW(params_lora, lr=cfg.entrainement.lr)

    images = images_entrainement(cfg)
    pixels = torch.stack([torch.from_numpy(np.array(im.resize((cfg.donnees.resolution,) * 2))).permute(2, 0, 1) for im in images])
    pixels = (pixels.float() / 127.5 - 1).to(dev, dtype)
    ids = tokenizer([cfg.donnees.prompt], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        latents_ref = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        texte = text_encoder(ids)[0]

    unet.train()
    pertes = []
    for pas in range(cfg.entrainement.max_steps):
        idx = torch.randint(0, len(latents_ref), (cfg.entrainement.batch,))
        latents = latents_ref[idx]
        eps = torch.randn_like(latents)
        t = torch.randint(0, bruit.config.num_train_timesteps, (len(idx),), device=dev)
        bruites = bruit.add_noise(latents, eps, t)
        cible = eps if bruit.config.prediction_type == "epsilon" else bruit.get_velocity(latents, eps, t)
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
            pred = unet(bruites, t, texte.expand(len(idx), -1, -1)).sample
        perte = F.mse_loss(pred.float(), cible.float())
        perte.backward()
        torch.nn.utils.clip_grad_norm_(params_lora, 1.0)
        optim.step()
        optim.zero_grad()
        pertes.append(perte.item())
        if pas % cfg.entrainement.logging_steps == 0:
            print(f"[diffusion-lora] pas {pas} perte {np.mean(pertes[-cfg.entrainement.logging_steps:]):.4f}")

    unet.eval()
    generer("apres")
    if cfg.get("sauvegarder", True):
        couches = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(f"{cfg.sortie}/lora", unet_lora_layers=couches, safe_serialization=True)
    rapport.enregistrer(cfg, entrainement={"perte_debut": round(float(np.mean(pertes[:10])), 4),
                                           "perte_fin": round(float(np.mean(pertes[-10:])), 4)},
                        images=f"{cfg.sortie}/avant-*.png, apres-*.png")


if __name__ == "__main__":
    main()
