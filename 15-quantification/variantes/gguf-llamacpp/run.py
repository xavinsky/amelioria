"""Export GGUF + quantification llama.cpp (Q8_0, Q5_K_M, Q4_K_M, IQ3...) avec matrice d'importance (imatrix).

Format utilisé par llama.cpp, Ollama, LM Studio : inférence CPU/GPU sur poste local.
Étapes : convertir en GGUF f16 -> calculer l'imatrix sur un texte de calibration -> quantifier chaque type ->
mesurer taille et perplexité (llama-perplexity) -> Modelfile Ollama.

Prérequis : un clone compilé de llama.cpp (`dossier_llamacpp`), voir README.
"""

import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from huggingface_hub import snapshot_download

from commun import config, donnees, rapport


def executer(cmd):
    print("[gguf] $", " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True)


def main():
    cfg = config.charger(__file__)
    lcpp = pathlib.Path(cfg.dossier_llamacpp).expanduser()
    bin_ = lcpp / "build" / "bin"
    sortie = pathlib.Path(cfg.sortie)
    local = pathlib.Path(cfg.modele.nom).expanduser()
    source = local if local.is_dir() else snapshot_download(cfg.modele.nom)
    f16 = sortie / "modele-f16.gguf"
    executer([sys.executable, lcpp / "convert_hf_to_gguf.py", source, "--outfile", f16, "--outtype", "f16"])

    calibration = sortie / "calibration.txt"
    calibration.write_text("\n\n".join(donnees.textes_generaux(cfg.calibration.n, graine=7)), encoding="utf-8")
    test = sortie / "test.txt"
    test.write_text("\n\n".join(donnees.textes_generaux(cfg.evaluation.n_textes, graine=123, split="test")), encoding="utf-8")
    imatrix = sortie / "imatrix.gguf"
    executer([bin_ / "llama-imatrix", "-m", f16, "-f", calibration, "-o", imatrix, "-ngl", cfg.couches_gpu, "-c", 512])

    def perplexite(gguf):
        sortie_ppl = executer([bin_ / "llama-perplexity", "-m", gguf, "-f", test, "-ngl", cfg.couches_gpu, "-c", 512])
        m = re.search(r"Final estimate: PPL = ([\d.]+)", sortie_ppl.stdout + sortie_ppl.stderr)
        return float(m.group(1)) if m else None

    res = {"f16": {"taille_Mo": round(f16.stat().st_size / 2**20, 1), "perplexite": perplexite(f16)}}
    for q in cfg.types:
        cible = sortie / f"modele-{q}.gguf"
        executer([bin_ / "llama-quantize", "--imatrix", imatrix, f16, cible, q])
        res[q] = {"taille_Mo": round(cible.stat().st_size / 2**20, 1), "perplexite": perplexite(cible)}

    (sortie / "Modelfile").write_text(f"FROM ./modele-{cfg.types[0]}.gguf\n", encoding="utf-8")
    print(f"[gguf] ollama create mon-modele -f {sortie / 'Modelfile'}")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
