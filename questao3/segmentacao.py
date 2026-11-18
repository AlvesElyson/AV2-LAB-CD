#!/usr/bin/env python3
"""
AV2 - PDI | Questão 3: Segmentação (Swedish Leaf)

Uso (na raiz do projeto, com o .venv ativo):
    python questao3/segmentacao.py

Entradas
    questao2/cache_256/   imagens pré-processadas da Questão 2 (256x256, fundo branco)
    questao1/resultados/split_treino_teste.csv   divisão treino/teste da Questão 1

O que o script responde
-----------------------
  1) Vale a pena segmentar?  Compara características SEM segmentação (imagem inteira,
     incluindo o fundo) e COM segmentação (só pixels da folha), com o mesmo proxy de
     classificação da Questão 2 (regressão logística, validação cruzada no TREINO).
     Também mede o quanto a "fração de folha no quadro" sozinha já prevê a classe (atalho).
  2) Qual método?  Compara Otsu (cinza), limiar fixo e k-means (Lab) por concordância (IoU),
     sensibilidade ao limiar e efeito nos descritores de forma.
  3) Limpeza morfológica: remove ruído, preenche pequenos buracos e mantém regiões
     desconexas (folíolos das classes 10 e 14).
  4) Pecíolo: avalia a remoção por abertura morfológica (raios 2, 3 e 5): perda de área por
     classe e efeito na classificação por forma.
  5) Limitações: lista e mostra os piores casos.

Saídas
    questao3/resultados/      tabelas (.csv), figuras (.png), resumo_q3.json, relatorio_q3_trechos.md
    questao3/mascaras_256/               máscaras finais com pecíolo   (0/255)
    questao3/mascaras_256_sem_peciolo/   máscaras com pecíolo removido (0/255)
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)

LIMIAR_FIXO = 200
R_LISTA = [2, 3, 5]
DESLOCAMENTOS = [-20, -10, 10, 20]
DESCRITORES = ["extent", "solidity", "eccentricity", "circularity", "aspecto"]


# ----------------------------------------------------------------------------
# Segmentação
# ----------------------------------------------------------------------------
def carregar_rgb(p) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def otsu_bruto(gray: np.ndarray):
    """Limiar de Otsu; a folha é a classe escura (o fundo foi normalizado para branco)."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t), (gray <= t).astype(np.uint8)


def limpar(mask: np.ndarray, min_abs: int = 40, rel: float = 0.01, fill_frac: float = 0.003) -> np.ndarray:
    """
    Remove componentes pequenos (< max(40 px, 1% do maior)) e preenche buracos internos
    pequenos (< 0,3% do quadro). Componentes grandes e separados são mantidos (folíolos).
    """
    m = mask.astype(np.uint8).copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        lim = max(min_abs, rel * areas.max())
        keep = np.zeros(n, bool)
        keep[1:] = areas >= lim
        m = keep[lab].astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        toca = x == 0 or y == 0 or x + ww == w or y + hh == h
        if not toca and a <= fill_frac * h * w:
            m[lab == i] = 1
    return m


def abrir(mask: np.ndarray, r: int) -> np.ndarray:
    """Abertura morfológica com elemento elíptico de raio r: remove estruturas mais finas que 2r+1 px (pecíolo)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return limpar(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k))


def seg_kmeans(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """Agrupamento de pixels (k=2) no espaço Lab; folha = cluster de menor luminosidade."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centros = cv2.kmeans(lab, 2, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    folha = int(np.argmin(centros[:, 0]))
    return (labels.reshape(img.shape[:2]) == folha).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def toca_borda(m: np.ndarray) -> bool:
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def n_componentes(m: np.ndarray) -> int:
    return int(cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1)


# ----------------------------------------------------------------------------
# Características (mesmas definições da Questão 2, para comparação justa)
# ----------------------------------------------------------------------------
def descritores_forma(mask: np.ndarray):
    if mask.sum() < 50:
        return None
    p = regionprops(mask.astype(np.uint8))[0]
    per = p.perimeter if p.perimeter > 0 else np.nan
    minr, minc, maxr, maxc = p.bbox
    hu = np.sign(p.moments_hu) * np.log10(np.abs(p.moments_hu) + 1e-30)
    return {"extent": float(p.extent), "solidity": float(p.solidity), "eccentricity": float(p.eccentricity),
            "circularity": float(4 * np.pi * p.area / per ** 2) if per == per else np.nan,
            "aspecto": float((maxr - minr) / max(1, (maxc - minc))), "hu": hu.astype(float)}


def vetor_forma(mask: np.ndarray) -> np.ndarray:
    d = descritores_forma(mask)
    if d is None:
        return np.zeros(12)
    return np.nan_to_num(np.array([d[k] for k in DESCRITORES] + list(d["hu"]), float))


def feats_cor(img: np.ndarray, sel: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    v = []
    for c in range(3):
        v += [img[..., c][sel].mean(), img[..., c][sel].std()]
    for c in range(3):
        v += [lab[..., c][sel].mean(), lab[..., c][sel].std()]
    return np.array(v, float)


def lbps(img: np.ndarray):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return [(P, local_binary_pattern(g, P, R, method="uniform")) for P, R in ((8, 1), (16, 2))]


def feats_tex(lbp_list, sel: np.ndarray) -> np.ndarray:
    v = []
    for P, l in lbp_list:
        h, _ = np.histogram(l[sel], bins=P + 2, range=(0, P + 2))
        v += list(h / max(1, h.sum()))
    return np.array(v, float)


# ----------------------------------------------------------------------------
# Auxiliares
# ----------------------------------------------------------------------------
def achar_cache(raiz: Path, arg: str):
    if arg:
        return Path(arg)
    for nome in ("questao2", "questão2"):
        c = raiz / nome / "cache_256"
        if c.is_dir():
            return c
    achados = list(raiz.glob("*2/cache_256"))
    return achados[0] if achados else None


def df_md(df: pd.DataFrame, casas: int = 3) -> str:
    cab = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    linhas = []
    for _, r in df.iterrows():
        cel = [f"{v:.{casas}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        linhas.append("| " + " | ".join(cel) + " |")
    return "\n".join([cab, sep] + linhas)


def contorno(img: np.ndarray, mask: np.ndarray, cor=(255, 0, 0), esp: int = 1) -> np.ndarray:
    out = img.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, cor, esp)
    return out


def ordem_classes(cs):
    return sorted(cs, key=lambda c: (len(str(c)), str(c)))


def avaliar_proxy(blocos: dict, conjuntos: dict, y, grupos, seed):
    """blocos: nome -> matriz (n, d). conjuntos: nome_do_teste -> lista de nomes de blocos a concatenar."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dobras = list(cv.split(np.zeros(len(y)), y, grupos))
    linhas = []
    d = cross_val_score(DummyClassifier(strategy="most_frequent"), np.zeros((len(y), 1)), y, cv=dobras)
    linhas.append({"teste": "referencia_classe_mais_frequente", "acc_media": d.mean(), "acc_desvio": d.std(),
                   "f1macro_media": np.nan})
    for nome, partes in conjuntos.items():
        X = np.nan_to_num(np.hstack([blocos[p] for p in partes]))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        acc = cross_val_score(clf, X, y, cv=dobras, scoring="accuracy")
        f1 = cross_val_score(clf, X, y, cv=dobras, scoring="f1_macro")
        linhas.append({"teste": nome, "acc_media": acc.mean(), "acc_desvio": acc.std(), "f1macro_media": f1.mean()})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def fig_histogramas(ids, caminhos, out):
    fig, ax = plt.subplots(1, len(ids), figsize=(3.4 * len(ids), 3), squeeze=False)
    for a, i in zip(ax[0], ids):
        g = cv2.cvtColor(carregar_rgb(caminhos[i]), cv2.COLOR_RGB2GRAY)
        t, _ = otsu_bruto(g)
        a.hist(g.ravel(), bins=64, color="#4C78A8", log=True)
        a.axvline(t, color="crimson", ls="--", label=f"Otsu = {t:.0f}")
        a.set_title(i, fontsize=9)
        a.set_xlabel("Tom de cinza")
        a.legend(fontsize=8)
    ax[0][0].set_ylabel("Pixels (escala log)")
    fig.suptitle("Histogramas bimodais: fundo claro (≈255) e folha escura")
    fig.tight_layout()
    fig.savefig(out / "fig_q3_histogramas.png", dpi=140)
    plt.close(fig)


def fig_pipeline(ids, caminhos, r, out):
    cols = ["Original", "Otsu bruto", "Otsu + limpeza", f"Sem pecíolo (r={r})"]
    fig, ax = plt.subplots(len(ids), 4, figsize=(2.6 * 4, 2.7 * len(ids)), squeeze=False)
    for k, i in enumerate(ids):
        img = carregar_rgb(caminhos[i])
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bruto = otsu_bruto(g)
        base = limpar(bruto)
        sp = abrir(base, r)
        for c, im in enumerate([img, bruto * 255, base * 255, sp * 255]):
            ax[k][c].imshow(im, cmap=None if c == 0 else "gray")
            ax[k][c].axis("off")
            if k == 0:
                ax[k][c].set_title(cols[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_q3_pipeline.png", dpi=130)
    plt.close(fig)


def fig_metodos(ids, caminhos, out):
    cols = ["Original", "Otsu (cinza)", f"Limiar fixo {LIMIAR_FIXO}", "k-means (Lab)", "Otsu vs k-means"]
    fig, ax = plt.subplots(len(ids), 5, figsize=(2.6 * 5, 2.7 * len(ids)), squeeze=False)
    for k, i in enumerate(ids):
        img = carregar_rgb(caminhos[i])
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mo = limpar(otsu_bruto(g)[1])
        mf = limpar((g <= LIMIAR_FIXO).astype(np.uint8))
        mk = limpar(seg_kmeans(img))
        dif = np.zeros(mo.shape + (3,), np.uint8)
        dif[mo.astype(bool) & mk.astype(bool)] = (255, 255, 255)
        dif[mo.astype(bool) & ~mk.astype(bool)] = (255, 0, 0)
        dif[~mo.astype(bool) & mk.astype(bool)] = (0, 120, 255)
        for c, im in enumerate([img, mo * 255, mf * 255, mk * 255, dif]):
            ax[k][c].imshow(im, cmap=None if c in (0, 4) else "gray")
            ax[k][c].axis("off")
            if k == 0:
                ax[k][c].set_title(cols[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_q3_metodos.png", dpi=130)
    plt.close(fig)


def fig_exemplos(df_tr, caminhos, seed, out):
    rng = np.random.RandomState(seed)
    cls = ordem_classes(df_tr["classe"].unique())
    fig, ax = plt.subplots(3, 5, figsize=(13, 8.4))
    for a, c in zip(ax.ravel(), cls):
        i = rng.choice(df_tr[df_tr.classe == c]["id_imagem"].values)
        img = carregar_rgb(caminhos[i])
        m = limpar(otsu_bruto(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[1])
        a.imshow(contorno(img, m))
        a.set_title(f"classe {c} ({i})", fontsize=8)
        a.axis("off")
    fig.suptitle("Contorno da máscara final (Otsu + limpeza) sobre a imagem")
    fig.tight_layout()
    fig.savefig(out / "fig_q3_exemplos_por_classe.png", dpi=130)
    plt.close(fig)


def fig_sensibilidade(tab_sens, ious, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(tab_sens["deslocamento"], tab_sens["iou_medio"], yerr=tab_sens["iou_desvio"], marker="o", capsize=3)
    ax[0].set_xlabel("Deslocamento do limiar em relação ao Otsu (níveis de cinza)")
    ax[0].set_ylabel("IoU com a máscara do Otsu")
    ax[0].set_title("Sensibilidade ao limiar")
    ax[0].set_ylim(0.9, 1.005)
    ax[1].boxplot([ious["fixo"], ious["kmeans"]], tick_labels=[f"Fixo {LIMIAR_FIXO}", "k-means (Lab)"])
    ax[1].set_ylabel("IoU com Otsu + limpeza")
    ax[1].set_title("Concordância entre métodos")
    fig.tight_layout()
    fig.savefig(out / "fig_q3_sensibilidade.png", dpi=140)
    plt.close(fig)


def fig_por_classe(met, out):
    cls = ordem_classes(met["classe"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].boxplot([100 * met[met.classe == c]["fracao_folha"].values for c in cls], tick_labels=cls)
    ax[0].set_xlabel("Classe")
    ax[0].set_ylabel("Folha no quadro 256×256 (%)")
    ax[0].set_title("Fração da imagem ocupada pela folha")
    for r, cor in zip(R_LISTA, ["#9ecae1", "#3182bd", "#08306b"]):
        medias = [100 * met[met.classe == c][f"perda_r{r}"].mean() for c in cls]
        ax[1].plot(cls, medias, marker="o", color=cor, label=f"r = {r}")
    ax[1].set_xlabel("Classe")
    ax[1].set_ylabel("Área removida pela abertura (%)")
    ax[1].set_title("Perda de área ao remover o pecíolo")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_q3_por_classe.png", dpi=140)
    plt.close(fig)


def fig_piores(met, caminhos, out, n=12):
    pior = met.assign(d=1 - met["iou_otsu_kmeans"]).sort_values("d", ascending=False).head(n)
    fig, ax = plt.subplots(3, 4, figsize=(11, 8.6))
    for a, (_, r) in zip(ax.ravel(), pior.iterrows()):
        img = carregar_rgb(caminhos[r.id_imagem])
        mo = limpar(otsu_bruto(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[1])
        mk = limpar(seg_kmeans(img))
        im = contorno(contorno(img, mk, (255, 200, 0)), mo, (255, 0, 0))
        a.imshow(im)
        a.set_title(f"{r.id_imagem} (cl. {r.classe}) IoU={r.iou_otsu_kmeans:.3f}", fontsize=8)
        a.axis("off")
    fig.suptitle("Maior discordância entre métodos — vermelho: Otsu, amarelo: k-means")
    fig.tight_layout()
    fig.savefig(out / "fig_q3_piores_casos.png", dpi=130)
    plt.close(fig)


def fig_proxy(tab, out):
    t = tab[tab.teste != "referencia_classe_mais_frequente"].copy()
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(t) + 1.2))
    y = np.arange(len(t))[::-1]
    ax.barh(y, 100 * t["acc_media"], xerr=100 * t["acc_desvio"], color="#4C78A8", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(t["teste"], fontsize=8)
    for yy, v in zip(y, t["acc_media"]):
        ax.text(100 * v + 0.5, yy, f"{100 * v:.1f}", va="center", fontsize=7)
    ax.set_xlabel("Acurácia (%) — validação cruzada no treino")
    ax.set_title("Proxy: efeito da segmentação nas características")
    fig.tight_layout()
    fig.savefig(out / "fig_q3_proxy.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    raiz = Path(__file__).resolve().parent.parent
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="AV2 PDI - Q3: segmentação")
    ap.add_argument("--cache", default="", help="pasta com as imagens da Questão 2 (cache_256)")
    ap.add_argument("--split", default=str(raiz / "questao1" / "resultados" / "split_treino_teste.csv"))
    ap.add_argument("--out", default=str(base / "resultados"))
    ap.add_argument("--mascaras", default=str(base / "mascaras_256"))
    ap.add_argument("--mascaras-sem-peciolo", default=str(base / "mascaras_256_sem_peciolo"))
    ap.add_argument("--r-peciolo", type=int, default=3, help="raio da abertura usada nas máscaras sem pecíolo salvas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache = achar_cache(raiz, args.cache)
    if cache is None or not cache.is_dir():
        sys.exit("Pasta cache_256 da Questão 2 não encontrada. Use --cache <pasta>.")
    if not Path(args.split).is_file():
        sys.exit(f"Divisão da Questão 1 não encontrada: {args.split}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pm, pm2 = Path(args.mascaras), Path(args.mascaras_sem_peciolo)

    split = pd.read_csv(args.split, dtype={"classe": str})
    caminhos = {p.stem: str(p) for p in cache.rglob("*.png")}
    faltando = [i for i in split["id_imagem"] if i not in caminhos]
    if faltando:
        sys.exit(f"{len(faltando)} imagens da divisão não estão em {cache} (ex.: {faltando[:3]})")
    rs = sorted(set(R_LISTA) | {args.r_peciolo})

    print(f"[1/3] Segmentando {len(split)} imagens (cache: {cache})...")
    t0 = time.time()
    metricas, bl = [], {}
    feats = {}
    for k, r_ in enumerate(split.itertuples(), 1):
        img = carregar_rgb(caminhos[r_.id_imagem])
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t, bruto = otsu_bruto(g)
        base_m = limpar(bruto)
        abertas = {r: abrir(base_m, r) for r in rs}
        fixo = limpar((g <= LIMIAR_FIXO).astype(np.uint8))
        km = limpar(seg_kmeans(img))
        a_base = max(1, int(base_m.sum()))
        reg = {"id_imagem": r_.id_imagem, "classe": r_.classe, "conjunto": r_.conjunto, "t_otsu": t,
               "fracao_folha": a_base / base_m.size, "n_comp": n_componentes(base_m),
               "toca_borda": toca_borda(base_m), "toca_borda_sem_peciolo": toca_borda(abertas[args.r_peciolo]),
               "iou_otsu_fixo": iou(base_m, fixo), "iou_otsu_kmeans": iou(base_m, km)}
        for r in rs:
            reg[f"perda_r{r}"] = 1 - abertas[r].sum() / a_base
        for d in DESLOCAMENTOS:
            reg[f"iou_sens_{d:+d}"] = iou(bruto, (g <= t + d).astype(np.uint8))
        metricas.append(reg)

        # máscaras salvas
        for pasta, m in ((pm, base_m), (pm2, abertas[args.r_peciolo])):
            dest = pasta / f"leaf{r_.classe}"
            dest.mkdir(parents=True, exist_ok=True)
            Image.fromarray((m * 255).astype(np.uint8)).save(dest / f"{r_.id_imagem}.png")

        # características do proxy (somente treino)
        if r_.conjunto == "treino":
            todos = np.ones(base_m.shape, bool)
            n = 2
            sel = cv2.erode(base_m, np.ones((2 * n + 1, 2 * n + 1), np.uint8)).astype(bool)
            if sel.sum() < 50:
                sel = base_m.astype(bool)
            lb = lbps(img)
            f = {"cor_global": feats_cor(img, todos), "textura_global": feats_tex(lb, todos),
                 "cor_folha": feats_cor(img, sel), "textura_folha": feats_tex(lb, sel),
                 "forma_otsu": vetor_forma(base_m), "forma_fixo": vetor_forma(fixo), "forma_kmeans": vetor_forma(km),
                 "fracao_folha": np.array([reg["fracao_folha"]])}
            for r in rs:
                f[f"forma_aberta_r{r}"] = vetor_forma(abertas[r])
            feats[r_.id_imagem] = f
        if k % 100 == 0 or k == len(split):
            print(f"      {k}/{len(split)}  ({time.time() - t0:.0f}s)", end="\r", flush=True)
    print()

    met = pd.DataFrame(metricas)
    met.to_csv(out / "q3_metricas_por_imagem.csv", index=False)
    tr = met[met.conjunto == "treino"].copy()

    print("[2/3] Tabelas e classificação de teste (treino)...")
    # métodos
    ious = {"fixo": tr["iou_otsu_fixo"].values, "kmeans": tr["iou_otsu_kmeans"].values}
    tab_met = pd.DataFrame([
        {"comparacao": f"Otsu vs limiar fixo {LIMIAR_FIXO}", "iou_medio": ious["fixo"].mean(),
         "iou_min": ious["fixo"].min(), "iou_p5": np.percentile(ious["fixo"], 5)},
        {"comparacao": "Otsu vs k-means (Lab)", "iou_medio": ious["kmeans"].mean(),
         "iou_min": ious["kmeans"].min(), "iou_p5": np.percentile(ious["kmeans"], 5)}])
    tab_met.to_csv(out / "q3_metodos.csv", index=False)
    tab_sens = pd.DataFrame([{"deslocamento": d, "iou_medio": tr[f"iou_sens_{d:+d}"].mean(),
                              "iou_desvio": tr[f"iou_sens_{d:+d}"].std(), "iou_min": tr[f"iou_sens_{d:+d}"].min()}
                             for d in DESLOCAMENTOS])
    tab_sens.to_csv(out / "q3_sensibilidade.csv", index=False)

    cls = ordem_classes(tr["classe"].unique())
    linhas = []
    for c in cls:
        s = tr[tr.classe == c]
        lin = {"classe": c, "folha_no_quadro_%": 100 * s.fracao_folha.mean(), "componentes_medios": s.n_comp.mean(),
               "toca_borda_%": 100 * s.toca_borda.mean()}
        for r in rs:
            lin[f"perda_area_r{r}_%"] = 100 * s[f"perda_r{r}"].mean()
        linhas.append(lin)
    tab_cls = pd.DataFrame(linhas)
    tab_cls.to_csv(out / "q3_por_classe.csv", index=False)

    # proxy
    ids = tr["id_imagem"].tolist()
    ytr = tr["classe"].astype(str).values
    gmap = split.set_index("id_imagem")["grupo_dup"]
    grupos = gmap.loc[ids].values
    nomes = list(feats[ids[0]].keys())
    blocos = {n: np.vstack([feats[i][n] for i in ids]) for n in nomes}
    conj = {
        "SEM seg.: cor global (imagem inteira)": ["cor_global"],
        "SEM seg.: textura global": ["textura_global"],
        "SEM seg.: cor + textura global": ["cor_global", "textura_global"],
        "COM seg.: cor na folha": ["cor_folha"],
        "COM seg.: textura na folha": ["textura_folha"],
        "COM seg.: cor + textura na folha": ["cor_folha", "textura_folha"],
        "COM seg.: forma (Otsu + limpeza)": ["forma_otsu"],
        "COM seg.: cor + textura + forma": ["cor_folha", "textura_folha", "forma_otsu"],
        "Atalho: só fração de folha no quadro": ["fracao_folha"],
        "Método: forma com limiar fixo": ["forma_fixo"],
        "Método: forma com k-means": ["forma_kmeans"],
    }
    for r in rs:
        conj[f"Pecíolo: forma com abertura r={r}"] = [f"forma_aberta_r{r}"]
    conj["Pecíolo: cor + textura + forma (abertura r=%d)" % args.r_peciolo] = ["cor_folha", "textura_folha",
                                                                              f"forma_aberta_r{args.r_peciolo}"]
    tab_proxy = avaliar_proxy(blocos, conj, ytr, grupos, args.seed)
    tab_proxy.to_csv(out / "q3_proxy_resultados.csv", index=False)

    print("[3/3] Figuras e relatório...")
    amostra = tr.groupby("classe").head(1).set_index("classe")["id_imagem"]
    pref = [amostra[c] for c in ["2", "10", "8", "7"] if c in amostra.index] or list(amostra.values[:4])
    fig_histogramas(pref, caminhos, out)
    fig_pipeline(pref, caminhos, args.r_peciolo, out)
    fig_metodos(pref, caminhos, out)
    fig_exemplos(tr, caminhos, args.seed, out)
    fig_sensibilidade(tab_sens, ious, out)
    fig_por_classe(tr, out)
    fig_piores(tr, caminhos, out)
    fig_proxy(tab_proxy, out)

    resumo = {
        "n_imagens": int(len(met)), "n_treino": int(len(tr)), "limiar_otsu_mediano": float(tr.t_otsu.median()),
        "limiar_otsu_min_max": [float(tr.t_otsu.min()), float(tr.t_otsu.max())],
        "fracao_folha_media_%": float(100 * tr.fracao_folha.mean()),
        "toca_borda_%": float(100 * tr.toca_borda.mean()),
        "toca_borda_sem_peciolo_%": float(100 * tr.toca_borda_sem_peciolo.mean()),
        "imagens_com_mais_de_1_componente_%": float(100 * (tr.n_comp > 1).mean()),
        "iou_otsu_fixo_medio": float(ious["fixo"].mean()), "iou_otsu_kmeans_medio": float(ious["kmeans"].mean()),
        "iou_otsu_kmeans_min": float(ious["kmeans"].min()),
        "sensibilidade_iou_medio": {f"{d:+d}": float(tr[f"iou_sens_{d:+d}"].mean()) for d in DESLOCAMENTOS},
        "perda_area_media_%": {f"r{r}": float(100 * tr[f"perda_r{r}"].mean()) for r in rs},
        "r_peciolo_salvo": args.r_peciolo,
        "proxy_acuracia_%": {t: round(100 * a, 2) for t, a in zip(tab_proxy.teste, tab_proxy.acc_media)},
        "mascaras": str(pm), "mascaras_sem_peciolo": str(pm2),
    }
    (out / "resumo_q3.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    tp = tab_proxy.copy()
    tp["acurácia (%)"] = [f"{100 * m:.1f} ± {100 * s:.1f}" for m, s in zip(tp.acc_media, tp.acc_desvio)]
    tp["F1 macro (%)"] = [f"{100 * f:.1f}" if f == f else "-" for f in tp.f1macro_media]
    md = f"""# Questão 3 — Segmentação (texto-base gerado automaticamente)

> Revise, interprete as figuras e complete as conclusões.

## 3.1 Método
Otsu sobre a imagem em cinza (cache 256 px da Questão 2) + limpeza morfológica (remoção de componentes < max(40 px, 1% do maior),
preenchimento de buracos internos < 0,3% do quadro). Limiar de Otsu mediano: {resumo['limiar_otsu_mediano']:.0f}
(mín–máx {resumo['limiar_otsu_min_max'][0]:.0f}–{resumo['limiar_otsu_min_max'][1]:.0f}). Figuras: `fig_q3_histogramas.png`, `fig_q3_pipeline.png`, `fig_q3_exemplos_por_classe.png`.

## 3.2 Comparação de métodos e sensibilidade
{df_md(tab_met, 4)}

Sensibilidade ao limiar (IoU com a máscara do Otsu):
{df_md(tab_sens, 4)}

Figuras: `fig_q3_metodos.png`, `fig_q3_sensibilidade.png`.

## 3.3 A segmentação contribui? (proxy — validação cruzada no treino)
Referência (classe mais frequente): {100 * tab_proxy.iloc[0].acc_media:.1f}%.

{df_md(tp[tp.teste != 'referencia_classe_mais_frequente'][['teste', 'acurácia (%)', 'F1 macro (%)']])}

Figura: `fig_q3_proxy.png`.

## 3.4 Pecíolo
Perda média de área na abertura morfológica: {', '.join(f"r={r}: {v:.1f}%" for r, v in resumo['perda_area_media_%'].items())}.
Máscaras tocando a borda do quadro: {resumo['toca_borda_%']:.1f}% (com pecíolo) vs {resumo['toca_borda_sem_peciolo_%']:.1f}% (sem).

{df_md(tab_cls, 2)}

Figura: `fig_q3_por_classe.png`.

## 3.5 Limitações
Imagens com mais de um componente: {resumo['imagens_com_mais_de_1_componente_%']:.1f}%. Piores casos: `fig_q3_piores_casos.png`.
Não há máscaras manuais (verdade de referência); a qualidade foi avaliada por consistência entre métodos, sensibilidade ao limiar,
inspeção visual e efeito na classificação.

## 3.6 Decisão
(completar após analisar as tabelas acima)
- Máscaras salvas em `{pm}` (com pecíolo) e `{pm2}` (abertura r={args.r_peciolo}).
"""
    (out / "relatorio_q3_trechos.md").write_text(md, encoding="utf-8")
    print(f"\nConcluído em {time.time() - t0:.0f}s. Resultados em: {out.resolve()}")


if __name__ == "__main__":
    main()