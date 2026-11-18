#!/usr/bin/env python3
"""
AV2 - PDI | Questão 2: Pré-processamento das imagens (Swedish Leaf)

Uso (na raiz do projeto, com o .venv ativo):
    pip install numpy pandas matplotlib pillow scikit-learn scikit-image opencv-python
    python questao2/preprocessamento.py

Por padrão lê:
    dataset/                                  (imagens originais)
    questao1/resultados/split_treino_teste.csv (divisão feita na Questão 1)

O que o script faz
------------------
Cada operação é AVALIADA com os dados antes de ser adotada:

  A) Normalização do fundo (balanço de branco pelo fundo)
       -> mede a variação da cor do fundo entre imagens, antes e depois.
  B) Redução de ruído
       -> estima o ruído do scanner no fundo (estimador robusto de Immerkaer) e compara
          filtros (gaussiano, mediana) pelo efeito na classificação.
  C) Ajuste de contraste (CLAHE)
       -> mede o contraste DENTRO da folha antes/depois e o efeito na classificação.
  D) Espaço de cor / canal para separar folha e fundo
       -> compara canais (cinza, R, G, B, S, V, L*, a*, b*, distância ao fundo) pela
          separabilidade de Otsu e pela concordância entre máscaras.
  E) Redimensionamento
       -> mede a convergência de descritores de forma em função da resolução e compara
          resoluções e interpolações pelo efeito na classificação.

"Efeito na classificação" = proxy: regressão logística com padronização, validação cruzada
estratificada por grupo (StratifiedGroupKFold) SOMENTE nas imagens de TREINO. O conjunto de
teste NÃO é usado para escolher nada; ele só recebe o pré-processamento final (cache).

Saídas
------
  questao2/resultados/   tabelas (.csv), figuras (.png), resumo_q2.json, relatorio_q2_trechos.md
  questao2/cache_256/    imagens pré-processadas finais (PNG 256x256), usadas nas Questões 3-5
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
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
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)  # evita competição de threads entre processos

EXTENSOES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
RESOLUCOES_FORMA = [64, 128, 256, 512]   # comparadas com a referência (até 1024)
REF_MAX = 1024
CANAIS = ["cinza", "R", "G", "B", "S", "V", "L", "a", "b", "dist_fundo"]
DESCRITORES = ["extent", "solidity", "eccentricity", "circularity", "aspecto"]

# Variantes avaliadas pelo proxy de classificação: nome -> configuração
VARIANTES = {
    "res128_area_wb":      dict(tam=128, interp="area",    wb=True,  op=None),
    "res256_area_wb":      dict(tam=256, interp="area",    wb=True,  op=None),      # base
    "res512_area_wb":      dict(tam=512, interp="area",    wb=True,  op=None),
    "res256_linear_wb":    dict(tam=256, interp="linear",  wb=True,  op=None),
    "res256_nearest_wb":   dict(tam=256, interp="nearest", wb=True,  op=None),
    "res256_area_raw":     dict(tam=256, interp="area",    wb=False, op=None),      # sem normalizar fundo
    "res256_area_wb_clahe": dict(tam=256, interp="area",   wb=True,  op="clahe"),
    "res256_area_wb_gauss": dict(tam=256, interp="area",   wb=True,  op="gauss"),
    "res256_area_wb_mediana": dict(tam=256, interp="area", wb=True,  op="mediana"),
}
BASE = "res256_area_wb"
CONJ_FEATS = ["cor", "textura", "forma", "todas"]
INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}


# ----------------------------------------------------------------------------
# Funções básicas
# ----------------------------------------------------------------------------
def carregar_rgb(caminho: str) -> np.ndarray:
    with Image.open(caminho) as im:
        return np.asarray(im.convert("RGB"))


def moldura(h: int, w: int, frac: float = 0.03) -> int:
    return max(2, int(frac * min(h, w)))


def pixels_moldura(img: np.ndarray, frac: float = 0.03) -> np.ndarray:
    """Pixels da moldura externa (assumida como fundo). Retorna (N, C) ou (N,)."""
    h, w = img.shape[:2]
    b = moldura(h, w, frac)
    partes = [img[:b].reshape(-1, *img.shape[2:]), img[-b:].reshape(-1, *img.shape[2:]),
              img[b:-b, :b].reshape(-1, *img.shape[2:]), img[b:-b, -b:].reshape(-1, *img.shape[2:])]
    return np.concatenate(partes, axis=0)


def estimar_ruido(rgb: np.ndarray) -> float:
    """
    Desvio-padrão do ruído (em níveis de cinza, escala 0-255) estimado no fundo, em
    resolução nativa. Método de Immerkaer com mediana (robusto a poeira e sombras):
    a resposta do núcleo [[1,-2,1],[-2,4,-2],[1,-2,1]] a ruído gaussiano tem desvio 6*sigma.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    b = max(8, moldura(h, w, 0.03))
    nucleo = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = []
    for faixa in (g[:b], g[-b:]):
        r = cv2.filter2D(faixa, -1, nucleo)[1:-1, 1:-1]
        resp.append(np.abs(r).ravel())
    resp = np.concatenate(resp)
    return float(np.median(resp) / (0.6745 * 6.0))


def redimensionar(rgb: np.ndarray, tam: int, interp: str, preenchimento) -> np.ndarray:
    """Mantém a proporção (maior lado = tam) e completa com o fundo até tam x tam."""
    h, w = rgb.shape[:2]
    esc = tam / max(h, w)
    nw, nh = max(1, round(w * esc)), max(1, round(h * esc))
    flag = INTERP[interp] if esc < 1 or interp == "nearest" else cv2.INTER_CUBIC
    r = cv2.resize(rgb, (nw, nh), interpolation=flag)
    tela = np.empty((tam, tam, 3), np.uint8)
    tela[:] = np.array(preenchimento, np.uint8)
    y0, x0 = (tam - nh) // 2, (tam - nw) // 2
    tela[y0:y0 + nh, x0:x0 + nw] = r
    return tela


def balanco_fundo(img: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """Escala cada canal para que o fundo estimado vire branco (255)."""
    ganho = 255.0 / np.maximum(bg_rgb.astype(np.float32), 1.0)
    return np.clip(np.rint(img.astype(np.float32) * ganho), 0, 255).astype(np.uint8)


def clahe_rgb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def aplicar_op(img: np.ndarray, op):
    """Retorna (img_para_caracteristicas, img_para_segmentacao)."""
    if op is None:
        return img, img
    if op == "clahe":
        return clahe_rgb(img), img   # máscara vem da imagem sem CLAHE
    if op == "gauss":
        f = cv2.GaussianBlur(img, (0, 0), 1.0)
        return f, f
    if op == "mediana":
        f = cv2.medianBlur(img, 3)
        return f, f
    raise ValueError(op)


# ----------------------------------------------------------------------------
# Segmentação simples (Otsu) usada só para AVALIAR o pré-processamento
# ----------------------------------------------------------------------------
def mascara_canal(ch: np.ndarray, limpar: bool = True) -> np.ndarray:
    t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fundo = np.median(pixels_moldura(ch))
    m = (ch <= t) if fundo > t else (ch > t)
    m = m.astype(np.uint8)
    if limpar:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        minimo = max(20, int(0.0005 * m.size))
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < minimo:
                m[lab == i] = 0
    return m


def eta_otsu(ch: np.ndarray) -> float:
    """Separabilidade (variância entre classes / variância total) no limiar de Otsu."""
    t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a, b = ch[ch <= t], ch[ch > t]
    if a.size == 0 or b.size == 0 or ch.var() == 0:
        return 0.0
    w0 = a.size / ch.size
    return float(w0 * (1 - w0) * (a.mean() - b.mean()) ** 2 / ch.var())


def canais_imagem(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    bg_lab = np.median(pixels_moldura(lab), axis=0)
    dist = np.sqrt(((lab.astype(np.float32) - bg_lab) ** 2).sum(-1))
    return {
        "cinza": cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
        "R": img[..., 0], "G": img[..., 1], "B": img[..., 2],
        "S": hsv[..., 1], "V": hsv[..., 2],
        "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2],
        "dist_fundo": np.clip(dist, 0, 255).astype(np.uint8),
    }


def descritores_forma(mask: np.ndarray):
    """Descritores de forma da máscara (união de todos os componentes). None se vazia."""
    if mask.sum() < 50:
        return None
    p = regionprops(mask.astype(np.uint8))[0]
    per = p.perimeter if p.perimeter > 0 else np.nan
    minr, minc, maxr, maxc = p.bbox
    hu = np.sign(p.moments_hu) * np.log10(np.abs(p.moments_hu) + 1e-30)
    return {
        "extent": float(p.extent),
        "solidity": float(p.solidity),
        "eccentricity": float(p.eccentricity),
        "circularity": float(4 * np.pi * p.area / per ** 2) if per == per else np.nan,
        "aspecto": float((maxr - minr) / max(1, (maxc - minc))),
        "hu": hu.astype(float),
    }


# ----------------------------------------------------------------------------
# Características usadas apenas como PROXY de desempenho
# ----------------------------------------------------------------------------
def feats_proxy(img_feat: np.ndarray, img_seg: np.ndarray) -> dict:
    tam = img_seg.shape[0]
    gray_seg = cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY)
    mask = mascara_canal(gray_seg)
    n = max(1, tam // 128)
    interior = cv2.erode(mask, np.ones((2 * n + 1, 2 * n + 1), np.uint8))
    dim_cor, dim_tex, dim_forma = 12, 10 + 18, 5 + 7
    if interior.sum() < 50:
        return {"cor": np.zeros(dim_cor), "textura": np.zeros(dim_tex),
                "forma": np.zeros(dim_forma), "vazia": True}
    sel = interior.astype(bool)
    lab = cv2.cvtColor(img_feat, cv2.COLOR_RGB2LAB).astype(np.float32)
    cor = []
    for c in range(3):
        cor += [img_feat[..., c][sel].mean(), img_feat[..., c][sel].std()]
    for c in range(3):
        cor += [lab[..., c][sel].mean(), lab[..., c][sel].std()]
    g = cv2.cvtColor(img_feat, cv2.COLOR_RGB2GRAY)
    tex = []
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(g, P, R, method="uniform")
        h, _ = np.histogram(lbp[sel], bins=P + 2, range=(0, P + 2))
        tex += list(h / max(1, h.sum()))
    d = descritores_forma(mask.astype(bool))
    forma = [d[k] for k in DESCRITORES] + list(d["hu"]) if d else [0.0] * dim_forma
    return {"cor": np.array(cor, float), "textura": np.array(tex, float),
            "forma": np.nan_to_num(np.array(forma, float)), "vazia": False,
            "contraste_folha": float(g[sel].std())}


# ----------------------------------------------------------------------------
# Trabalho por imagem (executado em processos paralelos)
# ----------------------------------------------------------------------------
def processar(args: dict) -> dict:
    caminho = args["caminho"]
    rgb = carregar_rgb(caminho)
    h, w = rgb.shape[:2]
    bg = np.median(pixels_moldura(rgb), axis=0)
    bg_std = float(pixels_moldura(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).std())
    out = {
        "id_imagem": args["id"], "largura_orig": w, "altura_orig": h,
        "bg_r": float(bg[0]), "bg_g": float(bg[1]), "bg_b": float(bg[2]),
        "bg_std": bg_std, "ruido_sigma": estimar_ruido(rgb),
    }

    # ---- cache final: balanço de fundo + resize 256 (área) com preenchimento branco
    tam = args["tam_final"]
    base_final = balanco_fundo(redimensionar(rgb, tam, "area", np.rint(bg)), bg)
    # a moldura de preenchimento acima era a cor do fundo; após o balanço vira branco (255)
    out["fator_escala"] = tam / max(h, w)
    if args["salvar_cache"]:
        destino = Path(args["cache_dir"]) / f"leaf{args['classe']}"
        destino.mkdir(parents=True, exist_ok=True)
        Image.fromarray(base_final).save(destino / f"{args['id']}.png")
        out["caminho_cache"] = str(destino / f"{args['id']}.png")

    # ---- avaliações (somente imagens de treino)
    if args["conjunto"] != "treino":
        return out

    # variantes -> características proxy
    out["proxy"] = {}
    for nome, cfg in VARIANTES.items():
        img = redimensionar(rgb, cfg["tam"], cfg["interp"], np.rint(bg))
        if cfg["wb"]:
            img = balanco_fundo(img, bg)
        f, s = aplicar_op(img, cfg["op"])
        out["proxy"][nome] = feats_proxy(f, s)
    out["contraste_raw"] = out["proxy"][BASE].get("contraste_folha", np.nan)
    out["contraste_clahe"] = out["proxy"]["res256_area_wb_clahe"].get("contraste_folha", np.nan)

    # amostra: separabilidade de canais e convergência de forma com a resolução
    if args["amostra"]:
        img256 = base_final
        chs = canais_imagem(img256)
        etas, masks = {}, {}
        for nome, ch in chs.items():
            etas[nome] = eta_otsu(ch)
            masks[nome] = mascara_canal(ch)
        voto = np.mean([m for m in masks.values()], axis=0) >= 0.5
        ious, falhas = {}, {}
        for nome, m in masks.items():
            mb = m.astype(bool)
            uni = (mb | voto).sum()
            ious[nome] = float((mb & voto).sum() / uni) if uni else 0.0
            fr = mb.mean()
            falhas[nome] = bool(fr < 0.01 or fr > 0.95)
        out["canais"] = {"eta": etas, "iou": ious, "falha": falhas}

        ref_tam = min(REF_MAX, max(h, w))
        ref = descritores_forma(mascara_canal(cv2.cvtColor(
            redimensionar(rgb, ref_tam, "area", np.rint(bg)), cv2.COLOR_RGB2GRAY)).astype(bool))
        resol = {}
        if ref is not None:
            for r in RESOLUCOES_FORMA:
                if r >= ref_tam:
                    continue
                im = redimensionar(rgb, r, "area", np.rint(bg))
                d = descritores_forma(mascara_canal(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)).astype(bool))
                if d is None:
                    continue
                resol[r] = {k: abs(d[k] - ref[k]) / max(abs(ref[k]), 1e-9) for k in DESCRITORES}
        out["resolucao"] = resol
    return out


# ----------------------------------------------------------------------------
# Auxiliares do programa principal
# ----------------------------------------------------------------------------
def indexar_dataset(raiz: Path) -> dict:
    idx = {}
    for p in sorted(raiz.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSOES and not p.name.startswith("._"):
            idx[p.stem] = str(p)
    return idx


def df_md(df: pd.DataFrame, casas: int = 3) -> str:
    cab = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    linhas = []
    for _, r in df.iterrows():
        cel = [f"{v:.{casas}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        linhas.append("| " + " | ".join(cel) + " |")
    return "\n".join([cab, sep] + linhas)


def avaliar_proxy(resultados: list, split: pd.DataFrame, seed: int) -> pd.DataFrame:
    tr = split[split["conjunto"] == "treino"].set_index("id_imagem")
    por_id = {r["id_imagem"]: r for r in resultados if "proxy" in r}
    ids = [i for i in tr.index if i in por_id]
    y = tr.loc[ids, "classe"].astype(str).values
    grupos = tr.loc[ids, "grupo_dup"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dobras = list(cv.split(np.zeros(len(ids)), y, grupos))
    linhas = []
    dummy = cross_val_score(DummyClassifier(strategy="most_frequent"), np.zeros((len(ids), 1)), y,
                            cv=dobras, scoring="accuracy")
    linhas.append({"variante": "referencia_classe_mais_frequente", "conjunto_feats": "-",
                   "acc_media": dummy.mean(), "acc_desvio": dummy.std(),
                   "f1macro_media": np.nan, "f1macro_desvio": np.nan})
    for nome in VARIANTES:
        blocos = {k: np.vstack([por_id[i]["proxy"][nome][k] for i in ids]) for k in ("cor", "textura", "forma")}
        blocos["todas"] = np.hstack([blocos["cor"], blocos["textura"], blocos["forma"]])
        for conj in CONJ_FEATS:
            X = np.nan_to_num(blocos[conj])
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
            acc = cross_val_score(clf, X, y, cv=dobras, scoring="accuracy")
            f1 = cross_val_score(clf, X, y, cv=dobras, scoring="f1_macro")
            linhas.append({"variante": nome, "conjunto_feats": conj,
                           "acc_media": acc.mean(), "acc_desvio": acc.std(),
                           "f1macro_media": f1.mean(), "f1macro_desvio": f1.std()})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def fig_diagnosticos(meta: pd.DataFrame, resultados: list, out: Path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for c, cor in zip(["bg_r", "bg_g", "bg_b"], ["#d62728", "#2ca02c", "#1f77b4"]):
        ax[0][0].hist(meta[c], bins=40, alpha=0.6, color=cor, label=c[-1].upper())
    ax[0][0].set_title("Cor do fundo por imagem (antes da normalização)")
    ax[0][0].set_xlabel("Valor (0-255)")
    ax[0][0].legend()
    ax[0][1].hist(meta["ruido_sigma"], bins=40, color="#9467bd")
    ax[0][1].set_title("Ruído do scanner no fundo (σ, níveis de cinza)")
    ax[0][1].set_xlabel("σ estimado (resolução nativa)")
    tr = [r for r in resultados if "contraste_raw" in r]
    cr = np.array([r["contraste_raw"] for r in tr])
    cc = np.array([r["contraste_clahe"] for r in tr])
    ax[1][0].scatter(cr, cc, s=6, alpha=0.5)
    lim = [min(cr.min(), cc.min()), max(cr.max(), cc.max())]
    ax[1][0].plot(lim, lim, "k--", lw=1)
    ax[1][0].set_xlabel("Contraste dentro da folha (σ cinza) sem CLAHE")
    ax[1][0].set_ylabel("com CLAHE")
    ax[1][0].set_title("Contraste interno da folha")
    ax[1][1].hist(meta["bg_std"], bins=40, color="#8c564b")
    ax[1][1].set_title("Desvio-padrão do fundo (sujeira/gradiente/ruído)")
    ax[1][1].set_xlabel("σ (níveis de cinza)")
    fig.tight_layout()
    fig.savefig(out / "fig_q2_diagnosticos.png", dpi=150)
    plt.close(fig)


def fig_canais(tab: pd.DataFrame, brutos: pd.DataFrame, out: Path):
    ordem = tab.sort_values("eta_media", ascending=False)["canal"].tolist()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].boxplot([brutos[brutos.canal == c]["eta"].values for c in ordem], tick_labels=ordem)
    ax[0].set_title("Separabilidade folha/fundo (η de Otsu)")
    ax[0].set_ylabel("η (1 = perfeitamente bimodal)")
    ax[1].boxplot([brutos[brutos.canal == c]["iou"].values for c in ordem], tick_labels=ordem)
    ax[1].set_title("Concordância (IoU) com a máscara de consenso")
    ax[1].set_ylabel("IoU")
    for a in ax:
        a.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "fig_q2_canais.png", dpi=150)
    plt.close(fig)


def fig_resolucao(tab: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in DESCRITORES:
        s = tab[tab.descritor == d].sort_values("resolucao")
        ax.plot(s["resolucao"], 100 * s["erro_relativo_mediano"], marker="o", label=d)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RESOLUCOES_FORMA)
    ax.set_xticklabels(RESOLUCOES_FORMA)
    ax.set_xlabel("Maior lado da imagem (px)")
    ax.set_ylabel("Erro relativo mediano vs. referência (%)")
    ax.set_title("Convergência dos descritores de forma com a resolução")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_q2_resolucao.png", dpi=150)
    plt.close(fig)


def fig_proxy(tab: pd.DataFrame, out: Path):
    t = tab[tab.variante != "referencia_classe_mais_frequente"]
    piv = t.pivot(index="variante", columns="conjunto_feats", values="acc_media")[CONJ_FEATS]
    piv = piv.loc[list(VARIANTES)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{100 * piv.values[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title("Acurácia (%) do proxy por variante de pré-processamento\n(validação cruzada no treino)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "fig_q2_proxy.png", dpi=150)
    plt.close(fig)


def fig_exemplos(amostra: pd.DataFrame, caminhos: dict, out: Path):
    ids = []
    for c in ["2", "9", "10", "7"]:
        s = amostra[amostra.classe.astype(str) == c]
        if len(s):
            ids.append(s.iloc[0]["id_imagem"])
    if not ids:
        ids = amostra["id_imagem"].tolist()[:4]
    cols = ["Original (maior lado=256)", "Fundo normalizado", "CLAHE", "Gaussiano σ=1", "Mediana 3×3",
            "Máscara (dist. ao fundo)"]
    fig, axes = plt.subplots(len(ids), len(cols), figsize=(2.3 * len(cols), 2.5 * len(ids)), squeeze=False)
    for r, i in enumerate(ids):
        rgb = carregar_rgb(caminhos[i])
        bg = np.median(pixels_moldura(rgb), axis=0)
        raw = redimensionar(rgb, 256, "area", np.rint(bg))
        wb = balanco_fundo(raw, bg)
        mk = mascara_canal(canais_imagem(wb)["dist_fundo"])
        imgs = [raw, wb, clahe_rgb(wb), cv2.GaussianBlur(wb, (0, 0), 1.0), cv2.medianBlur(wb, 3), mk * 255]
        for c, im in enumerate(imgs):
            axes[r][c].imshow(im, cmap="gray" if im.ndim == 2 else None)
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(cols[c], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_q2_exemplos_pipeline.png", dpi=130)
    plt.close(fig)

    # zoom em 512 px para ver o efeito sobre nervuras (textura)
    i = ids[min(1, len(ids) - 1)]
    rgb = carregar_rgb(caminhos[i])
    bg = np.median(pixels_moldura(rgb), axis=0)
    wb = balanco_fundo(redimensionar(rgb, 512, "area", np.rint(bg)), bg)
    mk = mascara_canal(cv2.cvtColor(wb, cv2.COLOR_RGB2GRAY))
    ys, xs = np.nonzero(mk)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (256, 256)
    y0, x0 = max(0, min(512 - 128, cy - 64)), max(0, min(512 - 128, cx - 64))
    vs = [("Base (512 px)", wb), ("CLAHE", clahe_rgb(wb)), ("Gaussiano σ=1", cv2.GaussianBlur(wb, (0, 0), 1.0)),
          ("Mediana 3×3", cv2.medianBlur(wb, 3))]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for a, (t, im) in zip(axes, vs):
        a.imshow(im[y0:y0 + 128, x0:x0 + 128])
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.suptitle("Zoom 128×128 na região da folha: efeito dos filtros na textura/nervuras")
    fig.tight_layout()
    fig.savefig(out / "fig_q2_zoom_textura.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    raiz_proj = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="AV2 PDI - Q2: pré-processamento")
    ap.add_argument("--data", default=str(raiz_proj / "dataset"))
    ap.add_argument("--split", default=str(raiz_proj / "questao1" / "resultados" / "split_treino_teste.csv"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "resultados"))
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parent / "cache_256"))
    ap.add_argument("--tamanho", type=int, default=256, help="maior lado do cache final")
    ap.add_argument("--amostra-por-classe", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sem-cache", action="store_true", help="não grava as imagens pré-processadas")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(args.split).is_file():
        sys.exit(f"Divisão da Questão 1 não encontrada: {args.split}")
    split = pd.read_csv(args.split, dtype={"classe": str})
    idx = indexar_dataset(Path(args.data))
    faltando = [i for i in split["id_imagem"] if i not in idx]
    if faltando:
        sys.exit(f"{len(faltando)} imagens da divisão não foram achadas em {args.data} (ex.: {faltando[:3]})")

    rng = np.random.RandomState(args.seed)
    amostra_ids = set()
    for c, g in split[split.conjunto == "treino"].groupby("classe"):
        amostra_ids |= set(rng.choice(g["id_imagem"].values, min(args.amostra_por_classe, len(g)), replace=False))

    tarefas = [{
        "id": r.id_imagem, "classe": r.classe, "conjunto": r.conjunto, "caminho": idx[r.id_imagem],
        "tam_final": args.tamanho, "salvar_cache": not args.sem_cache, "cache_dir": args.cache,
        "amostra": r.id_imagem in amostra_ids,
    } for r in split.itertuples()]
    print(f"[1/4] Processando {len(tarefas)} imagens com {args.jobs} processo(s)...")
    t0 = time.time()
    resultados = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for k, r in enumerate(ex.map(processar, tarefas, chunksize=4), 1):
                resultados.append(r)
                if k % 50 == 0 or k == len(tarefas):
                    print(f"      {k}/{len(tarefas)}  ({time.time() - t0:.0f}s)", end="\r", flush=True)
    else:
        for k, t in enumerate(tarefas, 1):
            resultados.append(processar(t))
            if k % 50 == 0 or k == len(tarefas):
                print(f"      {k}/{len(tarefas)}  ({time.time() - t0:.0f}s)", end="\r", flush=True)
    print()

    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ("proxy", "canais", "resolucao",
                                                                      "contraste_raw", "contraste_clahe")}
                         for r in resultados]).merge(split[["id_imagem", "classe", "conjunto"]], on="id_imagem")
    meta.to_csv(out / "metadados_preproc.csv", index=False)

    print("[2/4] Tabelas de avaliação...")
    # (A) fundo
    # estatísticas descritivas e de seleção usam só o TREINO (teste fica intocado)
    mt = meta[meta.conjunto == "treino"]
    fundo = {c: {"media": float(mt[c].mean()), "desvio": float(mt[c].std()),
                 "min": float(mt[c].min()), "max": float(mt[c].max())} for c in ("bg_r", "bg_g", "bg_b")}
    # (B) ruído
    ruido = {"mediana": float(mt["ruido_sigma"].median()), "p95": float(mt["ruido_sigma"].quantile(0.95)),
             "max": float(mt["ruido_sigma"].max())}
    # (C) contraste
    tr = [r for r in resultados if "contraste_raw" in r]
    cr = np.array([r["contraste_raw"] for r in tr])
    cc = np.array([r["contraste_clahe"] for r in tr])
    contraste = {"folha_sem_clahe_media": float(np.nanmean(cr)), "folha_com_clahe_media": float(np.nanmean(cc)),
                 "razao_media": float(np.nanmean(cc / np.maximum(cr, 1e-9)))}
    vazias = int(sum(v.get("vazia", False) for r in tr for v in r["proxy"].values()))

    # (D) canais
    linhas = []
    for r in resultados:
        if "canais" in r:
            for c in CANAIS:
                linhas.append({"id_imagem": r["id_imagem"], "canal": c, "eta": r["canais"]["eta"][c],
                               "iou": r["canais"]["iou"][c], "falha": r["canais"]["falha"][c]})
    brutos = pd.DataFrame(linhas)
    tab_canais = brutos.groupby("canal").agg(eta_media=("eta", "mean"), eta_desvio=("eta", "std"),
                                             iou_media=("iou", "mean"), falhas=("falha", "sum")).reset_index()
    tab_canais = tab_canais.sort_values("eta_media", ascending=False)
    tab_canais.to_csv(out / "q2_canais.csv", index=False)

    # (E) resolução
    linhas = []
    for r in resultados:
        for res, d in r.get("resolucao", {}).items():
            for k, v in d.items():
                linhas.append({"resolucao": res, "descritor": k, "erro": v})
    res_df = pd.DataFrame(linhas)
    tab_res = res_df.groupby(["resolucao", "descritor"])["erro"].median().reset_index() \
        .rename(columns={"erro": "erro_relativo_mediano"})
    tab_res.to_csv(out / "q2_resolucao.csv", index=False)

    print("[3/4] Proxy de classificação (validação cruzada no treino)...")
    tab_proxy = avaliar_proxy(resultados, split, args.seed)
    tab_proxy.to_csv(out / "q2_proxy_resultados.csv", index=False)

    print("[4/4] Figuras e relatório...")
    fig_diagnosticos(mt, resultados, out)
    if len(brutos):
        fig_canais(tab_canais, brutos, out)
    if len(tab_res):
        fig_resolucao(tab_res, out)
    fig_proxy(tab_proxy, out)
    fig_exemplos(split[split.id_imagem.isin(amostra_ids)].reset_index(drop=True), idx, out)

    resumo = {
        "n_imagens": int(len(meta)), "n_treino": int((meta.conjunto == "treino").sum()),
        "n_teste": int((meta.conjunto == "teste").sum()), "tamanho_final": args.tamanho,
        "cor_do_fundo": fundo, "ruido_sigma": ruido, "contraste_interno_folha": contraste,
        "mascaras_vazias_nas_avaliacoes": vazias,
        "canais": tab_canais.round(4).to_dict(orient="records"),
        "convergencia_forma_erro_mediano_%": {
            f"{int(r)}": {d: round(100 * float(tab_res[(tab_res.resolucao == r) & (tab_res.descritor == d)]
                                                   ["erro_relativo_mediano"].iloc[0]), 2) for d in DESCRITORES}
            for r in sorted(tab_res.resolucao.unique())} if len(tab_res) else {},
        "proxy_acuracia_percent": {
            f"{v}|{c}": round(100 * float(a), 2) for v, c, a in
            zip(tab_proxy.variante, tab_proxy.conjunto_feats, tab_proxy.acc_media)},
        "cache": None if args.sem_cache else str(args.cache),
    }
    (out / "resumo_q2.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    piv_acc = tab_proxy[tab_proxy.variante != "referencia_classe_mais_frequente"].copy()
    piv_acc["acc"] = [f"{100 * m:.1f} ± {100 * s:.1f}" for m, s in zip(piv_acc.acc_media, piv_acc.acc_desvio)]
    piv = piv_acc.pivot(index="variante", columns="conjunto_feats", values="acc")[CONJ_FEATS].loc[list(VARIANTES)]
    piv = piv.reset_index()
    ref = tab_proxy[tab_proxy.variante == "referencia_classe_mais_frequente"].iloc[0]
    md = f"""# Questão 2 — Pré-processamento (texto-base gerado automaticamente)

> Revise, interprete as figuras e complete as conclusões.

## 2.1 Normalização do fundo
Cor do fundo (moldura) em {len(mt)} imagens de treino — média ± desvio (mín–máx):
R {fundo['bg_r']['media']:.1f} ± {fundo['bg_r']['desvio']:.1f} ({fundo['bg_r']['min']:.0f}–{fundo['bg_r']['max']:.0f}),
G {fundo['bg_g']['media']:.1f} ± {fundo['bg_g']['desvio']:.1f} ({fundo['bg_g']['min']:.0f}–{fundo['bg_g']['max']:.0f}),
B {fundo['bg_b']['media']:.1f} ± {fundo['bg_b']['desvio']:.1f} ({fundo['bg_b']['min']:.0f}–{fundo['bg_b']['max']:.0f}).
Figura: `fig_q2_diagnosticos.png`.

## 2.2 Ruído
σ do ruído no fundo (níveis de cinza, resolução nativa): mediana {ruido['mediana']:.2f}, p95 {ruido['p95']:.2f}, máx {ruido['max']:.2f}.
Efeito visual dos filtros: `fig_q2_zoom_textura.png`.

## 2.3 Contraste
Contraste (σ de cinza) dentro da folha: {contraste['folha_sem_clahe_media']:.1f} sem CLAHE → {contraste['folha_com_clahe_media']:.1f} com CLAHE
(razão média {contraste['razao_media']:.2f}).

## 2.4 Canal/espaço de cor para separar folha e fundo
{df_md(tab_canais.rename(columns={'canal': 'canal', 'eta_media': 'η médio', 'eta_desvio': 'η desvio', 'iou_media': 'IoU médio', 'falhas': 'falhas'}), 3)}

Figura: `fig_q2_canais.png`.

## 2.5 Resolução e interpolação
Erro relativo mediano (%) dos descritores de forma vs. referência (até {REF_MAX} px):

{df_md(tab_res.assign(erro_relativo_mediano=100 * tab_res.erro_relativo_mediano).pivot(index='resolucao', columns='descritor', values='erro_relativo_mediano').reset_index(), 2)}

Figura: `fig_q2_resolucao.png`.

## 2.6 Efeito na classificação (proxy — validação cruzada somente no treino)
Regressão logística + padronização, 5 dobras estratificadas por grupo, acurácia média ± desvio (%).
Referência (classe mais frequente): {100 * ref.acc_media:.1f}%.

{df_md(piv)}

Figura: `fig_q2_proxy.png`. Diferenças menores que o desvio entre dobras não são conclusivas.

## 2.7 Pipeline adotado
(completar após analisar as tabelas acima)
- Imagens finais salvas em `{args.cache}` ({args.tamanho}×{args.tamanho}, maior lado = {args.tamanho}, fundo normalizado).
"""
    (out / "relatorio_q2_trechos.md").write_text(md, encoding="utf-8")
    print(f"\nConcluído em {time.time() - t0:.0f}s. Resultados em: {out.resolve()}")


if __name__ == "__main__":
    main()