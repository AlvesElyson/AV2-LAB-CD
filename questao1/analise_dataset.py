#!/usr/bin/env python3
"""
AV2 - PDI | Questão 1: Análise e preparação do dataset (Swedish Leaf)

Uso:
    python av2_q1_analise_dataset.py --data /caminho/para/dataset --out resultados_q1

O script percorre recursivamente a pasta do dataset, então não importa
como as imagens estão organizadas (uma pasta por classe, tudo misturado, etc.).

O que ele faz (itens da Questão 1):
  1. Identifica classes e quantidade de imagens por classe
  2. Verifica imagens corrompidas, duplicadas (exatas e quase-duplicadas)
     e possíveis problemas de rotulagem
  3. Analisa o balanceamento entre classes
  4. Gera exemplos de imagens e estatísticas das dificuldades do problema
  5. Divide em treino/teste (estratificado, respeitando grupos de duplicatas)

Saídas (em --out):
  manifest.csv               uma linha por imagem (id, caminho, classe, hashes, tamanho, ...)
  split_treino_teste.csv     id, classe, conjunto (treino/teste)
  resumo_q1.json             números principais para citar no relatório
  relatorio_q1_trechos.md    texto-base com tabelas para colar no relatório
  fig_*.png                  gráficos e grids de exemplo

Dependências: numpy pandas matplotlib pillow scikit-learn
    pip install numpy pandas matplotlib pillow scikit-learn
"""

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False  # truncadas devem ser detectadas como erro

EXTENSOES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}
# Padrão do Swedish Leaf: l<classe>nr<número>.tif  (ex.: l3nr012.tif = folha 12 da classe 3)
PADRAO_SWEDISH = re.compile(r"^l(\d+)nr(\d+)", re.IGNORECASE)

# Nomes das espécies do Swedish Leaf (ordem oficial das 15 classes).
# Usados apenas para deixar os gráficos legíveis; se a classe não estiver aqui, usa o rótulo bruto.
ESPECIES = {
    1: "Ulmus carpinifolia",
    2: "Acer",
    3: "Salix aurita",
    4: "Quercus",
    5: "Alnus incana",
    6: "Betula pubescens",
    7: "Salix alba 'Sericea'",
    8: "Populus tremula",
    9: "Ulmus glabra",
    10: "Sorbus aucuparia",
    11: "Salix sinerea",
    12: "Populus",
    13: "Tilia",
    14: "Sorbus intermedia",
    15: "Fagus silvatica",
}


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def rotulo_legivel(classe: str) -> str:
    m = re.fullmatch(r"(?:l|class|classe)?_?(\d+)", str(classe), re.IGNORECASE)
    if m and int(m.group(1)) in ESPECIES:
        n = int(m.group(1))
        return f"{n} - {ESPECIES[n]}"
    return str(classe)


def listar_imagens(raiz: Path):
    for p in sorted(raiz.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSOES and not p.name.startswith("._"):
            yield p


def inferir_classe(caminho: Path, raiz: Path):
    """Retorna (classe_por_nome, classe_por_pasta)."""
    por_nome = None
    m = PADRAO_SWEDISH.match(caminho.stem)
    if m:
        por_nome = str(int(m.group(1)))

    por_pasta = None
    rel = caminho.relative_to(raiz)
    if len(rel.parts) > 1:
        pasta = rel.parts[-2]
        m2 = re.search(r"(\d+)", pasta)
        # se a pasta tem número (ex.: leaf3, class_3) usa o número; senão, o nome da pasta
        por_pasta = str(int(m2.group(1))) if m2 and (por_nome is not None) else pasta
    return por_nome, por_pasta


def dhash(img_gray: Image.Image, tam: int = 8) -> int:
    """Hash perceptual de diferenças (64 bits) para achar quase-duplicatas."""
    p = img_gray.resize((tam + 1, tam), Image.LANCZOS)
    a = np.asarray(p, dtype=np.float32)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def md5_arquivo(p: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(bloco):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Etapa 1: varredura (um arquivo por vez para não estourar a memória)
# ----------------------------------------------------------------------------
def varrer(raiz: Path) -> pd.DataFrame:
    linhas = []
    arquivos = list(listar_imagens(raiz))
    total = len(arquivos)
    print(f"[1/5] {total} arquivos de imagem encontrados. Analisando...")

    for i, p in enumerate(arquivos, 1):
        por_nome, por_pasta = inferir_classe(p, raiz)
        reg = {
            "id_imagem": p.stem,
            "caminho": str(p),
            "extensao": p.suffix.lower(),
            "tamanho_bytes": p.stat().st_size,
            "classe_nome_arquivo": por_nome,
            "classe_pasta": por_pasta,
            "md5": None,
            "dhash": None,
            "largura": None,
            "altura": None,
            "modo": None,
            "brilho_medio": None,
            "contraste_std": None,
            "frac_borda_clara": None,
            "corrompida": False,
            "erro": "",
        }
        try:
            reg["md5"] = md5_arquivo(p)
            with Image.open(p) as im:
                im.verify()  # detecta arquivos truncados/corrompidos
            with Image.open(p) as im:  # verify() invalida o objeto: reabrir
                im.load()
                reg["largura"], reg["altura"] = im.size
                reg["modo"] = im.mode
                g = im.convert("L")
                arr = np.asarray(g, dtype=np.float32)
                reg["brilho_medio"] = float(arr.mean())
                reg["contraste_std"] = float(arr.std())
                # fração de pixels claros (>200) na moldura de 5% — indica fundo claro/uniforme
                h, w = arr.shape
                b = max(1, int(0.05 * min(h, w)))
                moldura = np.concatenate(
                    [arr[:b].ravel(), arr[-b:].ravel(), arr[:, :b].ravel(), arr[:, -b:].ravel()]
                )
                reg["frac_borda_clara"] = float((moldura > 200).mean())
                reg["dhash"] = dhash(g)
        except Exception as e:  # noqa: BLE001
            reg["corrompida"] = True
            reg["erro"] = f"{type(e).__name__}: {e}"[:200]
        linhas.append(reg)
        if i % 100 == 0 or i == total:
            print(f"      {i}/{total}", end="\r", flush=True)
    print()
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Etapa 2: classes, rótulos, duplicatas
# ----------------------------------------------------------------------------
def definir_classe(df: pd.DataFrame):
    """Define a coluna 'classe' e sinaliza conflito entre nome do arquivo e pasta."""
    def escolher(r):
        if r["classe_nome_arquivo"] is not None and not pd.isna(r["classe_nome_arquivo"]):
            return r["classe_nome_arquivo"]
        return r["classe_pasta"]

    df["classe"] = df.apply(escolher, axis=1)
    conflito = (
        df["classe_nome_arquivo"].notna()
        & df["classe_pasta"].notna()
        & (df["classe_nome_arquivo"] != df["classe_pasta"])
    )
    df["conflito_rotulo"] = conflito
    return df


def achar_duplicatas(df: pd.DataFrame, limiar_hamming: int = 4):
    """Duplicatas exatas (MD5) e quase-duplicatas (dHash). Retorna df com 'grupo_dup'."""
    ok = df[~df["corrompida"]].copy()

    # União por componentes conexos (union-find)
    pai = {i: i for i in ok.index}

    def find(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            pai[rb] = ra

    # exatas
    for _, idxs in ok.groupby("md5").groups.items():
        idxs = list(idxs)
        for j in idxs[1:]:
            union(idxs[0], j)
    n_exatas = int((ok.duplicated("md5", keep="first")).sum())

    # quase-duplicatas: compara hashes (N pequeno -> O(N^2) é aceitável até ~10k imagens)
    idx = list(ok.index)
    hashes = ok["dhash"].astype("uint64").tolist() if len(ok) else []
    pares_quase = []
    if len(idx) <= 15000:
        for a in range(len(idx)):
            ha = int(hashes[a])
            for b in range(a + 1, len(idx)):
                if hamming(ha, int(hashes[b])) <= limiar_hamming and ok.at[idx[a], "md5"] != ok.at[idx[b], "md5"]:
                    pares_quase.append((idx[a], idx[b]))

    else:
        print("      aviso: muitas imagens; busca de quase-duplicatas ignorada.")

    df["grupo_dup"] = -1
    raizes = {i: find(i) for i in ok.index}
    mapa = {r: k for k, r in enumerate(sorted(set(raizes.values())))}
    for i, r in raizes.items():
        df.at[i, "grupo_dup"] = mapa[r]
    # corrompidas: cada uma em seu próprio grupo
    base = (max(mapa.values()) + 1) if mapa else 0
    for k, i in enumerate(df.index[df["corrompida"]]):
        df.at[i, "grupo_dup"] = base + k

    tam = df.groupby("grupo_dup")["grupo_dup"].transform("size")
    df["em_grupo_duplicado"] = tam > 1
    return df, n_exatas, pares_quase


# ----------------------------------------------------------------------------
# Etapa 3: gráficos
# ----------------------------------------------------------------------------
def fig_balanceamento(contagem: pd.Series, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    rotulos = [rotulo_legivel(c) for c in contagem.index]
    barras = ax.bar(range(len(contagem)), contagem.values, color="#4C78A8")
    ax.axhline(contagem.mean(), color="crimson", ls="--", label=f"média = {contagem.mean():.1f}")
    ax.set_xticks(range(len(contagem)))
    ax.set_xticklabels(rotulos, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Nº de imagens")
    ax.set_title("Distribuição de imagens por classe")
    for b, v in zip(barras, contagem.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_balanceamento.png", dpi=150)
    plt.close(fig)


def fig_exemplos(df: pd.DataFrame, out: Path, por_classe: int = 4, seed: int = 42):
    classes = sorted(df["classe"].unique(), key=lambda c: (len(str(c)), str(c)))
    n = len(classes)
    fig, axes = plt.subplots(n, por_classe, figsize=(2.2 * por_classe, 1.9 * n), squeeze=False)
    rng = np.random.RandomState(seed)
    for r, c in enumerate(classes):
        sub = df[(df["classe"] == c) & (~df["corrompida"])]
        escolha = sub.sample(min(por_classe, len(sub)), random_state=rng) if len(sub) else sub
        for k in range(por_classe):
            ax = axes[r][k]
            ax.axis("off")
            if k < len(escolha):
                try:
                    with Image.open(escolha.iloc[k]["caminho"]) as im:
                        im = im.convert("RGB")
                        im.thumbnail((300, 300))
                        ax.imshow(im)
                except Exception:  # noqa: BLE001
                    pass
            if k == 0:
                ax.set_title(rotulo_legivel(c), fontsize=7, loc="left")
    fig.suptitle("Exemplos por classe", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig_exemplos_por_classe.png", dpi=120)
    plt.close(fig)


def fig_dificuldades(df: pd.DataFrame, out: Path):
    ok = df[~df["corrompida"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0][0].scatter(ok["largura"], ok["altura"], s=8, alpha=0.5)
    axes[0][0].set_xlabel("Largura (px)")
    axes[0][0].set_ylabel("Altura (px)")
    axes[0][0].set_title("Dimensões das imagens")

    axes[0][1].hist(ok["brilho_medio"], bins=30, color="#F58518")
    axes[0][1].set_title("Brilho médio (escala de cinza)")
    axes[0][1].set_xlabel("Brilho")

    axes[1][0].hist(ok["contraste_std"], bins=30, color="#54A24B")
    axes[1][0].set_title("Contraste (desvio-padrão dos tons de cinza)")
    axes[1][0].set_xlabel("Desvio-padrão")

    axes[1][1].hist(ok["frac_borda_clara"], bins=30, color="#B279A2")
    axes[1][1].set_title("Fração de moldura clara (fundo uniforme)")
    axes[1][1].set_xlabel("Fração")

    fig.tight_layout()
    fig.savefig(out / "fig_dificuldades.png", dpi=150)
    plt.close(fig)

    ok = df[~df["corrompida"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = sorted(ok["classe"].unique(), key=lambda c: (len(str(c)), str(c)))
    ax.boxplot([ok[ok["classe"] == c]["largura"].values for c in classes], tick_labels=classes)
    ax.set_xlabel("Classe");
    ax.set_ylabel("Largura (px)")
    ax.set_title("Largura da imagem por classe")
    fig.tight_layout();
    fig.savefig(out / "fig_largura_por_classe.png", dpi=150);
    plt.close(fig)

def fig_brilho_por_classe(df: pd.DataFrame, out: Path):
    ok = df[~df["corrompida"]]
    classes = sorted(ok["classe"].unique(), key=lambda c: (len(str(c)), str(c)))
    dados = [ok[ok["classe"] == c]["brilho_medio"].values for c in classes]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(dados, tick_labels=[str(c) for c in classes])
    ax.set_xlabel("Classe")
    ax.set_ylabel("Brilho médio")
    ax.set_title("Brilho médio por classe (possível viés de aquisição)")
    fig.tight_layout()
    fig.savefig(out / "fig_brilho_por_classe.png", dpi=150)
    plt.close(fig)


def fig_pares_duplicados(df: pd.DataFrame, pares, out: Path, maximo: int = 6):
    if not pares:
        return
    pares = pares[:maximo]
    fig, axes = plt.subplots(len(pares), 2, figsize=(5, 2.4 * len(pares)), squeeze=False)
    for r, (a, b) in enumerate(pares):
        for k, i in enumerate((a, b)):
            ax = axes[r][k]
            ax.axis("off")
            try:
                with Image.open(df.at[i, "caminho"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((300, 300))
                    ax.imshow(im)
                ax.set_title(f'{df.at[i, "id_imagem"]} (cl. {df.at[i, "classe"]})', fontsize=7)
            except Exception:  # noqa: BLE001
                pass
    fig.suptitle("Pares quase-duplicados")
    fig.tight_layout()
    fig.savefig(out / "fig_quase_duplicatas.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Etapa 4: divisão treino/teste
# ----------------------------------------------------------------------------
def dividir(df: pd.DataFrame, prop_teste: float, seed: int):
    """
    Divisão estratificada por classe, mantendo imagens do mesmo grupo
    (duplicatas/quase-duplicatas) no mesmo conjunto -> evita vazamento.
    Se não houver grupos duplicados, equivale a um train_test_split estratificado.
    """
    ok = df[~df["corrompida"]].copy()
    y = ok["classe"].values
    grupos = ok["grupo_dup"].values
    tem_grupos = ok["em_grupo_duplicado"].any()

    if tem_grupos:
        k = max(2, int(round(1 / prop_teste)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        tr, te = next(sgkf.split(ok, y, grupos))
        metodo = f"StratifiedGroupKFold(n_splits={k}) — primeira dobra como teste"
    else:
        idx = np.arange(len(ok))
        tr, te = train_test_split(idx, test_size=prop_teste, stratify=y, random_state=seed)
        metodo = f"train_test_split estratificado (test_size={prop_teste})"

    ok["conjunto"] = "treino"
    ok.iloc[te, ok.columns.get_loc("conjunto")] = "teste"

    # checagem anti-vazamento
    g_tr = set(ok.loc[ok["conjunto"] == "treino", "grupo_dup"])
    g_te = set(ok.loc[ok["conjunto"] == "teste", "grupo_dup"])
    assert not (g_tr & g_te), "Vazamento: grupo presente em treino e teste!"
    return ok[["id_imagem", "classe", "grupo_dup", "conjunto", "caminho"]], metodo


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AV2 PDI - Q1: análise e preparação do dataset")
    RAIZ_PROJETO = Path(__file__).resolve().parent.parent
    ap.add_argument("--data", default=str(RAIZ_PROJETO / "dataset"), help="pasta raiz do dataset (busca recursiva)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "resultados"), help="pasta de saída")
    ap.add_argument("--teste", type=float, default=0.2, help="proporção do conjunto de teste")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limiar-hamming", type=int, default=4,
                    help="distância máx. entre dHashes para considerar quase-duplicata")
    args = ap.parse_args()

    raiz = Path(args.data).expanduser().resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not raiz.is_dir():
        sys.exit(f"Pasta não encontrada: {raiz}")

    df = varrer(raiz)
    if df.empty:
        sys.exit(f"Nenhuma imagem ({', '.join(sorted(EXTENSOES))}) encontrada em {raiz}")

    print("[2/5] Classes, rótulos e duplicatas...")
    df = definir_classe(df)
    sem_classe = df["classe"].isna().sum()
    df["classe"] = df["classe"].fillna("desconhecida").astype(str)
    df, n_exatas, pares_quase = achar_duplicatas(df, args.limiar_hamming)

    ok = df[~df["corrompida"]]
    contagem = ok["classe"].value_counts()
    contagem = contagem.reindex(sorted(contagem.index, key=lambda c: (len(c), c)))
    razao = contagem.max() / contagem.min() if contagem.min() > 0 else float("inf")

    print("[3/5] Gerando figuras...")
    fig_balanceamento(contagem, out)
    fig_exemplos(df, out)
    fig_dificuldades(df, out)
    fig_brilho_por_classe(df, out)
    fig_pares_duplicados(df, pares_quase, out)

    print("[4/5] Dividindo treino/teste...")
    split, metodo = dividir(df, args.teste, args.seed)
    split.drop(columns=["caminho"]).to_csv(out / "split_treino_teste.csv", index=False)
    cont_split = split.groupby(["classe", "conjunto"]).size().unstack(fill_value=0)
    cont_split = cont_split.reindex(contagem.index)

    print("[5/5] Salvando resultados...")
    df.drop(columns=["dhash"]).to_csv(out / "manifest.csv", index=False)

    # outliers de tamanho/modo
    modos = ok["modo"].value_counts().to_dict()
    dims = ok.groupby(["largura", "altura"]).size().sort_values(ascending=False)
    dim_comum = dims.index[0] if len(dims) else None
    n_dim_diferente = int(len(ok) - dims.iloc[0]) if len(dims) else 0

    resumo = {
        "pasta_dataset": str(raiz),
        "total_arquivos": int(len(df)),
        "imagens_validas": int(len(ok)),
        "imagens_corrompidas": int(df["corrompida"].sum()),
        "num_classes": int(contagem.shape[0]),
        "imagens_por_classe": {str(k): int(v) for k, v in contagem.items()},
        "razao_maior_menor_classe": float(razao),
        "duplicatas_exatas_md5": int(n_exatas),
        "pares_quase_duplicatas": int(len(pares_quase)),
        "imagens_em_grupos_duplicados": int(df["em_grupo_duplicado"].sum()),
        "conflitos_rotulo_arquivo_vs_pasta": int(df["conflito_rotulo"].sum()),
        "sem_classe_identificada": int(sem_classe),
        "modos_de_cor": {str(k): int(v) for k, v in modos.items()},
        "dimensao_mais_comum": [int(x) for x in dim_comum] if dim_comum else None,
        "imagens_com_outra_dimensao": n_dim_diferente,
        "estrategia_divisao": metodo,
        "proporcao_teste": args.teste,
        "seed": args.seed,
        "treino": int((split["conjunto"] == "treino").sum()),
        "teste": int((split["conjunto"] == "teste").sum()),
    }
    (out / "resumo_q1.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- texto-base para o relatório ----
    tabela = pd.DataFrame(
        {
            "Classe": [rotulo_legivel(c) for c in contagem.index],
            "Imagens": contagem.values,
            "Treino": cont_split["treino"].values if "treino" in cont_split else 0,
            "Teste": cont_split["teste"].values if "teste" in cont_split else 0,
        }
    )
    linhas_md = ["| " + " | ".join(tabela.columns) + " |", "|" + "|".join(["---"] * len(tabela.columns)) + "|"]
    for _, r in tabela.iterrows():
        linhas_md.append("| " + " | ".join(str(v) for v in r.values) + " |")

    md = f"""# Questão 1 — Análise e preparação do dataset

> Texto-base gerado automaticamente. Revise, interprete os gráficos e complete com sua análise.

## 1.1 Classes e quantidade de imagens
O dataset contém **{resumo['num_classes']} classes** e **{resumo['imagens_validas']} imagens válidas**.

{chr(10).join(linhas_md)}

## 1.2 Integridade e rotulagem
- Imagens corrompidas: **{resumo['imagens_corrompidas']}**
- Duplicatas exatas (MD5): **{resumo['duplicatas_exatas_md5']}**
- Pares quase-duplicados (dHash, distância de Hamming ≤ {args.limiar_hamming}): **{resumo['pares_quase_duplicatas']}**
- Conflitos entre rótulo do nome do arquivo e da pasta: **{resumo['conflitos_rotulo_arquivo_vs_pasta']}**
- Imagens sem classe identificada: **{resumo['sem_classe_identificada']}**

## 1.3 Balanceamento
Razão entre a maior e a menor classe: **{razao:.2f}** (1,00 = perfeitamente balanceado).
Ver `fig_balanceamento.png`.

## 1.4 Exemplos e dificuldades
- Modos de cor: {modos}
- Dimensão mais comum: {dim_comum} ({n_dim_diferente} imagens com outra dimensão)
- Figuras: `fig_exemplos_por_classe.png`, `fig_dificuldades.png`, `fig_brilho_por_classe.png`
- (Complete com observações visuais: variação de forma intra-classe, semelhança entre espécies,
  folhas com pecíolo, sombras, fundo, escala/rotação etc.)

## 1.5 Divisão treino/teste
- Estratégia: **{metodo}**, semente {args.seed}.
- Treino: **{resumo['treino']}** | Teste: **{resumo['teste']}**
- Justificativa: estratificação mantém a proporção de cada classe nos dois conjuntos; grupos de
  duplicatas/quase-duplicatas ficam no mesmo conjunto para evitar vazamento. No Swedish Leaf cada
  imagem é uma folha distinta (sem sequências do mesmo indivíduo), então o agrupamento só atua se
  forem encontradas duplicatas.
- O conjunto de teste fica reservado para a avaliação final (Questão 5).
"""
    (out / "relatorio_q1_trechos.md").write_text(md, encoding="utf-8")

    print("\n=== RESUMO ===")
    print(json.dumps({k: v for k, v in resumo.items() if k != "imagens_por_classe"}, indent=2, ensure_ascii=False))
    print(f"\nResultados em: {out.resolve()}")


if __name__ == "__main__":
    main()