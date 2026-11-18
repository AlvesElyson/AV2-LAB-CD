# Questão 2 — Pré-processamento das imagens

**Código:** `questao2/preprocessamento.py` · **Saídas:** `questao2/resultados/` · **Imagens finais:** `questao2/cache_256/`

## 2.1 Abordagem

Cada operação foi **medida com os dados antes de ser adotada**, de três formas:
1. **diagnóstico direto** das imagens (cor do fundo, ruído, contraste, separação folha/fundo, convergência dos descritores de forma);
2. **inspeção visual** do efeito de cada operação (`fig_q2_exemplos_pipeline.png`, `fig_q2_zoom_textura.png`);
3. **um classificador de teste (proxy)**: regressão logística com padronização, validação cruzada de 5 dobras estratificada por grupo (`StratifiedGroupKFold`), usando **apenas as 900 imagens de treino**. A acurácia da classe mais frequente é 6,7%. O conjunto de teste não foi usado para nenhuma decisão; ele só recebeu o pré-processamento final.

O proxy usa características simples (cor média/desvio na folha em RGB e Lab, histogramas LBP e 12 descritores de forma, incluindo momentos de Hu) calculadas sobre uma máscara de Otsu. Ele serve para **comparar variantes de pré-processamento**, não é o classificador final (Questões 4 e 5). Diferenças menores que o desvio entre dobras (≈ 1 a 3 pontos percentuais) não são conclusivas.

## 2.2 Normalização do fundo

**Diagnóstico.** A cor do fundo (moldura) nas 900 imagens de treino é R 251,4 ± 1,8, G 247,4 ± 1,8 e B 253,2 ± 1,5 (`fig_q2_diagnosticos.png`). Ou seja, o fundo é **quase uniforme entre imagens (variação de ~2 níveis), mas não é branco puro**: tem tom levemente magenta (G ≈ 4 a 6 níveis abaixo de R e B).

**Operação adotada.** Balanço de branco pelo fundo: cada canal é multiplicado para que a mediana da moldura vire 255.

**Avaliação.** No proxy, a variante sem normalização (`res256_area_raw`) e com normalização (`res256_area_wb`) ficaram praticamente iguais (cor 69,1 vs 69,0; textura 88,7 vs 88,9; forma 86,4 vs 86,8; todas 98,3 vs 98,3). **Não há ganho mensurável de acurácia.**

**Justificativa para manter.** Como o custo é desprezível, a normalização torna o fundo exatamente branco, o que dá um limiar de segmentação mais estável na Questão 3 e permite preencher a moldura dos redimensionamentos com branco sem criar bordas artificiais. A decisão se apoia nessa consistência, e não em ganho de desempenho.

## 2.3 Redução de ruído

**Diagnóstico.** O ruído do scanner, estimado no fundo em resolução nativa (método de Immerkaer com mediana), é **muito baixo: σ mediano de 0,74 níveis de cinza (p95 = 0,99, máx = 0,99)**. O desvio-padrão do fundo (~2 a 3 níveis) vem de sombras, poeira e gradientes de iluminação, não de ruído.

**Avaliação.** Os filtros pioraram o resultado:

| Variante (256 px) | Textura | Forma | Todas |
|---|---|---|---|
| Sem filtro | 88,9 ± 3,2 | 86,8 ± 2,8 | 98,3 ± 1,0 |
| Gaussiano σ = 1 | 86,4 ± 3,1 | 84,0 ± 2,1 | 98,0 ± 0,7 |
| Mediana 3×3 | 84,9 ± 1,6 | 83,6 ± 3,5 | 98,1 ± 0,6 |

O zoom em `fig_q2_zoom_textura.png` mostra que o gaussiano borra as nervuras, que são justamente a textura discriminativa entre espécies, e a mediana também as atenua.

**Decisão: não aplicar redução de ruído.** Não há ruído relevante a remover, e os filtros removem informação útil.

## 2.4 Ajuste de contraste (CLAHE)

**Diagnóstico.** O contraste interno da folha (desvio-padrão dos tons de cinza dentro da máscara) é baixo, de 13,0 em média, porque as folhas são escuras e uniformes. Com CLAHE (clip 2,0, blocos 8×8, aplicado no canal L) ele sobe para 27,1 (×2,08), e as nervuras ficam bem mais visíveis (`fig_q2_zoom_textura.png`).

**Avaliação no proxy (256 px):**

| Variante | Cor | Textura | Forma | Todas |
|---|---|---|---|---|
| Sem CLAHE | 69,0 ± 2,6 | 88,9 ± 3,2 | 86,8 ± 2,8 | 98,3 ± 1,0 |
| Com CLAHE | 72,9 ± 3,4 | 87,1 ± 2,2 | 86,8 ± 2,8 | 98,4 ± 1,0 |

A cor subiu ≈ 4 pontos, mas dentro de ~1 desvio entre dobras, e a textura LBP caiu ≈ 2 pontos. Na forma não há diferença (a máscara é calculada sem CLAHE) e no conjunto completo é igual. **O resultado não é conclusivo.**

**Decisão: o CLAHE não entra no pipeline base.** O efeito não é demonstrado e ele altera a distribuição dos tons que as características de cor usam. Como ele aumenta o contraste das nervuras, pode ajudar características de gradiente (HOG, densidade de bordas), que o proxy não mede. Por isso ele será avaliado como **ablação nas Questões 4 a 6**, aplicado sobre as imagens do cache.

## 2.5 Espaço de cor / canal para separar folha e fundo

Foram comparados 10 canais em uma amostra de 75 imagens de treino (5 por classe), pela separabilidade de Otsu (η) e pela concordância (IoU) com a máscara de consenso (`fig_q2_canais.png`):

| Canal | η médio | IoU médio |
|---|---|---|
| B | 0,984 | 0,994 |
| Cinza | 0,981 | 0,999 |
| R | 0,980 | 0,996 |
| V (HSV) | 0,980 | 0,996 |
| G | 0,979 | 0,997 |
| Distância ao fundo (Lab) | 0,979 | 0,998 |
| L (Lab) | 0,978 | 0,997 |
| b (Lab) | 0,960 | 0,977 |
| S (HSV) | 0,949 | 0,963 |
| a (Lab) | 0,833 | 0,789 |

Todos os canais de luminância e do RGB separam folha e fundo quase igualmente bem (η ≈ 0,98), por causa do fundo claro e uniforme contra folhas escuras. Os canais de crominância (a, b) e a saturação são piores e menos estáveis. O IoU com o consenso é uma medida enviesada, porque o consenso é feito com os próprios canais (muito correlacionados), então ele serve só para detectar canais que destoam.

**Decisão:** converter para **escala de cinza** para a segmentação por limiarização de Otsu (Questão 3). É a opção mais simples, com desempenho equivalente ao melhor canal; a distância ao fundo não traz ganho. As características de cor (Questão 4) serão calculadas em RGB e Lab, apenas sobre os pixels da folha.

## 2.6 Redimensionamento

**Necessidade.** As imagens originais variam de 364 a 2.550 px de largura (Questão 1). O fator de redução ficou em média em 0,16 (≈ 6× menor), o que também reduz muito o custo de cálculo.

**Convergência dos descritores de forma.** O erro relativo mediano de cada descritor, comparado à imagem em até 1.024 px (`fig_q2_resolucao.png`), é:

| Maior lado | Aspecto | Extent | Solidity | Excentricidade | Circularidade |
|---|---|---|---|---|---|
| 64 | 11,3% | 15,2% | 6,4% | 0,7% | 104,4% |
| 128 | 1,3% | 2,7% | 1,8% | 0,4% | 67,7% |
| 256 | 0,3% | 0,5% | 0,3% | 0,1% | 36,0% |
| 512 | 0,2% | 0,2% | 0,1% | 0,1% | 15,6% |

Aspecto, extent, solidity e excentricidade **convergem a partir de 128 px** e ficam abaixo de 0,5% em 256 px. A **circularidade não converge**: ela depende do perímetro, que cresce com a resolução à medida que as serrilhas da borda aparecem. Consequência: perímetro e circularidade só têm sentido se calculados em **resolução fixa** (a mesma para todas as imagens) e não devem ser interpretados como valores absolutos.

**Resolução e classificação (proxy):**

| Maior lado | Cor | Textura | Forma | Todas |
|---|---|---|---|---|
| 128 | 68,2 | 83,9 | 84,4 | 98,1 |
| 256 | 69,0 | 88,9 | 86,8 | 98,3 |
| 512 | 71,2 | 90,4 | 87,1 | 98,7 |

Passar de 128 para 256 px melhora a textura em ≈ 5 pontos (maior que o desvio). De 256 para 512, o ganho é de ≈ 1,5 ponto (dentro do desvio), com 4× mais pixels.

**Interpolação.** Com `INTER_AREA` (média da área, adequada para reduzir ≈ 6×), a textura chega a 88,9%. Com interpolação bilinear ela cai para 78,6% e com vizinho mais próximo para 77,0%, uma perda de ≈ 10 pontos, bem acima do desvio. A explicação provável é o **aliasing**: sem pré-filtragem, a redução forte cria padrões falsos nas nervuras. (Na cor, o vizinho mais próximo teve acurácia maior, 74,4%, hipótese: o desvio-padrão da cor, ao capturar essas altas frequências, carrega informação de textura.)

**Decisão:** maior lado = **256 px, proporção preservada** (sem distorcer a forma), completando com branco até 256×256, usando `INTER_AREA`. A proporção preservada evita deformar a folha e a moldura branca mantém o fundo uniforme. O tamanho original fica só como metadado e **não** é usado como característica (Questão 1).

## 2.7 Pipeline final

| Etapa | Decisão | Justificativa |
|---|---|---|
| 1. Normalização do fundo | Balanço de branco pelo fundo | Fundo com leve tom magenta; garante fundo branco consistente. Sem ganho mensurável de acurácia |
| 2. Redimensionamento | Maior lado = 256 px, proporção preservada, moldura branca, `INTER_AREA` | Forma estável a partir de 128 px; ganho de textura de 128 para 256; `INTER_AREA` evita aliasing (+10 pontos) |
| 3. Redução de ruído | **Não aplicar** | Ruído desprezível (σ < 1); os filtros reduziram textura e forma |
| 4. Contraste (CLAHE) | **Fora do pipeline base**; ablação nas Questões 4 a 6 | Resultado inconclusivo; pode ajudar características de gradiente |
| 5. Espaço de cor | Cinza para segmentar (Otsu); RGB/Lab na folha para cor | Todos os canais de luminância separam bem; cinza é o mais simples |
| 6. Padronização numérica | Feita nas Questões 4 e 5, ajustada **apenas no treino** | Evita vazamento de dados |

As 1.125 imagens (900 treino e 225 teste) foram processadas com este pipeline e salvas em `questao2/cache_256/` (uma pasta por classe), e nenhuma máscara de avaliação resultou vazia.

**Limitações.** O proxy usa um único classificador simples e características básicas; com todas as características juntas a acurácia satura em ≈ 98% para qualquer variante, então a comparação só é informativa quando as características são avaliadas separadamente. Como a validação cruzada é feita no treino, esses números não são o desempenho final, que será medido no teste (Questão 5).