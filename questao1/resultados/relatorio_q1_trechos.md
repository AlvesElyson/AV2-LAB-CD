# Questão 1 — Análise e preparação do dataset

**Dataset:** Swedish Leaf (Linköping University) — imagens escaneadas de folhas de 15 espécies arbóreas suecas.
**Código:** `questao1/analise_dataset.py` · **Saídas:** `questao1/resultados/`

## 1.1 Classes e quantidade de imagens

O dataset contém **1.125 imagens em 15 classes, com exatamente 75 imagens por classe**. Todas as imagens são RGB e os rótulos vêm da estrutura de pastas (`leaf1` … `leaf15`), confirmada pelo padrão do nome do arquivo (`l<classe>nr<número>`).

| Classe | Espécie | Imagens | | Classe | Espécie | Imagens |
|---|---|---|---|---|---|---|
| 1 | *Ulmus carpinifolia* | 75 | | 9 | *Ulmus glabra* | 75 |
| 2 | *Acer* | 75 | | 10 | *Sorbus aucuparia* | 75 |
| 3 | *Salix aurita* | 75 | | 11 | *Salix sinerea* | 75 |
| 4 | *Quercus* | 75 | | 12 | *Populus* | 75 |
| 5 | *Alnus incana* | 75 | | 13 | *Tilia* | 75 |
| 6 | *Betula pubescens* | 75 | | 14 | *Sorbus intermedia* | 75 |
| 7 | *Salix alba 'Sericea'* | 75 | | 15 | *Fagus silvatica* | 75 |
| 8 | *Populus tremula* | 75 | | | **Total** | **1.125** |

## 1.2 Integridade, duplicatas e rotulagem

| Verificação | Resultado |
|---|---|
| Imagens corrompidas (abertura e `verify` do Pillow) | 0 |
| Conflitos entre rótulo do nome do arquivo e da pasta | 0 |
| Imagens sem classe identificada | 0 |
| Duplicatas exatas (MD5 do arquivo) | **3 pares**, todos na classe 8 |
| Pares apontados como quase-duplicatas (dHash 8×8, Hamming ≤ 4) | 153 |

**Duplicatas exatas.** Os três pares (`l8nr016/017`, `l8nr031/032`, `l8nr070/071`) têm numeração consecutiva e conteúdo byte a byte idêntico, ou seja, a mesma folha foi escaneada e salva duas vezes. Para evitar vazamento, cada par permanece no mesmo conjunto na divisão.

**Quase-duplicatas.** O hash perceptual de 64 bits sinalizou 153 pares, mas a inspeção visual (`fig_quase_duplicatas.png`, amostra dos 6 primeiros pares) mostra folhas **distintas**: furos, tons de cor e nervuras não coincidem. Isso é esperado, pois um hash de 8×8 pixels de uma silhueta oval sobre fundo branco é pouco discriminativo, e vários pares ligam classes diferentes (por exemplo, 1 e 9, ambas *Ulmus*). Como o Swedish Leaf tem uma folha distinta por imagem, esses pares foram tratados como falsos positivos e **não** foram usados para agrupar imagens. Só as duplicatas exatas formam grupos (6 imagens no total).

## 1.3 Balanceamento

As classes são **perfeitamente balanceadas** (razão maior/menor = 1,00; `fig_balanceamento.png`). Consequências para as etapas seguintes:
- a acurácia é uma métrica adequada, mas o F1 macro continua sendo reportado, como pede o enunciado;
- o classificador de referência (classe mais frequente) terá acurácia de ≈ 1/15 ≈ 6,7%;
- não é necessário balanceamento por pesos nem reamostragem.

## 1.4 Exemplos e principais dificuldades

`fig_exemplos_por_classe.png` mostra 4 amostras por classe. Pontos observados:

1. **Fundo claro e uniforme.** Em praticamente todas as imagens, a moldura é branca (fração de pixels claros na borda ≈ 1,0, `fig_dificuldades.png`). Isso favorece a separação folha/fundo por limiarização (Questão 3).
2. **Similaridade entre espécies.** Há grupos visualmente próximos: as classes 1 e 9 (*Ulmus*), as classes 3, 7 e 11 (*Salix*) e as folhas ovais serrilhadas de 3, 9 e 15. Espera-se que a confusão entre classes se concentre nesses grupos.
3. **Variabilidade intraclasse.** Dentro da mesma classe há variação de formato, rotação, orientação e danos (furos, bordas rasgadas, dobras), por exemplo na classe 6.
4. **Presença de pecíolo.** Algumas classes costumam ter pecíolo longo e visível (2, 5, 8, 12, 13) e outras não (1, 3, 9, 15). O pecíolo altera área, perímetro e proporções, e pode favorecer ou atrapalhar a classificação por forma.
5. **Folhas compostas.** As classes 10 e 14 têm folíolos separados. A segmentação precisa preservar regiões desconexas, e a área convexa ou preenchida terá comportamento diferente das folhas simples.
6. **Cor semelhante.** A cor é predominantemente verde-oliva escuro em todas as classes, o que limita o poder discriminativo das características de cor isoladas (a ser testado na Questão 4).

### Dimensões das imagens e risco de atalho

As imagens têm tamanhos muito variados: largura de 364 a 2.550 px (mediana 1.004 px) e altura de 652 a 4.125 px (mediana 1.655 px). Somente 3 imagens compartilham a dimensão mais comum (829×1128).

A distribuição **depende fortemente da classe** (`fig_largura_por_classe.png`): a largura mediana vai de ≈ 490 px (classe 7) a ≈ 2.035 px (classe 2). A origem provável é o corte das imagens ao redor da folha, de modo que o tamanho do arquivo reflete o tamanho da folha. Como diagnóstico exploratório, um Random Forest treinado **apenas com largura e altura** da imagem atingiu **≈ 38,6% de acurácia** (validação cruzada de 5 dobras, contra 6,7% do acaso). Ou seja, as dimensões do arquivo carregam informação sobre a classe sem que o classificador "olhe" para a folha.

Decisões decorrentes:
- largura, altura e resolução do arquivo **não serão usadas como características**;
- o redimensionamento da Questão 2 deve ser feito de forma controlada, preservando a proporção, para que o tamanho final não vire um atalho;
- se uma medida de tamanho da folha for usada como característica de forma (área, perímetro), isso será declarado e discutido na Questão 6, pois a escala pode refletir o corte da imagem e não só a morfologia.

### Brilho e contraste

O brilho médio varia por classe (`fig_brilho_por_classe.png`): as classes 2, 7 e 13 são mais claras (≈ 200 em escala de cinza) e as classes 9, 1, 3 e 11 mais escuras (≈ 167–172). Como o fundo é branco, o brilho médio mistura a **cor da folha** com a **proporção de fundo** no quadro (folhas pequenas ou estreitas dentro de um quadro grande deixam a imagem mais clara). Um Random Forest só com brilho e contraste médios chegou a ≈ 35,6% de acurácia, outro sinal de que estatísticas globais da imagem capturam o recorte e não apenas a folha. Por isso, as características de cor devem ser calculadas **apenas sobre os pixels da folha**, depois da segmentação.

## 1.5 Divisão treino/teste

- **Proporção:** 80% treino (900 imagens) e 20% teste (225 imagens), com semente fixa (42).
- **Estratégia:** divisão **estratificada por classe** (`StratifiedGroupKFold`, primeira dobra de 5 usada como teste), com grupos formados pelas duplicatas exatas. Assim, todas as classes aparecem nos dois conjuntos na mesma proporção (60/15 por classe, com ajuste mínimo só se um par duplicado exigir) e **nenhum par duplicado é separado** entre treino e teste, atendendo à exigência de evitar vazamento. O script verifica isso com uma checagem automática.
- **Por que não aleatória simples:** com 75 imagens por classe, uma divisão aleatória poderia desbalancear as classes no teste.
- **Uso do conjunto de teste:** ele fica **reservado para a avaliação final** (Questão 5). Padronização, seleção de características, PCA e ajuste de hiperparâmetros usam apenas o treino, com validação cruzada interna.
- **Reprodutibilidade:** a divisão está salva em `questao1/resultados/split_treino_teste.csv` e será reutilizada em todas as etapas seguintes.

## 1.6 Resumo das decisões desta etapa

| Achado | Decisão |
|---|---|
| Classes balanceadas (75 por classe) | Sem reamostragem; reportar também F1 macro |
| 3 duplicatas exatas (classe 8) | Mantidas no mesmo conjunto |
| 153 quase-duplicatas | Falsos positivos (inspeção visual); não agrupam |
| Dimensões da imagem preveem a classe (≈ 39%) | Não usar tamanho do arquivo como característica; redimensionar com cuidado |
| Brilho e contraste globais dependem do recorte | Calcular cor só sobre os pixels da folha |
| Fundo branco uniforme | Segmentação por limiarização é promissora |
| Pecíolo e folhas compostas | Avaliar remoção do pecíolo e preservar regiões desconexas |