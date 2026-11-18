# Questão 3 — Segmentação

**Código:** `questao3/segmentacao.py` · **Saídas:** `questao3/resultados/` · **Máscaras:** `questao3/mascaras_256/` e `questao3/mascaras_256_sem_peciolo/`

## 3.1 A segmentação é necessária?

Sim. As imagens têm fundo branco e uniforme (Questões 1 e 2) e folhas escuras, o que torna a separação objeto/fundo simples. Há três motivos para segmentar, e os dois últimos aparecem nos resultados da seção 3.4:

1. **Características de forma** (área, solidez, excentricidade, momentos de Hu) só existem com a silhueta da folha.
2. **Cor e textura** devem ser medidas só nos pixels da folha. Na imagem inteira, o fundo branco entra nas médias e histogramas.
3. A **fração de folha no quadro** varia muito entre as classes (de 7% na classe 7 a 28% nas classes 1 e 6; `fig_q3_por_classe.png`) e, sozinha, prevê a classe com 26,3% de acurácia, contra 6,7% do acaso. Esse é um atalho herdado do recorte das imagens (Questão 1), e a segmentação permite calcular características que não dependem dele.

## 3.2 Método

A segmentação foi feita sobre as imagens de 256 px da Questão 2 (fundo normalizado para branco):

1. **Conversão para escala de cinza** (canal escolhido na Questão 2).
2. **Limiarização de Otsu**, com a folha como a classe escura. O histograma é bimodal, com um pico escuro (folha) e um pico em ≈ 255 (fundo) (`fig_q3_histogramas.png`). O limiar é calculado por imagem: mediana 160, entre 144 e 174.
3. **Limpeza morfológica:** remoção de componentes menores que max(40 px, 1% do maior componente) e preenchimento de buracos internos menores que 0,3% do quadro. Componentes grandes e separados são mantidos, para preservar os folíolos das classes 10 e 14.

Exemplos em `fig_q3_pipeline.png` e `fig_q3_exemplos_por_classe.png` (uma imagem por classe, com o contorno da máscara). As máscaras seguem bem o contorno das 15 espécies, incluindo dentes, lobos e folíolos.

## 3.3 Comparação de métodos e robustez

Não há máscaras manuais de referência. A qualidade foi avaliada por consistência entre métodos independentes, sensibilidade ao limiar, inspeção visual e efeito na classificação.

| Comparação (900 imagens de treino) | IoU médio | IoU mínimo | IoU percentil 5 |
|---|---|---|---|
| Otsu vs k-means (k=2, espaço Lab) | 0,997 | 0,974 | 0,991 |
| Otsu vs limiar fixo (200) | 0,976 | 0,905 | 0,952 |

**Sensibilidade ao limiar** (IoU entre a máscara no limiar de Otsu e no limiar deslocado):

| Deslocamento | IoU médio ± desvio | IoU mínimo |
|---|---|---|
| −20 | 0,988 ± 0,005 | 0,963 |
| −10 | 0,994 ± 0,003 | 0,983 |
| +10 | 0,994 ± 0,003 | 0,982 |
| +20 | 0,989 ± 0,005 | 0,968 |

A máscara quase não muda com o limiar, o que reflete o forte contraste entre folha e fundo. O limiar fixo em 200 concorda menos (inclui mais halo e engrossa o pecíolo), e o k-means concorda quase totalmente com o Otsu. As diferenças entre Otsu e k-means ficam no pecíolo fino, que o k-means descarta, e em pequenas regiões claras entre folíolos (`fig_q3_metodos.png`, `fig_q3_piores_casos.png`).

Nos descritores de forma, os três métodos classificam de forma equivalente: 86,6% com Otsu, 87,3% com limiar fixo e 87,2% com k-means (desvio entre dobras de 2 a 3 pontos).

**Decisão:** Otsu sobre cinza com limpeza morfológica. O limiar se adapta a cada imagem, é estável, simples de explicar e equivale aos métodos alternativos.

## 3.4 A segmentação contribui para a classificação?

Teste (proxy) com regressão logística, validação cruzada de 5 dobras estratificada por grupo, somente no treino. Acurácia média ± desvio (%):

| Características | Sem segmentação (imagem inteira) | Com segmentação (só a folha) |
|---|---|---|
| Cor | 61,3 ± 4,9 | **70,8 ± 3,3** |
| Textura (LBP) | 90,2 ± 3,5 | 87,8 ± 3,2 |
| Cor + textura | 94,6 ± 1,8 | 94,3 ± 1,7 |
| Forma | não é possível | 86,6 ± 2,1 |
| Cor + textura + forma | não é possível | **98,4 ± 0,6** |

- **Cor:** a segmentação aumenta ≈ 9,5 pontos, mais que o desvio entre dobras. Sem ela, a média de cor é diluída pelo fundo branco.
- **Textura:** não há ganho (a imagem inteira ficou até 2,4 pontos acima, dentro do desvio). Uma hipótese, que não foi testada, é que o histograma LBP da imagem inteira também carrega a proporção de fundo e o padrão do contorno, ou seja, informação de forma e tamanho que a LBP restrita à folha não tem. Isso se alinha com o atalho de 26,3% da fração de folha no quadro.
- **Cor + textura:** equivalente nas duas formas (94,6 e 94,3), então o benefício vem do que a segmentação acrescenta.
- **Forma:** só é possível com a máscara, e adicioná-la leva a 98,4%, cerca de 4 pontos acima das características de imagem inteira (≈ 2 desvios), com desvio menor entre dobras.

**Conclusão:** a segmentação contribui principalmente por permitir as características de forma, que dão o maior ganho, e por tornar a cor representativa da folha. Para a textura isolada ela não trouxe ganho mensurável.

## 3.5 Pecíolo

O pecíolo é uma estrutura fina, presente e longa em algumas classes (2, 5, 8, 12, 13) e quase ausente em outras, e seu comprimento depende de como a folha foi cortada. Foi testada a remoção por **abertura morfológica** (elemento elíptico de raio r):

| Raio | Perda média de área | Classe 2 | Classe 10 | Classe 7 | Forma (acurácia %) | Cor + textura + forma (%) |
|---|---|---|---|---|---|---|
| Sem remoção | – | – | – | – | 86,6 ± 2,1 | 98,4 ± 0,6 |
| r = 2 | 1,0% | 3,2% | 2,2% | 0,9% | 86,1 ± 1,9 | – |
| r = 3 | 1,3% | 3,9% | 2,9% | 1,5% | 86,4 ± 1,8 | 98,6 ± 0,8 |
| r = 5 | 2,1% | 5,4% | 4,8% | 3,6% | 84,2 ± 1,9 | – |

A remoção **não melhora a classificação** (86,6 vs 86,4 com r = 3; 98,4 vs 98,6 no conjunto completo) e r = 5 piora. Além disso, a abertura tem efeitos colaterais visíveis (`fig_q3_pipeline.png`): arredonda as pontas dos lobos do bordo e dos dentes (classe 2), encurta a ponta da folha estreita (classe 7) e remove a nervura central entre os folíolos das folhas compostas (classe 10), separando-os.

**Decisão:** a máscara principal **mantém o pecíolo** (Otsu + limpeza). A versão sem pecíolo (r = 3) fica salva e será avaliada como ablação na análise das características de forma (Questões 4 a 6).

## 3.6 Limitações

- **Sem referência manual:** a avaliação é indireta (consistência entre métodos, sensibilidade, inspeção visual, classificação).
- **Pecíolo fino:** a máscara o captura de forma fragmentária. Em alguns casos o pecíolo claro é interrompido (classes 5 e 8 têm 1,5 e 1,2 componentes em média), e o k-means o descarta. Descritores como área, perímetro e proporção da caixa são sensíveis a isso.
- **Regiões claras em folhas compostas:** nas classes 10 e 14 há áreas claras entre folíolos sobrepostos (provável fundo visível ou reflexo), tratadas de forma diferente pelos métodos (`fig_q3_piores_casos.png`).
- **Folíolos sobrepostos** se fundem em um único componente.
- **Classe 7 (folha estreita):** ocupa só 7% do quadro, então tem poucos pixels em 256 px, e 1,7% das imagens da classe tocam a borda do quadro.
- **Imagens com mais de um componente:** 9,2%, concentradas nas classes 5, 10, 8, 2 e 13, por folíolos, pecíolo interrompido ou fragmentos.
- **Dependência de fundo claro:** o método (limiar escuro/claro) não funcionaria com fundos complexos, mas não é o caso deste dataset.

## 3.7 Resumo das decisões

| Decisão | Escolha | Justificativa |
|---|---|---|
| Segmentar? | Sim | Habilita forma (+4 pontos no conjunto completo) e torna a cor representativa (+9,5 pontos); reduz a dependência da fração de fundo |
| Método | Otsu (cinza) + limpeza morfológica | Limiar adaptativo; IoU 0,997 com k-means; estável a ±20 níveis |
| Folíolos | Componentes separados mantidos | Classes 10 e 14 |
| Pecíolo | Mantido na máscara principal; versão r = 3 salva para ablação | Remoção sem ganho mensurável e com efeitos colaterais nas pontas e nervuras |
| Uso nas Questões 4 e 5 | Características de cor, textura e gradiente só dentro da máscara; descritores de forma sem depender de tamanho absoluto | Evita o atalho de recorte (26,3% só com a fração de folha) |