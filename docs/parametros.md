# Parâmetros do código: valores e motivo de cada escolha

Este documento lista todo valor do código que pode ser alterado e que muda o resultado ou o
processo: onde está, quanto vale e por que vale isso. Os motivos são de quatro tipos, e cada
parâmetro diz qual é o seu:

- **Medido**: há um experimento neste projeto que sustenta o valor (os números estão aqui).
- **Aprendido / buscado**: o valor não é escrito à mão; sai dos dados de treino.
- **Convenção**: é o valor padrão da literatura ou da biblioteca, mantido sem ajuste.
- **Escolha manual**: foi fixado à mão e **não foi ajustado nem teve a sensibilidade medida**.
  Quando é este o caso, o documento diz isso, em vez de inventar uma justificativa.

Quase todos os parâmetros do pipeline ficam em `bottle_inspection/config.py`. Os demais estão
no módulo indicado em cada tabela. A interface web usa os mesmos valores: cada perfil começa
com os parâmetros com que o seu modelo foi treinado (`InspectionParams`, em `params.py`).

Dois experimentos sustentam os valores marcados como **Medido**. Ambos usam **somente as 1200
imagens de treino**; as 300 de validação ficam intocadas até a avaliação final, feita uma
única vez. Os resultados completos ficam em `results/parameter_search/`.

- `experiments/bottle_box_methods.py` (seção 2): compara três modos de tirar a caixa da garrafa
  da mesma máscara (todas as bordas pelos limiares; largura e altura fixas; largura fixa com
  centro e altura medidos), cada um com vários valores, contra a caixa anotada da garrafa
  (classes 0 e 1 do Kaggle). As anotações nunca são lidas pelo pipeline; servem só para
  conferir. Saída: `bottle_box_methods.csv` e `bottle_box_methods_best.csv`.
- `experiments/localization_sensitivity.py` (seção 2): varia um parâmetro da localização por
  vez, com os demais nos valores atuais, e mede a caixa contra a anotada. Saída:
  `localization_sensitivity.csv` e `localization_otsu.json`.
- `experiments/descriptor_search.py` (seção 5): varia um fator por vez em cada descritor. A
  nota de uma configuração é o F1-macro médio da validação cruzada estratificada em 5 dobras,
  nos 9 pares vetor × classificador do descritor (3 escopos × 3 classificadores, cada um com a
  sua busca em grade). **Regra de decisão:** um valor só substitui o de referência se a sua nota
  superar a da referência por mais de um erro padrão (desvio entre as dobras ÷ √5); diferenças
  dentro desse ruído não mudam nada. A exceção é o número de direções do HOG, escolhido pelo
  critério de um erro padrão na forma usual (seção 5). Saída: `descriptor_search.csv`,
  `descriptor_search_summary.csv` e `descriptor_decisions.json`.

## 1. Base de dados e reprodutibilidade

| Parâmetro | Valor | Motivo |
|---|---|---|
| `SEED` | 42 | **Convenção.** Semente única do sorteio das imagens do fundo, das dobras da validação cruzada, do SVM e do Random Forest. Qualquer inteiro serviria; fixá-la torna a execução repetível. |
| `dataset.SPLITS` | `train` (1200) / `val` (300) | **Decisão do trabalho.** Usa-se a divisão original do Kaggle, sem reorganizar, para que os resultados sejam comparáveis com os de quem usa a mesma base. |
| `YOLO_CLASS_NAMES` / `CLASSES` | as 7 classes do `data.yaml` | **Decisão do trabalho.** São exatamente as 7 classes da base. Só o id 0 foi renomeado (`bottle_ron88` → `bottle_no_defect`), para deixar claro que é a garrafa sem defeito. |
| `OK_CLASS` | `bottle_no_defect` | É a única classe que significa "aprovada"; todas as outras reprovam. |
| `BOTTLE_CLASS_IDS` | (0, 1) | São as duas classes cuja caixa anotada envolve a garrafa inteira; usadas só para validar a localização. |
| `INSPECTION_CRITERIA` | 5 quesitos | **Decisão do trabalho.** Agrupa as classes de defeito por tipo de inspeção (nível, tampa, rótulo, contaminação, marca) para as métricas binárias. |

## 2. Modelo de fundo e localização da garrafa

| Parâmetro | Valor | Motivo |
|---|---|---|
| `LOCALIZATION_SIZE` | 640×360 | **Escolha manual.** Metade da resolução original (1280×720), mesma proporção. A localização só precisa da caixa, então a resolução menor reduz o custo; os demais parâmetros desta seção estão em pixels desta resolução. Outras resoluções não foram testadas. |
| `N_BACKGROUND_IMAGES` | 200 | **Escolha manual.** A mediana por pixel precisa que a garrafa cubra cada pixel em menos da metade das imagens; como a garrafa muda de posição, 200 imagens bastam para o cenário ficar nítido (resta só um "fantasma" tênue da garrafa). O número não foi variado. |
| `preprocessing.MAX_DIFFERENCE` | 255 | **Convenção.** A diferença (0 a 765) é limitada a 255 porque o Otsu do OpenCV trabalha com imagens de 8 bits. Diferenças acima disso já são certamente garrafa. |
| Limiar da diferença | Otsu, por imagem | **Aprendido.** Não há limiar fixo: o método de Otsu escolhe, em cada imagem, o valor que melhor separa as diferenças em dois grupos. No treino ele ficou entre 44 e 115 (mediana 56). |
| `NOISE_FLOOR` | 20 | **Medido.** O Otsu sempre divide os pixels em dois grupos; numa cena vazia dividiria o ruído do sensor. O piso impede isso. Como o menor limiar de Otsu no treino é 44, o piso de 20 não altera nenhuma imagem da base. |
| `MASK_CLOSE_SIZE` | 15 px | **Medido.** O fechamento junta os furos da máscara da garrafa transparente. É o valor de maior IoU médio (tabela abaixo): abaixo dele a máscara fica fragmentada e o topo da garrafa é cortado; acima, o IoU volta a cair. |
| `MASK_OPEN_SIZE` | 5 px | **Medido (pouco sensível).** A abertura remove linhas finas, como a borda da mesa. De 1 a 5 px o IoU quase não muda; 5 px é o que dá o menor erro na base da caixa. Acima de 9 px o topo da garrafa começa a ser cortado. |
| `PROFILE_SMOOTHING` | 31 px | **Medido.** Média móvel do perfil de colunas, usada para achar o trecho da garrafa. A máscara de uma garrafa transparente às vezes se parte em duas metades verticais; com uma média estreita (até 11), o vale entre elas cai abaixo da fração do perfil, o trecho encontrado é só meia garrafa e o centro sai deslocado (IoU mínimo de 0,597 no treino, e 0,41 numa imagem de validação). A partir de 21 isso não acontece (mínimo 0,637). O IoU médio quase não muda entre 11 e 51 (0,935–0,937), então o valor foi escolhido pelo pior caso. |
| `COLUMN_PROFILE_FRACTION` | 0,4 | **Medido.** Fração do pico do perfil de colunas que delimita o trecho da garrafa; o centro é o meio entre a primeira e a última coluna desse trecho que atingem a mesma fração no perfil sem suavização. 0,4 dá o maior IoU médio (0,936) e o menor p95 do erro do centro entre 0,2 e 0,6. |
| `BOTTLE_WIDTH` | 0,145 (14,5% da largura da imagem) | **Escolha manual, com o valor medido.** Largura da caixa, centrada na garrafa encontrada. É um valor ad hoc: as garrafas da base têm a mesma largura e ficam à mesma distância da câmera, e as bordas laterais de uma garrafa transparente são a parte mais fraca da máscara. Entre 0,13 e 0,16, 0,145 dá o maior IoU (tabela abaixo); é também a mediana da largura das caixas anotadas. Em outra montagem de câmera ou com outro produto precisa ser redefinido. |
| `TOP_PROFILE_FRACTION` | 0,3 | **Medido (pelo pior caso).** O topo é a primeira linha que atinge esta fração da linha mais cheia. O gargalo e a tampa são mais estreitos que o corpo, e um gargalo transparente quase não entra na máscara. Com 0,5, o topo ficava cortado em mais de 15 px em 18 imagens de treino (até 129 px, perdendo tampa e gargalo); 0,3 é o maior valor sem nenhum corte acima de 15 px. O IoU médio é o mesmo de 0,1 a 0,45 (0,936). Abaixo de 0,4, em 58 imagens de garrafas de outra marca (mais baixas) a caixa sobe cerca de 20 px além do topo, por causa do resto da garrafa Ron 88 que fica no modelo de fundo. |
| `ROW_PROFILE_FRACTION` | 0,5 | **Medido.** A base é a última linha que atinge esta fração da linha mais cheia. Abaixo da garrafa fica a sombra na mesa; frações menores puxam a caixa para a sombra. De 0,5 a 0,6 o IoU é o maior (0,936). |

Comparação dos modos de tirar a caixa da mesma máscara, nas 1200 imagens de treino (melhor
combinação de valores de cada modo; fonte: `results/parameter_search/bottle_box_methods_best.csv`):

| Modo | IoU médio | 5% piores | Mínimo | Erro do centro (média / p95) | IoU médio na validação |
|---|---:|---:|---:|---:|---:|
| Automático: todas as bordas pelos limiares (suavização 31, colunas 0,3, topo 0,2, base 0,5) | 0,921 | 0,808 | 0,644 | 5,6 / 19,3 px | 0,919 |
| Tamanho ad hoc: largura 0,145 e altura 0,79 fixas | 0,919 | 0,825 | 0,588 | 3,9 / 14,1 px | 0,919 |
| Misto com os valores adotados: largura 0,145 fixa, centro e altura medidos | 0,936 | 0,847 | 0,637 | 4,0 / 14,4 px | 0,939 |

O modo misto foi adotado. A altura fixa perde porque as garrafas têm alturas diferentes (sem
tampa, com a tampa levantada, de outra marca). No modo misto, a combinação de maior IoU médio
no treino usa suavização 11 (0,938); a suavização 31 foi mantida por causa do pior caso
descrito acima.

Sensibilidade medida nas 1200 imagens de treino, variando um parâmetro por vez (os demais nos
valores atuais, em negrito). Erros em pixels na imagem de 1280×720. "Recuo" é quanto a caixa
detectada fica para dentro da caixa anotada em cada lado (positivo = corta a garrafa;
negativo = passa dela). Fonte: `results/parameter_search/localization_sensitivity.csv`.

| Parâmetro | Valor | IoU médio | IoU mínimo | Erro do centro (média / p95) | Recuo esq. / dir. | Recuo topo (média / máx.) | Topos cortados > 15 px | Recuo da base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `MASK_CLOSE_SIZE` | 1 | 0,841 | 0,329 | 14,0 / 36,8 | -7,0 / 7,3 | 3,5 / 184,4 | 21 | 8,8 |
|  | 5 | 0,899 | 0,389 | 7,9 / 27,0 | -4,5 / 4,8 | 2,4 / 157,7 | 17 | 4,8 |
|  | 9 | 0,924 | 0,595 | 5,5 / 19,8 | -2,8 / 3,1 | 1,5 / 91,6 | 9 | 1,6 |
|  | **15** | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,7 |
|  | 21 | 0,933 | 0,626 | 4,1 / 15,0 | -1,1 / 1,4 | 1,0 / 11,6 | 0 | -4,7 |
|  | 31 | 0,923 | 0,522 | 4,9 / 20,0 | -1,1 / 1,4 | 1,0 / 11,1 | 0 | -6,8 |
| `MASK_OPEN_SIZE` | 1 | 0,934 | 0,637 | 3,9 / 14,5 | -1,3 / 1,6 | 1,1 / 11,1 | 0 | -3,4 |
|  | 3 | 0,936 | 0,637 | 3,9 / 14,3 | -1,3 / 1,6 | 1,1 / 11,6 | 0 | -2,5 |
|  | **5** | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,7 |
|  | 9 | 0,935 | 0,637 | 4,2 / 15,5 | -1,8 / 2,1 | 1,1 / 15,7 | 1 | -1,2 |
|  | 13 | 0,933 | 0,629 | 4,5 / 15,8 | -2,0 / 2,3 | 1,3 / 33,2 | 6 | -1,0 |
| `PROFILE_SMOOTHING` | 1 | 0,935 | 0,597 | 4,1 / 15,4 | -1,4 / 1,7 | 1,0 / 13,1 | 0 | -2,2 |
|  | 5 | 0,937 | 0,597 | 3,9 / 14,7 | -1,3 / 1,6 | 1,0 / 13,1 | 0 | -2,2 |
|  | 11 | 0,937 | 0,597 | 3,9 / 14,1 | -1,4 / 1,7 | 1,0 / 13,1 | 0 | -2,2 |
|  | 21 | 0,936 | 0,637 | 4,0 / 14,2 | -1,4 / 1,7 | 1,0 / 13,1 | 0 | -2,0 |
|  | **31** | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,7 |
|  | 41 | 0,935 | 0,588 | 4,1 / 14,8 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,5 |
|  | 51 | 0,936 | 0,609 | 4,0 / 15,0 | -1,5 / 1,7 | 1,2 / 13,1 | 0 | -1,2 |
| `COLUMN_PROFILE_FRACTION` | 0,2 | 0,911 | 0,531 | 6,7 / 27,3 | -1,8 / 2,1 | 1,4 / 27,8 | 3 | -0,3 |
|  | 0,3 | 0,928 | 0,574 | 4,9 / 20,5 | -1,4 / 1,7 | 1,3 / 13,1 | 0 | -0,9 |
|  | **0,4** | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,5 | 0,932 | 0,622 | 4,5 / 14,1 | -2,2 / 2,5 | 0,9 / 11,6 | 0 | -2,8 |
|  | 0,6 | 0,909 | 0,635 | 7,0 / 25,8 | -4,0 / 4,3 | 0,8 / 11,6 | 0 | -3,8 |
| `BOTTLE_WIDTH` | 0,13 | 0,874 | 0,620 | 4,0 / 14,4 | 8,1 / 11,4 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,135 | 0,903 | 0,626 | 4,0 / 14,4 | 4,9 / 8,2 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,14 | 0,925 | 0,631 | 4,0 / 14,4 | 1,7 / 5,0 | 1,1 / 13,1 | 0 | -1,7 |
|  | **0,145** | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,15 | 0,929 | 0,642 | 4,0 / 14,4 | -4,7 / -1,4 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,155 | 0,910 | 0,647 | 4,0 / 14,4 | -7,9 / -4,6 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,16 | 0,887 | 0,652 | 4,0 / 14,4 | -11,1 / -7,8 | 1,1 / 13,1 | 0 | -1,7 |
| `TOP_PROFILE_FRACTION` | 0,1 | 0,937 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 0,1 / 8,8 | 0 | -1,7 |
|  | 0,2 | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 0,5 / 11,1 | 0 | -1,7 |
|  | 0,25 | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 0,8 / 11,1 | 0 | -1,7 |
|  | **0,3** | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,35 | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,5 / 27,8 | 1 | -1,7 |
|  | 0,4 | 0,936 | 0,635 | 4,0 / 14,4 | -1,5 / 1,8 | 2,7 / 31,8 | 3 | -1,7 |
|  | 0,45 | 0,936 | 0,635 | 4,0 / 14,4 | -1,5 / 1,8 | 3,6 / 82,3 | 12 | -1,7 |
|  | 0,5 | 0,934 | 0,635 | 4,0 / 14,4 | -1,5 / 1,8 | 4,7 / 128,6 | 18 | -1,7 |
| `ROW_PROFILE_FRACTION` | 0,3 | 0,933 | 0,635 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -6,5 |
|  | 0,4 | 0,935 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -4,1 |
|  | **0,5** | 0,936 | 0,637 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | -1,7 |
|  | 0,6 | 0,935 | 0,639 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | 0,8 |
|  | 0,7 | 0,934 | 0,639 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | 4,1 |
|  | 0,8 | 0,929 | 0,636 | 4,0 / 14,4 | -1,5 / 1,8 | 1,1 / 13,1 | 0 | 8,1 |

## 3. Normalização fotométrica

| Parâmetro | Valor | Motivo |
|---|---|---|
| `NORMALIZE_LIGHTING` | `True` | **Decisão do trabalho.** Cada imagem é reescalada, canal a canal, para que a região de referência fique igual à do fundo; isso compensa ganho e balanço de branco da câmera antes da subtração do fundo. A opção `--no-normalization` existe para desligar, mas os resultados do trabalho usam a normalização ligada. |
| `REFERENCE_PATCH` | (0,80; 0,02; 0,97; 0,25) | **Escolha manual (depende da cena).** Retângulo da parede, no canto superior direito, que nenhuma garrafa cobre em nenhuma imagem da base. É uma das poucas escolhas feitas olhando o cenário; em outra montagem de câmera precisa ser redefinido. Outras posições não foram testadas. |

## 4. Recorte da garrafa e regiões

| Parâmetro | Valor | Motivo |
|---|---|---|
| `BOTTLE_SIZE` | 96×384 | **Escolha manual.** Tamanho fixo do recorte da garrafa inteira: proporção 1:4, próxima da caixa detectada (cerca de 168×522 px), com lados múltiplos de 16 para que as células de HOG e LBP caibam inteiras. Outros tamanhos não foram testados. |
| `ROIS` | `cap` 0,00–0,12 · `label` 0,20–0,52 · `body` 0,12–1,00 | **Escolha manual; são as únicas posições escolhidas à mão.** Faixas verticais dentro da caixa detectada (0 = topo, 1 = base), definidas olhando onde ficam a tampa, o rótulo e o corpo nas garrafas da base. O `body` sobrepõe o `label` de propósito: cobre tudo abaixo da tampa. |
| `ROI_SIZE` | `cap` 64×48 · `label` 64×96 · `body` 48×192 | **Escolha manual.** Tamanho fixo de cada ROI antes do HOG e do LBP, com proporção próxima à do recorte e lados múltiplos de 16 (e de 8 na tampa). Definem o comprimento dos vetores. Não foram variados. |
| `AUTO_ROI_COUNT` | 3 | **Decisão do trabalho.** Mesmo número das ROIs ad hoc, para que a comparação entre as duas opções seja justa. É o único valor informado às regiões automáticas. |
| `AUTO_ROI_UNIT` | 16 linhas | Igual ao tamanho da célula de HOG e LBP: assim nenhuma célula fica dividida entre duas regiões. |
| `regions.MIN_BAND_STRIPS` | 2 faixas | Restrição técnica: o HOG precisa de pelo menos 2×2 células para formar um bloco. |
| `WIDE_AREA_RECT` | x 0,20–0,78 · y 0,02–0,97 da imagem | **Escolha manual (escopo sem localização).** Retângulo fixo da área ampla (`wide`), igual em todas as imagens e sem usar a caixa detectada: tem a largura de 3 a 4 garrafas e contém todas as garrafas da base (as caixas detectadas ficam entre x = 0,22 e 0,72 e entre y = 0,05 e 0,92). Serve para medir como os descritores lidam com mais fundo no recorte. |
| `WIDE_AREA_SIZE` | 256×240 | **Escolha manual.** Proporção próxima da área na foto (742×684 px), escala ~0,345. |
| `WHOLE_IMAGE_SIZE` | 448×256 | **Escolha manual.** Tamanho da imagem inteira (`image`), na mesma escala da área ampla: a garrafa fica com o mesmo tamanho em pixels nas duas e só muda a quantidade de fundo. |
| Faixas automáticas | `auto_1` 0–0,125 · `auto_2` 0,125–0,917 · `auto_3` 0,917–1 | **Aprendido.** Saem das garrafas de treino (divisão ótima da garrafa média em 3 grupos contíguos de faixas de cores parecidas, por programação dinâmica). Não são configuradas; ficam salvas em cada modelo. |

## 5. Descritores

| Parâmetro | Valor | Motivo |
|---|---|---|
| `DESCRIPTORS` | `hog`, `lbp`, `color` | **Decisão do trabalho.** Três descritores genéricos (forma, textura, cor), sem nenhuma medida feita sob medida para o produto. Cada um é classificado sozinho, para mostrar o efeito de cada descritor. |
| `HOG_ORIENTATIONS` | 6 | **Medido.** 6 direções de 30° entre 0° e 180°. O valor inicial era 9 (o de Dalal e Triggs, 2005, e padrão do scikit-image). Na busca: **6: 0,9944** · 9: 0,9932 · 12: 0,9939 · 18: 0,9921. A diferença de 6 para 9 (0,0012) ficou abaixo do erro padrão de 9 (0,0015), então a regra de troca da referência não a aceitaria sozinha; 6 foi adotado pelo critério de um erro padrão na forma usual (entre as configurações a até um erro padrão da melhor, fica a mais simples), por ter a melhor nota e o menor vetor (um terço menor). Ganha em 7 dos 9 pares e empata em mais 1. Com 6 como referência, a busca foi repetida e manteve 6. |
| `features.HOG_BLOCK` | 2×2 células, norma L2-Hys | **Convenção.** Mesma referência. |
| `HOG_CELL` | 8×8 na `cap`; 16×16 nas demais | **Medido.** A tampa é a menor região (64×48), então usa célula menor para ainda ter 8×6 células. Na busca (com 6 direções), esta combinação teve a maior nota (0,9944); 16×16 em tudo fica a 0,0010, dentro do erro padrão (0,9934), e 8×8 em tudo é pior (0,9869), com vetor 3,2 a 4,5 vezes maior e treino 6 vezes mais lento. |
| `LBP_POINTS` / `LBP_RADIUS` | 24 / 3 | **Medido.** Escolhido na busca entre as vizinhanças usuais de Ojala et al. (2002): (8, 1): 0,9722 · (8, 2): 0,9820 · (16, 2): 0,9805 · **(24, 3): 0,9871** · (24, 4): 0,9870 · (32, 4): 0,9862. O valor inicial era (8, 1); a busca com a localização anterior já tinha escolhido (24, 3), e a repetida com a localização atual o manteve. O custo é um vetor 2,6 vezes maior que com (8, 1). |
| `features.LBP_METHOD` | `uniform` | **Convenção.** Padrões uniformes invariantes à rotação: P + 2 = 26 códigos, o que mantém o histograma pequeno. Outros métodos não foram testados. |
| `LBP_CELL` | 16×16 em todas as regiões | **Medido.** Um histograma por célula preserva onde está cada textura. Na busca: 8 px: 0,9765 · **16 px: 0,9871** · 32 px: 0,9744. |
| `HSV_HISTOGRAM_BINS` | (36, 16, 16) | **Medido.** Escolhido na busca: (9, 4, 4): 0,9940 · (18, 8, 8): 0,9990 · **(36, 16, 16): 0,9994** · (72, 32, 32): 0,9992. O valor inicial era (18, 8, 8); a busca com a localização anterior já tinha escolhido (36, 16, 16), e a repetida o manteve. São 36 faixas de matiz (5° cada, já que o H do OpenCV vai de 0 a 180) e 16 para saturação e brilho; com média e desvio de cada canal, 74 valores por região. |
| `features.HSV_CHANNEL_RANGES` | (180, 256, 256) | Limites dos canais H, S e V no OpenCV; não é uma escolha. |

Comprimento de cada grupo, que decorre dos tamanhos e das células acima:

| | `cap` | `label` | `body` | `auto_1` | `auto_2` | `auto_3` | `bottle` | `wide` | `image` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hog` | 840 | 360 | 528 | 240 | 2160 | 120 | 2760 | 5040 | 9720 |
| `lbp` | 312 | 624 | 936 | 468 | 2964 | 312 | 3744 | 6240 | 11648 |
| `color` | 74 | 74 | 74 | 74 | 74 | 74 | 74 | 74 | 74 |

Resumo da busca dos descritores, repetida com a localização atual (F1-macro médio na
validação cruzada; em negrito, o valor adotado; a referência de cada descritor teve erro padrão
de 0,0011 no HOG, 0,0034 no LBP e 0,0004 na cor):

| Descritor | Fator | Valores e notas | Tamanho do vetor (mín.–máx. entre os escopos) |
|---|---|---|---|
| HOG | direções | **6: 0,9944** · 9: 0,9932 · 12: 0,9939 · 18: 0,9921 | 1728–2760 · 2592–4140 · 3456–5520 · 5184–8280 |
| HOG | célula (6 direções) | 8 px: 0,9869 · **8 px (tampa) e 16 px: 0,9944** · 16 px: 0,9934 | 5448–12408 · 1728–2760 · 1032–2760 |
| LBP | vizinhança | (8, 1): 0,9722 · (8, 2): 0,9820 · (16, 2): 0,9805 · **(24, 3): 0,9871** · (24, 4): 0,9870 · (32, 4): 0,9862 | 720–1440 · 720–1440 · 1296–2592 · 1872–3744 · 1872–3744 · 2448–4896 |
| LBP | célula | 8 px: 0,9765 · **16 px: 0,9871** · 32 px: 0,9744 | 7488–14976 · 1872–3744 · 364–936 |
| Cor | faixas H, S, V | (9, 4, 4): 0,9940 · (18, 8, 8): 0,9990 · **(36, 16, 16): 0,9994** · (72, 32, 32): 0,9992 | 23–69 · 40–120 · 74–222 · 142–426 |

Limites desta busca, que o texto deve declarar: um fator por vez (não garante o ótimo
conjunto); os tamanhos de redimensionamento das regiões, o bloco do HOG e o método do LBP não
foram variados.

As buscas anteriores ficam em `results/parameter_search/before_new_localization/` (localização
anterior: largura da caixa medida pelos limiares, suavização 11 e um único limiar de 50% para
as linhas) e `before_hog_6_orientations/` (localização atual, HOG com 9 direções).

Efeito na validação do Kaggle (avaliada depois de cada escolha, nunca usada para escolher),
em erros somados dos 27 modelos nas 300 imagens: com a busca dos descritores e do kernel, 52;
com a suavização do perfil em 31, 43; com o modo atual de tirar a caixa, 29; com 6 direções no
HOG, 25. No resultado final o SVM acerta todas as imagens nos 9 vetores com a garrafa localizada; o LBP fica em
0,980–1,000, a cor em 0,997–1,000 e o HOG em 0,997–1,000 (com 9 direções, 0,980–1,000). Com os valores iniciais o LBP ficava em 0,940–0,993.

`FEATURE_VECTORS` define os 15 vetores classificados (5 escopos × 3 descritores): `rois_*` (as
três ROIs ad hoc juntas), `auto_rois_*` (as três regiões automáticas), `bottle_*` (a garrafa
inteira) e, sem localização, `wide_*` (área ampla) e `image_*` (imagem inteira). Nenhuma ROI é
classificada sozinha e os descritores nunca são misturados. A busca dos descritores acima usou
só os três primeiros escopos; a área ampla e a imagem inteira usam os mesmos valores, com
células de 16×16 (`HOG_CELL` e `LBP_CELL` em `wide` e `image`), sem ajuste.

## 6. Treinamento e classificadores

| Parâmetro | Valor | Motivo |
|---|---|---|
| `N_FOLDS` | 5 | **Convenção.** Validação cruzada estratificada em 5 dobras, só nas imagens de treino. |
| `GRID_SEARCH_SCORING` | `f1_macro` | **Decisão do trabalho.** As classes são desbalanceadas (de 19 a 121 imagens na validação); o F1-macro dá o mesmo peso a todas. |
| `class_weight` (SVM e Random Forest) | `balanced` | Mesma razão: erros em classes raras pesam mais. |
| Padronização (`StandardScaler`) | SVM e KNN | Necessária para os dois: sem ela, atributos de escala maior dominariam o kernel e a distância. O Random Forest não precisa. |
| SVM: kernel | linear, RBF ou polinomial | **Buscado.** O kernel entra na busca em grade, como os demais hiperparâmetros. O linear foi o escolhido em 13 dos 15 vetores, o RBF em 1 e o polinomial em 1. |
| SVM: grade | linear: C ∈ {0,01; 0,1; 1} · RBF: C ∈ {1, 10, 100}, gamma ∈ {scale, 1e-3, 1e-4} · polinomial: grau ∈ {2, 3}, C ∈ {1, 10, 100}, gamma = scale | **Escolha manual da grade; valor final buscado.** Três ordens de grandeza de C por kernel. No linear, C = 0,01 (o menor da grade) venceu em 10 dos 13 vetores, então valores ainda menores poderiam ser testados. |
| KNN: grade | k ∈ {1, 3, 5, 7, 11, 15}; pesos ∈ {uniform, distance} | Idem. Valores ímpares de k evitam empates. A distância é a euclidiana; outras métricas não foram testadas. |
| Random Forest: grade | árvores ∈ {100, 300}; profundidade ∈ {sem limite, 20}; atributos por nó ∈ {√n, 10%} | Idem. |

Os hiperparâmetros finais **não são escritos à mão**: são os de maior F1-macro na validação
cruzada, para cada vetor (`best_params` em `results/feature_comparison.csv`):

| Vetor | SVM | KNN (k, pesos) | Random Forest (árvores, atributos por nó) |
|---|---|---|---|
| `rois_hog` | linear, C = 0,01 | 3, uniform | 300, √n |
| `auto_rois_hog` | linear, C = 0,01 | 1, uniform | 300, √n |
| `bottle_hog` | linear, C = 0,01 | 1, uniform | 300, √n |
| `wide_hog` | linear, C = 0,01 | 1, uniform | 300, √n |
| `image_hog` | linear, C = 0,01 | 3, distance | 300, 10% |
| `rois_lbp` | RBF, C = 10, gamma = 0.0001 | 3, distance | 300, 10% |
| `auto_rois_lbp` | linear, C = 0,01 | 1, uniform | 100, 10% |
| `bottle_lbp` | linear, C = 0,01 | 1, uniform | 300, 10% |
| `wide_lbp` | linear, C = 0,01 | 3, distance | 100, 10% |
| `image_lbp` | linear, C = 0,01 | 5, distance | 300, 10% |
| `rois_color` | linear, C = 0,01 | 1, uniform | 100, √n |
| `auto_rois_color` | polinomial grau 2, C = 100 | 1, uniform | 100, √n |
| `bottle_color` | linear, C = 1 | 1, uniform | 300, √n |
| `wide_color` | linear, C = 1 | 3, uniform | 100, 10% |
| `image_color` | linear, C = 0,1 | 1, uniform | 300, 10% |

A profundidade escolhida para o Random Forest foi "sem limite" em todos os vetores.

## 7. Avaliação e alertas

| Parâmetro | Valor | Motivo |
|---|---|---|
| `pipeline.MATRIX_METRICS` | acurácia e F1-macro | Métricas da matriz vetor × classificador. |
| `evaluation.ZERO_DIVISION` | 0,0 | Valor da métrica quando uma classe nunca é prevista; evita divisão por zero. |
| `evaluation.REFERENCE_IMAGE_WIDTH` | 1280 | Largura das imagens da base, usada só para expressar o erro do centro em pixels. |
| `ALERT_WINDOW` / `ALERT_THRESHOLD` | 10 / 4 | **Escolha manual.** Regra de alerta simulada: o mesmo defeito 4 vezes nas últimas 10 garrafas. É ilustrativa; os valores não foram ajustados a nenhuma linha real. |

## 8. Interface web

Estes valores não alteram os resultados do treinamento; mudam o comportamento da interface.

| Parâmetro | Valor | Motivo |
|---|---|---|
| Parâmetros de cada perfil | os do treinamento do modelo | Um perfil começa idêntico ao treino. Os que mudam o tamanho do vetor ficam travados; os demais (seções 2 a 4) podem ser editados e o perfil passa a ser marcado como "modificado". Os limites de edição estão em `params_catalog.py`. |
| Veredito: `reject_classes` | todas, exceto `bottle_no_defect` | Segue `OK_CLASS`. |
| Veredito: `min_confidence` | desligado | A confiança não é comparável entre algoritmos; por padrão não interfere no veredito. |
| `engine.analysis.LOW_FOREGROUND_FRACTION` | 0,04 | **Escolha manual.** Abaixo de 4% de primeiro plano na máscara a interface avisa que provavelmente não há garrafa. Só gera um aviso. |
| `engine.analysis.ASPECT_TOLERANCE` | 0,02 | **Escolha manual.** Diferença de proporção tolerada entre a imagem e o fundo antes do aviso de resolução diferente. |
| `engine.trace.OUTLIER_Z` | 3,0 | **Convenção.** No modo de depuração, atributos padronizados além de 3 desvios são contados como distantes do treino. |
| `settings.SETTINGS_DEFAULTS` | `max_profiles` 6, `default_fps` 3 (1 a 15), `motion_threshold` 0,04, `background_capture_frames` 30, `max_upload_mb` 25, `max_batch_files` 500, retenção de 30 dias | **Escolha manual.** Limites operacionais da interface, editáveis na tela Sistema. |

## 9. Só das figuras

`hog_view.TARGET_SIDE` (1600 px), `MIN_SCALE` (8) e `SUBPIXEL_BITS` (4) definem a resolução do
desenho do HOG nas figuras; `didactic.KNN_NEIGHBORS_SHOWN` (5) e `TREE_PATH_STEPS_SHOWN` (9),
quantos vizinhos e passos de árvore aparecem. Não entram em nenhum cálculo do pipeline.

## Resumo: o que ainda é escolhido à mão

- A largura da caixa da garrafa (`BOTTLE_WIDTH`), as faixas das ROIs ad hoc (`ROIS`), a região
  de referência (`REFERENCE_PATCH`) e o retângulo da área ampla (`WIDE_AREA_RECT`, só no escopo
  `wide`, sem localização): dependem do produto e do cenário.
- Tamanhos de redimensionamento das regiões, tamanho da localização e número de imagens do
  fundo: fixados sem ajuste.
- As grades de busca dos classificadores (os valores finais são buscados dentro delas).
