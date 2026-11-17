# Inspeção de garrafas plásticas com visão computacional clássica

Continuação do TCC "Sistema para inspeção automatizada de garrafas plásticas".
Sem acesso ao ambiente físico de captura, o trabalho usa a base do Kaggle
[Water Bottle Defect-Level Detection](https://www.kaggle.com/datasets/jihadakbr/water-bottle-defect-level-detection-dataset/).
A base não faz parte do repositório; veja [Reprodução](#reprodução). O notebook original (`ron88-prod-training.ipynb`) resolve a
tarefa com YOLOv8s. Aqui ela é resolvida com visão computacional clássica e três
classificadores: SVM, KNN e Random Forest. Os descritores (HOG, LBP e cor HSV) são
classificados **separadamente**, para comparar o impacto de cada um.

Todo o código (variáveis, constantes, saídas e nomes de arquivos) está em inglês;
esta documentação está em português.

## Reprodução

Requisitos: Python ≥ 3.11 (os resultados de `results/` foram gerados com Python 3.13.5 no Linux)
e `make`. A interface web também precisa do Node ≥ 20.19.

1. **Baixe a base do Kaggle**
   [Water Bottle Defect-Level Detection](https://www.kaggle.com/datasets/jihadakbr/water-bottle-defect-level-detection-dataset/)
   pelo botão *Download* da página ou pela CLI do Kaggle:

   ```bash
   kaggle datasets download jihadakbr/water-bottle-defect-level-detection-dataset -p ../archive --unzip
   ```

   A pasta da base deve ter esta estrutura (formato YOLO, com o split original do Kaggle):

   ```text
   <base>/
   ├── data.yaml
   ├── images/train/*.jpg   (1200)    images/val/*.jpg   (300)
   └── labels/train/*.txt              labels/val/*.txt
   ```

2. **Informe onde está a base** na variável de ambiente **`BOTTLE_DATA`**. O jeito mais simples
   é copiar `.env.example` para `.env` e editar o caminho:

   ```bash
   cp .env.example .env        # depois edite: BOTTLE_DATA=/caminho/da/base
   ```

   O `Makefile` e todos os scripts Python (`bottle_inspection/env.py`) leem o `.env`. Também
   vale exportar a variável (`export BOTTLE_DATA=/caminho/da/base`) ou passar o caminho
   direto (`make run DATA=/caminho/da/base`, `--data /caminho/da/base`). A ordem de
   prioridade é: `--data`/`DATA=`, a variável de ambiente, o `.env` e, por fim, o padrão
   `../archive` (ao lado da pasta `code/`). Um caminho relativo é relativo à pasta `code/`.

3. **Rode o pipeline:**

   ```bash
   make setup    # cria .venv e instala requirements.txt + requirements-dev.txt
   make run      # extrai os atributos, treina os 45 modelos e grava results/
   ```

   O resultado é determinístico (`SEED = 42` em `config.py`). Os números do repositório estão
   em `results/`. Os modelos treinados (`*.joblib`) e o cache de atributos não são versionados:
   o `make run` os recria. `requirements-lock.txt` traz as versões exatas do ambiente usado,
   para reproduzir os números ao pé da letra:
   `.venv/bin/python -m pip install -r requirements-lock.txt`.

4. **Opcional:** os experimentos de `experiments/` (busca de parâmetros; ver
   [docs/parametros.md](docs/parametros.md)), as figuras do TCC (`make thesis-figures
   THESIS_FIGURES=thesis/preview`; seção [Figuras do TCC](#figuras-do-tcc)), as figuras da
   apresentação (`presentation/src/README.md`) e a interface web (seção
   [Interface web](#interface-web)).

## Como rodar

```bash
make setup              # cria .venv e instala requirements.txt (+ requirements-dev.txt)
make run                # pipeline completo -> results/
make quick              # só os vetores rois_hog, rois_lbp e rois_color
make recompute          # ignora o cache de atributos
make no-normalization   # sem normalização fotométrica -> results_no_normalization/
make all                # run + no-normalization
make format             # black
make lint               # black --check
make clean              # remove as pastas de resultados
make distclean          # clean + remove .venv
make run MODELS="svm knn"
make run FEATURES="rois_hog bottle_lbp"   # só alguns vetores de atributos
make run DIDACTIC=1      # também gera a decisão real de cada modelo por imagem
make DATA=/outro/caminho OUTPUT=outra_pasta run
```

Variáveis do `Makefile`: `DATA`, `OUTPUT`, `MODELS` (padrão `"svm knn random_forest"`),
`FEATURES` (padrão: os 15 vetores) e
`DIDACTIC` (padrão `0`; com `1`, gera os gráficos didáticos).
O `make run` completo (45 treinos) leva cerca de 45 minutos numa CPU de 12 núcleos. Os atributos ficam em
cache em `results/feature_cache.joblib`. O Black é configurado em `pyproject.toml`
(`line-length = 100`); `requirements-dev.txt` adiciona o Black.

Sem o `make` (com `BOTTLE_DATA` definido, o `--data` pode ser omitido):

```bash
python -m bottle_inspection --data /caminho/da/base --output results \
    --models svm knn random_forest [--features rois_hog auto_rois_lbp ...] \
    [--recompute] [--no-normalization] [--didactic-plots]
```

## Estrutura do código

| pasta | conteúdo |
|---|---|
| `bottle_inspection/` | o pacote: pré-processamento, descritores, treino e avaliação (tabela abaixo) |
| `results/` | resultados oficiais do `make run` (métricas, relatórios, matrizes, gráficos; sem os modelos) |
| `experiments/` | buscas de parâmetros e comparações de métodos, só com as imagens de treino |
| `docs/` | `parametros.md` (cada parâmetro e o motivo do valor) e a especificação da interface web |
| `thesis/` | scripts das figuras do TCC; prévia das figuras e dos números em `thesis/preview/` |
| `presentation/` | apresentação (`index.html`) e os scripts que geram as figuras e os slides |
| `bottle_inspection_web/`, `frontend/` | interface web local (backend FastAPI e frontend React) |

Pacote `bottle_inspection/`:

| módulo | conteúdo |
|---|---|
| `config.py` | todas as constantes (classes, ROIs, parâmetros de HOG/LBP/HSV, vetores de atributos, alertas) |
| `env.py` | caminho da base (`BOTTLE_DATA`, lido do ambiente ou do `.env`) |
| `params.py` | os parâmetros agrupados (dataclasses imutáveis), usados também pela interface web |
| `dataset.py` | leitura da base, rótulos YOLO e split original do Kaggle |
| `preprocessing.py` | modelo de fundo, normalização fotométrica, detecção da caixa da garrafa e recortes |
| `regions.py` | regiões automáticas aprendidas das garrafas de treino |
| `features.py` | cálculo dos descritores (HOG, LBP, cor HSV) por região |
| `extraction.py` | extração dos atributos em lote, com cache |
| `models.py` | definição dos três classificadores e do grid search |
| `evaluation.py` | métricas, relatórios, critérios aprova/reprova, alertas |
| `didactic.py` | gráficos opcionais com a decisão real de cada classificador, por imagem de validação |
| `plots.py` | geração dos gráficos |
| `hog_view.py` | desenho do HOG de uma região (figuras) |
| `pipeline.py` | orquestração do fluxo completo |
| `cli.py` | argumentos de linha de comando |
| `__main__.py` | ponto de entrada de `python -m bottle_inspection` |

## A base

São 1500 imagens de 1280×720 com anotações YOLO de 7 classes. O código usa o split
original do Kaggle (1200 de treino e 300 de validação), sem reorganizá-lo. A checagem por
MD5 não encontra nenhuma imagem idêntica entre treino e validação (0 coincidências).

Cada imagem tem uma garrafa e **no máximo um defeito**, então o problema foi tratado como
**classificação da imagem nas mesmas 7 classes do `data.yaml` do Kaggle**:

| id | classe (Kaggle) | treino | val |
|---:|---|---:|---:|
| 0 | `bottle_no_defect` | 479 | 121 |
| 1 | `bottle_other_brand` | 81 | 19 |
| 2 | `defect_low_fill` | 209 | 41 |
| 3 | `defect_no_cap` | 122 | 28 |
| 4 | `defect_loose_cap` | 119 | 31 |
| 5 | `defect_debris` | 111 | 39 |
| 6 | `defect_label_damage` | 79 | 21 |

A classe de id 0 se chama `bottle_ron88` no `data.yaml` do Kaggle. No código ela foi
renomeada para `bottle_no_defect`, para deixar claro que é a garrafa sem defeitos; as demais
mantêm o nome original.

No YOLO, uma imagem com defeito tem duas caixas: a da garrafa (id 0) e a do defeito. Aqui cada
imagem recebe uma única classe: a do defeito, ou a da garrafa quando não há defeito
(`bottle_no_defect` é a garrafa boa). O agrupamento em `criteria_metrics.csv` (aprova/reprova
por tipo de inspeção) é uma análise extra feita sobre essas 7 classes.

## Método (constantes em `bottle_inspection/config.py`)

1. **Modelo de fundo.** A câmera é fixa, então a mediana de 200 imagens de treino dá o fundo.
2. **Normalização fotométrica.** Um patch fixo da parede (`REFERENCE_PATCH`) é usado como
   referência. Cada canal da imagem é reescalado para que esse patch fique igual ao do fundo.
3. **Detecção da garrafa por subtração de fundo.** Calcula-se |imagem − fundo| e o limiar
   é escolhido para cada imagem pelo método de Otsu. A máscara passa por um fechamento
   (`MASK_CLOSE_SIZE`) e uma abertura (`MASK_OPEN_SIZE`). A caixa da garrafa é tirada da
   máscara assim:
   - **Posição:** o perfil de brancos por coluna, suavizado por uma média móvel de 31 colunas
     (`PROFILE_SMOOTHING`), acha a garrafa no trecho em torno do pico (≥ 40% do pico); o
     centro é o meio entre a primeira e a última coluna desse trecho que atingem os mesmos
     40% no perfil sem suavização.
   - **Largura:** valor fixo, escolhido à mão (`BOTTLE_WIDTH` = 14,5% da largura da imagem),
     centrado nesse ponto. As bordas laterais de uma garrafa transparente são a parte mais
     fraca da máscara.
   - **Altura:** dentro dessas colunas, o topo é a primeira linha com ≥ 30% da linha mais
     cheia (`TOP_PROFILE_FRACTION`) e a base, a última com ≥ 50% (`ROW_PROFILE_FRACTION`). O
     gargalo é mais estreito que o corpo e a base tem a sombra, por isso cada borda tem o seu
     limiar.

   A posição e a altura são medidas; só a largura é informada. Comparado às caixas anotadas
   (usadas só para conferir), o erro médio do centro é de **4,0 px** (p95 de 14 px em
   1280 px), o IoU médio é **0,94** e o pior caso das 1500 imagens tem IoU 0,64. Três modos de
   tirar a caixa da mesma máscara foram comparados nas imagens de treino
   (`experiments/bottle_box_methods.py`):

   | modo | IoU médio | 5% piores | mínimo |
   |---|---:|---:|---:|
   | automático: todas as bordas pelos limiares (melhores valores) | 0,921 | 0,808 | 0,644 |
   | tamanho ad hoc: largura e altura fixas | 0,919 | 0,825 | 0,588 |
   | misto: largura fixa, centro e altura medidos (o adotado) | 0,936 | 0,847 | 0,637 |

4. **Regiões**, sempre relativas à caixa detectada (0 = topo, 1 = base). Há duas opções, que
   são comparadas:

   - **ROIs ad hoc** (`ROIS`), as únicas posições escolhidas à mão:

     | ROI | faixa dentro da caixa |
     |---|---|
     | `cap` | 0,00–0,12 |
     | `label` | 0,20–0,52 |
     | `body` | 0,12–1,00 |

   - **Regiões automáticas** (`regions.py`), aprendidas das garrafas de treino, sem rótulos:
     a garrafa média de treino (96×384) é dividida em faixas de 16 linhas, cada faixa é
     descrita pelas suas cores médias, e as faixas são separadas em 3 grupos contíguos o mais
     homogêneos possível (ótimo exato por programação dinâmica). Só o número de regiões é
     informado. Resultado: `auto_1` 0–0,125 (tampa), `auto_2` 0,125–0,917 e `auto_3`
     0,917–1 (base). As faixas aprendidas são salvas com cada modelo.

   A região `bottle` é a caixa inteira.

   - **Sem localização.** Duas regiões não usam a caixa detectada nem ROIs, para ver
     como os descritores lidam com mais fundo no recorte: a **área ampla** (`wide`), um
     retângulo fixo da imagem (`WIDE_AREA_RECT`: x de 0,20 a 0,78, y de 0,02 a 0,97; da largura
     de 3 a 4 garrafas, contém todas as garrafas da base, cujas caixas ficam entre x = 0,22 e
     0,72), reduzido para 256×240; e a **imagem inteira** (`image`), reduzida para 448×256. As
     duas usam a mesma escala (~0,345 da foto), então a garrafa tem o mesmo tamanho em pixels
     nas duas; só muda a quantidade de fundo. Os descritores são os mesmos, sem ajuste.

5. **Descritores.** Três descritores genéricos, calculados em cada região. Nenhum depende da
   cor ou da forma de um produto específico.

   | descritor | conteúdo | `cap` | `label` | `body` | `auto_1` | `auto_2` | `auto_3` | `bottle` | `wide` | `image` |
   |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | `hog` | histogramas de orientação do gradiente (6 direções de 30°) | 840 | 360 | 528 | 240 | 2160 | 120 | 2760 | 5040 | 9720 |
   | `lbp` | histogramas de padrões binários locais uniformes (P=24, R=3) por célula de 16×16 | 312 | 624 | 936 | 468 | 2964 | 312 | 3744 | 6240 | 11648 |
   | `color` | histogramas HSV (36+16+16) + média e desvio de cada canal | 74 | 74 | 74 | 74 | 74 | 74 | 74 | 74 | 74 |

6. **Vetores de atributos.** Os descritores **não são concatenados entre si** e nenhuma ROI é
   classificada sozinha. Cada vetor de `FEATURE_VECTORS` tem um único descritor:
   `rois_*` (as três ROIs ad hoc), `auto_rois_*` (as três regiões automáticas), `bottle_*`
   (garrafa inteira, sem região) e, sem localização, `wide_*` (área ampla) e `image_*` (imagem
   inteira). São 15 vetores (5 escopos × 3 descritores).

7. **Classificadores.** Cada classificador é treinado uma vez por vetor (45 modelos), sempre
   com o mesmo grid search (5 dobras estratificadas **somente no treino**, otimizando
   F1-macro), com `class_weight="balanced"` quando disponível.

   | modelo | pipeline | grade |
   |---|---|---|
   | SVM | `StandardScaler` + `SVC` | kernel linear: C ∈ {0,01; 0,1; 1} · RBF: C ∈ {1, 10, 100}, gamma ∈ {scale, 1e-3, 1e-4} · polinomial: grau ∈ {2, 3}, C ∈ {1, 10, 100} |
   | KNN | `StandardScaler` + `KNeighborsClassifier` | n_neighbors ∈ {1, 3, 5, 7, 11, 15}; weights ∈ {uniform, distance} |
   | Random Forest | sem escala | n_estimators ∈ {100, 300}; max_depth ∈ {None, 20}; max_features ∈ {sqrt, 0,1} |

   Os hiperparâmetros escolhidos para cada par ficam em `feature_comparison.csv`. No SVM, o
   kernel linear foi o escolhido em 13 dos 15 vetores, o RBF em `rois_lbp` e o polinomial em
   `auto_rois_color`.

Todos os parâmetros, seus valores e o motivo de cada escolha estão em
[`docs/parametros.md`](docs/parametros.md).

### Busca dos parâmetros (só com as imagens de treino)

Os parâmetros dos descritores e da localização foram medidos, e não apenas fixados à mão. As
300 imagens de validação nunca entram nessa escolha; são usadas uma única vez, no resultado
final.

```bash
.venv/bin/python experiments/bottle_box_methods.py         # ~15 min
.venv/bin/python experiments/localization_sensitivity.py   # ~4 min
.venv/bin/python experiments/descriptor_search.py          # ~2h40 (retoma se for interrompido)
```

- `bottle_box_methods.py` compara três modos de tirar a caixa da garrafa da mesma máscara
  (automático, tamanho ad hoc e misto), cada um com vários valores, contra as caixas anotadas.
- `localization_sensitivity.py` varia um parâmetro da localização por vez e compara a caixa
  detectada com a caixa anotada das 1200 imagens de treino.
- `descriptor_search.py` varia um fator por vez em cada descritor. A nota de uma configuração é
  o F1-macro médio da validação cruzada (5 dobras) nos 9 pares vetor × classificador do
  descritor. Um valor só substitui o de referência se ganhar por mais de um erro padrão
  (o erro padrão é a média, nos 9 pares, do desvio entre as dobras dividido por √5).

Valores medidos com a localização atual (em negrito, o valor adotado; erro padrão da
referência: HOG 0,0011, LBP 0,0034, cor 0,0004):

| descritor | fator | valores testados (F1-macro na validação cruzada) | decisão |
|---|---|---|---|
| HOG | direções | **6: 0,9944** · 9: 0,9932 · 12: 0,9939 · 18: 0,9921 | 9 → **6** |
| HOG | célula (com 6 direções) | 8 px: 0,9869 · **8 px na tampa e 16 px nas demais: 0,9944** · 16 px: 0,9934 | mantido |
| LBP | vizinhança (P, R) | (8, 1): 0,9722 · (8, 2): 0,9820 · (16, 2): 0,9805 · **(24, 3): 0,9871** · (24, 4): 0,9870 · (32, 4): 0,9862 | mantido em (24, 3) |
| LBP | célula | 8 px: 0,9765 · **16 px: 0,9871** · 32 px: 0,9744 | mantido em 16 px |
| cor | faixas de H, S, V | (9, 4, 4): 0,9940 · (18, 8, 8): 0,9990 · **(36, 16, 16): 0,9994** · (72, 32, 32): 0,9992 | mantido em (36, 16, 16) |

O valor de 6 direções do HOG não passou pela regra acima quando 9 era a referência: ganhou de
9 por 0,0012, menos que o erro padrão de 0,0015. Foi adotado pelo critério de um erro padrão
na forma usual (entre as configurações a até um erro padrão da melhor, fica a mais simples):
6 direções é ao mesmo tempo a melhor nota e o menor vetor (um terço menor que com 9), ganha
em 7 dos 9 pares e empata em mais 1. Com 6 como referência, a busca foi repetida e manteve 6
e as células.

Na primeira busca, ainda com a localização anterior, o LBP passou de (8, 1) para (24, 3) e a
cor de (18, 8, 8) para (36, 16, 16); a busca repetida com a localização atual confirmou os
dois. As tabelas completas e as decisões ficam em `results/parameter_search/`; as buscas
anteriores, em `before_new_localization/` (localização anterior) e
`before_hog_6_orientations/` (localização atual, HOG com 9 direções).

O que ainda é escolhido à mão: a largura da caixa da garrafa (`BOTTLE_WIDTH`), as faixas das
ROIs ad hoc, o patch de referência da normalização (`REFERENCE_PATCH`) e parâmetros genéricos
de processamento (tamanhos de redimensionamento, tamanho da localização).

## Resultados (`results/`, validação do Kaggle, 300 imagens)

Acurácia de cada vetor em cada classificador (`feature_matrix_accuracy.csv`,
`feature_matrix.png`; o F1-macro está em `feature_matrix_f1_macro.csv`):

| vetor | atributos | SVM | KNN | Random Forest |
|---|---:|---:|---:|---:|
| `rois_hog` | 1728 | 1,000 | 0,997 | 1,000 |
| `auto_rois_hog` | 2520 | 1,000 | 0,997 | 1,000 |
| `bottle_hog` | 2760 | 1,000 | 0,997 | 1,000 |
| `wide_hog` | 5040 | 1,000 | 0,960 | 1,000 |
| `image_hog` | 9720 | 0,997 | 0,987 | 0,997 |
| `rois_lbp` | 1872 | 1,000 | 0,993 | 0,990 |
| `auto_rois_lbp` | 3744 | 1,000 | 0,987 | 0,990 |
| `bottle_lbp` | 3744 | 1,000 | 0,980 | 0,990 |
| `wide_lbp` | 6240 | 1,000 | 0,963 | 0,977 |
| `image_lbp` | 11648 | 0,990 | 0,993 | 0,967 |
| `rois_color` | 222 | 1,000 | 1,000 | 1,000 |
| `auto_rois_color` | 222 | 1,000 | 1,000 | 1,000 |
| `bottle_color` | 74 | 1,000 | 1,000 | 0,997 |
| `wide_color` | 74 | 1,000 | 0,997 | 0,997 |
| `image_color` | 74 | 1,000 | 0,997 | 1,000 |

- **SVM.** Acerta as 300 imagens de validação com os 9 vetores em que a garrafa é localizada e
  com os 3 da área ampla; só erra com a imagem inteira (HOG 1 erro, LBP 3, todos de
  `defect_label_damage`).
- **ROIs ad hoc × regiões automáticas × garrafa inteira.** Com o SVM os três escopos empatam.
  Entre as regiões automáticas e as ROIs ad hoc há 8 empates e 1 par em que as automáticas
  perdem (LBP + KNN), por 0,7 ponto percentual (2 imagens). A garrafa inteira fica até 1,3
  ponto abaixo com o KNN (LBP).
- **Sem localização (área ampla e imagem inteira).** Os descritores continuam funcionando, mas o
  fundo a mais dilui a garrafa no vetor: com o KNN o HOG cai para 0,960 (área ampla) e 0,987
  (imagem inteira), e o LBP para 0,963 e 0,993; com o Random Forest o LBP cai para 0,977 e
  0,967. Na área ampla, 6 garrafas boas viram `defect_no_cap` com HOG + KNN; no LBP os erros
  se concentram em `defect_label_damage`. A cor quase não muda (0,997–1,000), porque o fundo é
  o mesmo em todas as fotos. Na validação cruzada do treino o efeito também aparece (SVM com
  LBP: 0,997 na garrafa inteira, 0,983 na área ampla, 0,962 na imagem inteira). Os 18 modelos
  sem localização somam 54 erros (SVM 4, KNN 31, Random Forest 19).
- **Descritores.** Com a garrafa localizada, o HOG e a cor ficam entre 0,997 e 1,000 em todos os pares; o LBP é o que mais
  depende do classificador (1,000 com SVM; 0,980–0,993 com KNN e Random Forest). A cor usa só
  74 a 222 valores.
- **Erros.** Os 27 modelos com a garrafa localizada somam 25 erros nas 300 imagens, em 15
  imagens diferentes; 12 são falsas aprovações (nenhuma com SVM, 8 com KNN e 4 com Random
  Forest). Com os 18 modelos sem localização, os 45 somam 79 erros em 48 imagens e 18 falsas
  aprovações (KNN 14, Random Forest 4).
- **Efeito das escolhas de parâmetros** (erros somados dos 27 modelos na validação): com a
  busca dos descritores e do kernel do SVM, 52; com a suavização do perfil em 31, 43; com o
  novo modo de tirar a caixa (largura fixa, topo em 30%), 29; com 6 direções no HOG em vez de
  9, 25 (o `bottle_hog` com KNN passou de 6 erros para 1, e o `auto_rois_hog` com KNN de 0
  para 1). Antes dessas mudanças o LBP ficava em 0,940–0,973 com KNN e Random Forest.
- **Tempo.** Com a garrafa localizada, o pré-processamento domina (~21 ms/imagem) e os 27 pares
  processam entre 29 e 40 garrafas/s (extração + predição). Sem localização, os recortes
  grandes custam mais (a cor da imagem inteira, calculada na resolução original, leva ~48 ms):
  14 a 34 garrafas/s.

## Arquivos gerados

Na pasta de saída:

| arquivo | conteúdo |
|---|---|
| `summary.json` | todas as métricas |
| `feature_comparison.csv` | uma linha por par (vetor, classificador): acurácia, F1-macro, F1 de cada uma das 7 classes, erros, tempos, hiperparâmetros |
| `feature_matrix_accuracy.csv`, `feature_matrix_f1_macro.csv`, `feature_matrix.png` | matriz vetor × classificador |
| `background_model.png`, `bottle_localization.png`, `rois_per_class.png`, `auto_rois_per_class.png` | figuras do método |
| `feature_cache.joblib` | cache dos atributos (não versionado) |

Em `<vetor>/` (ex.: `rois_hog/`):

| arquivo | conteúdo |
|---|---|
| `summary.json`, `model_comparison.{csv,png}` | métricas e comparação dos três modelos nesse vetor |
| `classifier_margins.png`, `didactic/<imagem>.png` | decisão real de cada modelo por imagem de validação (só com `DIDACTIC=1`) |

Em `<vetor>/<model>/` (`svm`, `knn`, `random_forest`):

| arquivo | conteúdo |
|---|---|
| `model.joblib` | modelo treinado (não versionado; o `make run` o recria) |
| `classification_report.txt` | relatório por classe |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | matrizes de confusão |
| `criteria_metrics.csv` | métricas binárias por quesito |
| `validation_predictions.csv` | predições na validação |
| `simulated_alerts.csv` | simulação de alertas |
| `validation_errors.png` | só é gerado se houver erros |
| `feature_importance.csv` | só para `random_forest` |

## Observações para o TCC

- A base é fácil: vários vetores saturam a validação. As diferenças aparecem na dependência do
  classificador e no tamanho do vetor.
- Os resultados não são diretamente comparáveis ao mAP do YOLO. O YOLO localiza a caixa do
  defeito; aqui a localização vem das ROIs fixas e o classificador rotula a imagem inteira.

## Figuras do TCC

`thesis/make_figures.py` gera as figuras do TCC 2 (rótulos em português) a partir dos
resultados já calculados em `results/` (`--results`), sem alterar nada nas pastas de
resultados. Precisa do `make run` antes.

```bash
.venv/bin/python thesis/make_figures.py --output thesis/preview   # prévia (a base vem de BOTTLE_DATA)
make thesis-figures                              # -> ../TCC_2/figuras/tcc2/ (sobrescreve as figuras do texto)
make thesis-figures THESIS_FIGURES=outra_pasta
```

Os números citados no texto ficam em `thesis_facts.json`, na pasta de saída. Uma prévia das
figuras está em `thesis/preview/`.

As figuras exportadas (TCC e apresentação) são vetoriais, para não perder qualidade ao
ampliar: o TCC recebe PDF (fontes TrueType embutidas, como pede o PDF/A) e a apresentação, SVG.
Traços, textos, eixos e marcadores ficam em vetor; o que não pode ser vetor (fotos, máscaras,
mapas de calor, desenhados com `imshow`) vai embutido como imagem a 300 DPI. As figuras que o
pipeline grava em `results/` continuam em PNG.

A apresentação está em `presentation/index.html` (fontes em `presentation/src/`).

## Interface web

Uma interface local (`bottle_inspection_web/` + `frontend/`) para analisar garrafas por envio
de imagens, amostras da base ou câmeras, comparar perfis lado a lado, ver relatórios e
disparar integrações. A especificação completa está em `docs/web-interface-spec.md`.

A web **não treina modelos**: ela procura modelos em `results/` (configurável em Sistema), no
formato `<vetor>/<algoritmo>/model.joblib`, e cria um perfil para cada modelo (45 em
`results/`). Um `model.joblib` que não pode ser carregado aparece na lista com o erro. Ela usa os
modelos treinados e validados pelo `make run` (ou `make quick`) e chama as mesmas funções de
pré-processamento e atributos da linha de comando, passando os parâmetros de cada perfil.

### Como rodar

```bash
make run          # (ou make quick) treina e valida os modelos -> results/
make web          # instala dependências, compila o frontend e abre http://127.0.0.1:8000
```

| alvo | o que faz |
|---|---|
| `make web-setup` | instala `requirements-web*.txt` no `.venv` e roda `npm ci` em `frontend/` |
| `make web-build` | compila o frontend (só quando os fontes mudaram) |
| `make web` | `web-setup` + `web-build` e serve em `127.0.0.1:8000` (`WEB_HOST`, `WEB_PORT`) |
| `make web-dev` | backend com recarga automática + servidor do Vite (proxy de `/api`) |
| `make web-test` | testes do backend (pytest), testes do frontend (vitest) e checagem das traduções |
| `make web-lint` | black + pyright no backend; eslint + prettier no frontend |
| `make web-e2e` | teste ponta a ponta e de acessibilidade com Playwright (precisa dos modelos do `make quick`) |

Requisitos: Python 3.13 (o mesmo `.venv`) e Node ≥ 20.19 (só para compilar o frontend). O app
funciona offline (fontes e bibliotecas embutidas). Sem login: por isso escuta só em
`127.0.0.1`; use `--host` apenas numa rede confiável.

### Telas

- **Inspecionar**: envio (arrastar e soltar ou lote), amostras da validação com rótulo real,
  câmera do navegador e câmeras do servidor (USB, RTSP, HTTP) nos modos único, contínuo e por
  movimento. Com vários perfis, cada um mostra seu próprio resultado lado a lado (não há
  veredito combinado): ROIs sobre a imagem, classe, confiança, pontuações das 7 classes e tempo
  de cada etapa.
- **Perfis**: receitas com o modelo treinado (vetor de atributos + algoritmo, ex.: `rois_hog · SVM`), os parâmetros e as regras de veredito.
  Parâmetros que mudam o tamanho do vetor de atributos ficam travados (vêm do treino); os
  demais podem ser editados com pré-visualização ao vivo e ficam marcados como "≠ treino".
  Cada alteração gera uma versão.
- **Depuração**: escolhe-se uma imagem (envio ou amostra da base) e a tela mostra cada etapa
  do pré-processamento: matriz da imagem, modelo de fundo, normalização (região de referência
  e ganhos), diferença para o fundo com o limiar de Otsu, máscara (limiar, fechamento,
  abertura), perfis de colunas e linhas com a caixa da garrafa e os recortes das regiões.
  Depois, N perfis podem ser adicionados, removidos e trocados à vontade sobre a mesma imagem:
  cada um mostra os grupos de atributos (HOG, LBP, cor), o vetor de atributos (bruto e
  padronizado) e a decisão do classificador. Os "ajustes temporários" (fechamento, abertura,
  suavização e frações dos perfis) valem só nessa tela. Nada é gravado: a depuração não aparece
  nos relatórios, e escolher outra imagem recomeça do zero.
- **Relatórios**: indicadores, distribuição das 7 classes, Pareto, séries temporais,
  comparação entre perfis, matriz de confusão das amostras da base e exportação CSV.
- **Integrações**: regras "quando → se → então" com e-mail, webhook assinado (Standard
  Webhooks), MQTT, arquivo, notificação e som; intervalo mínimo, registro e simulação.
- **Sistema**: câmeras, pastas de modelos, armazenamento ("Apagar todas as imagens"), base e
  limites.

### Dados

Tudo fica em `webdata/` (fora do git): `app.db` (SQLite), `images/`, `backgrounds/`,
`exports/`, `logs/` e `secrets.toml` (senhas das integrações, permissão 0600, nunca no banco).
Nenhum alvo do `Makefile` apaga essa pasta.

### Mudanças no pacote `bottle_inspection`

Para a web: `params.py` (`InspectionParams`, com padrões lidos do `config.py`), um
argumento opcional `params=` nas funções de pré-processamento e atributos (sem ele, valem
os padrões de `config.py`) e metadados extras salvos em `model.joblib` (parâmetros do treino,
versões das bibliotecas, vetor de atributos do modelo). Um `model.joblib` sem esses metadados
ou com outra `format_version` não é carregado pela web.

### Observações

- Câmeras USB no WSL2 precisam do `usbipd`. A câmera do navegador funciona sem configuração.
- A acurácia validada vale para a câmera e o cenário da base; outra câmera precisa de fundo
  próprio (perfil → "Capturar fundo") e não tem validação.
- As confianças são definidas por algoritmo e não são comparáveis entre SVM, KNN e Random
  Forest.

## Licença

Código sob a licença MIT (ver [LICENSE](LICENSE)). A base de imagens não faz parte do
repositório e segue os termos da página dela no Kaggle. O tema da apresentação em
`presentation/src/vendor/html-slides/` tem a sua própria licença MIT.
