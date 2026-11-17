# Prompt: atualizar o texto do TCC para o código atual

Você vai atualizar o texto do TCC em `../TCC_2/` (projeto LaTeX ao lado desta pasta, capítulos em
`capitulos/`) para que ele descreva o código atual, que está em
`code/`. Leia este arquivo inteiro antes de editar.

## Fontes da verdade

Não use números de memória nem deste prompt: leia-os dos arquivos.

- `code/README.md`: método atual, passo a passo, e a tabela de resultados.
- `code/docs/parametros.md`: todos os parâmetros, seus valores e o motivo de cada um.
- `code/results/summary.json` e `code/results/feature_comparison.csv`: todas as métricas.
- `code/thesis/preview/thesis_facts.json`: os números citados nas figuras.
- `code/thesis/preview/*.pdf`: as 33 figuras da versão atual, com rótulos em português.
- `code/results/parameter_search/`: a busca de parâmetros (tabelas `localization_sensitivity.csv`,
  `descriptor_search.csv`, `descriptor_search_summary.csv` e as decisões em
  `descriptor_decisions.json`).
- `code/bottle_inspection/`: o código, quando o README não bastar.

## Regras do autor (valem para todas as edições)

1. **Toda edição fica em vermelho e é reversível.** Texto novo ou alterado usa `\edicao{...}`
   (trechos curtos, títulos, legendas) ou o ambiente `blocoedicao` (blocos longos), ambos
   definidos em `configuracoes.tex` com a cor `corEdicao`. O texto original **nunca é apagado**:
   fica comentado com `%` logo acima do trecho novo.
2. **Sempre as 7 classes, pelo nome exato**, em toda tabela e gráfico: `bottle_no_defect`,
   `bottle_other_brand`, `defect_low_fill`, `defect_no_cap`, `defect_loose_cap`,
   `defect_debris`, `defect_label_damage`. O id 0 do Kaggle (`bottle_ron88`) foi renomeado para
   `bottle_no_defect`.
3. **A divisão do Kaggle é usada como está**: 1200 imagens de treino e 300 de validação, sem
   reorganizar.
4. **Nenhuma medida ad hoc.** As únicas posições escolhidas à mão são as faixas das ROIs. HOG,
   LBP e cor são classificados como vetores separados, nunca concatenados.
5. Não invente justificativas. Quando um valor foi fixado à mão sem ajuste, diga isso; o
   `parametros.md` indica o tipo de motivo de cada parâmetro.

## O que mudou no código e no processo (início × fim deste trabalho)

1. **A avaliação de robustez à iluminação foi removida por completo.** Não há mais imagens de
   validação com brilho ou gama alterados, nem as métricas, tabelas e figuras correspondentes.
   Os modelos são avaliados só nas imagens originais de treino e validação. A normalização
   fotométrica continua existindo como etapa do pré-processamento, e sua figura
   (`prep_normalizacao`) mostra apenas a imagem real, sem escurecimento artificial.
2. **A referência de 3 garrafas por segundo foi removida.** O texto deve informar quantas
   garrafas por segundo cada modelo processa, sem comparar com a vazão de uma linha de envase
   nem dizer que o sistema "atende" ou "supera" algum limite.
3. **O modo de tirar a caixa da garrafa mudou.** A máscara é a mesma (diferença para o fundo,
   Otsu, fechamento e abertura). A caixa agora é tirada assim: o perfil de colunas, suavizado
   por uma média móvel de 31 colunas, acha o trecho da garrafa (≥ 40% do pico) e o centro é o
   meio desse trecho; a **largura é um valor fixo, escolhido à mão** (14,5% da largura da
   imagem), centrado nesse ponto; a altura é medida, com um limiar para o topo (30% da linha
   mais cheia) e outro para a base (50%). O texto não pode mais dizer que nenhuma medida da
   garrafa é informada: a posição e a altura são medidas, a largura é ad hoc.
   Três modos foram comparados nas imagens de treino contra as caixas anotadas
   (`experiments/bottle_box_methods.py`): todas as bordas pelos limiares (IoU médio 0,921),
   largura e altura fixas (0,919) e o misto adotado (0,936; 0,939 na validação). Os motivos de
   cada valor (suavização 31, topo 30%, base 50%, largura 0,145) e a tabela de sensibilidade
   estão em `parametros.md`, seção 2.
4. **Todos os parâmetros estão documentados com o motivo** (`parametros.md`). O texto deve ser
   coerente com esse documento ao justificar valores, e dizer quando um valor é escolha manual.
5. **Houve uma busca de parâmetros, feita só com as 1200 imagens de treino.** Isto é conteúdo
   novo para o texto (método e resultados):
   - *Protocolo.* As 300 imagens de validação não participam de nenhuma escolha; são usadas
     uma única vez, no resultado final. Um fator é variado por vez.
   - *Localização* (`experiments/localization_sensitivity.py`): fechamento, abertura,
     suavização e frações dos perfis, comparando a caixa detectada com a anotada (IoU e erros
     das bordas). Figura `res_sensibilidade_localizacao`.
   - *Descritores* (`experiments/descriptor_search.py`): a nota de cada configuração é o
     F1-macro médio da validação cruzada em 5 dobras nos 9 pares vetor × classificador do
     descritor; um valor só substitui o de referência se ganhar por mais de um erro padrão.
     Figura `res_sensibilidade_descritores`.
   - *Decisões.* HOG: 9 direções e as células atuais foram mantidas (confirmadas). LBP: a
     vizinhança passou de (P=8, R=1) para **(P=24, R=3)**, com 26 códigos por célula. Cor: o
     histograma passou de (18, 8, 8) para **(36, 16, 16)** faixas, 74 valores por região.
   - *Efeito.* Compare, no texto, os resultados na validação antes e depois da busca (estão
     no README e em `parametros.md`): o LBP foi o que mais ganhou; o HOG não mudou.
   - *Limites a declarar.* Um fator por vez não garante o ótimo conjunto; tamanhos de
     redimensionamento, bloco do HOG e método do LBP não foram variados; a busca dos
     descritores foi feita antes das mudanças na localização e não foi repetida.
   - *Análise dos erros.* Refaça a contagem a partir dos `validation_predictions.csv` atuais:
     os 27 modelos somam 29 erros em 18 imagens, 16 deles falsas aprovações (nenhuma com SVM,
     12 com KNN, 4 com Random Forest). O SVM acerta as 300 imagens nos 9 vetores. A imagem
     `no_cap_0089`, que concentrava 16 erros, deixou de ser um caso especial.
6. **O kernel do SVM passou a ser escolhido pela busca em grade** (linear, RBF ou polinomial),
   junto com C e gamma. O linear foi o escolhido em 7 dos 9 vetores, o RBF em 1 e o polinomial
   em 1. O texto não deve mais
   chamar o classificador de "SVM RBF" nem descrever só o kernel RBF.
7. **As figuras de HOG foram refeitas em alta resolução** (traços nítidos ao ampliar). As
   figuras do LBP mostram a vizinhança de 24 pontos e raio 3, e as da cor, as novas faixas.
8. **Há matrizes de confusão para as 27 combinações** de vetor × classificador (na apresentação,
   `code/presentation/assets/steps/c_<vetor>.png`; no conjunto do TCC, `res_matrizes_confusao`
   e `res_matrizes_confusao_dificil`).
9. **Existe uma única versão do código.** Não há mais pastas nem nomes com v1/v2/v3: os
   resultados ficam em `results/` e a apresentação em `presentation/`. O texto não deve citar
   versões do código nem comparar com implementações anteriores.
10. **Interface web** (só se o texto a descrever): há um modo de depuração, que mostra cada
   etapa do pré-processamento de uma imagem e o resultado de vários perfis, e o histórico pode
   ser visto em cards, uma linha por imagem com os perfis lado a lado, com filtros.

## Atenção: o texto está mais defasado do que a lista acima

O texto atual do TCC descreve um método anterior a tudo isso: um único vetor de atributos com
HOG, cor, "perfil vertical do corpo" e "medidas ad hoc", localização por janela de largura
fixa, e as seções de robustez, ablação e importância dos atributos. O código atual é outro
(veja o README): detecção da garrafa por subtração de fundo com limiar de Otsu (posição e
altura medidas na máscara, largura da caixa fixa), ROIs ad hoc
ou regiões automáticas aprendidas do treino, descritores HOG, LBP e cor, 9 vetores e 27
modelos comparados numa matriz. Portanto, além dos itens acima, alinhe com o README e com o
código:

- `capitulos/cap-base-desenvolvimento.tex`: Pré-processamento (localização, recorte das
  regiões), Extração de atributos (entra o LBP, com P=24 e R=3; a cor passa a 36+16+16 faixas; saem "Perfil vertical do corpo" e "Medidas ad
  hoc"), Vetor de atributos, Classificadores (kernel do SVM buscado) e Protocolo de avaliação (a
  busca de parâmetros só no treino e a validação usada uma única vez).
- `capitulos/cap-experimentos-resultados.tex`: uma seção nova com a busca de parâmetros (antes
  dos resultados finais), Resultados na validação (matriz vetor × classificador), remoção das seções "Robustez à iluminação" e das que dependem do vetor
  único (ablação, importância dos atributos), Tempo de execução (sem a referência de 3
  garrafas/s) e o Comparativo.
- `capitulos/cap-conclusoes.tex` e as considerações finais dos capítulos: conclusões coerentes
  com os resultados atuais, sem afirmações sobre robustez à iluminação.
- `capitulos/cap-conceitos.tex` e `cap-trabalhos-relacionados.tex`: revise as menções a brilho
  e gama que só existiam para sustentar o experimento de robustez, e acrescente a
  fundamentação do LBP e do método de Otsu se faltar.

## Figuras

As figuras citadas no texto ficam em `TCC_2/figuras/tcc2/` e ainda são as do método antigo. As
atuais estão em `code/thesis/preview/`. Para substituí-las, rode em `code/`:
`make thesis-figures` (grava em `../TCC_2/figuras/tcc2/`). Depois:

- troque as figuras que deixaram de existir (`atr_adhoc`, `atr_perfil_corpo`,
  `res_importancia_atributos`, `res_robustez_com_normalizacao`,
  `res_robustez_sem_normalizacao`) pelas novas onde fizer sentido (`prep_otsu`,
  `prep_regioes_automaticas`, `atr_lbp_pixel`, `atr_lbp_celula`, `atr_lbp_mapa`, `res_regioes`,
  `res_f1_por_classe`, `res_custo`, `res_localizacao`, `res_importancia_regioes`,
  `res_matrizes_confusao_dificil`, `res_sensibilidade_localizacao`,
  `res_sensibilidade_descritores`);
- comente os `\includegraphics` antigos em vez de apagá-los, e marque legendas novas com
  `\edicao{}`.

## Como trabalhar

1. Leia o README, o `parametros.md` e os capítulos citados.
2. Faça um capítulo por vez; para cada trecho, comente o original e escreva o novo em vermelho.
3. Confira cada número com os arquivos de resultados antes de escrevê-lo.
4. Compile (`main.tex`) e verifique que não há referências quebradas nem figuras faltando.
5. Ao final, liste o que foi alterado em cada capítulo e o que ficou pendente ou em dúvida.
