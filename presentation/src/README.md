# Fontes da apresentação "Do Pixel aos Vetores"

Detecção da garrafa só por subtração de fundo, ROIs ad hoc × regiões automáticas e
descritores classificados separadamente (HOG, LBP, cor HSV).

- `make_steps.py`: gera as figuras de cada etapa e dos resultados com o código real do pacote.
  Rodar a partir de `code/`: `.venv/bin/python presentation/src/make_steps.py presentation/assets/steps [pasta de resultados]`
  (padrão: `results`). Os números citados nos slides ficam em `assets/steps/facts.json`.
  As figuras são SVG (vetoriais; fotos e mapas de calor embutidos a 300 DPI). O script também
  gera, com as funções de `bottle_inspection/plots.py`, as três figuras do pipeline usadas nos
  slides (`assets/bottle_localization.svg`, `assets/rois_per_class.svg` e
  `assets/auto_rois_per_class.svg`). A figura da busca de parâmetros
  (`assets/steps/p01_descritores.svg`) e a da área ampla e da imagem inteira
  (`assets/steps/p02_sem_localizacao.svg`) são as mesmas do TCC e vêm de
  `thesis/make_figures.py`.
- `body.html` + `custom.css`: conteúdo dos slides.
- `lightbox.js`: visualizador de imagens (clique para ampliar, scroll para zoom, arrastar, Esc).
- `build.py`: monta `presentation/index.html` com o tema Editorial Light do html-slides 0.9.4
  (MIT), copiado em `vendor/html-slides/`.
  Rodar: `python3 presentation/src/build.py`
