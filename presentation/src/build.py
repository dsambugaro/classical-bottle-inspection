"""Assembles the deck from the html-slides theme, components, runtime and slide body.

The slides with the confusion matrices are generated here, one per feature vector, from the
errors listed in `assets/steps/facts.json` (written by make_steps.py).
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SKILL = HERE / "vendor" / "html-slides"
"""Theme, components and runtime of html-slides 0.9.4 (MIT), vendored."""
OUT = HERE.parent / "index.html"
FACTS = HERE.parent / "assets" / "steps" / "facts.json"
MODEL_NAMES = {"svm": "SVM", "knn": "KNN", "random_forest": "Random Forest"}
SCOPE_NAMES = {
    "rois": "ROIs ad hoc",
    "auto_rois": "regiões automáticas",
    "bottle": "garrafa inteira",
    "wide": "área ampla",
    "image": "imagem inteira",
}
DESCRIPTOR_NAMES = {"hog": "HOG", "lbp": "LBP", "color": "cor HSV"}
DESCRIPTOR_CLASSES = {
    "hog": "highlight-orange",
    "lbp": "highlight-green",
    "color": "highlight-blue",
}


def error_list(errors):
    """The confusions of one classifier as text: "real → prevista (n)"."""
    if not errors:
        return "nenhum erro"
    return "; ".join(
        f"<code>{e['true']}</code> → <code>{e['predicted']}</code> ({e['count']})" for e in errors
    )


def confusion_slides(confusions):
    """One slide per feature vector with the confusion matrix of the three classifiers."""
    slides = []
    for vector, by_model in confusions.items():
        scope, _, descriptor = vector.rpartition("_")
        totals = {m: sum(e["count"] for e in errors) for m, errors in by_model.items()}
        lines = " · ".join(
            f"<strong>{MODEL_NAMES[m]}</strong>: {error_list(errors)}"
            for m, errors in by_model.items()
        )
        notes = {
            "title": f"Matrizes de confusão: {vector}",
            "script": f"Estas são as matrizes de confusão dos três classificadores no vetor {vector}, "
            f"{DESCRIPTOR_NAMES[descriptor]} sobre {SCOPE_NAMES[scope]}. A diagonal são os acertos; "
            "os números em vermelho fora dela são os erros.",
            "notes": [
                f"{MODEL_NAMES[m]}: {n} {'erro' if n == 1 else 'erros'} em 300"
                for m, n in totals.items()
            ],
        }
        slides.append(f"""<div class="slide" data-slide="0">
  <p class="slide-tag anim-1">Resultado · matrizes de confusão · {DESCRIPTOR_NAMES[descriptor]} · {SCOPE_NAMES[scope]}</p>
  <h2 class="anim-2">Onde cada classificador erra: <span class="{DESCRIPTOR_CLASSES[descriptor]}">{vector}</span></h2>
  <div class="image-frame image-screenshot anim-3" style="margin-top:12px;max-width:1250px;">
    <img src="assets/steps/c_{vector}.svg" alt="Matrizes de confusão do SVM, do KNN e do Random Forest no vetor {vector}" class="slide-image" style="max-height:min(56vh,470px);">
  </div>
  <p class="subtitle anim-4" style="font-size:clamp(12px,1.1vw,14px);">Erros (classe real → classe prevista): {lines}</p>
  <script type="application/json" class="slide-notes">
  {json.dumps(notes, ensure_ascii=False)}
  </script>
</div>

""")
    return "".join(slides)


def numbered(body):
    """Numbers the slides in order (data-slide="0", "1", ...)."""
    counter = iter(range(10_000))
    return re.sub(r'data-slide="\d+"', lambda _match: f'data-slide="{next(counter)}"', body)


theme = (SKILL / "themes/editorial-light.css").read_text()
components = (SKILL / "components.css").read_text()
runtime = (SKILL / "slides-runtime.js").read_text()
custom = (HERE / "custom.css").read_text()
body = (HERE / "body.html").read_text()
confusions = json.loads(FACTS.read_text())["confusions"]
body = numbered(body.replace("<!-- CONFUSION_SLIDES -->\n\n", confusion_slides(confusions)))
lightbox = (HERE / "lightbox.js").read_text()
OUT.write_text(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="generator" content="html-slides v0.9.4">
<title>Do Pixel aos Vetores</title>
<style>
/* === THEME: EDITORIAL LIGHT (verbatim) === */
{theme}
/* === SHARED COMPONENTS (verbatim) === */
{components}
{custom}
</style>
</head>
<body>
<div class="particles" id="particles"></div>
<div class="branding"><div class="branding-icon">T</div> TCC · Descritores separados</div>
<div class="slide-nav" id="slideNav"></div>
<div class="progress-bar" id="progress"></div>
<div class="slide-counter" id="counter"></div>
<div class="nav-hints"><span><kbd>&larr;</kbd><kbd>&rarr;</kbd> navegar</span></div>
<div class="deck" id="deck">
{body}
</div>
<script>
/* === NAVIGATION RUNTIME (verbatim) === */
{runtime}
</script>
<script>
{lightbox}
</script>
</body>
</html>
""")
print(OUT)
