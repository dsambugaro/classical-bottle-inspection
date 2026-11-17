/* === IMAGE VIEWER (deck-specific): click an image to inspect it === */
(function () {
  const MIN_SCALE = 1, MAX_SCALE = 12, WHEEL_STEP = 1.2;
  const overlay = document.createElement('div');
  overlay.className = 'lightbox';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', 'Visualizador de imagem');
  overlay.innerHTML =
    '<img class="lightbox-image" alt="" draggable="false">' +
    '<div class="lightbox-bar">' +
    '<button type="button" data-action="out" aria-label="Diminuir zoom">−</button>' +
    '<span class="lightbox-zoom" aria-live="polite">100%</span>' +
    '<button type="button" data-action="in" aria-label="Aumentar zoom">+</button>' +
    '<button type="button" data-action="reset" aria-label="Ajustar à tela">Ajustar</button>' +
    '<button type="button" data-action="close" class="lightbox-close" aria-label="Fechar (Esc)">✕ Fechar</button>' +
    '</div>' +
    '<p class="lightbox-hint">Role para dar zoom · arraste para mover · duplo clique para ajustar · Esc para fechar</p>';
  document.body.appendChild(overlay);
  const image = overlay.querySelector('.lightbox-image');
  const zoomLabel = overlay.querySelector('.lightbox-zoom');
  let scale = 1, x = 0, y = 0, drag = null, lastFocus = null;

  function isOpen() { return overlay.classList.contains('open'); }
  function render() {
    image.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + scale + ')';
    zoomLabel.textContent = Math.round(scale * 100) + '%';
    overlay.classList.toggle('zoomed', scale > 1);
  }
  function reset() { scale = 1; x = 0; y = 0; render(); }
  /* Zooms keeping the point (cx, cy) of the viewport under the cursor. */
  function zoomAt(factor, cx, cy) {
    const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
    if (next === scale) return;
    const ox = cx - window.innerWidth / 2, oy = cy - window.innerHeight / 2;
    x = ox - (ox - x) * (next / scale);
    y = oy - (oy - y) * (next / scale);
    scale = next;
    if (scale === 1) { x = 0; y = 0; }
    render();
  }
  function open(source) {
    lastFocus = document.activeElement;
    image.src = source.currentSrc || source.src;
    image.alt = source.alt || '';
    reset();
    overlay.classList.add('open');
    overlay.querySelector('.lightbox-close').focus();
  }
  function close() {
    overlay.classList.remove('open');
    drag = null;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  document.querySelectorAll('.deck img').forEach(function (img) {
    img.classList.add('zoomable');
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.setAttribute('aria-label', (img.alt ? img.alt + ' — ' : '') + 'ampliar imagem');
    img.addEventListener('click', function () { open(img); });
    img.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); open(img); }
    });
  });

  overlay.addEventListener('click', function (e) {
    const action = e.target.dataset && e.target.dataset.action;
    const center = [window.innerWidth / 2, window.innerHeight / 2];
    if (action === 'close' || e.target === overlay) close();
    else if (action === 'in') zoomAt(WHEEL_STEP * WHEEL_STEP, center[0], center[1]);
    else if (action === 'out') zoomAt(1 / (WHEEL_STEP * WHEEL_STEP), center[0], center[1]);
    else if (action === 'reset') reset();
  });
  overlay.addEventListener('wheel', function (e) {
    e.preventDefault();
    e.stopPropagation();  /* the deck changes slide on wheel */
    zoomAt(e.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP, e.clientX, e.clientY);
  }, { passive: false });
  image.addEventListener('dblclick', function (e) {
    if (scale > 1) reset(); else zoomAt(3, e.clientX, e.clientY);
  });
  image.addEventListener('pointerdown', function (e) {
    e.preventDefault();
    drag = { px: e.clientX, py: e.clientY, x: x, y: y };
    image.setPointerCapture(e.pointerId);
    overlay.classList.add('dragging');
  });
  image.addEventListener('pointermove', function (e) {
    if (!drag) return;
    x = drag.x + e.clientX - drag.px;
    y = drag.y + e.clientY - drag.py;
    render();
  });
  ['pointerup', 'pointercancel'].forEach(function (name) {
    image.addEventListener(name, function () { drag = null; overlay.classList.remove('dragging'); });
  });
  ['touchstart', 'touchend'].forEach(function (name) {
    overlay.addEventListener(name, function (e) { e.stopPropagation(); });  /* the deck swipes slides */
  });
  /* Capture phase: while the viewer is open the deck must not react to the keyboard. */
  window.addEventListener('keydown', function (e) {
    if (!isOpen()) return;
    const center = [window.innerWidth / 2, window.innerHeight / 2];
    if (e.key === 'Escape') close();
    else if (e.key === '+' || e.key === '=') zoomAt(WHEEL_STEP, center[0], center[1]);
    else if (e.key === '-') zoomAt(1 / WHEEL_STEP, center[0], center[1]);
    else if (e.key === '0') reset();
    else if (e.key === 'Tab' || e.key === 'Enter' || e.key === ' ') return;  /* buttons stay usable */
    e.stopPropagation();
    if (e.key !== 'Escape') e.preventDefault();
  }, true);
})();
