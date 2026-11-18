/**
 * CS8850: Advanced Machine Learning - Lecture 22 (Word Embedding)
 * Interactive Pedagogical Demonstrations Suite
 *
 * 0. Dynamic Bridge Slide Connectors (Precision SVG alignment)
 * 1. Co-Occurrence, PPMI & Truncated SVD Explorer (#emb-ppmi-svd-canvas)
 * 2. Skip-Gram with Negative Sampling Live Trainer (#emb-sgns-canvas)
 * 3. Semantic Analogies, Drift & Creativity Explorer (#emb-analogy-canvas)
 * 4. Subword BPE & Contextual Polysemy Studio (#emb-modern-canvas)
 *
 * All 2D spatial plots and embedding maps enforce an exact 1:1 geometric aspect ratio.
 */

(function () {
  'use strict';

  const SOL = {
    base03: '#002b36',
    base02: '#073642',
    base01: '#586e75',
    base00: '#657b83',
    base0:  '#839496',
    base1:  '#93a1a1',
    base2:  '#eee8d5',
    base3:  '#fdf6e3',
    yellow: '#b58900',
    orange: '#cb4b16',
    red:    '#dc322f',
    magenta:'#d33682',
    violet: '#6c71c4',
    blue:   '#268bd2',
    cyan:   '#2aa198',
    green:  '#859900'
  };

  function reqEl(id, context) {
    const el = document.getElementById(id);
    if (!el) {
      console.error(`[embedding_demos] Missing DOM element #${id} in ${context}`);
    }
    return el;
  }

  function setupHiDPI(canvas) {
    const dpr = window.devicePixelRatio || 1;
    if (!canvas._logicalW) {
      canvas._logicalW = parseInt(canvas.getAttribute('width'), 10) || 1180;
      canvas._logicalH = parseInt(canvas.getAttribute('height'), 10) || 450;
    }
    const logW = canvas._logicalW;
    const logH = canvas._logicalH;
    const targetW = Math.round(logW * dpr);
    const targetH = Math.round(logH * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    return { ctx, width: logW, height: logH };
  }

  function canvasPoint(canvas, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const logW = canvas._logicalW || parseInt(canvas.getAttribute('width'), 10) || 1180;
    const logH = canvas._logicalH || parseInt(canvas.getAttribute('height'), 10) || 450;
    return {
      x: (clientX - rect.left) * (logW / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (logH / Math.max(1, rect.height))
    };
  }

  function stopSwipePropagation(el) {
    if (!el) return;
    ['mousedown', 'pointerdown', 'touchstart'].forEach((evt) => {
      el.addEventListener(evt, (e) => {
        e.stopPropagation();
      }, { passive: true });
    });
  }

  function bindRange(slider, onUpdate) {
    if (!slider) return;
    stopSwipePropagation(slider);
    slider.addEventListener('input', onUpdate);
    slider.addEventListener('change', onUpdate);
  }

  function registerSlideListener(canvas, drawFn) {
    let animFrame = null;
    const safeDraw = () => {
      if (animFrame) cancelAnimationFrame(animFrame);
      animFrame = requestAnimationFrame(drawFn);
    };
    window.addEventListener('resize', safeDraw);
    if (window.ResizeObserver && canvas) {
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.contentRect.width > 50) safeDraw();
        }
      });
      ro.observe(canvas);
    }
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && canvas && e.currentSlide.contains(canvas)) {
          safeDraw();
          setTimeout(safeDraw, 80);
          setTimeout(safeDraw, 250);
        }
      });
      Reveal.addEventListener('ready', (e) => {
        if (e.currentSlide && canvas && e.currentSlide.contains(canvas)) {
          setTimeout(safeDraw, 100);
        }
      });
    }
    return safeDraw;
  }

  function drawRoundRect(ctx, x, y, w, h, r, fill, stroke, lineW) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineW || 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, lineWidth, headLen, dashed) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 2) return;
    const h = headLen || 8;
    const angle = Math.atan2(dy, dx);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth || 2;
    if (dashed) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - h * Math.cos(angle - Math.PI / 6), y2 - h * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - h * Math.cos(angle + Math.PI / 6), y2 - h * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function dotVec(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function normVec(a) {
    return Math.sqrt(dotVec(a, a));
  }

  function cosineSim(a, b) {
    const na = normVec(a);
    const nb = normVec(b);
    if (na < 1e-10 || nb < 1e-10) return 0;
    return dotVec(a, b) / (na * nb);
  }

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }

  function heatColor(val, minV, maxV) {
    if (val < 0) {
      const u = Math.min(1, Math.abs(val) / Math.max(1e-6, Math.abs(minV)));
      const r = Math.round(253 + (38 - 253) * u);
      const g = Math.round(246 + (139 - 246) * u);
      const b = Math.round(227 + (210 - 227) * u);
      return `rgb(${r},${g},${b})`;
    }
    const t = Math.min(1, val / Math.max(1e-6, maxV));
    const r = Math.round(253 + (203 - 253) * t);
    const g = Math.round(246 + (75 - 246) * t);
    const b = Math.round(227 + (22 - 227) * t);
    return `rgb(${r},${g},${b})`;
  }

  // Symmetric Jacobi eigensolver for small N×N matrices (deterministic, zero external deps)
  function symmetricEigen(mat) {
    const n = mat.length;
    const A = mat.map((row) => row.slice());
    const V = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (__, j) => (i === j ? 1 : 0))
    );
    for (let iter = 0; iter < 80 * n; iter++) {
      let p = 0, q = 1, maxOff = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const av = Math.abs(A[i][j]);
          if (av > maxOff) {
            maxOff = av;
            p = i;
            q = j;
          }
        }
      }
      if (maxOff < 1e-9) break;
      const app = A[p][p];
      const aqq = A[q][q];
      const apq = A[p][q];
      const tau = (aqq - app) / (2 * apq);
      const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
      const c = 1 / Math.sqrt(1 + t * t);
      const s = t * c;
      A[p][p] = app - t * apq;
      A[q][q] = aqq + t * apq;
      A[p][q] = 0;
      A[q][p] = 0;
      for (let r = 0; r < n; r++) {
        if (r !== p && r !== q) {
          const arp = A[r][p];
          const arq = A[r][q];
          A[r][p] = c * arp - s * arq;
          A[p][r] = A[r][p];
          A[r][q] = s * arp + c * arq;
          A[q][r] = A[r][q];
        }
      }
      for (let r = 0; r < n; r++) {
        const vrp = V[r][p];
        const vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq;
        V[r][q] = s * vrp + c * vrq;
      }
    }
    const pairs = [];
    for (let i = 0; i < n; i++) {
      pairs.push({
        val: A[i][i],
        vec: V.map((row) => row[i])
      });
    }
    pairs.sort((a, b) => b.val - a.val);
    return pairs;
  }

  // =========================================================================
  // 0. DYNAMIC BRIDGE SLIDE CONNECTORS
  // =========================================================================
  function updateBridgeConnectors() {
    const container = document.getElementById('bridge-container');
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    if (cRect.width < 50) return;
    const baseW = container.offsetWidth || 960;
    const scale = cRect.width / baseW;
    if (!scale || scale <= 0) return;

    function getCardRightCenter(id) {
      const el = document.getElementById(id);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return {
        x: (r.right - cRect.left) / scale,
        y: (r.top + r.height / 2 - cRect.top) / scale
      };
    }

    function getRowLeftCenter(id) {
      const el = document.getElementById(id);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return {
        x: (r.left - cRect.left) / scale,
        y: (r.top + r.height / 2 - cRect.top) / scale
      };
    }

    function setCurve(pathId, p1, p2) {
      const path = document.getElementById(pathId);
      if (!path || !p1 || !p2) return;
      const dx = Math.max(16, (p2.x - p1.x) * 0.45);
      path.setAttribute(
        'd',
        `M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} C ${(p1.x + dx).toFixed(1)} ${p1.y.toFixed(1)}, ${(p2.x - dx).toFixed(1)} ${p2.y.toFixed(1)}, ${(p2.x - 4).toFixed(1)} ${p2.y.toFixed(1)}`
      );
    }

    const c0 = getCardRightCenter('bridge-card-0');
    const c1 = getCardRightCenter('bridge-card-1');
    const c2 = getCardRightCenter('bridge-card-2');
    const c3 = getCardRightCenter('bridge-card-3');
    const c4 = getCardRightCenter('bridge-card-4');

    setCurve('conn-0-4', c0, getRowLeftCenter('bridge-row-4'));
    setCurve('conn-0-7', c0, getRowLeftCenter('bridge-row-7'));
    setCurve('conn-1-9', c1, getRowLeftCenter('bridge-row-9'));
    setCurve('conn-1-11', c1, getRowLeftCenter('bridge-row-11'));
    setCurve('conn-2-6', c2, getRowLeftCenter('bridge-row-6'));
    setCurve('conn-2-14', c2, getRowLeftCenter('bridge-row-14'));
    setCurve('conn-2-23', c2, getRowLeftCenter('bridge-row-23'));
    setCurve('conn-3-21', c3, getRowLeftCenter('bridge-row-21'));
    setCurve('conn-3-22', c3, getRowLeftCenter('bridge-row-22'));
    setCurve('conn-4-25', c4, getRowLeftCenter('bridge-row-25'));
    setCurve('conn-4-26', c4, getRowLeftCenter('bridge-row-26'));
  }

  function initBridgeSlide() {
    window.addEventListener('resize', updateBridgeConnectors);
    const container = document.getElementById('bridge-container');
    if (container && window.ResizeObserver) {
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.contentRect.width > 50) updateBridgeConnectors();
        }
      });
      ro.observe(container);
    }
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        const slide = document.getElementById('bridge-slide');
        if (e.currentSlide === slide) {
          setTimeout(updateBridgeConnectors, 60);
          setTimeout(updateBridgeConnectors, 250);
        }
      });
      Reveal.addEventListener('fragmentshown', updateBridgeConnectors);
      Reveal.addEventListener('fragmenthidden', updateBridgeConnectors);
      Reveal.addEventListener('ready', () => {
        setTimeout(updateBridgeConnectors, 120);
      });
    }
    setTimeout(updateBridgeConnectors, 300);
  }

  // =========================================================================
  // 1. CO-OCCURRENCE, PPMI & TRUNCATED SVD EXPLORER
  // =========================================================================
  function initPpmiSvdDemo() {
    const canvas = document.getElementById('emb-ppmi-svd-canvas');
    if (!canvas) return;

    const banner = reqEl('emb-svd-banner', 'Demo 1');
    const selMode = reqEl('emb-svd-mode', 'Demo 1');
    const sliderWin = reqEl('emb-svd-win', 'Demo 1');
    const valWin = reqEl('emb-svd-win-val', 'Demo 1');
    const chkNorm = reqEl('emb-svd-norm', 'Demo 1');
    const btnReset = reqEl('emb-svd-reset', 'Demo 1');
    const readFocus = reqEl('emb-svd-focus-val', 'Demo 1');
    const readNbr = reqEl('emb-svd-nbr-val', 'Demo 1');

    const WORDS = [
      { w: 'the',      cat: 'stop',    col: SOL.red },
      { w: 'is',       cat: 'stop',    col: SOL.magenta },
      { w: 'spinach',  cat: 'greens',  col: SOL.green },
      { w: 'chard',    cat: 'greens',  col: SOL.green },
      { w: 'garlic',   cat: 'greens',  col: SOL.green },
      { w: 'rice',     cat: 'greens',  col: SOL.green },
      { w: 'king',     cat: 'royalty', col: SOL.orange },
      { w: 'queen',    cat: 'royalty', col: SOL.orange },
      { w: 'crown',    cat: 'royalty', col: SOL.orange },
      { w: 'gpu',      cat: 'ml',      col: SOL.blue },
      { w: 'gradient', cat: 'ml',      col: SOL.blue },
      { w: 'tensor',   cat: 'ml',      col: SOL.blue }
    ];
    const V = WORDS.length;
    const IDX = Object.fromEntries(WORDS.map((d, i) => [d.w, i]));

    const CORPUS = [
      'the spinach is over the rice',
      'the chard with garlic is over rice',
      'spinach garlic and the rice is delicious',
      'the chard spinach is sauteed with garlic',
      'spinach over the rice is the best',
      'garlic spinach chard the rice is fresh',
      'the king is wearing the crown',
      'the queen is beside the king',
      'the crown is for the queen and king',
      'the king and the queen rule the crown',
      'queen wears the royal crown',
      'the gpu runs the gradient on the tensor',
      'the tensor gradient is fast on the gpu',
      'the gpu is computing the tensor gradient',
      'gradient descent on gpu updates the tensor',
      'the tensor on the gpu is huge',
      'the king tasted the garlic rice' // subtle cross-link to make rare PMI pop!
    ];

    const state = {
      mode: selMode ? selMode.value : 'ppmi',
      win: sliderWin ? parseInt(sliderWin.value, 10) : 2,
      unitNorm: chkNorm ? chkNorm.checked : true,
      focusIdx: 2 // spinach
    };

    let hitAreas = { gridOriginX: 0, gridOriginY: 0, cellSize: 28, scatterPts: [] };

    function computeMatrices(win) {
      const C = Array.from({ length: V }, () => Array(V).fill(0));
      const wordCounts = Array(V).fill(0);
      let totalPairs = 0;

      for (const sent of CORPUS) {
        const toks = sent.split(/\s+/).filter((t) => t in IDX);
        for (let i = 0; i < toks.length; i++) {
          const wi = IDX[toks[i]];
          wordCounts[wi] += 1;
          for (let d = -win; d <= win; d++) {
            if (d === 0) continue;
            const j = i + d;
            if (j >= 0 && j < toks.length) {
              const wj = IDX[toks[j]];
              C[wi][wj] += 1;
              totalPairs += 1;
            }
          }
        }
      }

      const rowSum = C.map((r) => r.reduce((a, b) => a + b, 0));
      const M = Array.from({ length: V }, () => Array(V).fill(0));

      if (state.mode === 'raw') {
        for (let i = 0; i < V; i++) {
          for (let j = 0; j < V; j++) M[i][j] = C[i][j];
        }
      } else if (state.mode === 'tfidf') {
        for (let i = 0; i < V; i++) {
          const df = C[i].filter((v) => v > 0).length;
          const idf = Math.log((V + 1) / (df + 1));
          for (let j = 0; j < V; j++) {
            M[i][j] = C[i][j] * idf;
          }
        }
        // Symmetrize softly so SVD gives clean shared word coordinates
        for (let i = 0; i < V; i++) {
          for (let j = i + 1; j < V; j++) {
            const avg = 0.5 * (M[i][j] + M[j][i]);
            M[i][j] = avg;
            M[j][i] = avg;
          }
        }
      } else if (state.mode === 'pmi' || state.mode === 'ppmi') {
        for (let i = 0; i < V; i++) {
          for (let j = 0; j < V; j++) {
            if (C[i][j] === 0) {
              M[i][j] = state.mode === 'pmi' ? -2.8 : 0;
            } else {
              const pxy = C[i][j] / Math.max(1, totalPairs);
              const px = rowSum[i] / Math.max(1, totalPairs);
              const py = rowSum[j] / Math.max(1, totalPairs);
              const pmi = Math.log2(pxy / Math.max(1e-12, px * py));
              M[i][j] = state.mode === 'ppmi' ? Math.max(0, pmi) : pmi;
            }
          }
        }
      } else {
        // Smoothed PPMI (Add-0.5 + alpha=0.75 context prior)
        const colPow = rowSum.map((s) => Math.pow(s + 1, 0.75));
        const powTot = colPow.reduce((a, b) => a + b, 0);
        const rowTot = rowSum.map((s) => s + V * 0.5);
        const totSmooth = rowTot.reduce((a, b) => a + b, 0);
        for (let i = 0; i < V; i++) {
          for (let j = 0; j < V; j++) {
            const cSmooth = C[i][j] + 0.5;
            const pxy = cSmooth / totSmooth;
            const px = rowTot[i] / totSmooth;
            const py = colPow[j] / powTot;
            const pmi = Math.log2(pxy / Math.max(1e-12, px * py));
            M[i][j] = Math.max(0, pmi);
          }
        }
      }

      // Compute top-3 symmetric eigenvectors / SVD proxy of M^T M or M
      // Using M M^T gives positive eigenvalues equal to singularValues^2 for all modes!
      const Gram = Array.from({ length: V }, () => Array(V).fill(0));
      for (let i = 0; i < V; i++) {
        for (let j = i; j < V; j++) {
          const s = dotVec(M[i], M[j]);
          Gram[i][j] = s;
          Gram[j][i] = s;
        }
      }
      const eig = symmetricEigen(Gram);
      // Canonicalize signs deterministically so clusters don't flip upside down across modes
      const u0 = eig[0].vec.slice();
      const u1 = eig[1].vec.slice();
      if (u0[IDX['gpu']] < 0) {
        for (let i = 0; i < V; i++) u0[i] = -u0[i];
      }
      if (u1[IDX['spinach']] < 0) {
        for (let i = 0; i < V; i++) u1[i] = -u1[i];
      }
      const s0 = Math.sqrt(Math.max(1e-6, eig[0].val));
      const s1 = Math.sqrt(Math.max(1e-6, eig[1].val));
      const coords = [];
      for (let i = 0; i < V; i++) {
        let x = u0[i] * Math.sqrt(s0);
        let y = u1[i] * Math.sqrt(s1);
        if (state.unitNorm) {
          const len = Math.hypot(x, y) || 1;
          x /= len;
          y /= len;
        }
        coords.push([x, y]);
      }

      return { C, M, coords, eig };
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      if (valWin) valWin.textContent = `±${state.win}`;
      const { M, coords } = computeMatrices(state.win);

      const leftBox = { x: 16, y: 12, w: 540, h: 426 };
      const rightBox = { x: 572, y: 12, w: 592, h: 426 };
      drawRoundRect(ctx, leftBox.x, leftBox.y, leftBox.w, leftBox.h, 8, SOL.base2, SOL.base1, 1);
      drawRoundRect(ctx, rightBox.x, rightBox.y, rightBox.w, rightBox.h, 8, SOL.base2, SOL.base1, 1);

      // ----- LEFT PANEL: 12×12 Co-occurrence / PPMI Matrix (1:1 square cells) -----
      const modeNames = {
        raw: 'Raw Co-Occurrence Counts',
        tfidf: 'TF-IDF Context Matrix',
        pmi: 'PMI (Unclipped)',
        ppmi: 'Positive PMI (PPMI)',
        sppmi: 'Smoothed PPMI (Add-0.5, α=0.75)'
      };
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`1. ${modeNames[state.mode]} (w = ±${state.win})`, leftBox.x + 12, leftBox.y + 22);

      let minV = 0, maxV = 1e-6;
      for (let i = 0; i < V; i++) {
        for (let j = 0; j < V; j++) {
          if (M[i][j] < minV) minV = M[i][j];
          if (M[i][j] > maxV) maxV = M[i][j];
        }
      }

      const cell = 26; // exact 1:1 square cell
      const ox = leftBox.x + 114;
      const oy = leftBox.y + 70;
      hitAreas.gridOriginX = ox;
      hitAreas.gridOriginY = oy;
      hitAreas.cellSize = cell;

      for (let i = 0; i < V; i++) {
        const isFocus = i === state.focusIdx;
        // Row labels
        ctx.fillStyle = isFocus ? SOL.orange : WORDS[i].col;
        ctx.font = isFocus ? 'bold 12.5px monospace' : '12px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(WORDS[i].w, ox - 8, oy + i * cell + 17);

        // Col labels (rotated slightly at top)
        ctx.save();
        ctx.translate(ox + i * cell + 15, oy - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'left';
        ctx.fillStyle = isFocus ? SOL.orange : WORDS[i].col;
        ctx.font = isFocus ? 'bold 11.5px monospace' : '11px monospace';
        ctx.fillText(WORDS[i].w, 0, 0);
        ctx.restore();
      }

      for (let i = 0; i < V; i++) {
        for (let j = 0; j < V; j++) {
          const val = M[i][j];
          const cx = ox + j * cell;
          const cy = oy + i * cell;
          ctx.fillStyle = heatColor(val, minV, maxV);
          ctx.fillRect(cx, cy, cell, cell);
          ctx.strokeStyle = 'rgba(147,161,161,0.45)';
          ctx.lineWidth = 0.7;
          ctx.strokeRect(cx, cy, cell, cell);

          if (Math.abs(val) > 0.05) {
            const bright = (val / Math.max(1e-6, maxV)) > 0.58 || (val < -1.5);
            ctx.fillStyle = bright ? SOL.base3 : SOL.base02;
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            const txt = state.mode === 'raw' ? `${Math.round(val)}` : val.toFixed(1);
            ctx.fillText(txt, cx + cell / 2, cy + 16);
          }
        }
      }
      // Highlight focus row
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 2.2;
      ctx.strokeRect(ox - 1, oy + state.focusIdx * cell - 1, V * cell + 2, cell + 2);
      ctx.textAlign = 'left';
      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.fillText('Click any row or dot to inspect nearest neighbors in 2D SVD space.', leftBox.x + 14, leftBox.y + 416);

      // ----- RIGHT PANEL: Exact 1:1 Aspect Ratio 2D SVD Embedding Plot -----
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`2. Truncated SVD 2D Word Space (${state.unitNorm ? 'Unit Cosine Sphere' : 'Raw Weighted'})`, rightBox.x + 12, rightBox.y + 22);

      // Square plot area with exact 1:1 pxPerUnitX === pxPerUnitY
      const plotSize = 352;
      const px0 = rightBox.x + 28;
      const py0 = rightBox.y + 40;
      drawRoundRect(ctx, px0, py0, plotSize, plotSize, 6, SOL.base3, SOL.base1, 1);

      let maxR = 1.12;
      if (!state.unitNorm) {
        maxR = 0.25;
        for (const [x, y] of coords) {
          maxR = Math.max(maxR, Math.abs(x), Math.abs(y));
        }
        maxR *= 1.18;
      }
      const scale = (plotSize * 0.45) / maxR; // IDENTICAL x and y scale -> exact 1:1!
      const midX = px0 + plotSize / 2;
      const midY = py0 + plotSize / 2;

      // Grid axes
      ctx.strokeStyle = 'rgba(147,161,161,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px0 + 8, midY); ctx.lineTo(px0 + plotSize - 8, midY);
      ctx.moveTo(midX, py0 + 8); ctx.lineTo(midX, py0 + plotSize - 8);
      ctx.stroke();

      if (state.unitNorm) {
        ctx.save();
        ctx.strokeStyle = 'rgba(38,139,210,0.42)';
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(midX, midY, scale * 1.0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Rank neighbors in full row space of M (or 2D coords if unnormalized)
      const focusVec = M[state.focusIdx];
      const nbrs = [];
      for (let i = 0; i < V; i++) {
        if (i === state.focusIdx) continue;
        const sim = state.unitNorm ? cosineSim(focusVec, M[i]) : cosineSim(coords[state.focusIdx], coords[i]);
        nbrs.push({ i, w: WORDS[i].w, col: WORDS[i].col, sim });
      }
      nbrs.sort((a, b) => b.sim - a.sim);

      const fWord = WORDS[state.focusIdx];
      if (readFocus) readFocus.textContent = fWord.w;
      if (readNbr) readNbr.textContent = nbrs.slice(0, 3).map((n) => n.w).join(', ');
      if (banner) {
        if (state.mode === 'raw') {
          banner.textContent = 'Raw Counts: Stop words ("the", "is") dominate inner products and pull content words together!';
        } else if (state.mode === 'pmi') {
          banner.textContent = 'Unclipped PMI: Co-occurring pairs glow warm, while unseen pairs sink into noisy negative values (-2.8).';
        } else {
          banner.textContent = `PPMI + SVD: Stop words drop toward 0; "${fWord.w}" clusters cleanly with ${nbrs.slice(0, 3).map((n) => n.w).join(', ')}!`;
        }
      }

      // Draw neighbor rays for top-2
      const fx = midX + coords[state.focusIdx][0] * scale;
      const fy = midY - coords[state.focusIdx][1] * scale;
      for (let k = 0; k < 2; k++) {
        const ni = nbrs[k].i;
        const nx = midX + coords[ni][0] * scale;
        const ny = midY - coords[ni][1] * scale;
        ctx.save();
        ctx.strokeStyle = 'rgba(133,153,0,0.55)';
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(fx, fy); ctx.lineTo(nx, ny);
        ctx.stroke();
        ctx.restore();
      }

      // Plot points & labels (draw all dots first, pre-seed placedRects with dots)
      hitAreas.scatterPts = [];
      const scatterBox = { x: px0, y: py0, w: plotSize, h: plotSize };
      const placedRects = [];
      const dotPts = [];
      for (let i = 0; i < V; i++) {
        const px = midX + coords[i][0] * scale;
        const py = midY - coords[i][1] * scale;
        const r = i === state.focusIdx ? 7.5 : 5.5;
        hitAreas.scatterPts.push({ i, x: px, y: py });
        dotPts.push({ i, px, py, r });
        placedRects.push({ x: px - r, y: py - r, w: r * 2, h: r * 2 });

        const isFocus = i === state.focusIdx;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = WORDS[i].col;
        ctx.fill();
        ctx.strokeStyle = isFocus ? SOL.base03 : SOL.base3;
        ctx.lineWidth = isFocus ? 2.2 : 1.3;
        ctx.stroke();
      }
      // Place focused word first so it gets prime anchor placement, then remaining words
      const order = [state.focusIdx, ...Array.from({ length: V }, (_, idx) => idx).filter((idx) => idx !== state.focusIdx)];
      for (const i of order) {
        const isFocus = i === state.focusIdx;
        const { px, py } = dotPts[i];
        placeLabelNonOverlapping(
          ctx,
          WORDS[i].w,
          px,
          py,
          scatterBox,
          placedRects,
          isFocus ? 'bold 13px monospace' : 'bold 11.5px monospace',
          isFocus ? SOL.base03 : SOL.base02
        );
      }

      // Right sidebar inside rightBox: Nearest Neighbor Cosine Ranking
      const sbX = px0 + plotSize + 14;
      const sbW = rightBox.x + rightBox.w - sbX - 14;
      drawRoundRect(ctx, sbX, py0, sbW, plotSize, 6, SOL.base3, SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(`Focus: "${fWord.w}"`, sbX + 10, py0 + 22);
      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.fillText('Nearest Neighbors (Cos):', sbX + 10, py0 + 42);

      for (let k = 0; k < Math.min(7, nbrs.length); k++) {
        const nb = nbrs[k];
        const yRow = py0 + 64 + k * 38;
        ctx.fillStyle = nb.col;
        ctx.font = 'bold 12px monospace';
        ctx.fillText(`${k + 1}. ${nb.w}`, sbX + 10, yRow);
        ctx.fillStyle = SOL.base02;
        ctx.font = '11.5px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(nb.sim.toFixed(2), sbX + sbW - 10, yRow);
        ctx.textAlign = 'left';

        // Bar
        const barW = sbW - 20;
        drawRoundRect(ctx, sbX + 10, yRow + 6, barW, 8, 3, SOL.base2, null);
        const fillW = Math.max(2, Math.round(Math.max(0, nb.sim) * barW));
        drawRoundRect(ctx, sbX + 10, yRow + 6, fillW, 8, 3, nb.col, null);
      }

      // Legend bottom
      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.fillText('Aspect Ratio: exact 1:1 (circles & cosine angles preserved)', px0, rightBox.y + 416);
    }

    canvas.addEventListener('click', (e) => {
      const pt = canvasPoint(canvas, e.clientX, e.clientY);
      // Check matrix click
      const gx = pt.x - hitAreas.gridOriginX;
      const gy = pt.y - hitAreas.gridOriginY;
      if (gy >= 0 && gy < V * hitAreas.cellSize && gx >= -95 && gx < V * hitAreas.cellSize) {
        const r = Math.floor(gy / hitAreas.cellSize);
        if (r >= 0 && r < V) {
          state.focusIdx = r;
          render();
          return;
        }
      }
      // Check scatter click
      let best = -1, bestD = 26;
      for (const sp of hitAreas.scatterPts) {
        const d = Math.hypot(pt.x - sp.x, pt.y - sp.y);
        if (d < bestD) {
          bestD = d;
          best = sp.i;
        }
      }
      if (best >= 0) {
        state.focusIdx = best;
        render();
      }
    });
    stopSwipePropagation(canvas);

    if (selMode) {
      stopSwipePropagation(selMode);
      selMode.addEventListener('change', () => { state.mode = selMode.value; render(); });
    }
    bindRange(sliderWin, () => { state.win = parseInt(sliderWin.value, 10); render(); });
    if (chkNorm) {
      stopSwipePropagation(chkNorm);
      chkNorm.addEventListener('change', () => { state.unitNorm = chkNorm.checked; render(); });
    }
    if (btnReset) {
      stopSwipePropagation(btnReset);
      btnReset.addEventListener('click', () => {
        state.mode = 'ppmi';
        state.win = 2;
        state.unitNorm = true;
        state.focusIdx = 2;
        if (selMode) selMode.value = 'ppmi';
        if (sliderWin) sliderWin.value = '2';
        if (chkNorm) chkNorm.checked = true;
        render();
      });
    }

    registerSlideListener(canvas, render);
    render();
  }

  function rectsOverlap(a, b, pad = 2) {
    return !(
      a.x + a.w + pad <= b.x ||
      b.x + b.w + pad <= a.x ||
      a.y + a.h + pad <= b.y ||
      b.y + b.h + pad <= a.y
    );
  }

  function placeLabelNonOverlapping(ctx, text, px, py, box, placedRects, font, color) {
    ctx.save();
    ctx.font = font;
    const tw = ctx.measureText(text).width;
    const th = 13;
    const padX = 5;
    const padY = 5;
    const minX = box.x + padX;
    const maxX = box.x + box.w - tw - padX;
    const minY = box.y + padY + th;
    const maxY = box.y + box.h - padY;

    const offsets = [
      [8, -7],
      [8, 13],
      [-tw - 8, -7],
      [-tw - 8, 13],
      [-tw / 2, -12],
      [-tw / 2, 18],
      [12, -16],
      [-tw - 12, -16],
      [12, 22],
      [-tw - 12, 22]
    ];

    let chosen = null;
    for (const [dx, dy] of offsets) {
      const lx = Math.min(maxX, Math.max(minX, px + dx));
      const ly = Math.min(maxY, Math.max(minY, py + dy));
      const rect = { x: lx, y: ly - th + 1, w: tw, h: th };
      let hits = false;
      for (const prev of placedRects) {
        if (rectsOverlap(rect, prev, 2)) {
          hits = true;
          break;
        }
      }
      if (!hits) {
        chosen = { lx, ly, rect };
        break;
      }
    }

    if (!chosen) {
      let bestScore = Infinity;
      for (const [dx, dy] of offsets) {
        const lx = Math.min(maxX, Math.max(minX, px + dx));
        const ly = Math.min(maxY, Math.max(minY, py + dy));
        const rect = { x: lx, y: ly - th + 1, w: tw, h: th };
        let area = 0;
        for (const prev of placedRects) {
          const ox = Math.max(0, Math.min(rect.x + rect.w, prev.x + prev.w) - Math.max(rect.x, prev.x));
          const oy = Math.max(0, Math.min(rect.y + rect.h, prev.y + prev.h) - Math.max(rect.y, prev.y));
          area += ox * oy;
        }
        if (area < bestScore) {
          bestScore = area;
          chosen = { lx, ly, rect };
        }
      }
    }

    placedRects.push(chosen.rect);
    ctx.fillStyle = color;
    ctx.textAlign = 'left';
    ctx.fillText(text, chosen.lx, chosen.ly);
    ctx.restore();
    return chosen.rect;
  }

  // =========================================================================
  // 2. LIVE SKIP-GRAM WITH NEGATIVE SAMPLING TRAINER ("apricot jam")
  // =========================================================================
  function initSgnsDemo() {
    const canvas = document.getElementById('emb-sgns-canvas');
    if (!canvas) return;

    const banner = reqEl('emb-sgns-banner', 'Demo 2');
    const btnPlay = reqEl('emb-sgns-play', 'Demo 2');
    const btnStep = reqEl('emb-sgns-step', 'Demo 2');
    const btnBurst = reqEl('emb-sgns-burst', 'Demo 2');
    const btnReset = reqEl('emb-sgns-reset', 'Demo 2');
    const sliderK = reqEl('emb-sgns-k', 'Demo 2');
    const valK = reqEl('emb-sgns-k-val', 'Demo 2');
    const sliderLr = reqEl('emb-sgns-lr', 'Demo 2');
    const valLr = reqEl('emb-sgns-lr-val', 'Demo 2');
    const readStep = reqEl('emb-sgns-step-val', 'Demo 2');
    const readLoss = reqEl('emb-sgns-loss-val', 'Demo 2');

    const VOCAB = [
      { w: 'apricot',  col: SOL.orange, group: 'fruit' },
      { w: 'jam',      col: SOL.orange, group: 'fruit' },
      { w: 'teaspoon', col: SOL.orange, group: 'fruit' },
      { w: 'peach',    col: SOL.orange, group: 'fruit' },
      { w: 'sweet',    col: SOL.orange, group: 'fruit' },
      { w: 'garlic',   col: SOL.green,  group: 'savory' },
      { w: 'pizza',    col: SOL.green,  group: 'savory' },
      { w: 'cheese',   col: SOL.green,  group: 'savory' },
      { w: 'freeze',   col: SOL.blue,   group: 'cold' },
      { w: 'ice',      col: SOL.blue,   group: 'cold' },
      { w: 'snow',     col: SOL.blue,   group: 'cold' },
      { w: 'a',        col: SOL.base01, group: 'stop' },
      { w: 'of',       col: SOL.base01, group: 'stop' }
    ];
    const V = VOCAB.length;
    const IDX = Object.fromEntries(VOCAB.map((d, i) => [d.w, i]));

    const SENTENCES = [
      ['a', 'teaspoon', 'of', 'apricot', 'jam'],
      ['sweet', 'apricot', 'jam', 'of', 'peach'],
      ['peach', 'jam', 'a', 'teaspoon', 'sweet'],
      ['garlic', 'pizza', 'cheese'],
      ['cheese', 'pizza', 'garlic'],
      ['ice', 'snow', 'freeze'],
      ['freeze', 'ice', 'snow']
    ];

    // Build all sliding-window positive pairs (win = ±2)
    const PAIRS = [];
    SENTENCES.forEach((sent, sIdx) => {
      for (let i = 0; i < sent.length; i++) {
        for (let d = -2; d <= 2; d++) {
          if (d === 0) continue;
          const j = i + d;
          if (j >= 0 && j < sent.length) {
            PAIRS.push({ sIdx, posT: i, posC: j, t: IDX[sent[i]], c: IDX[sent[j]] });
          }
        }
      }
    });

    // Compute unigram noise distribution P_0.75
    const counts = Array(V).fill(0);
    SENTENCES.forEach((s) => s.forEach((w) => { counts[IDX[w]] += 1; }));
    const powCounts = counts.map((c) => Math.pow(c, 0.75));
    const powSum = powCounts.reduce((a, b) => a + b, 0);
    const cdf = [];
    let acc = 0;
    for (let i = 0; i < V; i++) {
      acc += powCounts[i] / powSum;
      cdf.push(acc);
    }

    let rngState = 1337;
    function rand() {
      rngState = (rngState * 1664525 + 1013904223) >>> 0;
      return rngState / 4294967296;
    }

    function sampleNoise(excludeA, excludeB) {
      for (let tries = 0; tries < 20; tries++) {
        const u = rand();
        for (let i = 0; i < V; i++) {
          if (u <= cdf[i] && i !== excludeA && i !== excludeB) return i;
        }
      }
      return (excludeA + 3) % V;
    }

    const state = {
      K: sliderK ? parseInt(sliderK.value, 10) : 3,
      lr: sliderLr ? parseFloat(sliderLr.value) : 0.20,
      step: 0,
      pairCursor: 3, // starts right on "apricot" -> "jam" in "a teaspoon of apricot jam"!
      playing: false,
      timer: null,
      We: [],
      Wd: [],
      lastEvent: null,
      lossHistory: [],
      emaLoss: null
    };

    function resetModel() {
      rngState = 202611;
      state.step = 0;
      // Place cursor on center='apricot' (pos 3), ctx='jam' (pos 4) in sentence 0
      state.pairCursor = PAIRS.findIndex((p) => p.sIdx === 0 && p.t === IDX['apricot'] && p.c === IDX['jam']);
      if (state.pairCursor < 0) state.pairCursor = 0;

      state.We = [];
      state.Wd = [];
      for (let i = 0; i < V; i++) {
        // Scramble initial positions around the ring so semantic clusters do not start grouped
        const ang = (((i * 5 + 2) % V) / V) * Math.PI * 2 + 0.25;
        const r = 0.55 + 0.45 * rand();
        state.We.push([r * Math.cos(ang), r * Math.sin(ang)]);
        state.Wd.push([(rand() - 0.5) * 0.4, (rand() - 0.5) * 0.4]);
      }
      state.lossHistory = [];
      state.emaLoss = null;
      // Pre-build preview event without mutating weights so step 0 shows what's about to update
      const pair = PAIRS[state.pairCursor];
      const negs = [];
      for (let k = 0; k < state.K; k++) negs.push(sampleNoise(pair.t, pair.c));
      const posScore = sigmoid(dotVec(state.We[pair.t], state.Wd[pair.c]));
      const negScores = negs.map((n) => sigmoid(-dotVec(state.We[pair.t], state.Wd[n])));
      state.lastEvent = { pair, negs, posScore, negScores };
    }

    function stepOnce() {
      const pair = PAIRS[state.pairCursor];
      const t = pair.t;
      const c = pair.c;
      const negs = [];
      for (let k = 0; k < state.K; k++) {
        negs.push(sampleNoise(t, c));
      }

      const vt = state.We[t].slice();
      const uc = state.Wd[c].slice();
      const posP = sigmoid(dotVec(vt, uc));
      const posLoss = -Math.log(Math.max(1e-7, posP));
      let negLoss = 0;
      const negProbs = [];

      // Accumulate gradient on v_t
      const gradVt = [(posP - 1) * uc[0], (posP - 1) * uc[1]];
      // Update u_c
      state.Wd[c][0] -= state.lr * (posP - 1) * vt[0];
      state.Wd[c][1] -= state.lr * (posP - 1) * vt[1];

      for (const n of negs) {
        const un = state.Wd[n].slice();
        const sigPos = sigmoid(dotVec(vt, un)); // 1 - sigma(-vt·un)
        const sigNeg = 1 - sigPos;
        negProbs.push(sigNeg);
        negLoss += -Math.log(Math.max(1e-7, sigNeg));
        gradVt[0] += sigPos * un[0];
        gradVt[1] += sigPos * un[1];
        state.Wd[n][0] -= state.lr * sigPos * vt[0];
        state.Wd[n][1] -= state.lr * sigPos * vt[1];
      }

      // Small weight decay when K>0 keeps vectors bounded cleanly; when K=0 let them collapse!
      const decay = state.K > 0 ? 0.002 : 0.0;
      state.We[t][0] -= state.lr * (gradVt[0] + decay * vt[0]);
      state.We[t][1] -= state.lr * (gradVt[1] + decay * vt[1]);

      // Clamp extreme radius for display stability while preserving collapse direction when K=0
      const clampNorm = (vec, maxR) => {
        const n = Math.hypot(vec[0], vec[1]);
        if (n > maxR) {
          vec[0] = (vec[0] / n) * maxR;
          vec[1] = (vec[1] / n) * maxR;
        }
      };
      clampNorm(state.We[t], 2.75);
      clampNorm(state.Wd[c], 2.75);
      for (const n of negs) clampNorm(state.Wd[n], 2.75);

      const totalLoss = posLoss + negLoss;
      state.emaLoss = state.emaLoss === null ? totalLoss : (0.90 * state.emaLoss + 0.10 * totalLoss);
      state.lossHistory.push({ raw: totalLoss, ema: state.emaLoss });
      if (state.lossHistory.length > 160) state.lossHistory.shift();

      state.step += 1;
      state.lastEvent = {
        pair,
        negs,
        posScore: posP,
        negScores: negProbs,
        loss: totalLoss
      };
      state.pairCursor = (state.pairCursor + 1) % PAIRS.length;
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      if (valK) valK.textContent = `${state.K}`;
      if (valLr) valLr.textContent = state.lr.toFixed(2);
      if (readStep) readStep.textContent = `${state.step}`;
      const ev = state.lastEvent;
      if (readLoss && ev) {
        const curLoss = ev.loss !== undefined ? ev.loss : (-Math.log(Math.max(1e-6, ev.posScore)) + ev.negScores.reduce((a, b) => a - Math.log(Math.max(1e-6, b)), 0));
        readLoss.textContent = curLoss.toFixed(2);
      }

      if (banner && ev) {
        const tw = VOCAB[ev.pair.t].w;
        const cw = VOCAB[ev.pair.c].w;
        const negStr = ev.negs.length ? ev.negs.map((n) => VOCAB[n].w).join(', ') : 'NONE (vectors collapse!)';
        banner.textContent = `Target "${tw}"  |  +Positive: "${cw}" (σ=${ev.posScore.toFixed(2)})  |  −Negatives: [${negStr}]`;
      }

      const leftBox = { x: 16, y: 12, w: 530, h: 426 };
      const rightBox = { x: 562, y: 12, w: 602, h: 426 };
      drawRoundRect(ctx, leftBox.x, leftBox.y, leftBox.w, leftBox.h, 8, SOL.base2, SOL.base1, 1);
      drawRoundRect(ctx, rightBox.x, rightBox.y, rightBox.w, rightBox.h, 8, SOL.base2, SOL.base1, 1);

      // ----- LEFT PANEL: Active Corpus Window + Logistic Sigmoid Bars + Loss Curve -----
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('1. Active Sliding Window & Logistic Pairs', leftBox.x + 12, leftBox.y + 22);

      if (ev) {
        const sent = SENTENCES[ev.pair.sIdx];
        const chipY = leftBox.y + 42;
        let cx = leftBox.x + 16;
        sent.forEach((tok, idx) => {
          const isT = idx === ev.pair.posT;
          const isC = idx === ev.pair.posC;
          ctx.font = 'bold 14px monospace';
          const tw = ctx.measureText(tok).width + 18;
          const fill = isT ? SOL.orange : (isC ? SOL.green : SOL.base3);
          const stroke = isT ? SOL.orange : (isC ? SOL.green : SOL.base1);
          drawRoundRect(ctx, cx, chipY, tw, 30, 5, fill, stroke, 1.5);
          ctx.fillStyle = (isT || isC) ? SOL.base3 : SOL.base02;
          ctx.fillText(tok, cx + 9, chipY + 20);
          cx += tw + 8;
        });

        // Positive + Negative Sigmoid Bars
        const boxY = leftBox.y + 92;
        drawRoundRect(ctx, leftBox.x + 14, boxY, leftBox.w - 28, 198, 6, SOL.base3, SOL.base1, 1);

        // Positive row
        const twName = VOCAB[ev.pair.t].w;
        const cwName = VOCAB[ev.pair.c].w;
        ctx.fillStyle = SOL.green;
        ctx.font = 'bold 13px monospace';
        ctx.fillText(`+ (${twName}, ${cwName})`, leftBox.x + 26, boxY + 24);
        ctx.fillStyle = SOL.base02;
        ctx.fillText(`σ(v·u) = ${ev.posScore.toFixed(2)}`, leftBox.x + 240, boxY + 24);
        drawRoundRect(ctx, leftBox.x + 360, boxY + 13, 146, 12, 3, SOL.base2, null);
        drawRoundRect(ctx, leftBox.x + 360, boxY + 13, Math.round(146 * ev.posScore), 12, 3, SOL.green, null);

        if (ev.negs.length === 0) {
          ctx.fillStyle = SOL.red;
          ctx.font = 'bold 13px sans-serif';
          ctx.fillText('K = 0: No negative repulsion → all embeddings collapse together!', leftBox.x + 26, boxY + 70);
        } else {
          for (let k = 0; k < Math.min(6, ev.negs.length); k++) {
            const ry = boxY + 52 + k * 24;
            const nw = VOCAB[ev.negs[k]].w;
            const np = ev.negScores[k];
            ctx.fillStyle = SOL.red;
            ctx.font = 'bold 12px monospace';
            ctx.fillText(`− (${twName}, ${nw})`, leftBox.x + 26, ry);
            ctx.fillStyle = SOL.base01;
            ctx.fillText(`σ(-v·u) = ${np.toFixed(2)}`, leftBox.x + 240, ry);
            drawRoundRect(ctx, leftBox.x + 360, ry - 10, 146, 10, 3, SOL.base2, null);
            drawRoundRect(ctx, leftBox.x + 360, ry - 10, Math.round(146 * np), 10, 3, SOL.red, null);
          }
        }

        // Loss Sparkline
        const ly0 = leftBox.y + 306;
        const lh = 114;
        const lw = leftBox.w - 28;
        drawRoundRect(ctx, leftBox.x + 14, ly0, lw, lh, 6, SOL.base3, SOL.base1, 1);
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 12.5px sans-serif';
        ctx.fillText('SGNS Batch Loss Trajectory (−log σ(+ ) − Σ log σ(−))', leftBox.x + 24, ly0 + 18);

        if (state.lossHistory.length > 1) {
          const maxL = Math.max(4.0, ...state.lossHistory.map((d) => d.raw));
          // Faint raw per-step trace
          ctx.strokeStyle = 'rgba(203,75,22,0.28)';
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          state.lossHistory.forEach((d, idx) => {
            const x = leftBox.x + 26 + (idx / Math.max(1, state.lossHistory.length - 1)) * (lw - 24);
            const y = ly0 + lh - 12 - (d.raw / maxL) * (lh - 38);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.stroke();

          // Smooth EMA trendline
          ctx.strokeStyle = SOL.orange;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          state.lossHistory.forEach((d, idx) => {
            const x = leftBox.x + 26 + (idx / Math.max(1, state.lossHistory.length - 1)) * (lw - 24);
            const y = ly0 + lh - 12 - (d.ema / maxL) * (lh - 38);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.stroke();
        } else {
          ctx.fillStyle = SOL.base01;
          ctx.font = '12px sans-serif';
          ctx.fillText('Press Play or Train 100 Steps to observe loss & clustering...', leftBox.x + 24, ly0 + 66);
        }
      }

      // ----- RIGHT PANEL: Exact 1:1 2D Embedding Space (W_e vectors) -----
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('2. Live 2D Embedding Space Wₑ (Exact 1:1 Aspect Ratio)', rightBox.x + 12, rightBox.y + 22);

      const plotBox = { x: rightBox.x + 16, y: rightBox.y + 36, w: rightBox.w - 32, h: 356 };
      drawRoundRect(ctx, plotBox.x, plotBox.y, plotBox.w, plotBox.h, 6, SOL.base3, SOL.base1, 1);

      // Find symmetric data range & lock pxPerUnitX === pxPerUnitY
      let maxCoord = 1.6;
      for (let i = 0; i < V; i++) {
        maxCoord = Math.max(maxCoord, Math.abs(state.We[i][0]) + 0.35, Math.abs(state.We[i][1]) + 0.35);
      }
      const scale = Math.min(plotBox.w, plotBox.h) * 0.45 / maxCoord; // EXACT 1:1!
      const cx0 = plotBox.x + plotBox.w / 2;
      const cy0 = plotBox.y + plotBox.h / 2;

      const toPx = (v) => [cx0 + v[0] * scale, cy0 - v[1] * scale];

      // Subtle origin axes & reference unit circle
      ctx.strokeStyle = 'rgba(147,161,161,0.40)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotBox.x + 8, cy0); ctx.lineTo(plotBox.x + plotBox.w - 8, cy0);
      ctx.moveTo(cx0, plotBox.y + 8); ctx.lineTo(cx0, plotBox.y + plotBox.h - 8);
      ctx.stroke();

      ctx.save();
      ctx.strokeStyle = 'rgba(147,161,161,0.35)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(cx0, cy0, scale * 1.0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Draw gradient arrows from target to positive & negative words
      if (ev) {
        const [tx, ty] = toPx(state.We[ev.pair.t]);
        const [px, py] = toPx(state.We[ev.pair.c]);
        drawArrow(ctx, tx, ty, px, py, SOL.green, 2.4, 8, false);
        for (const n of ev.negs) {
          const [nx, ny] = toPx(state.We[n]);
          drawArrow(ctx, tx, ty, nx, ny, SOL.red, 1.5, 7, true);
        }
      }

      // Draw all word vectors (dots first, pre-seed placedRects with dots, then non-overlapping labels)
      const placedRects = [];
      const dotList = [];
      for (let i = 0; i < V; i++) {
        const [px, py] = toPx(state.We[i]);
        const isT = ev && i === ev.pair.t;
        const isC = ev && i === ev.pair.c;
        const isN = ev && ev.negs.includes(i);
        const r = isT ? 8.5 : (isC ? 7.5 : 6);
        dotList.push({ i, px, py, isT, isC });
        placedRects.push({ x: px - r, y: py - r, w: r * 2, h: r * 2 });

        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = VOCAB[i].col;
        ctx.fill();
        ctx.strokeStyle = isT ? SOL.base03 : (isC ? SOL.green : (isN ? SOL.red : SOL.base3));
        ctx.lineWidth = (isT || isC || isN) ? 2.4 : 1.3;
        ctx.stroke();
      }
      dotList.sort((a, b) => (b.isT ? 2 : (b.isC ? 1 : 0)) - (a.isT ? 2 : (a.isC ? 1 : 0)));
      for (const { i, px, py, isT, isC } of dotList) {
        placeLabelNonOverlapping(
          ctx,
          VOCAB[i].w,
          px,
          py,
          plotBox,
          placedRects,
          (isT || isC) ? 'bold 13px monospace' : 'bold 11.5px monospace',
          isT ? SOL.orange : SOL.base02
        );
      }

      // Legend
      ctx.font = '11.5px sans-serif';
      ctx.fillStyle = SOL.orange; ctx.fillText('● Fruit/Jam', rightBox.x + 18, rightBox.y + 414);
      ctx.fillStyle = SOL.green;  ctx.fillText('● Savory', rightBox.x + 118, rightBox.y + 414);
      ctx.fillStyle = SOL.blue;   ctx.fillText('● Cold', rightBox.x + 195, rightBox.y + 414);
      ctx.fillStyle = SOL.base01; ctx.fillText('— Solid Green: +Pull   -- Dashed Red: −Push', rightBox.x + 265, rightBox.y + 414);
    }

    function stopPlay() {
      state.playing = false;
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
      if (btnPlay) {
        btnPlay.textContent = 'Play';
        btnPlay.style.background = SOL.orange;
      }
    }

    if (btnPlay) {
      stopSwipePropagation(btnPlay);
      btnPlay.addEventListener('click', () => {
        if (state.playing) {
          stopPlay();
        } else {
          state.playing = true;
          btnPlay.textContent = 'Pause';
          btnPlay.style.background = SOL.red;
          state.timer = setInterval(() => {
            stepOnce();
            render();
          }, 120);
        }
      });
    }
    if (btnStep) {
      stopSwipePropagation(btnStep);
      btnStep.addEventListener('click', () => {
        stopPlay();
        stepOnce();
        render();
      });
    }
    if (btnBurst) {
      stopSwipePropagation(btnBurst);
      btnBurst.addEventListener('click', () => {
        stopPlay();
        for (let i = 0; i < 100; i++) stepOnce();
        render();
      });
    }
    if (btnReset) {
      stopSwipePropagation(btnReset);
      btnReset.addEventListener('click', () => {
        stopPlay();
        resetModel();
        render();
      });
    }
    bindRange(sliderK, () => { state.K = parseInt(sliderK.value, 10); render(); });
    bindRange(sliderLr, () => { state.lr = parseFloat(sliderLr.value); render(); });

    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && !e.currentSlide.contains(canvas)) stopPlay();
      });
    }

    resetModel();
    registerSlideListener(canvas, render);
    render();
  }

  // =========================================================================
  // 3. SEMANTIC ANALOGIES, DIACHRONIC DRIFT & DAT CREATIVITY EXPLORER
  // =========================================================================
  function initAnalogyDemo() {
    const canvas = document.getElementById('emb-analogy-canvas');
    if (!canvas) return;

    const banner = reqEl('emb-analogy-banner', 'Demo 3');
    const selMode = reqEl('emb-an-mode', 'Demo 3');
    const selPreset = reqEl('emb-an-preset', 'Demo 3');
    const wrapPreset = reqEl('emb-an-preset-wrap', 'Demo 3');
    const wrapEra = reqEl('emb-an-era-wrap', 'Demo 3');
    const sliderEra = reqEl('emb-an-era', 'Demo 3');
    const valEra = reqEl('emb-an-era-val', 'Demo 3');
    const btnAction = reqEl('emb-an-action', 'Demo 3');
    const btnReset = reqEl('emb-an-reset', 'Demo 3');
    const readRes = reqEl('emb-an-res-val', 'Demo 3');

    // Curated 2D (display 1:1) + 6D semantic embeddings for clean analogies & DAT
    const WORDS_AN = [
      // [word, x2d, y2d, extraDims(4), color]
      { w: 'man',      p: [-1.15, -0.55], e: [0.0,  0.0,  0.0, 0.2], col: SOL.blue },
      { w: 'woman',    p: [-0.60,  0.42], e: [0.0,  0.0,  0.0, 0.2], col: SOL.blue },
      { w: 'uncle',    p: [-0.45, -0.78], e: [0.2,  0.0,  0.0, 0.2], col: SOL.blue },
      { w: 'aunt',     p: [ 0.10,  0.19], e: [0.2,  0.0,  0.0, 0.2], col: SOL.blue },
      { w: 'king',     p: [ 0.35, -1.00], e: [0.8,  0.0,  0.0, 0.2], col: SOL.orange },
      { w: 'queen',    p: [ 0.90, -0.03], e: [0.8,  0.0,  0.0, 0.2], col: SOL.orange },
      { w: 'kings',    p: [ 0.10, -0.25], e: [0.8,  0.9,  0.0, 0.2], col: SOL.orange },
      { w: 'queens',   p: [ 0.65,  0.72], e: [0.8,  0.9,  0.0, 0.2], col: SOL.orange },

      { w: 'France',   p: [-1.25,  0.85], e: [0.0,  0.0,  1.0, 0.0], col: SOL.cyan },
      { w: 'Paris',    p: [-0.42,  1.25], e: [0.0,  0.0,  1.0, 0.5], col: SOL.cyan },
      { w: 'Italy',    p: [-1.10,  0.10], e: [0.0,  0.0,  1.0, 0.0], col: SOL.cyan },
      { w: 'Rome',     p: [-0.27,  0.50], e: [0.0,  0.0,  1.0, 0.5], col: SOL.cyan },

      { w: 'slow',     p: [ 0.55,  0.95], e: [0.0,  0.8, -0.8, 0.0], col: SOL.violet },
      { w: 'slower',   p: [ 1.05,  1.15], e: [0.0,  0.8, -0.8, 0.0], col: SOL.violet },
      { w: 'strong',   p: [ 0.65,  0.35], e: [0.0,  0.8, -0.8, 0.0], col: SOL.violet },
      { w: 'stronger', p: [ 1.15,  0.55], e: [0.0,  0.8, -0.8, 0.0], col: SOL.violet },

      { w: 'apricot',  p: [-0.70, -1.10], e: [-0.9, -0.5, 0.0, 0.8], col: SOL.green },
      { w: 'sweet',    p: [-0.15, -1.25], e: [-0.9, -0.5, 0.0, 0.8], col: SOL.green },
      { w: 'spinach',  p: [ 0.45, -1.35], e: [-0.9, -0.5, 0.0, 0.8], col: SOL.green },
      { w: 'salty',    p: [ 1.00, -1.50], e: [-0.9, -0.5, 0.0, 0.8], col: SOL.green }
    ];
    const AN_IDX = Object.fromEntries(WORDS_AN.map((d, i) => [d.w, i]));

    const PRESETS = {
      king:     { a: 'man',    b: 'woman',  c: 'king',    ans: 'queen' },
      capitals: { a: 'France', b: 'Paris',  c: 'Italy',   ans: 'Rome' },
      degree:   { a: 'slow',   b: 'slower', c: 'strong',  ans: 'stronger' },
      plural:   { a: 'king',   b: 'kings',  c: 'queen',   ans: 'queens' },
      culinary: { a: 'apricot',b: 'sweet',  c: 'spinach', ans: 'salty' }
    };

    // Diachronic Drift data (Procrustes aligned across 1850s, 1900s, 1990s)
    const DRIFT_ANCHORS = [
      { w: 'seed',       p: [-1.92,  1.10], era: 0, col: SOL.green },
      { w: 'scatter',    p: [-0.92,  1.26], era: 0, col: SOL.green },
      { w: 'newspaper',  p: [-1.95,  0.22], era: 1, col: SOL.green },
      { w: 'radio',      p: [-1.92, -1.08], era: 2, col: SOL.green },
      { w: 'television', p: [-0.82, -1.26], era: 2, col: SOL.green },

      { w: 'cheerful',   p: [ 1.38,  1.16], era: 0, col: SOL.blue },
      { w: 'witty',      p: [ 1.42,  0.28], era: 1, col: SOL.blue },
      { w: 'identity',   p: [ 1.46, -1.04], era: 2, col: SOL.blue },

      { w: 'majestic',   p: [ 0.14,  1.18], era: 0, col: SOL.orange },
      { w: 'horrible',   p: [ 0.12, -0.32], era: 1, col: SOL.orange },
      { w: 'weird',      p: [ 0.30, -1.08], era: 2, col: SOL.orange }
    ];
    const DRIFT_PATHS = [
      { w: 'broadcast', col: SOL.green,  pts: [[-1.46, 0.85], [-1.38, 0.02], [-1.30, -0.88]] },
      { w: 'awful',     col: SOL.orange, pts: [[-0.12, 0.92], [-0.10,-0.10], [ 0.02, -0.78]] },
      { w: 'gay',       col: SOL.blue,   pts: [[ 1.10, 0.90], [ 1.16, 0.08], [ 1.22, -0.76]] }
    ];
    const ERAS = ['1850s', '1900s', '1990s'];

    // DAT Creativity nouns in 6D unit space (displayed via clean 2D coordinates)
    const DAT_WORDS = [
      { w: 'apricot',    p: [-1.10, -0.85], v: [ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0], col: SOL.orange },
      { w: 'spinach',    p: [-0.85, -1.05], v: [ 0.9,  0.3,  0.0,  0.0,  0.0,  0.0], col: SOL.orange },
      { w: 'garlic',     p: [-0.65, -0.75], v: [ 0.85, 0.4,  0.1,  0.0,  0.0,  0.0], col: SOL.orange },
      { w: 'theorem',    p: [ 1.05,  0.90], v: [-0.3,  1.0,  0.2,  0.0,  0.0,  0.0], col: SOL.blue },
      { w: 'quantum',    p: [ 1.15,  0.60], v: [-0.3,  0.9,  0.4,  0.0,  0.0,  0.0], col: SOL.blue },
      { w: 'microscope', p: [ 0.85,  0.50], v: [-0.2,  0.6,  0.8,  0.1,  0.0,  0.0], col: SOL.blue },
      { w: 'violin',     p: [-0.15,  1.05], v: [ 0.0, -0.3,  0.0,  1.0,  0.0,  0.0], col: SOL.magenta },
      { w: 'sonnet',     p: [-0.45,  0.90], v: [ 0.0, -0.2,  0.0,  0.9,  0.3,  0.0], col: SOL.magenta },
      { w: 'cathedral',  p: [-0.85,  0.60], v: [ 0.0,  0.0, -0.3,  0.4,  0.9,  0.0], col: SOL.violet },
      { w: 'pyramid',    p: [-1.15,  0.25], v: [ 0.0,  0.0, -0.2,  0.2,  0.95, 0.1], col: SOL.violet },
      { w: 'glacier',    p: [ 0.25, -1.05], v: [ 0.0,  0.0,  0.0, -0.3,  0.0,  1.0], col: SOL.cyan },
      { w: 'volcano',    p: [ 0.65, -0.95], v: [ 0.0,  0.0,  0.1, -0.2,  0.0,  0.95], col: SOL.cyan },
      { w: 'tornado',    p: [ 0.95, -0.60], v: [ 0.0,  0.0,  0.2,  0.0, -0.2,  0.9], col: SOL.cyan },
      { w: 'bicycle',    p: [ 0.15, -0.15], v: [ 0.0,  0.1,  0.5,  0.0,  0.3, -0.5], col: SOL.green },
      { w: 'anchor',     p: [-0.25, -0.20], v: [ 0.1, -0.1,  0.3,  0.0,  0.5,  0.4], col: SOL.green }
    ];

    const state = {
      mode: selMode ? selMode.value : 'analogy',
      presetKey: selPreset ? selPreset.value : 'king',
      eraIdx: sliderEra ? parseInt(sliderEra.value, 10) : 2,
      datSelected: [0, 3, 6, 9, 10], // apricot, theorem, violin, pyramid, glacier
      hitPoints: []
    };

    function getFullVec(d) {
      return [d.p[0], d.p[1], d.e[0], d.e[1], d.e[2], d.e[3]];
    }

    function computeDatScore(indices) {
      if (indices.length < 2) return 0;
      let sum = 0, cnt = 0;
      for (let i = 0; i < indices.length; i++) {
        for (let j = i + 1; j < indices.length; j++) {
          const d = 1 - cosineSim(DAT_WORDS[indices[i]].v, DAT_WORDS[indices[j]].v);
          sum += d;
          cnt += 1;
        }
      }
      // Scale cosine distance [0..1.4] into canonical 0..100 DAT scale (~78 typical, ~94 spread)
      return Math.min(100, (sum / cnt) * 86);
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      if (wrapPreset) wrapPreset.style.display = state.mode === 'analogy' ? 'inline-block' : 'none';
      if (wrapEra) wrapEra.style.display = state.mode === 'drift' ? 'inline-flex' : 'none';
      if (btnAction) {
        btnAction.style.display = state.mode === 'dat' ? 'inline-block' : 'none';
        btnAction.textContent = 'Greedy Optimal Set';
      }
      if (valEra) valEra.textContent = ERAS[state.eraIdx];

      const leftBox = { x: 16, y: 12, w: 690, h: 426 };
      const rightBox = { x: 722, y: 12, w: 442, h: 426 };
      drawRoundRect(ctx, leftBox.x, leftBox.y, leftBox.w, leftBox.h, 8, SOL.base2, SOL.base1, 1);
      drawRoundRect(ctx, rightBox.x, rightBox.y, rightBox.w, rightBox.h, 8, SOL.base2, SOL.base1, 1);

      const plotBox = { x: leftBox.x + 16, y: leftBox.y + 36, w: leftBox.w - 32, h: 364 };
      drawRoundRect(ctx, plotBox.x, plotBox.y, plotBox.w, plotBox.h, 6, SOL.base3, SOL.base1, 1);

      // Exact 1:1 geometric aspect ratio in left semantic plane
      const maxRange = 1.65;
      const scale = Math.min(plotBox.w, plotBox.h) * 0.46 / maxRange; // pxPerUnitX === pxPerUnitY!
      const cx0 = plotBox.x + plotBox.w / 2;
      const cy0 = plotBox.y + plotBox.h / 2;
      const toPx = (p) => [cx0 + p[0] * scale, cy0 - p[1] * scale];

      ctx.strokeStyle = 'rgba(147,161,161,0.38)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotBox.x + 8, cy0); ctx.lineTo(plotBox.x + plotBox.w - 8, cy0);
      ctx.moveTo(cx0, plotBox.y + 8); ctx.lineTo(cx0, plotBox.y + plotBox.h - 8);
      ctx.stroke();

      state.hitPoints = [];

      if (state.mode === 'analogy') {
        const pr = PRESETS[state.presetKey];
        const da = WORDS_AN[AN_IDX[pr.a]];
        const db = WORDS_AN[AN_IDX[pr.b]];
        const dc = WORDS_AN[AN_IDX[pr.c]];
        const va = getFullVec(da), vb = getFullVec(db), vc = getFullVec(dc);
        const query = va.map((_, i) => vb[i] - va[i] + vc[i]);

        const ranked = [];
        for (let i = 0; i < WORDS_AN.length; i++) {
          const w = WORDS_AN[i].w;
          if (w === pr.a || w === pr.b || w === pr.c) continue;
          ranked.push({ w, col: WORDS_AN[i].col, cos: cosineSim(getFullVec(WORDS_AN[i]), query) });
        }
        ranked.sort((a, b) => b.cos - a.cos);

        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`1. Analogy Parallelogram (1:1 Plane):  ${pr.b} − ${pr.a} + ${pr.c} ≈ ${ranked[0].w}`, leftBox.x + 14, leftBox.y + 22);

        if (readRes) readRes.textContent = `${ranked[0].w} (cos=${ranked[0].cos.toFixed(2)})`;
        if (banner) banner.textContent = `Analogy Query: "${pr.a}" is to "${pr.b}" as "${pr.c}" is to ? → Top match: "${ranked[0].w}" (cos = ${ranked[0].cos.toFixed(3)})`;

        const pa = toPx(da.p), pb = toPx(db.p), pc = toPx(dc.p);
        const pq = toPx([query[0], query[1]]);
        // Parallelogram edges
        drawArrow(ctx, pa[0], pa[1], pb[0], pb[1], SOL.blue, 2.8, 9, false);
        drawArrow(ctx, pc[0], pc[1], pq[0], pq[1], SOL.orange, 2.8, 9, true);
        ctx.save();
        ctx.strokeStyle = SOL.base1;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pc[0], pc[1]);
        ctx.moveTo(pb[0], pb[1]); ctx.lineTo(pq[0], pq[1]);
        ctx.stroke();
        ctx.restore();

        // Draw all words (dots first, pre-seed placedRects with dots, key words get label priority)
        const placedRects = [];
        const anDots = [];
        for (let i = 0; i < WORDS_AN.length; i++) {
          const d = WORDS_AN[i];
          const [px, py] = toPx(d.p);
          const isKey = d.w === pr.a || d.w === pr.b || d.w === pr.c || d.w === ranked[0].w;
          const r = isKey ? 7 : 4.5;
          anDots.push({ d, px, py, isKey, isAns: d.w === ranked[0].w });
          placedRects.push({ x: px - r, y: py - r, w: r * 2, h: r * 2 });

          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fillStyle = d.w === ranked[0].w ? SOL.orange : d.col;
          ctx.fill();
          ctx.strokeStyle = isKey ? SOL.base03 : SOL.base3;
          ctx.lineWidth = isKey ? 2 : 1;
          ctx.stroke();
        }
        anDots.sort((x, y) => (y.isKey ? 1 : 0) - (x.isKey ? 1 : 0));
        for (const { d, px, py, isKey } of anDots) {
          placeLabelNonOverlapping(
            ctx,
            d.w,
            px,
            py,
            plotBox,
            placedRects,
            isKey ? 'bold 12.5px monospace' : '11px monospace',
            isKey ? SOL.base03 : SOL.base01
          );
        }

        // Right panel: Top-5 3CosAdd Ranking
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('2. 3CosAdd Top-5 Candidates', rightBox.x + 14, rightBox.y + 24);
        drawRoundRect(ctx, rightBox.x + 14, rightBox.y + 36, rightBox.w - 28, 364, 6, SOL.base3, SOL.base1, 1);
        for (let k = 0; k < 5; k++) {
          const item = ranked[k];
          const ry = rightBox.y + 72 + k * 64;
          ctx.fillStyle = k === 0 ? SOL.orange : item.col;
          ctx.font = k === 0 ? 'bold 16px monospace' : 'bold 14px monospace';
          ctx.fillText(`${k + 1}. ${item.w}`, rightBox.x + 28, ry);
          ctx.fillStyle = SOL.base02;
          ctx.textAlign = 'right';
          ctx.fillText(`cos = ${item.cos.toFixed(3)}`, rightBox.x + rightBox.w - 28, ry);
          ctx.textAlign = 'left';

          const bw = rightBox.w - 56;
          drawRoundRect(ctx, rightBox.x + 28, ry + 10, bw, 12, 4, SOL.base2, null);
          drawRoundRect(ctx, rightBox.x + 28, ry + 10, Math.max(4, Math.round(Math.max(0, item.cos) * bw)), 12, 4, k === 0 ? SOL.orange : item.col, null);
        }
      } else if (state.mode === 'drift') {
        const eraName = ERAS[state.eraIdx];
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`1. Procrustes-Aligned Drift Space — Focusing ${eraName}`, leftBox.x + 14, leftBox.y + 22);
        if (readRes) readRes.textContent = `${eraName} neighborhoods`;
        if (banner) banner.textContent = `Diachronic Semantic Drift (${eraName}): Watch "broadcast", "awful", and "gay" migrate across aligned decade spaces!`;

        const placedRects = [];
        const driftLabels = [];

        for (const anc of DRIFT_ANCHORS) {
          const [px, py] = toPx(anc.p);
          const active = anc.era === state.eraIdx;
          const r = active ? 5.5 : 4;
          placedRects.push({ x: px - r, y: py - r, w: r * 2, h: r * 2 });
          ctx.globalAlpha = active ? 1.0 : 0.35;
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fillStyle = anc.col;
          ctx.fill();
          ctx.globalAlpha = 1.0;
          driftLabels.push({
            prio: active ? 2 : 0,
            alpha: active ? 1.0 : 0.45,
            txt: `${anc.w} (${ERAS[anc.era]})`,
            px,
            py,
            font: active ? 'bold 11.5px monospace' : '10.5px monospace',
            col: SOL.base01
          });
        }

        for (const tr of DRIFT_PATHS) {
          const p0 = toPx(tr.pts[0]);
          const p1 = toPx(tr.pts[1]);
          const p2 = toPx(tr.pts[2]);
          drawArrow(ctx, p0[0], p0[1], p1[0], p1[1], tr.col, 2.5, 8, false);
          drawArrow(ctx, p1[0], p1[1], p2[0], p2[1], tr.col, 2.5, 8, false);
          for (let e = 0; e < 3; e++) {
            const pe = toPx(tr.pts[e]);
            const isNow = e === state.eraIdx;
            const r = isNow ? 8.5 : 5;
            placedRects.push({ x: pe[0] - r, y: pe[1] - r, w: r * 2, h: r * 2 });
            ctx.beginPath();
            ctx.arc(pe[0], pe[1], r, 0, Math.PI * 2);
            ctx.fillStyle = isNow ? SOL.base3 : tr.col;
            ctx.fill();
            ctx.strokeStyle = tr.col;
            ctx.lineWidth = isNow ? 3 : 1.5;
            ctx.stroke();

            driftLabels.push({
              prio: isNow ? 3 : 1,
              alpha: 1.0,
              txt: `${tr.w} (${ERAS[e]})`,
              px: pe[0],
              py: pe[1],
              font: isNow ? 'bold 12.5px monospace' : '10.5px monospace',
              col: tr.col
            });
          }
        }

        driftLabels.sort((a, b) => b.prio - a.prio);
        for (const lbl of driftLabels) {
          ctx.save();
          ctx.globalAlpha = lbl.alpha;
          placeLabelNonOverlapping(ctx, lbl.txt, lbl.px, lbl.py, plotBox, placedRects, lbl.font, lbl.col);
          ctx.restore();
        }

        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`2. Semantic Trajectory Summary (${eraName})`, rightBox.x + 14, rightBox.y + 24);
        drawRoundRect(ctx, rightBox.x + 14, rightBox.y + 36, rightBox.w - 28, 364, 6, SOL.base3, SOL.base1, 1);
        const notes = [
          { w: 'broadcast', col: SOL.green,  txt: ['1850s: scatter seeds in fields', '1900s: circulate newspapers', '1990s: radio & television signal'], short: 'scatter seeds → newspapers → radio/TV' },
          { w: 'awful',     col: SOL.orange, txt: ['1850s: majestic, awe-inspiring', '1900s: horrible, terrible, dread', '1990s: weird, awfully (adverb)'], short: 'majestic → horrible → weird' },
          { w: 'gay',       col: SOL.blue,   txt: ['1850s: cheerful, sweet, daft', '1900s: bright, witty, pleasant', '1990s: modern identity & community'], short: 'cheerful → bright → identity' }
        ];
        notes.forEach((n, idx) => {
          const ny = rightBox.y + 68 + idx * 112;
          ctx.fillStyle = n.col;
          ctx.font = 'bold 15px monospace';
          ctx.fillText(n.w, rightBox.x + 28, ny);
          ctx.fillStyle = SOL.base02;
          ctx.font = '13px sans-serif';
          ctx.fillText(n.txt[state.eraIdx], rightBox.x + 28, ny + 26);
          ctx.fillStyle = SOL.base01;
          ctx.font = '11.5px sans-serif';
          ctx.fillText(`All eras: ${n.short}`, rightBox.x + 28, ny + 50);
        });
      } else {
        // DAT Creativity Mini-Game
        const datScore = computeDatScore(state.datSelected);
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`1. Click 2–7 Words to Maximize Semantic Spread (DAT = ${datScore.toFixed(1)})`, leftBox.x + 14, leftBox.y + 22);
        if (readRes) readRes.textContent = `DAT Score: ${datScore.toFixed(1)} / 100 (${state.datSelected.length} words)`;
        if (banner) banner.textContent = `Divergent Association Task (Olson et al., PNAS 2021): Selected ${state.datSelected.length}/7 words — DAT Score = ${datScore.toFixed(1)} (avg human ≈ 78)`;

        // Draw pairwise chords
        for (let a = 0; a < state.datSelected.length; a++) {
          for (let b = a + 1; b < state.datSelected.length; b++) {
            const ia = state.datSelected[a], ib = state.datSelected[b];
            const dist = 1 - cosineSim(DAT_WORDS[ia].v, DAT_WORDS[ib].v);
            const pa = toPx(DAT_WORDS[ia].p), pb = toPx(DAT_WORDS[ib].p);
            ctx.strokeStyle = dist > 0.85 ? 'rgba(133,153,0,0.65)' : (dist > 0.45 ? 'rgba(38,139,210,0.45)' : 'rgba(220,50,47,0.65)');
            ctx.lineWidth = dist > 0.85 ? 2.2 : 1.5;
            ctx.beginPath();
            ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
            ctx.stroke();
          }
        }

        const placedRects = [];
        const datDots = [];
        for (let i = 0; i < DAT_WORDS.length; i++) {
          const d = DAT_WORDS[i];
          const [px, py] = toPx(d.p);
          state.hitPoints.push({ i, x: px, y: py });
          const sel = state.datSelected.includes(i);
          const r = sel ? 8 : 5.5;
          datDots.push({ d, px, py, sel });
          placedRects.push({ x: px - r, y: py - r, w: r * 2, h: r * 2 });

          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fillStyle = sel ? SOL.orange : d.col;
          ctx.fill();
          ctx.strokeStyle = sel ? SOL.base03 : SOL.base3;
          ctx.lineWidth = sel ? 2.4 : 1.2;
          ctx.stroke();
        }
        datDots.sort((a, b) => (b.sel ? 1 : 0) - (a.sel ? 1 : 0));
        for (const { d, px, py, sel } of datDots) {
          placeLabelNonOverlapping(
            ctx,
            d.w,
            px,
            py,
            plotBox,
            placedRects,
            sel ? 'bold 12.5px monospace' : '11.5px monospace',
            sel ? SOL.base03 : SOL.base01
          );
        }

        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('2. Creativity Score Breakdown', rightBox.x + 14, rightBox.y + 24);
        drawRoundRect(ctx, rightBox.x + 14, rightBox.y + 36, rightBox.w - 28, 364, 6, SOL.base3, SOL.base1, 1);

        ctx.fillStyle = datScore >= 82 ? SOL.green : (datScore >= 65 ? SOL.blue : SOL.red);
        ctx.font = 'bold 42px monospace';
        ctx.fillText(datScore.toFixed(1), rightBox.x + 32, rightBox.y + 98);
        ctx.fillStyle = SOL.base01;
        ctx.font = '13px sans-serif';
        ctx.fillText('out of 100 (Human median ≈ 78)', rightBox.x + 150, rightBox.y + 92);

        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 13px sans-serif';
        ctx.fillText(`Selected Set (${state.datSelected.length}/7):`, rightBox.x + 32, rightBox.y + 138);
        ctx.fillStyle = SOL.orange;
        ctx.font = '13px monospace';
        const line1 = state.datSelected.slice(0, 4).map((i) => DAT_WORDS[i].w).join(', ');
        const line2 = state.datSelected.slice(4).map((i) => DAT_WORDS[i].w).join(', ');
        ctx.fillText(line1 || '(click words on left)', rightBox.x + 32, rightBox.y + 164);
        if (line2) ctx.fillText(line2, rightBox.x + 32, rightBox.y + 186);

        ctx.fillStyle = SOL.base01;
        ctx.font = '12.5px sans-serif';
        ctx.fillText('• Green chords: semantically distant pairs (high novelty)', rightBox.x + 32, rightBox.y + 240);
        ctx.fillText('• Red chords: clustered synonyms (low DAT score)', rightBox.x + 32, rightBox.y + 266);
        ctx.fillText('• Click any word on the left to add/remove up to 7.', rightBox.x + 32, rightBox.y + 292);
        ctx.fillText('• Click "Greedy Optimal Set" for max semantic spread!', rightBox.x + 32, rightBox.y + 318);
      }

      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.fillText('Exact 1:1 geometric aspect ratio', leftBox.x + 16, leftBox.y + 416);
    }

    canvas.addEventListener('click', (e) => {
      if (state.mode !== 'dat') return;
      const pt = canvasPoint(canvas, e.clientX, e.clientY);
      let best = -1, bestD = 24;
      for (const hp of state.hitPoints) {
        const d = Math.hypot(pt.x - hp.x, pt.y - hp.y);
        if (d < bestD) { bestD = d; best = hp.i; }
      }
      if (best >= 0) {
        const pos = state.datSelected.indexOf(best);
        if (pos >= 0) {
          if (state.datSelected.length > 2) state.datSelected.splice(pos, 1);
        } else {
          if (state.datSelected.length >= 7) state.datSelected.shift();
          state.datSelected.push(best);
        }
        render();
      }
    });
    stopSwipePropagation(canvas);

    if (selMode) {
      stopSwipePropagation(selMode);
      selMode.addEventListener('change', () => { state.mode = selMode.value; render(); });
    }
    if (selPreset) {
      stopSwipePropagation(selPreset);
      selPreset.addEventListener('change', () => { state.presetKey = selPreset.value; render(); });
    }
    bindRange(sliderEra, () => { state.eraIdx = parseInt(sliderEra.value, 10); render(); });
    if (btnAction) {
      stopSwipePropagation(btnAction);
      btnAction.addEventListener('click', () => {
        // Greedy farthest-point set of 7 unrelated nouns
        state.datSelected = [0, 3, 6, 9, 10, 13, 5]; // apricot, theorem, violin, pyramid, glacier, bicycle, microscope
        render();
      });
    }
    if (btnReset) {
      stopSwipePropagation(btnReset);
      btnReset.addEventListener('click', () => {
        state.presetKey = 'king';
        state.eraIdx = 2;
        state.datSelected = [0, 3, 6, 9, 10];
        if (selPreset) selPreset.value = 'king';
        if (sliderEra) sliderEra.value = '2';
        render();
      });
    }

    registerSlideListener(canvas, render);
    render();
  }

  // =========================================================================
  // 4. SUBWORD BPE & CONTEXTUAL POLYSEMY STUDIO
  // =========================================================================
  function initModernDemo() {
    const canvas = document.getElementById('emb-modern-canvas');
    if (!canvas) return;

    const banner = reqEl('emb-mod-banner', 'Demo 4');
    const selMode = reqEl('emb-mod-mode', 'Demo 4');
    const wrapWord = reqEl('emb-mod-word-wrap', 'Demo 4');
    const selWord = reqEl('emb-mod-word', 'Demo 4');
    const wrapBpe = reqEl('emb-mod-bpe-wrap', 'Demo 4');
    const selPhrase = reqEl('emb-mod-phrase', 'Demo 4');
    const slider = reqEl('emb-mod-slider', 'Demo 4');
    const sliderLbl = reqEl('emb-mod-slider-lbl', 'Demo 4');
    const sliderVal = reqEl('emb-mod-slider-val', 'Demo 4');
    const btnToggle = reqEl('emb-mod-toggle', 'Demo 4');
    const btnReset = reqEl('emb-mod-reset', 'Demo 4');
    const readStat = reqEl('emb-mod-stat-val', 'Demo 4');

    const POLY_DATA = {
      bank: {
        sentA: { label: 'Sentence A (Nature)', tokens: ['kayak', 'near', 'the', 'muddy', 'river', 'bank'], target: 5, ctxIdx: [0, 3, 4], weights: [0.16, 0.04, 0.02, 0.24, 0.54, 0.0], pole: [-1.10, 0.72], poleLabel: 'Nature / River', anchors: [['river', -1.20, 0.90], ['muddy', -0.95, 0.45], ['kayak', -1.35, 0.30]] },
        sentB: { label: 'Sentence B (Finance)', tokens: ['deposit', 'cash', 'at', 'the', 'central', 'bank'], target: 5, ctxIdx: [0, 1, 4], weights: [0.28, 0.52, 0.02, 0.02, 0.16, 0.0], pole: [1.10, -0.72], poleLabel: 'Finance / Money', anchors: [['cash', 1.25, -0.85], ['deposit', 0.95, -0.45], ['loan', 1.30, -0.30]] }
      },
      apple: {
        sentA: { label: 'Sentence A (Orchard)', tokens: ['bake', 'a', 'sweet', 'apple', 'pie', 'today'], target: 3, ctxIdx: [0, 2, 4], weights: [0.22, 0.03, 0.32, 0.0, 0.40, 0.03], pole: [-1.05, 0.80], poleLabel: 'Food / Fruit', anchors: [['sweet', -1.15, 0.95], ['pie', -0.90, 0.50], ['apricot', -1.35, 0.55]] },
        sentB: { label: 'Sentence B (Tech)', tokens: ['upgrade', 'the', 'new', 'apple', 'iphone', 'chip'], target: 3, ctxIdx: [0, 4, 5], weights: [0.20, 0.02, 0.08, 0.0, 0.48, 0.22], pole: [1.05, -0.78], poleLabel: 'Tech / Hardware', anchors: [['iphone', 1.20, -0.90], ['chip', 0.95, -0.45], ['laptop', 1.30, -0.40]] }
      },
      cell: {
        sentA: { label: 'Sentence A (Biology)', tokens: ['stain', 'the', 'living', 'cell', 'membrane'], target: 3, ctxIdx: [0, 2, 4], weights: [0.20, 0.03, 0.32, 0.0, 0.45], pole: [-1.05, 0.78], poleLabel: 'Biology / Tissue', anchors: [['membrane', -1.20, 0.92], ['living', -0.92, 0.45], ['protein', -1.30, 0.42]] },
        sentB: { label: 'Sentence B (Energy)', tokens: ['charge', 'the', 'lithium', 'cell', 'battery'], target: 3, ctxIdx: [0, 2, 4], weights: [0.24, 0.02, 0.34, 0.0, 0.40], pole: [1.05, -0.78], poleLabel: 'Battery / Energy', anchors: [['battery', 1.22, -0.90], ['lithium', 0.92, -0.45], ['voltage', 1.30, -0.35]] }
      }
    };

    const BPE_PHRASES = {
      apricot: {
        text: 'unbelievably sweet apricot jam',
        steps: [
          ['u','n','b','e','l','i','e','v','a','b','l','y',' ','s','w','e','e','t',' ','a','p','r','i','c','o','t',' ','j','a','m'],
          ['u','n','b','e','l','i','e','v','a','b','l','y',' ','s','w','ee','t',' ','a','p','r','i','c','o','t',' ','j','a','m'],
          ['u','n','b','e','l','i','e','v','ab','l','y',' ','s','w','ee','t',' ','a','p','r','i','c','o','t',' ','j','a','m'],
          ['un','b','e','l','i','e','v','ab','l','y',' ','s','w','ee','t',' ','a','p','r','i','c','o','t',' ','j','a','m'],
          ['un','b','e','l','i','e','v','ably',' ','s','w','ee','t',' ','a','p','r','i','c','o','t',' ','j','a','m'],
          ['un','be','l','ie','v','ably',' ','sw','ee','t',' ','ap','r','i','c','o','t',' ','j','am'],
          ['un','believ','ably',' ','sweet',' ','ap','ric','ot',' ','jam'],
          ['un','believ','ably','_sweet','_ap','ricot','_jam'],
          ['un','believ','ably','_sweet','_apricot','_jam']
        ],
        merges: ['(start: raw characters)', 'e + e → ee', 'a + b → ab', 'u + n → un', 'ab + ly → ably', 's+w, i+e, a+p, a+m', 'believ, sweet, jam', 'merge leading spaces (_)', 'ap + ricot → apricot']
      },
      morph: {
        text: 'playing replay player plays',
        steps: [
          ['p','l','a','y','i','n','g',' ','r','e','p','l','a','y',' ','p','l','a','y','e','r',' ','p','l','a','y','s'],
          ['pl','a','y','i','n','g',' ','r','e','pl','a','y',' ','pl','a','y','e','r',' ','pl','a','y','s'],
          ['pla','y','i','n','g',' ','r','e','pla','y',' ','pla','y','e','r',' ','pla','y','s'],
          ['play','i','n','g',' ','r','e','play',' ','play','e','r',' ','play','s'],
          ['play','ing',' ','re','play',' ','play','er',' ','play','s'],
          ['play','ing','_re','play','_play','er','_play','s'],
          ['playing','_re','play','_player','_plays']
        ],
        merges: ['(start: raw characters)', 'p + l → pl (4×)', 'pl + a → pla (4×)', 'pla + y → play (4×)', 'i+ng, r+e, e+r', 'merge spaces (_)', 'shared stem "play" across all 4!']
      },
      bio: {
        text: 'neuroimaging tokenization',
        steps: [
          ['n','e','u','r','o','i','m','a','g','i','n','g',' ','t','o','k','e','n','i','z','a','t','i','o','n'],
          ['n','e','u','r','o','i','m','a','g','in','g',' ','t','o','k','e','n','i','z','a','ti','o','n'],
          ['n','eu','r','o','im','ag','ing',' ','t','ok','en','iz','a','tion'],
          ['neuro','im','ag','ing',' ','token','iz','ation'],
          ['neuro','imag','ing','_token','iz','ation'],
          ['neuro','imaging','_token','ization']
        ],
        merges: ['(start: raw characters)', 'i + n → in, t + i → ti', 'e+u, a+t+ion → ation', 'n+eu+ro → neuro, tok+en → token', 'imag+ing, _+token', '2 rare words → 4 known subwords, 0 <UNK>!']
      }
    };

    const state = {
      mode: selMode ? selMode.value : 'polysemy',
      wordKey: selWord ? selWord.value : 'bank',
      phraseKey: selPhrase ? selPhrase.value : 'apricot',
      stepVal: slider ? parseInt(slider.value, 10) : 3,
      compareBoth: true
    };

    function syncControls() {
      const isPoly = state.mode === 'polysemy';
      if (wrapWord) wrapWord.style.display = isPoly ? 'inline-block' : 'none';
      if (wrapBpe) wrapBpe.style.display = isPoly ? 'none' : 'inline-block';
      if (btnToggle) btnToggle.style.display = isPoly ? 'inline-block' : 'none';
      if (slider) {
        if (isPoly) {
          slider.max = '4';
          if (state.stepVal > 4) state.stepVal = 4;
        } else {
          const maxB = BPE_PHRASES[state.phraseKey].steps.length - 1;
          slider.max = `${maxB}`;
          if (state.stepVal > maxB) state.stepVal = maxB;
        }
        slider.value = `${state.stepVal}`;
      }
      if (sliderLbl) sliderLbl.textContent = isPoly ? 'Layer Depth:' : 'Merge Step:';
      if (sliderVal) sliderVal.textContent = isPoly ? `L=${state.stepVal}` : `${state.stepVal}/${slider ? slider.max : 8}`;
    }

    function render() {
      syncControls();
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      const leftBox = { x: 16, y: 12, w: 566, h: 426 };
      const rightBox = { x: 598, y: 12, w: 566, h: 426 };
      drawRoundRect(ctx, leftBox.x, leftBox.y, leftBox.w, leftBox.h, 8, SOL.base2, SOL.base1, 1);
      drawRoundRect(ctx, rightBox.x, rightBox.y, rightBox.w, rightBox.h, 8, SOL.base2, SOL.base1, 1);

      if (state.mode === 'polysemy') {
        const pd = POLY_DATA[state.wordKey];
        const alphaMix = state.stepVal / 4; // 0 (static) -> 1 (full contextual)
        const vecA = [pd.sentA.pole[0] * alphaMix, pd.sentA.pole[1] * alphaMix];
        const vecB = [pd.sentB.pole[0] * alphaMix, pd.sentB.pole[1] * alphaMix];
        // Shared static lexical anchor dimension z = 1.45 -> cos=1.00 at L=0, splitting smoothly to ~0.10 at L=4
        const cosSplit = state.stepVal === 0 ? 1.0 : cosineSim([vecA[0], vecA[1], 1.45], [vecB[0], vecB[1], 1.45]);

        if (readStat) readStat.textContent = state.stepVal === 0 ? 'Collapsed cos=1.00' : `Split cos=${cosSplit.toFixed(2)}`;
        if (banner) {
          banner.textContent = state.stepVal === 0
            ? `Layer 0 (Static Lookup): Both occurrences of "${state.wordKey}" share the identical Wₑ row (cosine = 1.00 — polysemy collapse)!`
            : `Layer ${state.stepVal} Self-Attention: Context weights α pull "${state.wordKey}" into ${pd.sentA.poleLabel} vs. ${pd.sentB.poleLabel}!`;
        }

        // Left Panel: Attention Arcs over Both Sentences
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`1. Self-Attention Context Arcs onto "${state.wordKey}" (L=${state.stepVal})`, leftBox.x + 14, leftBox.y + 24);

        const drawSentenceArcs = (sentObj, yBase, accent) => {
          drawRoundRect(ctx, leftBox.x + 14, yBase, leftBox.w - 28, 174, 6, SOL.base3, accent, 1.5);
          ctx.fillStyle = accent;
          ctx.font = 'bold 13px sans-serif';
          ctx.fillText(sentObj.label, leftBox.x + 26, yBase + 22);

          const toks = sentObj.tokens;
          const slotW = (leftBox.w - 56) / toks.length;
          const tokY = yBase + 120;
          const xs = toks.map((_, i) => leftBox.x + 28 + i * slotW + slotW / 2);
          const tx = xs[sentObj.target];

          // Attention arcs when Layer > 0
          if (state.stepVal > 0) {
            for (let j = 0; j < toks.length; j++) {
              if (j === sentObj.target) continue;
              const w = sentObj.weights[j] * alphaMix;
              if (w < 0.05) continue;
              const jx = xs[j];
              const arcH = 26 + Math.abs(sentObj.target - j) * 14;
              ctx.save();
              ctx.strokeStyle = accent;
              ctx.globalAlpha = Math.min(1, 0.25 + w * 1.4);
              ctx.lineWidth = Math.max(1.2, w * 8);
              ctx.beginPath();
              ctx.moveTo(jx, tokY - 4);
              ctx.quadraticCurveTo((jx + tx) / 2, tokY - arcH, tx, tokY - 4);
              ctx.stroke();
              ctx.restore();
            }
          }

          toks.forEach((tk, i) => {
            const isT = i === sentObj.target;
            const bw = Math.min(slotW - 6, 78);
            drawRoundRect(ctx, xs[i] - bw / 2, tokY, bw, 28, 5, isT ? accent : SOL.base2, accent, isT ? 2 : 1);
            ctx.fillStyle = isT ? SOL.base3 : SOL.base02;
            ctx.font = isT ? 'bold 12.5px monospace' : '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(tk, xs[i], tokY + 18);
            if (state.stepVal > 0 && !isT) {
              ctx.fillStyle = SOL.base01;
              ctx.font = '10.5px monospace';
              ctx.fillText(`α=${(sentObj.weights[i] * alphaMix).toFixed(2)}`, xs[i], tokY + 42);
            }
          });
          ctx.textAlign = 'left';
        };

        drawSentenceArcs(pd.sentA, leftBox.y + 42, SOL.cyan);
        if (state.compareBoth) {
          drawSentenceArcs(pd.sentB, leftBox.y + 230, SOL.orange);
        } else {
          drawRoundRect(ctx, leftBox.x + 14, leftBox.y + 230, leftBox.w - 28, 174, 6, SOL.base3, SOL.base1, 1);
          ctx.fillStyle = SOL.base01;
          ctx.font = '13px sans-serif';
          ctx.fillText('Click "Compare Both" to show Sentence B arcs side-by-side.', leftBox.x + 30, leftBox.y + 320);
        }

        // Right Panel: Exact 1:1 2D Contextual Semantic Plane
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('2. Contextual Representation Trajectory (Exact 1:1)', rightBox.x + 14, rightBox.y + 24);
        const plotBox = { x: rightBox.x + 18, y: rightBox.y + 38, w: rightBox.w - 36, h: 356 };
        drawRoundRect(ctx, plotBox.x, plotBox.y, plotBox.w, plotBox.h, 6, SOL.base3, SOL.base1, 1);

        const maxR = 1.65;
        const scale = Math.min(plotBox.w, plotBox.h) * 0.46 / maxR; // EXACT 1:1!
        const cx0 = plotBox.x + plotBox.w / 2;
        const cy0 = plotBox.y + plotBox.h / 2;
        const toPx = (p) => [cx0 + p[0] * scale, cy0 - p[1] * scale];

        ctx.strokeStyle = 'rgba(147,161,161,0.38)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plotBox.x + 8, cy0); ctx.lineTo(plotBox.x + plotBox.w - 8, cy0);
        ctx.moveTo(cx0, plotBox.y + 8); ctx.lineTo(cx0, plotBox.y + plotBox.h - 8);
        ctx.stroke();

        // Draw anchor sense halos, corner labels, vectors & dots first, then non-overlapping labels
        const placedRects = [];
        const drawHalo = (anchors, poleLabel, col, isTopLeft) => {
          const [ax, ay] = toPx([anchors[0][1], anchors[0][2]]);
          ctx.save();
          ctx.fillStyle = col;
          ctx.globalAlpha = 0.12;
          ctx.beginPath();
          ctx.arc(ax, ay, 72, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1.0;
          ctx.font = 'bold 12px sans-serif';
          const tw = ctx.measureText(poleLabel).width;
          if (isTopLeft) {
            const lx = plotBox.x + 12;
            const ly = plotBox.y + 18;
            ctx.textAlign = 'left';
            ctx.fillText(poleLabel, lx, ly);
            placedRects.push({ x: lx - 2, y: ly - 13, w: tw + 4, h: 15 });
          } else {
            const rx = plotBox.x + plotBox.w - 12;
            const ly = plotBox.y + plotBox.h - 10;
            ctx.textAlign = 'right';
            ctx.fillText(poleLabel, rx, ly);
            placedRects.push({ x: rx - tw - 2, y: ly - 13, w: tw + 4, h: 15 });
          }
          ctx.restore();
        };
        drawHalo(pd.sentA.anchors, pd.sentA.poleLabel, SOL.cyan, true);
        drawHalo(pd.sentB.anchors, pd.sentB.poleLabel, SOL.orange, false);

        const anchorDots = [];
        const drawAnchorDots = (anchors, col) => {
          anchors.forEach(([w, x, y]) => {
            const [px, py] = toPx([x, y]);
            anchorDots.push({ w, px, py });
            placedRects.push({ x: px - 4.5, y: py - 4.5, w: 9, h: 9 });
            ctx.beginPath();
            ctx.arc(px, py, 4.5, 0, Math.PI * 2);
            ctx.fillStyle = col;
            ctx.fill();
          });
        };
        drawAnchorDots(pd.sentA.anchors, SOL.cyan);
        drawAnchorDots(pd.sentB.anchors, SOL.orange);

        // Origin static vector + contextual vectors (drawn before labels)
        placedRects.push({ x: cx0 - 6.5, y: cy0 - 6.5, w: 13, h: 13 });
        ctx.beginPath();
        ctx.arc(cx0, cy0, 6.5, 0, Math.PI * 2);
        ctx.fillStyle = SOL.red;
        ctx.fill();

        let paX = cx0, paY = cy0, pbX = cx0, pbY = cy0;
        if (state.stepVal > 0) {
          [paX, paY] = toPx(vecA);
          [pbX, pbY] = toPx(vecB);
          drawArrow(ctx, cx0, cy0, paX, paY, SOL.cyan, 2.8, 9, false);
          drawArrow(ctx, cx0, cy0, pbX, pbY, SOL.orange, 2.8, 9, false);
          placedRects.push({ x: paX - 8, y: paY - 8, w: 16, h: 16 });
          placedRects.push({ x: pbX - 8, y: pbY - 8, w: 16, h: 16 });

          ctx.beginPath();
          ctx.arc(paX, paY, 8, 0, Math.PI * 2);
          ctx.fillStyle = SOL.cyan;
          ctx.fill();
          ctx.strokeStyle = SOL.base03; ctx.lineWidth = 2; ctx.stroke();

          ctx.beginPath();
          ctx.arc(pbX, pbY, 8, 0, Math.PI * 2);
          ctx.fillStyle = SOL.orange;
          ctx.fill();
          ctx.strokeStyle = SOL.base03; ctx.lineWidth = 2; ctx.stroke();

          placeLabelNonOverlapping(ctx, `h_${state.wordKey}(A)`, paX, paY, plotBox, placedRects, 'bold 12.5px monospace', SOL.base03);
          placeLabelNonOverlapping(ctx, `h_${state.wordKey}(B)`, pbX, pbY, plotBox, placedRects, 'bold 12.5px monospace', SOL.base03);
        }
        placeLabelNonOverlapping(ctx, `x_${state.wordKey} (L=0)`, cx0, cy0, plotBox, placedRects, 'bold 12px monospace', SOL.red);
        for (const { w, px, py } of anchorDots) {
          placeLabelNonOverlapping(ctx, w, px, py, plotBox, placedRects, '11px monospace', SOL.base02);
        }
        ctx.fillStyle = SOL.base01;
        ctx.font = '11.5px sans-serif';
        ctx.fillText('Scrub Layer Depth L=0→4 to split static centroid into contextual senses.', rightBox.x + 18, rightBox.y + 414);
      } else {
        // BPE Merge Stepper
        const bp = BPE_PHRASES[state.phraseKey];
        const stepIdx = Math.min(bp.steps.length - 1, state.stepVal);
        const toks = bp.steps[stepIdx];
        const mergeDesc = bp.merges[stepIdx];

        if (readStat) readStat.textContent = `${toks.length} tokens | 0 <UNK>`;
        if (banner) banner.textContent = `BPE Merge Step ${stepIdx}/${bp.steps.length - 1}: ${mergeDesc} → ${toks.length} subword tokens`;

        // Left Panel: Live Subword Token Chips
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`1. Input: "${bp.text}"`, leftBox.x + 14, leftBox.y + 24);
        drawRoundRect(ctx, leftBox.x + 14, leftBox.y + 38, leftBox.w - 28, 368, 6, SOL.base3, SOL.base1, 1);

        ctx.fillStyle = SOL.blue;
        ctx.font = 'bold 13px sans-serif';
        ctx.fillText(`Step ${stepIdx}: ${mergeDesc}`, leftBox.x + 28, leftBox.y + 66);

        const palette = [SOL.blue, SOL.cyan, SOL.green, SOL.orange, SOL.violet];
        let tx = leftBox.x + 28;
        let ty = leftBox.y + 92;
        toks.forEach((tk, idx) => {
          const disp = tk === ' ' ? '␣' : tk;
          ctx.font = 'bold 14px monospace';
          const tw = Math.max(24, ctx.measureText(disp).width + 14);
          if (tx + tw > leftBox.x + leftBox.w - 28) {
            tx = leftBox.x + 28;
            ty += 38;
          }
          const c = tk.length === 1 ? SOL.base01 : palette[idx % palette.length];
          drawRoundRect(ctx, tx, ty, tw, 28, 5, tk.length > 1 ? c : SOL.base2, c, 1.5);
          ctx.fillStyle = tk.length > 1 ? SOL.base3 : SOL.base02;
          ctx.textAlign = 'center';
          ctx.fillText(disp, tx + tw / 2, ty + 19);
          ctx.textAlign = 'left';
          tx += tw + 6;
        });

        ctx.fillStyle = SOL.green;
        ctx.font = 'bold 13.5px sans-serif';
        ctx.fillText(`Token count: ${toks.length}   |   Unknown <UNK> tokens: 0`, leftBox.x + 28, leftBox.y + 386);

        // Right Panel: Compression Curve (Token Count vs. BPE Merge Step)
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('2. Sequence Length Compression vs. Merge Steps', rightBox.x + 14, rightBox.y + 24);
        const plotBox = { x: rightBox.x + 18, y: rightBox.y + 38, w: rightBox.w - 36, h: 368 };
        drawRoundRect(ctx, plotBox.x, plotBox.y, plotBox.w, plotBox.h, 6, SOL.base3, SOL.base1, 1);

        const ox = plotBox.x + 56, oy = plotBox.y + plotBox.h - 54;
        const gw = plotBox.w - 86, gh = plotBox.h - 94;
        ctx.strokeStyle = SOL.base01;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(ox, oy - gh); ctx.lineTo(ox, oy); ctx.lineTo(ox + gw, oy);
        ctx.stroke();

        const maxLen = bp.steps[0].length;
        const nSteps = bp.steps.length - 1;
        ctx.strokeStyle = SOL.blue;
        ctx.lineWidth = 2.8;
        ctx.beginPath();
        bp.steps.forEach((st, i) => {
          const x = ox + (i / Math.max(1, nSteps)) * gw;
          const y = oy - (st.length / maxLen) * gh;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        bp.steps.forEach((st, i) => {
          const x = ox + (i / Math.max(1, nSteps)) * gw;
          const y = oy - (st.length / maxLen) * gh;
          const active = i === stepIdx;
          ctx.beginPath();
          ctx.arc(x, y, active ? 7.5 : 4.5, 0, Math.PI * 2);
          ctx.fillStyle = active ? SOL.orange : SOL.blue;
          ctx.fill();
          if (active) {
            ctx.fillStyle = SOL.orange;
            ctx.font = 'bold 12.5px monospace';
            ctx.fillText(`${st.length} tok`, Math.min(ox + gw - 44, x + 8), y - 8);
          }
          ctx.fillStyle = SOL.base01;
          ctx.font = '11px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`${i}`, x, oy + 18);
          ctx.textAlign = 'left';
        });

        ctx.fillStyle = SOL.base01;
        ctx.font = '12px sans-serif';
        ctx.fillText('BPE Merge Step', ox + gw / 2 - 48, oy + 40);
        ctx.fillText(`${maxLen}`, ox - 28, oy - gh + 6);
        ctx.fillText('0', ox - 18, oy + 4);
      }
    }

    if (selMode) {
      stopSwipePropagation(selMode);
      selMode.addEventListener('change', () => {
        state.mode = selMode.value;
        state.stepVal = state.mode === 'polysemy' ? 3 : 4;
        render();
      });
    }
    if (selWord) {
      stopSwipePropagation(selWord);
      selWord.addEventListener('change', () => { state.wordKey = selWord.value; render(); });
    }
    if (selPhrase) {
      stopSwipePropagation(selPhrase);
      selPhrase.addEventListener('change', () => {
        state.phraseKey = selPhrase.value;
        state.stepVal = Math.min(state.stepVal, BPE_PHRASES[state.phraseKey].steps.length - 1);
        render();
      });
    }
    bindRange(slider, () => { state.stepVal = parseInt(slider.value, 10); render(); });
    if (btnToggle) {
      stopSwipePropagation(btnToggle);
      btnToggle.addEventListener('click', () => {
        state.compareBoth = !state.compareBoth;
        render();
      });
    }
    if (btnReset) {
      stopSwipePropagation(btnReset);
      btnReset.addEventListener('click', () => {
        state.stepVal = state.mode === 'polysemy' ? 3 : 0;
        state.compareBoth = true;
        render();
      });
    }

    registerSlideListener(canvas, render);
    render();
  }

  // =========================================================================
  // BOOTSTRAP
  // =========================================================================
  function initAll() {
    initBridgeSlide();
    initPpmiSvdDemo();
    initSgnsDemo();
    initAnalogyDemo();
    initModernDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
