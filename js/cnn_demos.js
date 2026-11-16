/**
 * CS8850: Advanced Machine Learning - Lecture 21 (Convolutional Neural Networks)
 * Interactive Pedagogical Demonstrations Suite
 *
 * 0. Dynamic Bridge Slide Connectors (Precision SVG alignment)
 * 1. Convolution Arithmetic & Shift Equivariance Explorer (#cnn-conv-canvas)
 * 2. Receptive Field Growth: Pooling vs. Dilation (#cnn-rf-canvas)
 * 3. Deep Gradient Flow: Plain vs. Highway vs. ResNet (#cnn-skip-canvas)
 * 4. Causal Masked & Deformable Receptive Fields (#cnn-advanced-canvas)
 *
 * All 2D spatial grids and geometric plots enforce an exact 1:1 aspect ratio.
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
      console.error(`[cnn_demos] Missing DOM element #${id} in ${context}`);
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

  function drawArrow(ctx, x1, y1, x2, y2, color, lineWidth, headLen) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 1.5) return;
    const h = headLen || 8;
    const angle = Math.atan2(dy, dx);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth || 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - h * Math.cos(angle - Math.PI / 6), y2 - h * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - h * Math.cos(angle + Math.PI / 6), y2 - h * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function divergingColor(val, maxAbs) {
    const m = Math.max(1e-6, maxAbs);
    const t = Math.max(-1, Math.min(1, val / m));
    if (t >= 0) {
      // cream (#fdf6e3) -> Solarized orange/red (#dc322f)
      const r = Math.round(253 + (220 - 253) * t);
      const g = Math.round(246 + (50 - 246) * t);
      const b = Math.round(227 + (47 - 227) * t);
      return `rgb(${r},${g},${b})`;
    }
    const u = -t;
    // cream (#fdf6e3) -> Solarized blue (#268bd2)
    const r = Math.round(253 + (38 - 253) * u);
    const g = Math.round(246 + (139 - 246) * u);
    const b = Math.round(227 + (210 - 227) * u);
    return `rgb(${r},${g},${b})`;
  }

  function inputGrayColor(v) {
    // 0 -> cream (#fdf6e3), 1 -> deep slate (#073642)
    const t = Math.max(0, Math.min(1, v));
    const r = Math.round(253 + (7 - 253) * t);
    const g = Math.round(246 + (54 - 246) * t);
    const b = Math.round(227 + (66 - 227) * t);
    return `rgb(${r},${g},${b})`;
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
    setCurve('conn-1-12', c1, getRowLeftCenter('bridge-row-12'));
    setCurve('conn-1-13', c1, getRowLeftCenter('bridge-row-13'));
    setCurve('conn-2-21', c2, getRowLeftCenter('bridge-row-21'));
    setCurve('conn-2-22', c2, getRowLeftCenter('bridge-row-22'));
    setCurve('conn-3-24', c3, getRowLeftCenter('bridge-row-24'));
    setCurve('conn-4-25', c4, getRowLeftCenter('bridge-row-25'));
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
  // 1. CONVOLUTION ARITHMETIC & SHIFT EQUIVARIANCE EXPLORER
  // =========================================================================
  function initConvDemo() {
    const canvas = document.getElementById('cnn-conv-canvas');
    if (!canvas) return;

    const banner = reqEl('cnn-conv-banner', 'Demo 1');
    const selImg = reqEl('cnn-conv-img', 'Demo 1');
    const selKer = reqEl('cnn-conv-ker', 'Demo 1');
    const selPad = reqEl('cnn-conv-pad', 'Demo 1');
    const selStride = reqEl('cnn-conv-stride', 'Demo 1');
    const selDil = reqEl('cnn-conv-dil', 'Demo 1');
    const sliderDx = reqEl('cnn-conv-dx', 'Demo 1');
    const valDx = reqEl('cnn-conv-dx-val', 'Demo 1');
    const sliderDy = reqEl('cnn-conv-dy', 'Demo 1');
    const valDy = reqEl('cnn-conv-dy-val', 'Demo 1');
    const btnMode = reqEl('cnn-conv-mode', 'Demo 1');
    const chkRelu = reqEl('cnn-conv-relu', 'Demo 1');
    const btnStep = reqEl('cnn-conv-step', 'Demo 1');
    const btnReset = reqEl('cnn-conv-reset', 'Demo 1');
    const readShape = reqEl('cnn-conv-shape-val', 'Demo 1');
    const readParams = reqEl('cnn-conv-params-val', 'Demo 1');

    const KERNELS = {
      sobel_x:   { name: 'Sobel ∂x (Vertical Edges)',   k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
      sobel_y:   { name: 'Sobel ∂y (Horizontal Edges)', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
      laplacian: { name: 'Laplacian Outline',           k: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]] },
      blur:      { name: 'Box / Smoothing',             k: [[1, 1, 1], [1, 2, 1], [1, 1, 1]] },
      sharpen:   { name: 'Sharpen High-Pass',           k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
      gabor:     { name: 'Diagonal 45° Edge',           k: [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]] }
    };

    const H_IN = 9;
    const W_IN = 9;
    const CYCLE_VALS = [-2, -1, 0, 1, 2];

    const state = {
      imgType: selImg ? selImg.value : 'shapes',
      kerKey: selKer ? selKer.value : 'sobel_x',
      kernel: KERNELS.sobel_x.k.map((r) => r.slice()),
      pad: selPad ? parseInt(selPad.value, 10) : 1,
      stride: selStride ? parseInt(selStride.value, 10) : 1,
      dilation: selDil ? parseInt(selDil.value, 10) : 1,
      dx: sliderDx ? parseInt(sliderDx.value, 10) : 0,
      dy: sliderDy ? parseInt(sliderDy.value, 10) : 0,
      flippedConv: false,
      applyRelu: chkRelu ? chkRelu.checked : false,
      activeOutR: 3,
      activeOutC: 3
    };

    let hitAreas = { kernelCells: [], outGrid: null };

    function makeBaseInput(type) {
      const g = Array.from({ length: H_IN }, () => Array(W_IN).fill(0));
      if (type === 'shapes') {
        for (let r = 2; r <= 6; r++) {
          for (let c = 2; c <= 5; c++) g[r][c] = 0.9;
        }
        g[4][4] = 0.15;
      } else if (type === 'mri') {
        const cr = 4, cc = 4;
        for (let r = 0; r < H_IN; r++) {
          for (let c = 0; c < W_IN; c++) {
            const dist = Math.hypot((r - cr) * 0.95, (c - cc) * 1.15);
            if (dist < 1.35) g[r][c] = 0.12; // ventricle CSF
            else if (dist < 2.9) g[r][c] = 0.72; // white/gray matter
            else if (dist < 3.9) g[r][c] = 0.95; // cortex / skull rim
            else g[r][c] = 0.0;
          }
        }
      } else if (type === 'digit') {
        // Stylized digit '7' or '4'
        const coords = [
          [1,2],[1,3],[1,4],[1,5],[1,6],
          [2,6],[3,5],[4,4],[5,4],[6,3],[7,3]
        ];
        coords.forEach(([r, c]) => { g[r][c] = 0.95; });
      } else {
        // single impulse at center
        g[4][4] = 1.0;
      }
      return g;
    }

    function getShiftedInput() {
      const base = makeBaseInput(state.imgType);
      const out = Array.from({ length: H_IN }, () => Array(W_IN).fill(0));
      for (let r = 0; r < H_IN; r++) {
        for (let c = 0; c < W_IN; c++) {
          const sr = r - state.dy;
          const sc = c - state.dx;
          if (sr >= 0 && sr < H_IN && sc >= 0 && sc < W_IN) {
            out[r][c] = base[sr][sc];
          }
        }
      }
      return out;
    }

    function computeOutput() {
      const x = getShiftedInput();
      const k = 3;
      const p = state.pad;
      const s = state.stride;
      const d = state.dilation;
      const hOut = Math.floor((H_IN + 2 * p - d * (k - 1) - 1) / s) + 1;
      const wOut = Math.floor((W_IN + 2 * p - d * (k - 1) - 1) / s) + 1;

      const kerEff = state.flippedConv
        ? [
            [state.kernel[2][2], state.kernel[2][1], state.kernel[2][0]],
            [state.kernel[1][2], state.kernel[1][1], state.kernel[1][0]],
            [state.kernel[0][2], state.kernel[0][1], state.kernel[0][0]]
          ]
        : state.kernel;

      const y = Array.from({ length: hOut }, () => Array(wOut).fill(0));
      let maxAbs = 0.5;
      for (let ro = 0; ro < hOut; ro++) {
        for (let co = 0; co < wOut; co++) {
          let acc = 0;
          for (let u = 0; u < 3; u++) {
            for (let v = 0; v < 3; v++) {
              const ri = ro * s - p + u * d;
              const ci = co * s - p + v * d;
              const xv = (ri >= 0 && ri < H_IN && ci >= 0 && ci < W_IN) ? x[ri][ci] : 0;
              acc += xv * kerEff[u][v];
            }
          }
          if (state.applyRelu) acc = Math.max(0, acc);
          y[ro][co] = acc;
          if (Math.abs(acc) > maxAbs) maxAbs = Math.abs(acc);
        }
      }
      return { x, y, hOut, wOut, kerEff, maxAbs };
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      const { x, y, hOut, wOut, kerEff, maxAbs } = computeOutput();
      state.activeOutR = Math.max(0, Math.min(hOut - 1, state.activeOutR));
      state.activeOutC = Math.max(0, Math.min(wOut - 1, state.activeOutC));

      const p = state.pad;
      const s = state.stride;
      const d = state.dilation;
      const hPad = H_IN + 2 * p;
      const wPad = W_IN + 2 * p;

      // Update toolbar readouts
      if (valDx) valDx.textContent = (state.dx >= 0 ? `+${state.dx}` : `${state.dx}`);
      if (valDy) valDy.textContent = (state.dy >= 0 ? `+${state.dy}` : `${state.dy}`);
      if (readShape) readShape.textContent = `${H_IN}×${W_IN} → ${hOut}×${wOut}`;
      const denseParams = H_IN * W_IN * hOut * wOut;
      if (readParams) readParams.textContent = `10 vs ${denseParams.toLocaleString()}`;
      if (btnMode) {
        btnMode.textContent = state.flippedConv ? 'Mode: Flipped Conv (*)' : 'Mode: Cross-Corr';
        btnMode.style.background = state.flippedConv ? SOL.violet : SOL.blue;
      }

      // Panel layout: Left (Input 1:1), Center (Kernel & Dot Product), Right (Output 1:1)
      const leftBox = { x: 18, y: 12, w: 390, h: 426 };
      const midBox  = { x: 424, y: 12, w: 332, h: 426 };
      const rightBox = { x: 772, y: 12, w: 390, h: 426 };

      drawRoundRect(ctx, leftBox.x, leftBox.y, leftBox.w, leftBox.h, 8, SOL.base2, SOL.base1, 1);
      drawRoundRect(ctx, midBox.x, midBox.y, midBox.w, midBox.h, 8, SOL.base3, SOL.base1, 1);
      drawRoundRect(ctx, rightBox.x, rightBox.y, rightBox.w, rightBox.h, 8, SOL.base2, SOL.base1, 1);

      // ----- LEFT PANEL: 1:1 Input + Padding Grid -----
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`1. Input X (${H_IN}×${W_IN}) + Pad p=${p}  [Shift Δx=${state.dx}, Δy=${state.dy}]`, leftBox.x + 12, leftBox.y + 22);

      const maxGridIn = 340;
      const cellIn = Math.floor(maxGridIn / Math.max(hPad, wPad)); // EXACT 1:1 square
      const gridInW = cellIn * wPad;
      const gridInH = cellIn * hPad;
      const inOx = leftBox.x + Math.round((leftBox.w - gridInW) / 2);
      const inOy = leftBox.y + 38 + Math.round((maxGridIn - gridInH) / 2);

      // Active receptive field cells
      const rfCoords = new Map();
      const ar = state.activeOutR;
      const ac = state.activeOutC;
      const terms = [];
      let rawSum = 0;
      for (let u = 0; u < 3; u++) {
        for (let v = 0; v < 3; v++) {
          const ri = ar * s - p + u * d;
          const ci = ac * s - p + v * d;
          const xv = (ri >= 0 && ri < H_IN && ci >= 0 && ci < W_IN) ? x[ri][ci] : 0;
          const kw = kerEff[u][v];
          const prod = xv * kw;
          rawSum += prod;
          terms.push({ u, v, xv, kw, prod, isPad: (ri < 0 || ri >= H_IN || ci < 0 || ci >= W_IN) });
          rfCoords.set(`${ri + p},${ci + p}`, { u, v });
        }
      }

      for (let rp = 0; rp < hPad; rp++) {
        for (let cp = 0; cp < wPad; cp++) {
          const cx = inOx + cp * cellIn;
          const cy = inOy + rp * cellIn;
          const ri = rp - p;
          const ci = cp - p;
          const isPad = ri < 0 || ri >= H_IN || ci < 0 || ci >= W_IN;
          const val = isPad ? 0 : x[ri][ci];

          ctx.fillStyle = isPad ? '#eae3cd' : inputGrayColor(val);
          ctx.fillRect(cx, cy, cellIn, cellIn);
          ctx.strokeStyle = isPad ? '#b8b19c' : SOL.base1;
          ctx.lineWidth = 0.8;
          if (isPad) ctx.setLineDash([2, 2]);
          ctx.strokeRect(cx, cy, cellIn, cellIn);
          ctx.setLineDash([]);

          if (cellIn >= 22) {
            ctx.fillStyle = isPad ? SOL.base01 : (val > 0.55 ? SOL.base3 : SOL.base02);
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(isPad ? '0' : val.toFixed(1), cx + cellIn / 2, cy + cellIn / 2);
          }

          if (rfCoords.has(`${rp},${cp}`)) {
            ctx.strokeStyle = SOL.orange;
            ctx.lineWidth = 2.6;
            ctx.strokeRect(cx + 1.5, cy + 1.5, cellIn - 3, cellIn - 3);
          }
        }
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';

      // Bounding box for dilated/strided kernel footprint
      const topLeftR = ar * s;
      const topLeftC = ac * s;
      const spanCells = d * 2 + 1;
      ctx.save();
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(
        inOx + topLeftC * cellIn,
        inOy + topLeftR * cellIn,
        spanCells * cellIn,
        spanCells * cellIn
      );
      ctx.restore();

      ctx.fillStyle = SOL.base01;
      ctx.font = '12px sans-serif';
      ctx.fillText('Dashed border = zero padding (p); orange cells = active kernel taps', leftBox.x + 14, leftBox.y + leftBox.h - 14);

      // ----- CENTER PANEL: 3x3 Kernel (Clickable) + Dot-Product Breakdown -----
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('2. Shared Kernel K (Click Cell to Edit)', midBox.x + 14, midBox.y + 22);

      const kCell = 48;
      const kOx = midBox.x + Math.round((midBox.w - 3 * kCell) / 2);
      const kOy = midBox.y + 38;
      hitAreas.kernelCells = [];

      for (let u = 0; u < 3; u++) {
        for (let v = 0; v < 3; v++) {
          const val = kerEff[u][v];
          const kx = kOx + v * kCell;
          const ky = kOy + u * kCell;
          ctx.fillStyle = divergingColor(val, 4);
          ctx.fillRect(kx, ky, kCell, kCell);
          ctx.strokeStyle = SOL.base02;
          ctx.lineWidth = 1.4;
          ctx.strokeRect(kx, ky, kCell, kCell);

          ctx.fillStyle = Math.abs(val) >= 3 ? SOL.base3 : SOL.base03;
          ctx.font = 'bold 16px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(val), kx + kCell / 2, ky + kCell / 2);

          // map back to unflipped indices when clicked
          const origU = state.flippedConv ? 2 - u : u;
          const origV = state.flippedConv ? 2 - v : v;
          hitAreas.kernelCells.push({ x: kx, y: ky, w: kCell, h: kCell, u: origU, v: origV });
        }
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';

      ctx.fillStyle = state.flippedConv ? SOL.violet : SOL.blue;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        state.flippedConv
          ? 'True Convolution (*): kernel flipped 180° before dot product'
          : 'Cross-Correlation: PyTorch / JAX Conv2d default',
        midBox.x + midBox.w / 2,
        kOy + 3 * kCell + 18
      );
      ctx.textAlign = 'left';

      // Dot-product grid (3x3 elementwise products)
      const boxY = kOy + 3 * kCell + 30;
      drawRoundRect(ctx, midBox.x + 12, boxY, midBox.w - 24, 134, 6, SOL.base2, SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12.5px sans-serif';
      ctx.fillText(`Active Cell Y[${ar}, ${ac}] = ∑ X_patch ⊙ K:`, midBox.x + 22, boxY + 18);

      ctx.font = '11.5px monospace';
      for (let u = 0; u < 3; u++) {
        const rowTerms = terms.slice(u * 3, u * 3 + 3).map((t) => {
          const pStr = t.prod >= 0 ? `+${t.prod.toFixed(1)}` : t.prod.toFixed(1);
          return `${t.xv.toFixed(1)}·(${t.kw})=${pStr}`;
        });
        ctx.fillStyle = SOL.base01;
        ctx.fillText(rowTerms.join('  '), midBox.x + 20, boxY + 42 + u * 22);
      }

      const finalVal = state.applyRelu ? Math.max(0, rawSum) : rawSum;
      ctx.fillStyle = SOL.orange;
      ctx.font = 'bold 13.5px monospace';
      ctx.fillText(
        state.applyRelu
          ? `Raw = ${rawSum.toFixed(2)}  →  ReLU = ${finalVal.toFixed(2)}`
          : `Dot Sum = ${finalVal.toFixed(2)}`,
        midBox.x + 22,
        boxY + 122
      );

      // Arithmetic formula box at bottom of middle panel
      const fY = boxY + 144;
      drawRoundRect(ctx, midBox.x + 12, fY, midBox.w - 24, 50, 6, '#e6dfc7', SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 11.2px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`H_out = ⌊(${H_IN}+2·${p}-${d}·(3-1)-1)/${s}⌋+1 = ${hOut}`, midBox.x + midBox.w / 2, fY + 21);
      ctx.fillStyle = SOL.base01;
      ctx.font = '11px sans-serif';
      ctx.fillText(`Shared Conv: 9+1=10 params | Dense: ${denseParams.toLocaleString()} params`, midBox.x + midBox.w / 2, fY + 40);
      ctx.textAlign = 'left';

      // ----- RIGHT PANEL: 1:1 Output Feature Map Y -----
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`3. Feature Map Y (${hOut}×${wOut})  [Click/Hover Any Cell]`, rightBox.x + 12, rightBox.y + 22);

      const maxGridOut = 340;
      const cellOut = Math.floor(maxGridOut / Math.max(1, hOut, wOut)); // EXACT 1:1 square
      const gridOutW = cellOut * wOut;
      const gridOutH = cellOut * hOut;
      const outOx = rightBox.x + Math.round((rightBox.w - gridOutW) / 2);
      const outOy = rightBox.y + 38 + Math.round((maxGridOut - gridOutH) / 2);

      hitAreas.outGrid = { x: outOx, y: outOy, cell: cellOut, hOut, wOut };

      for (let ro = 0; ro < hOut; ro++) {
        for (let co = 0; co < wOut; co++) {
          const cx = outOx + co * cellOut;
          const cy = outOy + ro * cellOut;
          const val = y[ro][co];
          ctx.fillStyle = divergingColor(val, maxAbs);
          ctx.fillRect(cx, cy, cellOut, cellOut);
          ctx.strokeStyle = SOL.base1;
          ctx.lineWidth = 0.9;
          ctx.strokeRect(cx, cy, cellOut, cellOut);

          if (cellOut >= 22) {
            ctx.fillStyle = Math.abs(val) > maxAbs * 0.65 ? SOL.base3 : SOL.base02;
            ctx.font = cellOut >= 34 ? 'bold 11.5px monospace' : '10px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(val.toFixed(1), cx + cellOut / 2, cy + cellOut / 2);
          }

          if (ro === ar && co === ac) {
            ctx.strokeStyle = SOL.orange;
            ctx.lineWidth = 3;
            ctx.strokeRect(cx + 1.5, cy + 1.5, cellOut - 3, cellOut - 3);
          }
        }
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';

      ctx.fillStyle = SOL.base01;
      ctx.font = '12px sans-serif';
      ctx.fillText('Blue = negative response, Cream = 0, Red = strong match', rightBox.x + 14, rightBox.y + rightBox.h - 14);

      if (banner) {
        const shiftMsg = (state.dx !== 0 || state.dy !== 0)
          ? `Translation Equivariance active: shifted input by (Δx=${state.dx}, Δy=${state.dy}) → feature map shifted identically!`
          : `Output ${hOut}×${wOut} using 10 shared params (${(denseParams / 10).toFixed(0)}× compression vs. dense). Drag Δx/Δy to test shift equivariance!`;
        banner.textContent = shiftMsg;
      }
    }

    // Event wiring
    if (selImg) {
      stopSwipePropagation(selImg);
      selImg.addEventListener('change', () => { state.imgType = selImg.value; render(); });
    }
    if (selKer) {
      stopSwipePropagation(selKer);
      selKer.addEventListener('change', () => {
        state.kerKey = selKer.value;
        state.kernel = KERNELS[state.kerKey].k.map((r) => r.slice());
        render();
      });
    }
    if (selPad) {
      stopSwipePropagation(selPad);
      selPad.addEventListener('change', () => { state.pad = parseInt(selPad.value, 10); render(); });
    }
    if (selStride) {
      stopSwipePropagation(selStride);
      selStride.addEventListener('change', () => { state.stride = parseInt(selStride.value, 10); render(); });
    }
    if (selDil) {
      stopSwipePropagation(selDil);
      selDil.addEventListener('change', () => { state.dilation = parseInt(selDil.value, 10); render(); });
    }
    bindRange(sliderDx, () => { state.dx = parseInt(sliderDx.value, 10); render(); });
    bindRange(sliderDy, () => { state.dy = parseInt(sliderDy.value, 10); render(); });

    if (btnMode) {
      stopSwipePropagation(btnMode);
      btnMode.addEventListener('click', () => { state.flippedConv = !state.flippedConv; render(); });
    }
    if (chkRelu) {
      stopSwipePropagation(chkRelu);
      chkRelu.addEventListener('change', () => { state.applyRelu = chkRelu.checked; render(); });
    }
    if (btnStep) {
      stopSwipePropagation(btnStep);
      btnStep.addEventListener('click', () => {
        const { hOut, wOut } = computeOutput();
        state.activeOutC += 1;
        if (state.activeOutC >= wOut) {
          state.activeOutC = 0;
          state.activeOutR = (state.activeOutR + 1) % hOut;
        }
        render();
      });
    }
    if (btnReset) {
      stopSwipePropagation(btnReset);
      btnReset.addEventListener('click', () => {
        state.dx = 0;
        state.dy = 0;
        if (sliderDx) sliderDx.value = '0';
        if (sliderDy) sliderDy.value = '0';
        state.pad = 1; if (selPad) selPad.value = '1';
        state.stride = 1; if (selStride) selStride.value = '1';
        state.dilation = 1; if (selDil) selDil.value = '1';
        state.flippedConv = false;
        state.applyRelu = false; if (chkRelu) chkRelu.checked = false;
        state.kerKey = selKer ? selKer.value : 'sobel_x';
        state.kernel = KERNELS[state.kerKey].k.map((r) => r.slice());
        state.activeOutR = 3;
        state.activeOutC = 3;
        render();
      });
    }

    stopSwipePropagation(canvas);
    canvas.addEventListener('click', (e) => {
      const pt = canvasPoint(canvas, e.clientX, e.clientY);
      for (const kc of hitAreas.kernelCells) {
        if (pt.x >= kc.x && pt.x <= kc.x + kc.w && pt.y >= kc.y && pt.y <= kc.y + kc.h) {
          const cur = state.kernel[kc.u][kc.v];
          const idx = CYCLE_VALS.indexOf(cur);
          state.kernel[kc.u][kc.v] = CYCLE_VALS[(idx + 1) % CYCLE_VALS.length];
          render();
          return;
        }
      }
      const og = hitAreas.outGrid;
      if (og) {
        const co = Math.floor((pt.x - og.x) / og.cell);
        const ro = Math.floor((pt.y - og.y) / og.cell);
        if (ro >= 0 && ro < og.hOut && co >= 0 && co < og.wOut) {
          state.activeOutR = ro;
          state.activeOutC = co;
          render();
        }
      }
    });

    canvas.addEventListener('mousemove', (e) => {
      const pt = canvasPoint(canvas, e.clientX, e.clientY);
      const og = hitAreas.outGrid;
      if (og) {
        const co = Math.floor((pt.x - og.x) / og.cell);
        const ro = Math.floor((pt.y - og.y) / og.cell);
        if (ro >= 0 && ro < og.hOut && co >= 0 && co < og.wOut) {
          if (state.activeOutR !== ro || state.activeOutC !== co) {
            state.activeOutR = ro;
            state.activeOutC = co;
            render();
          }
        }
      }
    });

    registerSlideListener(canvas, render);
    render();
  }

  // =========================================================================
  // 2. RECEPTIVE FIELD GROWTH: POOLING VS. DILATION
  // =========================================================================
  function initReceptiveFieldDemo() {
    const canvas = document.getElementById('cnn-rf-canvas');
    if (!canvas) return;

    const banner = reqEl('cnn-rf-banner', 'Demo 2');
    const selArch = reqEl('cnn-rf-arch', 'Demo 2');
    const sliderDepth = reqEl('cnn-rf-depth', 'Demo 2');
    const valDepth = reqEl('cnn-rf-depth-val', 'Demo 2');
    const btnView = reqEl('cnn-rf-view', 'Demo 2');
    const btnCenter = reqEl('cnn-rf-center', 'Demo 2');
    const readSize = reqEl('cnn-rf-size-val', 'Demo 2');
    const readOut = reqEl('cnn-rf-out-val', 'Demo 2');
    const readMem = reqEl('cnn-rf-mem-val', 'Demo 2');

    const ARCHS = {
      plain: {
        name: 'Plain 3×3 Stack (d=1,1,1,1, s=1)',
        layers: [
          { k: 3, d: 1, s: 1 },
          { k: 3, d: 1, s: 1 },
          { k: 3, d: 1, s: 1 },
          { k: 3, d: 1, s: 1 }
        ],
        params3D: '84K',
        memText: 'No Pooling (RF too small: 9³)'
      },
      pooled: {
        name: 'Pooled / Strided Encoder (s=2 downsampling, U-Net)',
        layers: [
          { k: 3, d: 1, s: 2 },
          { k: 3, d: 1, s: 2 },
          { k: 3, d: 1, s: 2 },
          { k: 3, d: 1, s: 1 }
        ],
        params3D: '23.5M',
        memText: '~2 GB VRAM (Skip tensors + Decoder)'
      },
      dilated: {
        name: 'Exponential Dilated Stack (d=1,2,4,8, MeshNet / WaveNet)',
        layers: [
          { k: 3, d: 1, s: 1 },
          { k: 3, d: 2, s: 1 },
          { k: 3, d: 4, s: 1 },
          { k: 3, d: 8, s: 1 }
        ],
        params3D: '72.5K',
        memText: '~600 KB (Full-Res, Zero Skip Buffers)'
      }
    };

    const IN_N = 32;
    const state = {
      arch: selArch ? selArch.value : 'dilated',
      depth: sliderDepth ? parseInt(sliderDepth.value, 10) : 4,
      showEffective: true,
      queryR: 16,
      queryC: 16
    };

    let activeGridHit = null;

    function computeChain() {
      const cfg = ARCHS[state.arch];
      const grids = [{ N: IN_N, rf: 1, jump: 1, label: 'Input X' }];
      let n = IN_N;
      let rf = 1;
      let jump = 1;
      for (let i = 0; i < state.depth; i++) {
        const L = cfg.layers[i];
        rf = rf + (L.k - 1) * L.d * jump;
        jump = jump * L.s;
        n = Math.max(1, Math.floor(n / L.s));
        grids.push({
          N: n,
          rf,
          jump,
          d: L.d,
          s: L.s,
          label: `Layer ${i + 1} (d=${L.d}, s=${L.s})`
        });
      }
      return grids;
    }

    // Propagate exact path weights backward in 1D (separable -> outer product for 2D ERF)
    function compute1DWeights(depth, archKey, queryIdx) {
      const cfg = ARCHS[archKey];
      const sizes = [IN_N];
      for (let i = 0; i < depth; i++) {
        sizes.push(Math.max(1, Math.floor(sizes[i] / cfg.layers[i].s)));
      }
      const q = Math.max(0, Math.min(sizes[depth] - 1, queryIdx));
      const maps = Array.from({ length: depth + 1 }, (_, i) => new Float64Array(sizes[i]));
      maps[depth][q] = 1.0;

      for (let l = depth - 1; l >= 0; l--) {
        const { d, s } = cfg.layers[l];
        const curN = sizes[l];
        const nextN = sizes[l + 1];
        for (let j = 0; j < nextN; j++) {
          const w = maps[l + 1][j];
          if (w === 0) continue;
          const center = j * s + (s > 1 ? 0 : 0);
          for (const off of [-d, 0, d]) {
            const idx = center + off;
            if (idx >= 0 && idx < curN) {
              maps[l][idx] += w;
            }
          }
        }
      }
      return { maps, sizes, q };
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      const grids = computeChain();
      const topG = grids[grids.length - 1];
      const maxQ = topG.N - 1;
      const qr = Math.max(0, Math.min(maxQ, Math.round(state.queryR * topG.N / IN_N)));
      const qc = Math.max(0, Math.min(maxQ, Math.round(state.queryC * topG.N / IN_N)));

      const wR = compute1DWeights(state.depth, state.arch, qr);
      const wC = compute1DWeights(state.depth, state.arch, qc);

      const cfg = ARCHS[state.arch];
      if (valDepth) valDepth.textContent = String(state.depth);
      if (readSize) readSize.textContent = `${topG.rf}×${topG.rf}`;
      if (readOut) readOut.textContent = `${topG.N}×${topG.N}`;
      if (readMem) {
        const memShort = state.arch === 'dilated' ? '~600 KB' : (state.arch === 'pooled' ? '~2 GB' : 'Small RF');
        readMem.textContent = `${cfg.params3D} (${memShort})`;
      }
      if (btnView) {
        btnView.textContent = state.showEffective ? 'View: Box + Effective RF' : 'View: Theoretical Box Only';
      }

      // Draw up to 5 1:1 square panels cleanly across 1180px
      const count = grids.length;
      const slotW = Math.floor((width - 40) / count);
      const maxBox = 186;
      const centerY = 185;
      const layerRects = [];

      for (let i = 0; i < count; i++) {
        const g = grids[i];
        // Scale physical box size slightly with resolution when downsampled, keeping exact 1:1 square
        const resRatio = Math.max(0.40, g.N / IN_N);
        const targetSide = Math.round(maxBox * (state.arch === 'pooled' ? (0.38 + 0.62 * resRatio) : 1.0));
        const cell = Math.max(3, targetSide / g.N);
        const side = cell * g.N; // EXACT 1:1 square
        const ox = 20 + i * slotW + Math.round((slotW - side) / 2);
        const oy = centerY - Math.round(side / 2);

        layerRects.push({ ox, oy, side, cell, N: g.N });

        // Card header
        ctx.fillStyle = i === count - 1 ? SOL.orange : SOL.base02;
        ctx.font = 'bold 12.5px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(g.label, ox + side / 2, 34);
        ctx.fillStyle = SOL.base01;
        ctx.font = '11.5px monospace';
        ctx.fillText(`Grid: ${g.N}×${g.N}  |  RF: ${g.rf}×${g.rf}`, ox + side / 2, 52);

        // Compute max 2D path weight in this layer for ERF normalization
        let maxR = 1e-9, maxC = 1e-9;
        for (let r = 0; r < g.N; r++) maxR = Math.max(maxR, wR.maps[i][r]);
        for (let c = 0; c < g.N; c++) maxC = Math.max(maxC, wC.maps[i][c]);
        const max2D = maxR * maxC;

        // Track theoretical bounding box in this layer
        let rMin = g.N, rMax = -1, cMin = g.N, cMax = -1;
        for (let r = 0; r < g.N; r++) {
          if (wR.maps[i][r] > 0) { rMin = Math.min(rMin, r); rMax = Math.max(rMax, r); }
        }
        for (let c = 0; c < g.N; c++) {
          if (wC.maps[i][c] > 0) { cMin = Math.min(cMin, c); cMax = Math.max(cMax, c); }
        }

        // Draw pixels
        ctx.fillStyle = SOL.base2;
        ctx.fillRect(ox, oy, side, side);

        for (let r = 0; r < g.N; r++) {
          for (let c = 0; c < g.N; c++) {
            const w2 = wR.maps[i][r] * wC.maps[i][c];
            if (w2 > 0) {
              if (state.showEffective) {
                const alpha = Math.pow(w2 / max2D, 0.45);
                ctx.fillStyle = `rgba(203, 75, 22, ${(0.18 + 0.82 * alpha).toFixed(3)})`;
              } else {
                ctx.fillStyle = 'rgba(38, 139, 210, 0.55)';
              }
              ctx.fillRect(ox + c * cell, oy + r * cell, Math.max(1, cell), Math.max(1, cell));
            }
          }
        }

        // Grid lines for coarse grids (N <= 16)
        if (g.N <= 16) {
          ctx.strokeStyle = 'rgba(88, 110, 117, 0.25)';
          ctx.lineWidth = 0.7;
          for (let k = 1; k < g.N; k++) {
            ctx.beginPath();
            ctx.moveTo(ox + k * cell, oy);
            ctx.lineTo(ox + k * cell, oy + side);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(ox, oy + k * cell);
            ctx.lineTo(ox + side, oy + k * cell);
            ctx.stroke();
          }
        }

        // Outer grid frame
        ctx.strokeStyle = i === count - 1 ? SOL.orange : SOL.base01;
        ctx.lineWidth = i === count - 1 ? 2 : 1.2;
        ctx.strokeRect(ox, oy, side, side);

        // Theoretical RF bounding box (exact reach across all paths including gaps!)
        if (rMax >= rMin && cMax >= cMin) {
          ctx.strokeStyle = SOL.blue;
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          ctx.strokeRect(
            ox + cMin * cell,
            oy + rMin * cell,
            (cMax - cMin + 1) * cell,
            (rMax - rMin + 1) * cell
          );
          ctx.setLineDash([]);
        }

        // Highlight single queried neuron in top layer
        if (i === count - 1) {
          ctx.fillStyle = SOL.orange;
          ctx.fillRect(ox + qc * cell, oy + qr * cell, Math.max(4, cell), Math.max(4, cell));
          ctx.strokeStyle = SOL.base03;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(ox + qc * cell, oy + qr * cell, Math.max(4, cell), Math.max(4, cell));
          activeGridHit = { ox, oy, side, cell, N: g.N };
        }

        // Connect arrows between layers
        if (i > 0) {
          const prev = layerRects[i - 1];
          drawArrow(ctx, prev.ox + prev.side + 6, centerY, ox - 6, centerY, SOL.base1, 2, 7);
        }
      }

      // Bottom Scorecard Comparing All 3 Architectures
      const cardY = 308;
      const cardW = 368;
      const cardH = 128;
      const archKeys = ['plain', 'pooled', 'dilated'];
      const summaries = {
        plain: {
          title: 'Plain 3×3 (d=1,1,1,1, s=1)',
          rf: 'RF₄ = 9×9 (Linear growth)',
          res: 'Output: 32×32 (100% spatial res)',
          cost: 'Params (3D C=21): ~84K  |  Tiny context'
        },
        pooled: {
          title: 'Pooled Encoder (s=2 stages, U-Net)',
          rf: 'RF₄ = 31×31 (Exponential via stride)',
          res: 'Output: 4×4 (64× area lost! Needs decoder)',
          cost: 'Params (3D U-Net): 23.5M  |  ~2 GB VRAM'
        },
        dilated: {
          title: 'Dilated Stack (d=1,2,4,8, MeshNet)',
          rf: 'RF₄ = 31×31 (Exponential via dilation!)',
          res: 'Output: 32×32 (100% spatial res preserved)',
          cost: 'Params (3D MeshNet): 72.5K  |  ~600 KB'
        }
      };

      ctx.textAlign = 'left';
      archKeys.forEach((key, idx) => {
        const cx = 20 + idx * (cardW + 18);
        const active = key === state.arch;
        const s = summaries[key];
        drawRoundRect(
          ctx, cx, cardY, cardW, cardH, 8,
          active ? '#fdf6e3' : SOL.base2,
          active ? SOL.orange : SOL.base1,
          active ? 2.5 : 1
        );
        ctx.fillStyle = active ? SOL.orange : SOL.base02;
        ctx.font = 'bold 13.5px sans-serif';
        ctx.fillText((active ? '▶ ' : '') + s.title, cx + 14, cardY + 25);

        ctx.fillStyle = SOL.base02;
        ctx.font = '12.5px monospace';
        ctx.fillText(s.rf, cx + 14, cardY + 54);
        ctx.fillStyle = key === 'pooled' ? SOL.red : SOL.green;
        ctx.fillText(s.res, cx + 14, cardY + 80);
        ctx.fillStyle = SOL.base01;
        ctx.fillText(s.cost, cx + 14, cardY + 106);
      });

      if (banner) {
        banner.textContent =
          `${cfg.name} at depth L=${state.depth}: RF = ${topG.rf}×${topG.rf}, output grid = ${topG.N}×${topG.N}. Notice Gaussian concentration inside the blue theoretical RF box!`;
      }
    }

    if (selArch) {
      stopSwipePropagation(selArch);
      selArch.addEventListener('change', () => { state.arch = selArch.value; render(); });
    }
    bindRange(sliderDepth, () => { state.depth = parseInt(sliderDepth.value, 10); render(); });
    if (btnView) {
      stopSwipePropagation(btnView);
      btnView.addEventListener('click', () => { state.showEffective = !state.showEffective; render(); });
    }
    if (btnCenter) {
      stopSwipePropagation(btnCenter);
      btnCenter.addEventListener('click', () => {
        state.queryR = 16;
        state.queryC = 16;
        render();
      });
    }

    stopSwipePropagation(canvas);
    let dragging = false;
    function updateQueryFromMouse(e) {
      if (!activeGridHit) return;
      const pt = canvasPoint(canvas, e.clientX, e.clientY);
      const g = activeGridHit;
      if (pt.x >= g.ox && pt.x <= g.ox + g.side && pt.y >= g.oy && pt.y <= g.oy + g.side) {
        const c = Math.max(0, Math.min(g.N - 1, Math.floor((pt.x - g.ox) / g.cell)));
        const r = Math.max(0, Math.min(g.N - 1, Math.floor((pt.y - g.oy) / g.cell)));
        state.queryR = Math.round((r + 0.5) * IN_N / g.N);
        state.queryC = Math.round((c + 0.5) * IN_N / g.N);
        render();
      }
    }
    canvas.addEventListener('mousedown', (e) => { dragging = true; updateQueryFromMouse(e); });
    window.addEventListener('mouseup', () => { dragging = false; });
    canvas.addEventListener('mousemove', (e) => { if (dragging) updateQueryFromMouse(e); });
    canvas.addEventListener('click', updateQueryFromMouse);

    registerSlideListener(canvas, render);
    render();
  }

  // =========================================================================
  // 3. DEEP GRADIENT FLOW: PLAIN VS. HIGHWAY VS. RESNET
  // =========================================================================
  function initSkipDemo() {
    const canvas = document.getElementById('cnn-skip-canvas');
    if (!canvas) return;

    const banner = reqEl('cnn-skip-banner', 'Demo 3');
    const sliderDepth = reqEl('cnn-skip-depth', 'Demo 3');
    const valDepth = reqEl('cnn-skip-depth-val', 'Demo 3');
    const sliderBias = reqEl('cnn-skip-bias', 'Demo 3');
    const valBias = reqEl('cnn-skip-bias-val', 'Demo 3');
    const sliderWscale = reqEl('cnn-skip-wscale', 'Demo 3');
    const valWscale = reqEl('cnn-skip-wscale-val', 'Demo 3');
    const btnBadGate = reqEl('cnn-skip-badgate', 'Demo 3');
    const btnGoodGate = reqEl('cnn-skip-goodgate', 'Demo 3');
    const btnReset = reqEl('cnn-skip-reset', 'Demo 3');
    const readPlain = reqEl('cnn-skip-plain-grad', 'Demo 3');
    const readHw = reqEl('cnn-skip-hw-grad', 'Demo 3');
    const readRes = reqEl('cnn-skip-res-grad', 'Demo 3');

    const state = {
      L: sliderDepth ? parseInt(sliderDepth.value, 10) : 56,
      bT: sliderBias ? parseFloat(sliderBias.value) : -2.5,
      sigmaW: sliderWscale ? parseFloat(sliderWscale.value) : 0.85
    };

    function sigmoid(z) {
      return 1 / (1 + Math.exp(-z));
    }

    function computeProfiles() {
      const L = state.L;
      const tGate = sigmoid(state.bT);
      // Effective per-layer Jacobian spectral gain
      // Plain: sigmaW * 0.90 -> vanishes (<1) below 1.11, explodes (>1) above 1.11
      const gPlain = Math.max(0.05, state.sigmaW * 0.90);
      // Highway: T * gPlain + (1 - T) * 1.0
      const gHw = tGate * gPlain + (1 - tGate) * 1.0;
      // ResNet with residual branch scaled/normalized (~0.15/sqrt(L) residual variance)
      const epsRes = 0.015 * state.sigmaW;
      const gRes = 1.0 + epsRes;

      const plain = new Array(L);
      const highway = new Array(L);
      const resnet = new Array(L);

      // Backward accumulation from layer L down to 1
      let pAcc = 1.0, hAcc = 1.0, rAcc = 1.0;
      for (let idx = L - 1; idx >= 0; idx--) {
        // Gentle periodic modulation to mimic real conv Jacobian variations
        const jitter = 1.0 + 0.06 * Math.sin(idx * 0.7);
        plain[idx] = Math.log10(Math.max(1e-16, Math.min(1e12, pAcc)));
        highway[idx] = Math.log10(Math.max(1e-16, Math.min(1e12, hAcc)));
        resnet[idx] = Math.log10(Math.max(1e-16, Math.min(1e12, rAcc)));
        pAcc *= gPlain * jitter;
        hAcc *= gHw * (1 + (tGate * 0.04) * Math.sin(idx * 0.7));
        rAcc *= gRes;
      }
      return { plain, highway, resnet, tGate };
    }

    function drawMiniLandscape(ctx, boxX, boxY, side, mode) {
      // Exact 1:1 square contour plot (side x side)
      drawRoundRect(ctx, boxX, boxY, side, side, 6, SOL.base3, SOL.base1, 1);
      const pad = 8;
      const pw = side - 2 * pad; // 1:1 square plot area
      const px0 = boxX + pad;
      const py0 = boxY + pad;

      const N = 26;
      const cs = pw / N;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const u = ((j + 0.5) / N) * 2 - 1;
          const v = ((i + 0.5) / N) * 2 - 1;
          let z;
          if (mode === 'plain') {
            z = 0.45 * (u * u + 1.8 * v * v) +
                0.28 * Math.sin(7.5 * u) * Math.cos(7.5 * v) +
                0.18 * Math.cos(12 * (u - v));
          } else {
            z = 0.45 * (u * u + 1.25 * v * v) + 0.02 * Math.sin(4 * u);
          }
          const t = Math.max(0, Math.min(1, z / 1.4));
          // Smooth cream -> yellow/orange -> deep slate contour shading
          const r = Math.round(253 - 190 * t);
          const g = Math.round(246 - 160 * t);
          const b = Math.round(227 - 115 * t);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(px0 + j * cs, py0 + i * cs, cs + 0.5, cs + 0.5);
        }
      }

      // Draw SGD trajectory
      ctx.beginPath();
      const pts = mode === 'plain'
        ? [[-0.78, -0.70], [-0.52, -0.58], [-0.62, -0.35], [-0.42, -0.44], [-0.48, -0.26]]
        : [[-0.78, -0.70], [-0.54, -0.48], [-0.33, -0.28], [-0.15, -0.12], [0.0, 0.0]];
      pts.forEach(([u, v], k) => {
        const sx = px0 + ((u + 1) / 2) * pw;
        const sy = py0 + ((v + 1) / 2) * pw;
        if (k === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.strokeStyle = mode === 'plain' ? SOL.red : SOL.green;
      ctx.lineWidth = 2.4;
      ctx.stroke();

      // Minimum star
      const cx = px0 + 0.5 * pw;
      const cy = py0 + 0.5 * pw;
      ctx.fillStyle = SOL.base3;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = SOL.base03;
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Label badge
      drawRoundRect(ctx, boxX + 6, boxY + 6, side - 12, 22, 4, 'rgba(253,246,227,0.88)', null, 0);
      ctx.fillStyle = mode === 'plain' ? SOL.red : SOL.green;
      ctx.font = 'bold 11.5px sans-serif';
      ctx.fillText(
        mode === 'plain' ? 'Plain-56: Chaotic Local Traps' : 'ResNet-56: Smooth Convex-Like Basin',
        boxX + 12,
        boxY + 21
      );
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      const { plain, highway, resnet, tGate } = computeProfiles();
      if (valDepth) valDepth.textContent = String(state.L);
      if (valBias) valBias.textContent = state.bT.toFixed(2);
      if (valWscale) valWscale.textContent = state.sigmaW.toFixed(2);

      const fmtLog = (v) => (Math.abs(v) < 0.05 ? '10^0.0' : `10^${v.toFixed(1)}`);
      if (readPlain) readPlain.textContent = fmtLog(plain[0]);
      if (readHw) readHw.textContent = fmtLog(highway[0]);
      if (readRes) readRes.textContent = fmtLog(resnet[0]);

      // Left Panel: Reverse-Mode AD Gradient Flow Curve
      const leftW = 738;
      drawRoundRect(ctx, 16, 12, leftW, 426, 8, SOL.base2, SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14.5px sans-serif';
      ctx.fillText(
        `Reverse-Mode AD Gradient Norm Across Depth (L=${state.L}, T(b_T)=${tGate.toFixed(2)}, Carry (1-T)=${(1 - tGate).toFixed(2)})`,
        32, 36
      );

      const plot = { x: 84, y: 54, w: 644, h: 320 };
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);

      const yMin = -12;
      const yMax = 4;
      const mapX = (lIdx) => plot.x + (lIdx / Math.max(1, state.L - 1)) * plot.w;
      const mapY = (logVal) => {
        const clamped = Math.max(yMin, Math.min(yMax, logVal));
        return plot.y + plot.h - ((clamped - yMin) / (yMax - yMin)) * plot.h;
      };

      // Horizontal grid lines
      for (let yTick = yMin; yTick <= yMax; yTick += 2) {
        const py = mapY(yTick);
        ctx.strokeStyle = yTick === 0 ? SOL.base01 : '#e4dec9';
        ctx.lineWidth = yTick === 0 ? 1.4 : 0.8;
        ctx.beginPath();
        ctx.moveTo(plot.x, py);
        ctx.lineTo(plot.x + plot.w, py);
        ctx.stroke();

        ctx.fillStyle = SOL.base01;
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`10^${yTick}`, plot.x - 8, py + 4);
      }

      // Vertical ticks
      ctx.textAlign = 'center';
      const stepTick = state.L <= 24 ? 4 : (state.L <= 60 ? 10 : 20);
      for (let l = 1; l <= state.L; l += stepTick) {
        const px = mapX(l - 1);
        ctx.strokeStyle = '#eee8d5';
        ctx.beginPath();
        ctx.moveTo(px, plot.y);
        ctx.lineTo(px, plot.y + plot.h);
        ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.fillText(String(l), px, plot.y + plot.h + 16);
      }
      ctx.fillText(String(state.L), mapX(state.L - 1), plot.y + plot.h + 16);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12.5px sans-serif';
      ctx.fillText('Layer Index ℓ  (Input ℓ=1  ←  Backprop Direction  ←  Loss at ℓ=L)', plot.x + plot.w / 2, plot.y + plot.h + 38);

      ctx.save();
      ctx.translate(28, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText('log₁₀ ||∂L / ∂x_ℓ||', 0, 0);
      ctx.restore();
      ctx.textAlign = 'left';

      // Shaded vanishing / exploding zones
      ctx.fillStyle = 'rgba(220, 50, 47, 0.08)';
      ctx.fillRect(plot.x, mapY(-6), plot.w, mapY(yMin) - mapY(-6));
      ctx.fillStyle = SOL.red;
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText('Vanishing Gradient Zone (< 10⁻⁶)', plot.x + 12, mapY(-11));

      function plotSeries(arr, color, dash) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        if (dash) ctx.setLineDash(dash);
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
          const px = mapX(i);
          const py = mapY(arr[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.restore();
      }

      plotSeries(plain, SOL.red, null);
      plotSeries(highway, SOL.blue, [8, 4]);
      plotSeries(resnet, SOL.green, null);

      // Legend inside plot top-left
      drawRoundRect(ctx, plot.x + 12, plot.y + 10, 405, 66, 6, 'rgba(253,246,227,0.94)', SOL.base1, 1);
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.red;
      ctx.fillText('━━  Plain Net:  ∏ (∂f_k / ∂x_k)  → collapses or explodes', plot.x + 22, plot.y + 29);
      ctx.fillStyle = SOL.blue;
      ctx.fillText('┅┅  Highway Net: T·∂H + (1-T)·I  (depends on gate bias b_T)', plot.x + 22, plot.y + 48);
      ctx.fillStyle = SOL.green;
      ctx.fillText('━━  ResNet:  I + ∂F/∂x  → clean identity gradient highway!', plot.x + 22, plot.y + 67);

      // Right Panel: Exact 1:1 Square Loss Landscapes (Li et al. 2018)
      const rightX = 770;
      const rightW = 394;
      drawRoundRect(ctx, rightX, 12, rightW, 426, 8, SOL.base2, SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.fillText('Loss Surface Effect (Li et al., NeurIPS 2018)', rightX + 14, 34);

      const sqSide = 186; // exact 1:1 square
      const sqX = rightX + Math.round((rightW - sqSide) / 2);
      drawMiniLandscape(ctx, sqX, 46, sqSide, 'plain');
      drawMiniLandscape(ctx, sqX, 244, sqSide, 'resnet');

      if (banner) {
        if (state.bT > 0.25) {
          banner.textContent = `Warning: positive gate bias b_T = +${state.bT.toFixed(2)} shuts the carry gate (1-T=${(1 - tGate).toFixed(2)}), choking Highway gradients just like a Plain Net!`;
        } else if (state.sigmaW > 1.15) {
          banner.textContent = `Large weight scale σ_w=${state.sigmaW.toFixed(2)} drives Plain Net gradients into explosion while ResNet stays well-conditioned!`;
        } else {
          banner.textContent = `At L=${state.L}: Plain Net early-layer gradient collapses to 10^${plain[0].toFixed(1)}, while ResNet preserves 10^${resnet[0].toFixed(1)} via +I shortcut!`;
        }
      }
    }

    bindRange(sliderDepth, () => { state.L = parseInt(sliderDepth.value, 10); render(); });
    bindRange(sliderBias, () => { state.bT = parseFloat(sliderBias.value); render(); });
    bindRange(sliderWscale, () => { state.sigmaW = parseFloat(sliderWscale.value); render(); });
    if (btnBadGate) {
      stopSwipePropagation(btnBadGate);
      btnBadGate.addEventListener('click', () => {
        state.bT = 2.0;
        if (sliderBias) sliderBias.value = '2.0';
        render();
      });
    }
    if (btnGoodGate) {
      stopSwipePropagation(btnGoodGate);
      btnGoodGate.addEventListener('click', () => {
        state.bT = -3.0;
        if (sliderBias) sliderBias.value = '-3.0';
        render();
      });
    }
    if (btnReset) {
      stopSwipePropagation(btnReset);
      btnReset.addEventListener('click', () => {
        state.L = 56; if (sliderDepth) sliderDepth.value = '56';
        state.bT = -2.5; if (sliderBias) sliderBias.value = '-2.5';
        state.sigmaW = 0.85; if (sliderWscale) sliderWscale.value = '0.85';
        render();
      });
    }

    registerSlideListener(canvas, render);
    render();
  }

  // =========================================================================
  // 4. CAUSAL MASKED (PIXELCNN) & DEFORMABLE CONVOLUTION EXPLORER
  // =========================================================================
  function initAdvancedDemo() {
    const canvas = document.getElementById('cnn-advanced-canvas');
    if (!canvas) return;

    const banner = reqEl('cnn-adv-banner', 'Demo 4');
    const selMode = reqEl('cnn-adv-mode', 'Demo 4');
    const selStack = reqEl('cnn-adv-stack', 'Demo 4');
    const sliderLayers = reqEl('cnn-adv-layers', 'Demo 4');
    const valLayers = reqEl('cnn-adv-layers-val', 'Demo 4');
    const sliderAngle = reqEl('cnn-adv-angle', 'Demo 4');
    const valAngle = reqEl('cnn-adv-angle-val', 'Demo 4');
    const sliderScale = reqEl('cnn-adv-scale', 'Demo 4');
    const valScale = reqEl('cnn-adv-scale-val', 'Demo 4');
    const btnStep = reqEl('cnn-adv-step', 'Demo 4');
    const btnReset = reqEl('cnn-adv-reset', 'Demo 4');

    const N = 11;
    const state = {
      mode: selMode ? selMode.value : 'pixelcnn',
      stack: selStack ? selStack.value : 'standard',
      layers: sliderLayers ? parseInt(sliderLayers.value, 10) : 3,
      targetR: 6,
      targetC: 5,
      angleDeg: sliderAngle ? parseFloat(sliderAngle.value) : 35,
      scale: sliderScale ? parseFloat(sliderScale.value) : 1.25,
      showRigid: true
    };

    let gridHit = null;

    function traceReach(stackType) {
      if (stackType === 'standard') {
        let reachable = Array.from({ length: N }, () => Array(N).fill(false));
        reachable[state.targetR][state.targetC] = true;
        for (let l = state.layers; l >= 1; l--) {
          const next = Array.from({ length: N }, () => Array(N).fill(false));
          const isFirstLayer = l === 1;
          for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
              if (!reachable[r][c]) continue;
              for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                  const allow = (dr < 0) || (dr === 0 && (isFirstLayer ? dc < 0 : dc <= 0));
                  if (allow) {
                    const pr = r + dr;
                    const pc = c + dc;
                    if (pr >= 0 && pr < N && pc >= 0 && pc < N) next[pr][pc] = true;
                  }
                }
              }
            }
          }
          reachable = next;
        }
        return reachable;
      }

      // Gated Vertical + Horizontal Stack (van den Oord et al., NeurIPS 2016):
      // Vertical stack V_l propagates rows strictly above targetR across all horizontal offsets [-1, 0, +1];
      // Horizontal stack H_l covers current row targetR to the left of targetC.
      const L = state.layers;
      const reached = Array.from({ length: N }, () => Array(N).fill(false));
      for (let r = Math.max(0, state.targetR - L); r < state.targetR; r++) {
        for (let c = Math.max(0, state.targetC - L); c <= Math.min(N - 1, state.targetC + L); c++) {
          reached[r][c] = true;
        }
      }
      for (let c = Math.max(0, state.targetC - L); c < state.targetC; c++) {
        reached[state.targetR][c] = true;
      }
      return reached;
    }

    function computePixelCnnReach() {
      return {
        reach: traceReach(state.stack),
        idealReach: traceReach('vh')
      };
    }

    function drawMaskGrid(ctx, ox, oy, cell, title, includeCenter) {
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12.5px sans-serif';
      ctx.fillText(title, ox, oy - 8);
      for (let u = 0; u < 3; u++) {
        for (let v = 0; v < 3; v++) {
          const isPast = (u < 1) || (u === 1 && v < 1);
          const isCenter = (u === 1 && v === 1);
          const val = isPast ? 1 : (isCenter ? (includeCenter ? 1 : 0) : 0);
          const cx = ox + v * cell;
          const cy = oy + u * cell;
          ctx.fillStyle = val === 1 ? '#d7e6ba' : (isCenter ? '#f8d7da' : '#e6e0cc');
          ctx.fillRect(cx, cy, cell, cell);
          ctx.strokeStyle = SOL.base01;
          ctx.lineWidth = 1.2;
          ctx.strokeRect(cx, cy, cell, cell);
          ctx.fillStyle = val === 1 ? SOL.green : SOL.red;
          ctx.font = 'bold 16px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(val), cx + cell / 2, cy + cell / 2);
        }
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
    }

    function renderPixelCNN(ctx, width, height) {
      const leftBox = { x: 18, y: 12, w: 460, h: 426 };
      const rightBox = { x: 494, y: 12, w: 668, h: 426 };
      drawRoundRect(ctx, leftBox.x, leftBox.y, leftBox.w, leftBox.h, 8, SOL.base2, SOL.base1, 1);
      drawRoundRect(ctx, rightBox.x, rightBox.y, rightBox.w, rightBox.h, 8, SOL.base2, SOL.base1, 1);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`1. Raster-Scan Autoregressive RF (${N}×${N}, Click Target Pixel)`, leftBox.x + 14, leftBox.y + 24);

      const { reach, idealReach } = computePixelCnnReach();
      const cell = 32; // exact 1:1 square
      const side = cell * N;
      const ox = leftBox.x + Math.round((leftBox.w - side) / 2);
      const oy = leftBox.y + 42;
      gridHit = { ox, oy, cell, N };

      let totalIdeal = 0;
      let capturedPast = 0;
      let blindSpotCount = 0;

      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          const isTarget = (r === state.targetR && c === state.targetC);
          const isPast = (r < state.targetR) || (r === state.targetR && c < state.targetC);
          const isIdeal = isPast && idealReach[r][c];
          const isCaptured = isPast && reach[r][c];
          const isBlind = isIdeal && !isCaptured;
          if (isIdeal) totalIdeal++;
          if (isCaptured) capturedPast++;
          if (isBlind) blindSpotCount++;

          const cx = ox + c * cell;
          const cy = oy + r * cell;
          let fill = '#e5dfca'; // future
          if (isTarget) fill = SOL.orange;
          else if (isCaptured) fill = '#b8d8be'; // captured past
          else if (isBlind) fill = '#f5d487'; // blind spot!
          else if (isPast) fill = '#eee9d6'; // past outside L-layer RF

          ctx.fillStyle = fill;
          ctx.fillRect(cx, cy, cell, cell);
          ctx.strokeStyle = SOL.base1;
          ctx.lineWidth = 0.8;
          ctx.strokeRect(cx, cy, cell, cell);

          if (isTarget) {
            ctx.fillStyle = SOL.base3;
            ctx.font = 'bold 13px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('xᵢ', cx + cell / 2, cy + cell / 2);
          } else if (isBlind) {
            ctx.fillStyle = SOL.orange;
            ctx.font = 'bold 12px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('?', cx + cell / 2, cy + cell / 2);
          }
        }
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';

      // Legend below left grid
      const ly = oy + side + 18;
      ctx.font = '11.5px sans-serif';
      const legItems = [
        ['#b8d8be', 'Captured RF'],
        ['#f5d487', 'Blind Spot (?)'],
        [SOL.orange, 'Target xᵢ'],
        ['#e5dfca', 'Masked Future']
      ];
      let lx = leftBox.x + 16;
      legItems.forEach(([col, lbl]) => {
        ctx.fillStyle = col;
        ctx.fillRect(lx, ly - 10, 12, 12);
        ctx.strokeStyle = SOL.base01;
        ctx.strokeRect(lx, ly - 10, 12, 12);
        ctx.fillStyle = SOL.base02;
        ctx.fillText(lbl, lx + 16, ly);
        lx += 112;
      });

      // Right Panel: Mask A vs Mask B + Blind Spot Explanation + Metrics
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14.5px sans-serif';
      ctx.fillText('2. Mask A vs. Mask B & Why Standard 3×3 Stacks Have a Blind Spot', rightBox.x + 16, rightBox.y + 26);

      drawMaskGrid(ctx, rightBox.x + 28, rightBox.y + 72, 46, 'Mask A (Layer 1: center = 0)', false);
      drawMaskGrid(ctx, rightBox.x + 235, rightBox.y + 72, 46, 'Mask B (Layers 2..L: center = 1)', true);

      drawRoundRect(ctx, rightBox.x + 425, rightBox.y + 58, 224, 152, 6, SOL.base3, SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(`Causal Coverage (L=${state.layers})`, rightBox.x + 439, rightBox.y + 80);
      ctx.font = '12.5px monospace';
      ctx.fillStyle = SOL.green;
      ctx.fillText(`Captured Past: ${capturedPast}/${totalIdeal}`, rightBox.x + 439, rightBox.y + 108);
      ctx.fillStyle = blindSpotCount > 0 && state.stack === 'standard' ? SOL.red : SOL.base01;
      ctx.fillText(`Blind-Spot Cells: ${blindSpotCount}`, rightBox.x + 439, rightBox.y + 132);
      ctx.fillStyle = SOL.blue;
      ctx.fillText(`Future Leakage: 0 (Exact)`, rightBox.x + 439, rightBox.y + 156);
      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.fillText(
        state.stack === 'standard' ? 'Standard 3×3 misses upper-right!' : 'V+H stack covers upper-right rows.',
        rightBox.x + 439,
        rightBox.y + 186
      );

      // Equations & Training vs Sampling note
      drawRoundRect(ctx, rightBox.x + 20, rightBox.y + 232, 628, 178, 6, SOL.base3, SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.fillText('Autoregressive Factorization & Training vs. Sampling Speed', rightBox.x + 34, rightBox.y + 258);
      ctx.font = '13px monospace';
      ctx.fillStyle = SOL.blue;
      ctx.fillText('p(x) = ∏ᵢ p(xᵢ | x₁, …, xᵢ₋₁)   in raster scan order', rightBox.x + 34, rightBox.y + 288);
      ctx.fillStyle = SOL.base02;
      ctx.font = '12.5px sans-serif';
      ctx.fillText('• Parallel Training: One masked forward pass evaluates all N×N conditionals simultaneously!', rightBox.x + 34, rightBox.y + 320);
      ctx.fillText('• Sequential Sampling: Generating an image requires N×N forward passes, pixel by pixel.', rightBox.x + 34, rightBox.y + 346);
      ctx.fillStyle = state.stack === 'standard' ? SOL.orange : SOL.green;
      ctx.font = 'bold 12.5px sans-serif';
      ctx.fillText(
        state.stack === 'standard'
          ? '• Try selecting "V+H Gated Stack (Fixed!)" above to eliminate the amber ? blind spot.'
          : '• Gated Vertical + Horizontal Stack captures the upper-right cone with zero future leakage!',
        rightBox.x + 34,
        rightBox.y + 378
      );

      if (banner) {
        banner.textContent = state.stack === 'standard'
          ? `PixelCNN (Standard 3×3, L=${state.layers}): Zero future leakage, but ${blindSpotCount} upper-right past pixels fall into the blind spot!`
          : `Gated V+H PixelCNN (L=${state.layers}): Vertical + Horizontal stacks capture upper-right past pixels while keeping zero future leakage!`;
      }
    }

    function renderDeformable(ctx, width, height) {
      gridHit = null;
      const leftBox = { x: 18, y: 12, w: 690, h: 426 };
      const rightBox = { x: 724, y: 12, w: 438, h: 426 };
      drawRoundRect(ctx, leftBox.x, leftBox.y, leftBox.w, leftBox.h, 8, SOL.base2, SOL.base1, 1);
      drawRoundRect(ctx, rightBox.x, rightBox.y, rightBox.w, rightBox.h, 8, SOL.base2, SOL.base1, 1);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(
        `1. Deformable 3×3 Sampling Offsets (Rotation θ=${state.angleDeg}°, Scale s=${state.scale.toFixed(2)})`,
        leftBox.x + 14,
        leftBox.y + 24
      );

      // Exact 1:1 square coordinate canvas
      const sqSide = 360;
      const ox = leftBox.x + Math.round((leftBox.w - sqSide) / 2);
      const oy = leftBox.y + 42;
      const G = 9;
      const cell = sqSide / G;

      ctx.fillStyle = SOL.base3;
      ctx.fillRect(ox, oy, sqSide, sqSide);

      // Draw rotated/scaled object shape (smooth ellipse) centered at (4.5, 4.5) in grid coords
      const rad = (state.angleDeg * Math.PI) / 180;
      const cxPix = ox + 4.5 * cell;
      const cyPix = oy + 4.5 * cell;
      ctx.save();
      ctx.translate(cxPix, cyPix);
      ctx.rotate(rad);
      ctx.fillStyle = 'rgba(38, 139, 210, 0.18)';
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.ellipse(0, 0, 2.3 * state.scale * cell, 0.95 * state.scale * cell, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Grid lines (1:1 squares)
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.25)';
      ctx.lineWidth = 0.8;
      for (let i = 0; i <= G; i++) {
        ctx.beginPath(); ctx.moveTo(ox + i * cell, oy); ctx.lineTo(ox + i * cell, oy + sqSide); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ox, oy + i * cell); ctx.lineTo(ox + sqSide, oy + i * cell); ctx.stroke();
      }
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.4;
      ctx.strokeRect(ox, oy, sqSide, sqSide);

      // Compute rigid 3x3 points and deformed points p_0 + p_n + \Delta p_n
      let samplePt = null;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const rx = cxPix + dc * cell;
          const ry = cyPix + dr * cell;

          // Learned offset tracks rotated & scaled ellipse axes
          const lx = dc * 1.45 * state.scale;
          const ly = dr * 0.65 * state.scale;
          const dxRot = lx * Math.cos(rad) - ly * Math.sin(rad);
          const dyRot = lx * Math.sin(rad) + ly * Math.cos(rad);
          const tx = cxPix + dxRot * cell;
          const ty = cyPix + dyRot * cell;

          if (state.showRigid) {
            ctx.fillStyle = SOL.base1;
            ctx.beginPath();
            ctx.arc(rx, ry, 5, 0, Math.PI * 2);
            ctx.fill();
            drawArrow(ctx, rx, ry, tx, ty, SOL.orange, 2, 7);
          }

          ctx.fillStyle = (dr === 0 && dc === 0) ? SOL.red : SOL.green;
          ctx.beginPath();
          ctx.arc(tx, ty, 6.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = SOL.base03;
          ctx.lineWidth = 1.2;
          ctx.stroke();

          if (dr === -1 && dc === 1) {
            samplePt = {
              gx: 4.5 + dxRot,
              gy: 4.5 + dyRot
            };
          }
        }
      }

      // ----- RIGHT PANEL: Bilinear Interpolation Zoom (1:1 Square) -----
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('2. Differentiable Fractional Bilinear Sampling', rightBox.x + 14, rightBox.y + 24);

      const sp = samplePt || { gx: 5.4, gy: 3.7 };
      const fx = sp.gx - Math.floor(sp.gx);
      const fy = sp.gy - Math.floor(sp.gy);
      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      const zSide = 180; // exact 1:1 square
      const zx = rightBox.x + Math.round((rightBox.w - zSide) / 2);
      const zy = rightBox.y + 60;
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(zx, zy, zSide, zSide);
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(zx, zy, zSide, zSide);

      const px = zx + fx * zSide;
      const py = zy + fy * zSide;

      // Crosshairs to 4 integer neighbors
      ctx.strokeStyle = SOL.blue;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(px, zy); ctx.lineTo(px, zy + zSide); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(zx, py); ctx.lineTo(zx + zSide, py); ctx.stroke();
      ctx.setLineDash([]);

      // Draw 4 corner weights
      const corners = [
        [zx, zy, `q₀₀: ${w00.toFixed(2)}`, -4, -6],
        [zx + zSide, zy, `q₁₀: ${w10.toFixed(2)}`, -44, -6],
        [zx, zy + zSide, `q₀₁: ${w01.toFixed(2)}`, -4, 16],
        [zx + zSide, zy + zSide, `q₁₁: ${w11.toFixed(2)}`, -44, 16]
      ];
      corners.forEach(([cx0, cy0, lbl, dxL, dyL]) => {
        ctx.fillStyle = SOL.blue;
        ctx.beginPath();
        ctx.arc(cx0, cy0, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 11.5px monospace';
        ctx.fillText(lbl, cx0 + dxL, cy0 + dyL);
      });

      // Fractional point p = p0 + pn + Delta pn
      ctx.fillStyle = SOL.green;
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = SOL.base03;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      drawRoundRect(ctx, rightBox.x + 16, rightBox.y + 276, rightBox.w - 32, 134, 6, SOL.base3, SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12.5px monospace';
      ctx.fillText('y(p₀) = ∑ w(pₙ) · x(p₀ + pₙ + Δpₙ)', rightBox.x + 28, rightBox.y + 300);
      ctx.fillStyle = SOL.blue;
      ctx.font = '12px sans-serif';
      ctx.fillText('Because Δpₙ is continuous (fractional), x(p) is computed', rightBox.x + 28, rightBox.y + 328);
      ctx.fillText('via bilinear interpolation of its 4 integer neighbors:', rightBox.x + 28, rightBox.y + 348);
      ctx.fillStyle = SOL.orange;
      ctx.font = 'bold 12px monospace';
      ctx.fillText(`x(p) = ${w00.toFixed(2)}·q₀₀ + ${w10.toFixed(2)}·q₁₀ + ${w01.toFixed(2)}·q₀₁ + ${w11.toFixed(2)}·q₁₁`, rightBox.x + 28, rightBox.y + 374);
      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.fillText('This makes ∂x/∂Δpₙ piecewise smooth for reverse-mode AD!', rightBox.x + 28, rightBox.y + 398);

      if (banner) {
        banner.textContent = `Deformable Conv (θ=${state.angleDeg}°, s=${state.scale.toFixed(2)}): Learned offsets Δpₙ adapt receptive field geometry & backprop via bilinear interpolation!`;
      }
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);
      if (valLayers) valLayers.textContent = String(state.layers);
      if (valAngle) valAngle.textContent = `${state.angleDeg >= 0 ? '+' : ''}${state.angleDeg}°`;
      if (valScale) valScale.textContent = state.scale.toFixed(2);
      if (selMode && selMode.value !== state.mode) selMode.value = state.mode;
      if (btnStep) {
        btnStep.textContent = state.mode === 'pixelcnn' ? 'Step Pixel xᵢ' : (state.showRigid ? 'Hide Rigid 3×3' : 'Show Rigid 3×3');
      }

      if (state.mode === 'pixelcnn') {
        renderPixelCNN(ctx, width, height);
      } else {
        renderDeformable(ctx, width, height);
      }
    }

    if (selMode) {
      stopSwipePropagation(selMode);
      selMode.addEventListener('change', () => { state.mode = selMode.value; render(); });
    }
    if (selStack) {
      stopSwipePropagation(selStack);
      selStack.addEventListener('change', () => {
        state.mode = 'pixelcnn';
        state.stack = selStack.value;
        render();
      });
    }
    bindRange(sliderLayers, () => {
      state.mode = 'pixelcnn';
      state.layers = parseInt(sliderLayers.value, 10);
      render();
    });
    bindRange(sliderAngle, () => {
      state.mode = 'deformable';
      state.angleDeg = parseFloat(sliderAngle.value);
      render();
    });
    bindRange(sliderScale, () => {
      state.mode = 'deformable';
      state.scale = parseFloat(sliderScale.value);
      render();
    });

    if (btnStep) {
      stopSwipePropagation(btnStep);
      btnStep.addEventListener('click', () => {
        if (state.mode === 'pixelcnn') {
          state.targetC += 1;
          if (state.targetC >= N) {
            state.targetC = 0;
            state.targetR = (state.targetR + 1) % N;
            if (state.targetR === 0) state.targetR = 1;
          }
        } else {
          state.showRigid = !state.showRigid;
        }
        render();
      });
    }
    if (btnReset) {
      stopSwipePropagation(btnReset);
      btnReset.addEventListener('click', () => {
        state.targetR = 6;
        state.targetC = 5;
        state.layers = 3; if (sliderLayers) sliderLayers.value = '3';
        state.stack = 'standard'; if (selStack) selStack.value = 'standard';
        state.angleDeg = 35; if (sliderAngle) sliderAngle.value = '35';
        state.scale = 1.25; if (sliderScale) sliderScale.value = '1.25';
        state.showRigid = true;
        render();
      });
    }

    stopSwipePropagation(canvas);
    canvas.addEventListener('click', (e) => {
      if (state.mode === 'pixelcnn' && gridHit) {
        const pt = canvasPoint(canvas, e.clientX, e.clientY);
        const c = Math.floor((pt.x - gridHit.ox) / gridHit.cell);
        const r = Math.floor((pt.y - gridHit.oy) / gridHit.cell);
        if (r >= 0 && r < gridHit.N && c >= 0 && c < gridHit.N) {
          state.targetR = r;
          state.targetC = c;
          render();
        }
      }
    });

    registerSlideListener(canvas, render);
    render();
  }

  // =========================================================================
  // 0B. ANIMATED CONVOLUTION VS. CROSS-CORRELATION VISUALIZER
  // =========================================================================
  function initXcorrAnimDemo() {
    const canvas = document.getElementById('cnn-xcorr-anim-canvas');
    if (!canvas) return;

    const btnPlay = reqEl('cnn-xc-play', 'Xcorr Anim');
    const sliderPos = reqEl('cnn-xc-pos', 'Xcorr Anim');
    const valPos = reqEl('cnn-xc-pos-val', 'Xcorr Anim');
    const selKer = reqEl('cnn-xc-ker', 'Xcorr Anim');

    const N = 15;
    // 1D Signal with two clear landmarks:
    // n=3: Unit Impulse delta[n-3] = 1.0
    // n=9..11: Asymmetric Rising Ramp [0.2, 0.5, 1.0]
    const SIGNAL = [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0.2, 0.5, 1.0, 0, 0, 0];

    const KERNELS = {
      ramp: {
        name: 'Asymmetric Ramp [0.2, 0.5, 1.0]',
        k: [0.2, 0.5, 1.0] // indices u = -1, 0, +1
      },
      edge: {
        name: 'Asymmetric Edge [-1.0, 0.0, +1.0]',
        k: [-1.0, 0.0, 1.0]
      }
    };

    const state = {
      pos: sliderPos ? parseInt(sliderPos.value, 10) : 3,
      kerKey: selKer ? selKer.value : 'ramp',
      playing: true,
      timer: null
    };

    function computeTraces() {
      const K = KERNELS[state.kerKey].k;
      const conv = new Array(N).fill(0);
      const xcorr = new Array(N).fill(0);
      for (let n = 0; n < N; n++) {
        let cSum = 0;
        let xSum = 0;
        for (let idx = 0; idx < 3; idx++) {
          const u = idx - 1; // -1, 0, +1
          const iConv = n - u; // convolution: I[n - u] * K[u] -> flips K
          const iXcorr = n + u; // cross-corr:  I[n + u] * K[u] -> unflipped K
          if (iConv >= 0 && iConv < N) cSum += SIGNAL[iConv] * K[idx];
          if (iXcorr >= 0 && iXcorr < N) xSum += SIGNAL[iXcorr] * K[idx];
        }
        conv[n] = cSum;
        xcorr[n] = xSum;
      }
      return { K, conv, xcorr };
    }

    function isSlideActive() {
      if (!window.Reveal || !Reveal.getCurrentSlide) return true;
      const cur = Reveal.getCurrentSlide();
      return cur && cur.contains(canvas);
    }

    function render() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.fillStyle = SOL.base3;
      ctx.fillRect(0, 0, width, height);

      const { K, conv, xcorr } = computeTraces();
      const nCur = Math.max(0, Math.min(N - 1, state.pos));
      if (valPos) valPos.textContent = String(nCur);
      if (sliderPos && parseInt(sliderPos.value, 10) !== nCur) sliderPos.value = String(nCur);
      if (btnPlay) {
        btnPlay.textContent = state.playing ? 'Pause' : 'Play';
        btnPlay.style.background = state.playing ? SOL.orange : SOL.green;
      }

      const padL = 42;
      const padR = 18;
      const plotW = width - padL - padR;
      const stepX = plotW / N;
      const xCenter = (n) => padL + (n + 0.5) * stepX;

      const bands = [
        { y: 10, h: 118, title: '1. Input Signal I[n]  (Impulse at n=3,  Ramp at n=9..11)', color: SOL.base02, bg: SOL.base2 },
        { y: 138, h: 120, title: `2. Convolution (I * K)[${nCur}] = ${conv[nCur].toFixed(2)}  [Flipped: ${K.slice().reverse().map((v) => v.toFixed(1)).join(', ')}]`, color: SOL.violet, bg: '#f3efe0' },
        { y: 268, h: 122, title: `3. Cross-Corr (I ★ K)[${nCur}] = ${xcorr[nCur].toFixed(2)}  [Unflipped: ${K.map((v) => v.toFixed(1)).join(', ')}]`, color: SOL.green, bg: '#f3efe0' }
      ];

      bands.forEach((b) => {
        drawRoundRect(ctx, 10, b.y, width - 20, b.h, 6, b.bg, SOL.base1, 1);
        ctx.fillStyle = b.color;
        ctx.font = 'bold 11.5px sans-serif';
        ctx.fillText(b.title, 18, b.y + 17);
      });

      // Right-aligned dynamic insight badges on the header line of Bands 1 & 2 (never collide with stems!)
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'right';
      if (nCur >= 2 && nCur <= 4) {
        ctx.fillStyle = SOL.violet;
        ctx.fillText('Impulse Response = K (upright)', width - 18, bands[1].y + 17);
        ctx.fillStyle = SOL.green;
        ctx.fillText('Impulse Response = K_flip', width - 18, bands[2].y + 17);
      } else if (nCur >= 9 && nCur <= 11) {
        ctx.fillStyle = SOL.violet;
        ctx.fillText(state.kerKey === 'ramp' ? 'Flipped kernel misaligns (0.65)' : 'Derivative sign inverted', width - 18, bands[1].y + 17);
        ctx.fillStyle = SOL.green;
        ctx.fillText(state.kerKey === 'ramp' ? 'Matched filter peaks (1.29)!' : 'Matched template orientation!', width - 18, bands[2].y + 17);
      }
      ctx.textAlign = 'left';

      // Highlight sliding 3-tap window across Track 0
      const winLeft = padL + Math.max(0, nCur - 1) * stepX;
      const winRight = padL + Math.min(N, nCur + 2) * stepX;
      ctx.fillStyle = 'rgba(203, 75, 22, 0.12)';
      ctx.fillRect(winLeft, bands[0].y + 24, winRight - winLeft, bands[0].h - 32);
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(winLeft, bands[0].y + 24, winRight - winLeft, bands[0].h - 32);

      // Helper to draw a 1D stem chart cleanly
      function drawStemTrack(band, values, minV, maxV, activeColor, highlightLimit, isInputTrack) {
        const basePlotY = band.y + (isInputTrack ? 42 : 34);
        const plotH = band.h - (isInputTrack ? 58 : 50);
        const zeroY = basePlotY + plotH * (maxV / (maxV - minV));

        // Zero axis
        ctx.strokeStyle = SOL.base1;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL - 4, zeroY);
        ctx.lineTo(padL + plotW, zeroY);
        ctx.stroke();

        for (let n = 0; n < N; n++) {
          const cx = xCenter(n);
          const v = values[n];
          const py = basePlotY + plotH * ((maxV - v) / (maxV - minV));
          const drawn = highlightLimit === null || n <= highlightLimit;

          // Index tick on bottom of band
          ctx.fillStyle = n === nCur ? SOL.orange : SOL.base01;
          ctx.font = n === nCur ? 'bold 10.5px monospace' : '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(String(n), cx, band.y + band.h - 5);

          if (!drawn) continue;

          ctx.strokeStyle = n === nCur ? SOL.orange : activeColor;
          ctx.lineWidth = n === nCur ? 3.2 : 2.2;
          ctx.beginPath();
          ctx.moveTo(cx, zeroY);
          ctx.lineTo(cx, py);
          ctx.stroke();

          ctx.fillStyle = n === nCur ? SOL.orange : activeColor;
          ctx.beginPath();
          ctx.arc(cx, py, n === nCur ? 4.8 : 3.5, 0, Math.PI * 2);
          ctx.fill();

          // Numeric labels only where clean: for input, show compact values below dot or offset;
          // for output tracks, show active nCur or key peaks cleanly
          if (Math.abs(v) > 0.05 && (isInputTrack || n === nCur || Math.abs(v) > 0.9)) {
            ctx.fillStyle = n === nCur ? SOL.orange : activeColor;
            ctx.font = 'bold 9.5px monospace';
            const lblY = v >= 0 ? Math.max(band.y + (isInputTrack ? 38 : 30), py - 6) : Math.min(band.y + band.h - 17, py + 13);
            ctx.fillText(v.toFixed(1), cx, lblY);
          }
        }
        ctx.textAlign = 'left';
      }

      drawStemTrack(bands[0], SIGNAL, -0.15, 1.25, SOL.blue, null, true);

      // Put flipped vs unflipped sliding tap weights cleanly in the top-right badge of Track 0
      const kRev = K.slice().reverse();
      ctx.font = 'bold 10.5px monospace';
      ctx.textAlign = 'right';
      ctx.fillStyle = SOL.violet;
      ctx.fillText(`*:[${kRev.map((v) => v.toFixed(1)).join(',')}]`, width - 126, bands[0].y + 17);
      ctx.fillStyle = SOL.green;
      ctx.fillText(`★:[${K.map((v) => v.toFixed(1)).join(',')}]`, width - 18, bands[0].y + 17);
      ctx.textAlign = 'left';

      const minOut = state.kerKey === 'edge' ? -1.4 : -0.2;
      const maxOut = 1.5;
      drawStemTrack(bands[1], conv, minOut, maxOut, SOL.violet, nCur, false);
      drawStemTrack(bands[2], xcorr, minOut, maxOut, SOL.green, nCur, false);
    }

    function startTimer() {
      if (state.timer) clearInterval(state.timer);
      state.timer = setInterval(() => {
        if (!state.playing || !isSlideActive()) return;
        state.pos = (state.pos + 1) % N;
        render();
      }, 650);
    }

    if (btnPlay) {
      stopSwipePropagation(btnPlay);
      btnPlay.addEventListener('click', () => {
        state.playing = !state.playing;
        render();
      });
    }
    bindRange(sliderPos, () => {
      state.playing = false;
      state.pos = parseInt(sliderPos.value, 10);
      render();
    });
    if (selKer) {
      stopSwipePropagation(selKer);
      selKer.addEventListener('change', () => {
        state.kerKey = selKer.value;
        render();
      });
    }
    stopSwipePropagation(canvas);
    canvas.addEventListener('click', (e) => {
      const pt = canvasPoint(canvas, e.clientX, e.clientY);
      const padL = 46;
      const plotW = canvas._logicalW - padL - 16;
      const nClicked = Math.floor(((pt.x - padL) / plotW) * N);
      if (nClicked >= 0 && nClicked < N) {
        state.playing = false;
        state.pos = nClicked;
        render();
      }
    });

    registerSlideListener(canvas, render);
    startTimer();
    render();
  }

  // =========================================================================
  // BOOTSTRAP ALL DEMOS ON DOMContentLoaded
  // =========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    initBridgeSlide();
    initXcorrAnimDemo();
    initConvDemo();
    initReceptiveFieldDemo();
    initSkipDemo();
    initAdvancedDemo();
  });
})();
