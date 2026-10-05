/**
 * CS8850: Advanced Machine Learning - Lecture 13 (Support Vector Machines)
 * Interactive Pedagogical Demonstrations Suite
 * 
 * 1. Dynamic Bridge Slide Connectors (Precision SVG alignment)
 * 2. Decision Surface & Margin Band Explorer (Geometry, Scale Invariance, Snap-to-Optimal)
 * 3. Interactive Lagrange Multipliers & Stationary Gradient Alignment
 * 4. Primal-Dual Saddle Explorer for Example 1 (min x^2 s.t. x >= b)
 * 5. Interactive 2D Kernel SVM Playground (Live in-browser SMO, Linear/Poly/RBF, Support Vectors)
 * 6. Loss Function Zoo (0-1 vs Hinge vs Logistic vs Squared Loss & Sparsity)
 */

(function() {
  'use strict';

  // Solarized Color Palette
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

  /**
   * High-DPI canvas setup with permanent logical dimension caching.
   * Prevents canvas.width DOM attribute reflection from doubling backing-store size.
   */
  function setupHiDPI(canvas) {
    const dpr = window.devicePixelRatio || 1;
    if (!canvas._logicalW) {
      canvas._logicalW = parseInt(canvas.getAttribute('width'), 10) || 940;
      canvas._logicalH = parseInt(canvas.getAttribute('height'), 10) || 340;
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
    return { ctx, width: logW, height: logH };
  }

  /**
   * Helper to ensure canvases render immediately when navigated to without runaway event loops.
   */
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
          if (entry.contentRect.width > 50) {
            safeDraw();
          }
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
  }

  // Seeded Pseudo-Random Generator
  function pseudoRandom(seed) {
    let s = seed % 2147483647;
    if (s <= 0) s += 2147483646;
    return function() {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  }

  function sampleGaussian(rng, mean, std) {
    const u1 = Math.max(1e-10, rng());
    const u2 = rng();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z * std;
  }

  // =========================================================================
  // 1. DYNAMIC BRIDGE SLIDE CONNECTORS (Pixel-perfect alignment)
  // =========================================================================
  function updateBridgeConnectors() {
    const bridgeContainer = document.getElementById('bridge-container');
    const svg = document.getElementById('bridge-svg');
    if (!bridgeContainer || !svg) return;

    const contRect = bridgeContainer.getBoundingClientRect();
    if (contRect.width < 100) return;

    const mapping = [
      { cardId: 'bridge-card-0', rowId: 'bridge-row-2', pathId: 'conn-0-2' },
      { cardId: 'bridge-card-0', rowId: 'bridge-row-3', pathId: 'conn-0-3' },
      { cardId: 'bridge-card-1', rowId: 'bridge-row-4', pathId: 'conn-1-4' },
      { cardId: 'bridge-card-1', rowId: 'bridge-row-6', pathId: 'conn-1-6' },
      { cardId: 'bridge-card-2', rowId: 'bridge-row-8', pathId: 'conn-2-8' },
      { cardId: 'bridge-card-3', rowId: 'bridge-row-9', pathId: 'conn-3-9' },
      { cardId: 'bridge-card-3', rowId: 'bridge-row-10', pathId: 'conn-3-10' },
      { cardId: 'bridge-card-3', rowId: 'bridge-row-11', pathId: 'conn-3-11' },
      { cardId: 'bridge-card-4', rowId: 'bridge-row-12', pathId: 'conn-4-12' },
      { cardId: 'bridge-card-5', rowId: 'bridge-row-13', pathId: 'conn-5-13' }
    ];

    mapping.forEach(({ cardId, rowId, pathId }) => {
      const cardEl = document.getElementById(cardId);
      const rowEl = document.getElementById(rowId);
      const pathEl = document.getElementById(pathId);
      if (!cardEl || !rowEl || !pathEl) return;

      const cRect = cardEl.getBoundingClientRect();
      const rRect = rowEl.getBoundingClientRect();

      const x1 = cRect.right - contRect.left;
      const y1 = cRect.top + cRect.height * 0.5 - contRect.top;
      const x2 = rRect.left - contRect.left;
      const y2 = rRect.top + rRect.height * 0.5 - contRect.top;

      const dx = Math.max(30, (x2 - x1) * 0.5);
      pathEl.setAttribute('d', `M ${x1.toFixed(1)},${y1.toFixed(1)} C ${(x1 + dx).toFixed(1)},${y1.toFixed(1)} ${(x2 - dx).toFixed(1)},${y2.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`);
    });
  }

  function initBridgeConnectors() {
    window.addEventListener('resize', updateBridgeConnectors);
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.querySelector('#bridge-container')) {
          requestAnimationFrame(updateBridgeConnectors);
          setTimeout(updateBridgeConnectors, 80);
          setTimeout(updateBridgeConnectors, 250);
        }
      });
      Reveal.addEventListener('fragmentshown', updateBridgeConnectors);
      Reveal.addEventListener('fragmenthidden', updateBridgeConnectors);
      Reveal.addEventListener('ready', () => setTimeout(updateBridgeConnectors, 200));
    }
    setTimeout(updateBridgeConnectors, 150);
  }

  // =========================================================================
  // 2. DEMO 1: DECISION SURFACE & MARGIN BAND EXPLORER
  // =========================================================================
  function initDecisionSurfaceDemo() {
    const canvas = document.getElementById('svm-decision-canvas');
    if (!canvas) return;

    // Fixed reproducible dataset
    const rng = pseudoRandom(42);
    const greenPoints = [];
    const redPoints = [];

    // Class +1 (Green) centered near (200, 340)
    for (let i = 0; i < 22; i++) {
      greenPoints.push({
        x: sampleGaussian(rng, 220, 65),
        y: sampleGaussian(rng, 330, 45),
        label: 1
      });
    }
    // Class -1 (Red) centered near (580, 160)
    for (let i = 0; i < 20; i++) {
      redPoints.push({
        x: sampleGaussian(rng, 570, 70),
        y: sampleGaussian(rng, 170, 50),
        label: -1
      });
    }

    const allPoints = [...greenPoints, ...redPoints];

    // Precomputed optimal max-margin hyperplane parameters
    const optAngle = 46.5;
    const optMidX = 395;
    const optMidY = 250;

    const declaredW = parseInt(canvas.getAttribute('width'), 10) || 940;
    const declaredH = parseInt(canvas.getAttribute('height'), 10) || 340;

    let lineAngle = 20.0;
    let lineMidX = 400;
    let lineMidY = 260;
    let wScale = 1.0;

    let isDragging = false;
    let dragMode = null; // 'translate' or 'rotate'
    let lastPointerAngle = 0;
    let lastDragX = 0;
    let lastDragY = 0;

    // DOM displays
    const w1Disp = document.getElementById('svm-w1-val');
    const w2Disp = document.getElementById('svm-w2-val');
    const bDisp = document.getElementById('svm-b-val');
    const normDisp = document.getElementById('svm-norm-val');
    const geomMarginDisp = document.getElementById('svm-geom-margin');
    const minFuncDisp = document.getElementById('svm-min-func');
    const statusBox = document.getElementById('svm-margin-status');

    const snapBtn = document.getElementById('svm-btn-snap');
    const resetBtn = document.getElementById('svm-btn-reset');
    const scaleSlider = document.getElementById('svm-scale-slider');
    const scaleValSpan = document.getElementById('svm-scale-val');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      // Background subtle grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.15)';
      ctx.lineWidth = 1;
      for (let x = 0; x < W; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
      for (let y = 0; y < H; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }

      const angleRad = lineAngle * Math.PI / 180;
      const dx = Math.cos(angleRad);
      const dy = Math.sin(angleRad);
      // Normal vector pointing toward Class +1
      const nx = -dy;
      const ny = dx;

      // Base unscaled unit normal
      // w = scale * (nx, ny)
      const w1 = nx * wScale;
      const w2 = ny * wScale;
      const b = -(w1 * lineMidX + w2 * lineMidY);
      const normW = Math.sqrt(w1 * w1 + w2 * w2);

      // Calculate distances for all points
      let minGeomMargin = Infinity;
      let minFuncMargin = Infinity;
      let numViolations = 0;
      let closestPoints = [];

      allPoints.forEach(p => {
        const funcVal = p.label * (w1 * p.x + w2 * p.y + b);
        const geomDist = funcVal / normW;
        if (funcVal < 0) numViolations++;
        if (geomDist < minGeomMargin) {
          minGeomMargin = geomDist;
        }
        if (funcVal < minFuncMargin) {
          minFuncMargin = funcVal;
        }
      });

      // Tolerance for points lying on margin boundary
      allPoints.forEach(p => {
        const geomDist = (p.label * (w1 * p.x + w2 * p.y + b)) / normW;
        if (Math.abs(geomDist - minGeomMargin) < 4.0) {
          closestPoints.push(p);
        }
      });

      // Margin width in pixels: D = 2 / normW (or based on closest point distance)
      const marginPx = Math.max(0, minGeomMargin);

      // 1. Draw Margin Corridor (Shaded band)
      if (marginPx > 0 && numViolations === 0) {
        ctx.save();
        ctx.fillStyle = 'rgba(181, 137, 0, 0.12)';
        ctx.beginPath();
        const L = 2000;
        ctx.moveTo((lineMidX - dx * L) + nx * marginPx, (lineMidY - dy * L) + ny * marginPx);
        ctx.lineTo((lineMidX + dx * L) + nx * marginPx, (lineMidY + dy * L) + ny * marginPx);
        ctx.lineTo((lineMidX + dx * L) - nx * marginPx, (lineMidY + dy * L) - ny * marginPx);
        ctx.lineTo((lineMidX - dx * L) - nx * marginPx, (lineMidY - dy * L) - ny * marginPx);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        // Margin boundary lines (+1 and -1)
        ctx.strokeStyle = SOL.yellow;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);

        // Plane +1
        ctx.beginPath();
        ctx.moveTo((lineMidX - dx * L) + nx * marginPx, (lineMidY - dy * L) + ny * marginPx);
        ctx.lineTo((lineMidX + dx * L) + nx * marginPx, (lineMidY + dy * L) + ny * marginPx);
        ctx.stroke();

        // Plane -1
        ctx.beginPath();
        ctx.moveTo((lineMidX - dx * L) - nx * marginPx, (lineMidY - dy * L) - ny * marginPx);
        ctx.lineTo((lineMidX + dx * L) - nx * marginPx, (lineMidY + dy * L) - ny * marginPx);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 2. Draw Main Decision Hyperplane (w^T x + b = 0)
      const L = 2000;
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(lineMidX - dx * L, lineMidY - dy * L);
      ctx.lineTo(lineMidX + dx * L, lineMidY + dy * L);
      ctx.stroke();

      // 3. Draw Perpendicular Projections from closest points to decision plane
      closestPoints.forEach(p => {
        const dot = (p.x - lineMidX) * nx + (p.y - lineMidY) * ny;
        const projX = p.x - dot * nx;
        const projY = p.y - dot * ny;

        ctx.strokeStyle = SOL.orange;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(projX, projY);
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // 4. Draw Data Points
      allPoints.forEach(p => {
        const isClosest = closestPoints.includes(p);
        const funcVal = p.label * (w1 * p.x + w2 * p.y + b);

        if (isClosest) {
          ctx.strokeStyle = SOL.base03;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 13, 0, Math.PI * 2);
          ctx.stroke();

          // SV ring
          ctx.strokeStyle = SOL.yellow;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.fillStyle = p.label === 1 ? SOL.green : SOL.red;
        ctx.strokeStyle = p.label === 1 ? '#2d6b2f' : '#8b0000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });

      // 5. Draw Interactive Drag Handles
      // Translation handle at midpoint
      ctx.fillStyle = SOL.orange;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(lineMidX, lineMidY, 11, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Move icon inside translation handle
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('✢', lineMidX, lineMidY);

      // Normal rotation handle
      const handleDist = 70;
      const rotHX = lineMidX + nx * handleDist;
      const rotHY = lineMidY + ny * handleDist;

      ctx.strokeStyle = SOL.cyan;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(lineMidX, lineMidY);
      ctx.lineTo(rotHX, rotHY);
      ctx.stroke();

      ctx.fillStyle = SOL.blue;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(rotHX, rotHY, 13, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Rotation icon inside rotation handle
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('↻', rotHX, rotHY);

      // Label on normal
      ctx.fillStyle = SOL.blue;
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('w (rotate)', rotHX + 16, rotHY);

      // Reset alignment for other text
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';

      // Update Text Displays
      if (w1Disp) w1Disp.textContent = w1.toFixed(2);
      if (w2Disp) w2Disp.textContent = w2.toFixed(2);
      if (bDisp) bDisp.textContent = b.toFixed(1);
      if (normDisp) normDisp.textContent = normW.toFixed(2);
      if (geomMarginDisp) geomMarginDisp.textContent = (minGeomMargin > 0 ? (2 * minGeomMargin).toFixed(1) : '0.0') + ' px';
      if (minFuncDisp) minFuncDisp.textContent = minFuncMargin.toFixed(2);

      if (statusBox) {
        if (numViolations > 0) {
          statusBox.style.color = SOL.red;
          statusBox.textContent = `❌ Infeasible: ${numViolations} point(s) misclassified! Margin invalid.`;
        } else if (Math.abs(lineAngle - optAngle) < 2.0 && Math.abs(lineMidX - optMidX) < 15) {
          statusBox.style.color = SOL.green;
          statusBox.textContent = `★ Max-Margin Hyperplane Optimal! Margin gap maximized (2 / ||w|| = ${(2 * minGeomMargin).toFixed(1)} px).`;
        } else {
          statusBox.style.color = SOL.base01;
          statusBox.textContent = `✓ Linearly separating. Rotate & translate to maximize the gap (support vectors highlighted).`;
        }
      }
    }

    // Interactive event handling with exact logical canvas scaling
    function getCanvasCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: (clientX - rect.left) * (declaredW / rect.width),
        y: (clientY - rect.top) * (declaredH / rect.height)
      };
    }

    function onPointerDown(e) {
      const pos = getCanvasCoords(e);
      const angleRad = lineAngle * Math.PI / 180;
      const dx = Math.cos(angleRad);
      const dy = Math.sin(angleRad);
      const nx = -dy;
      const ny = dx;
      const rotHX = lineMidX + nx * 70;
      const rotHY = lineMidY + ny * 70;

      const dRot = Math.hypot(pos.x - rotHX, pos.y - rotHY);
      const dMid = Math.hypot(pos.x - lineMidX, pos.y - lineMidY);
      const distToLine = Math.abs((pos.x - lineMidX) * nx + (pos.y - lineMidY) * ny);
      const projOnLine = (pos.x - lineMidX) * dx + (pos.y - lineMidY) * dy;

      if (dRot < 28) {
        isDragging = true;
        dragMode = 'rotate';
        lastPointerAngle = Math.atan2(pos.y - lineMidY, pos.x - lineMidX);
        canvas.style.cursor = 'grabbing';
        e.preventDefault();
      } else if (dMid < 24 || (distToLine < 14 && Math.abs(projOnLine) < 450)) {
        isDragging = true;
        dragMode = 'translate';
        lastDragX = pos.x;
        lastDragY = pos.y;
        canvas.style.cursor = 'grabbing';
        e.preventDefault();
      }
    }

    function onPointerMove(e) {
      const pos = getCanvasCoords(e);
      const angleRad = lineAngle * Math.PI / 180;
      const dx = Math.cos(angleRad);
      const dy = Math.sin(angleRad);
      const nx = -dy;
      const ny = dx;
      const rotHX = lineMidX + nx * 70;
      const rotHY = lineMidY + ny * 70;

      if (!isDragging) {
        const dRot = Math.hypot(pos.x - rotHX, pos.y - rotHY);
        const dMid = Math.hypot(pos.x - lineMidX, pos.y - lineMidY);
        const distToLine = Math.abs((pos.x - lineMidX) * nx + (pos.y - lineMidY) * ny);
        const projOnLine = (pos.x - lineMidX) * dx + (pos.y - lineMidY) * dy;

        if (dRot < 28) {
          canvas.style.cursor = 'grab';
          canvas.title = 'Drag to rotate decision plane';
        } else if (dMid < 24 || (distToLine < 14 && Math.abs(projOnLine) < 450)) {
          canvas.style.cursor = 'move';
          canvas.title = 'Drag to translate decision plane';
        } else {
          canvas.style.cursor = 'default';
          canvas.title = '';
        }
        return;
      }

      if (dragMode === 'rotate') {
        const distFromCenter = Math.hypot(pos.x - lineMidX, pos.y - lineMidY);
        if (distFromCenter > 15) {
          const curPointerAngle = Math.atan2(pos.y - lineMidY, pos.x - lineMidX);
          let delta = curPointerAngle - lastPointerAngle;
          while (delta > Math.PI) delta -= 2 * Math.PI;
          while (delta < -Math.PI) delta += 2 * Math.PI;

          lineAngle += delta * 180 / Math.PI;
          while (lineAngle > 180) lineAngle -= 360;
          while (lineAngle < -180) lineAngle += 360;

          lastPointerAngle = curPointerAngle;
        }
      } else if (dragMode === 'translate') {
        const deltaX = pos.x - lastDragX;
        const deltaY = pos.y - lastDragY;
        lineMidX = Math.max(60, Math.min(declaredW - 60, lineMidX + deltaX));
        lineMidY = Math.max(60, Math.min(declaredH - 60, lineMidY + deltaY));
        lastDragX = pos.x;
        lastDragY = pos.y;
      }
      requestAnimationFrame(draw);
      if (e.cancelable) e.preventDefault();
    }

    function onPointerUp() {
      isDragging = false;
      dragMode = null;
      canvas.style.cursor = 'default';
      canvas.title = '';
      requestAnimationFrame(draw);
    }

    canvas.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);

    canvas.addEventListener('touchstart', onPointerDown, { passive: false });
    window.addEventListener('touchmove', onPointerMove, { passive: false });
    window.addEventListener('touchend', onPointerUp);

    if (snapBtn) {
      snapBtn.addEventListener('click', () => {
        isDragging = false;
        dragMode = null;
        canvas.style.cursor = 'default';
        canvas.title = '';
        lineAngle = optAngle;
        lineMidX = optMidX;
        lineMidY = optMidY;
        requestAnimationFrame(draw);
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        isDragging = false;
        dragMode = null;
        canvas.style.cursor = 'default';
        canvas.title = '';
        lineAngle = 20.0;
        lineMidX = 400;
        lineMidY = 260;
        wScale = 1.0;
        if (scaleSlider) scaleSlider.value = 1.0;
        if (scaleValSpan) scaleValSpan.textContent = '1.0x';
        requestAnimationFrame(draw);
      });
    }

    if (scaleSlider) {
      scaleSlider.addEventListener('input', (e) => {
        wScale = parseFloat(e.target.value);
        if (scaleValSpan) scaleValSpan.textContent = wScale.toFixed(1) + 'x';
        requestAnimationFrame(draw);
      });
    }

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // 3. DEMO 2: INTERACTIVE LAGRANGE MULTIPLIERS (REPAIRED & UNCLIPPED)
  // =========================================================================
  function initLagrangeDemo() {
    const canvas2D = document.getElementById('lagrange-canvas-2d-v4');
    const canvas3D = document.getElementById('lagrange-canvas-3d-v4');
    if (!canvas2D || !canvas3D) return;

    const ctx2D = canvas2D.getContext('2d');
    const ctx3D = canvas3D.getContext('2d');
    const zValueLabel = document.getElementById('z-value-label-v4');
    const statusLabel = document.getElementById('lagrange-status-badge');

    const width = 420, height = 420;
    const plotRange = { min: -2.2, max: 2.2 };
    const scale = width / (plotRange.max - plotRange.min);
    let angle = Math.PI / 3;
    let isDragging = false;

    // Off-screen canvas for pre-rendering static contour heatmaps
    const contourCanvas = document.createElement('canvas');
    contourCanvas.width = width;
    contourCanvas.height = height;
    const contourCtx = contourCanvas.getContext('2d');

    const toPixel = (x, y) => ({ x: (x - plotRange.min) * scale, y: height - (y - plotRange.min) * scale });
    const toPlot = (px, py) => ({ x: (px / scale) + plotRange.min, y: ((height - py) / scale) + plotRange.min });

    function getColorForZ(z) {
      const zClamped = Math.max(-2, Math.min(2, z));
      const t = (zClamped + 2) / 4;
      if (t < 0.5) {
        const s = t * 2;
        return `rgb(${Math.floor(40 + 180 * s)}, ${Math.floor(100 + 100 * s)}, ${Math.floor(210 * (1 - s * 0.5))})`;
      } else {
        const s = (t - 0.5) * 2;
        return `rgb(${Math.floor(220 + 35 * s)}, ${Math.floor(200 * (1 - s * 0.6))}, ${Math.floor(80 * (1 - s))})`;
      }
    }

    function drawArrow(ctx, fromX, fromY, toX, toY, color, width = 3) {
      const headlen = 11, dx = toX - fromX, dy = toY - fromY, a = Math.atan2(dy, dx);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(toX, toY);
      ctx.lineTo(toX - headlen * Math.cos(a - Math.PI / 6), toY - headlen * Math.sin(a - Math.PI / 6));
      ctx.lineTo(toX - headlen * Math.cos(a + Math.PI / 6), toY - headlen * Math.sin(a + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
    }

    function drawContourPlot() {
      const resolution = 4;
      for (let i = 0; i < width; i += resolution) {
        for (let j = 0; j < height; j += resolution) {
          const { x, y } = toPlot(i, j);
          contourCtx.fillStyle = getColorForZ(x * x - y * y);
          contourCtx.fillRect(i, j, resolution, resolution);
        }
      }
    }

    function draw3DView(ang) {
      ctx3D.clearRect(0, 0, width, height);
      ctx3D.fillStyle = '#ffffff';
      ctx3D.fillRect(0, 0, width, height);

      const cx = width * 0.5, cy = height * 0.6, gridRes = 22, xRange = 2.2, yRange = 2.2;
      const scale3d = 32;

      for (let i = 0; i < gridRes; i++) {
        for (let j = 0; j < gridRes; j++) {
          const x1 = -xRange + (i / gridRes) * 2 * xRange;
          const y1 = -yRange + (j / gridRes) * 2 * yRange;
          const x2 = -xRange + ((i + 1) / gridRes) * 2 * xRange;
          const y2 = -yRange + ((j + 1) / gridRes) * 2 * yRange;
          const z1 = x1 * x1 - y1 * y1, z2 = x2 * x2 - y2 * y2;

          const px1 = cx + x1 * scale3d - y1 * scale3d * 0.5;
          const py1 = cy - z1 * scale3d * 0.75 - y1 * scale3d * 0.3;
          const px2 = cx + x2 * scale3d - y1 * scale3d * 0.5;
          const py2 = cy - z2 * scale3d * 0.75 - y1 * scale3d * 0.3;
          const px3 = cx + x1 * scale3d - y2 * scale3d * 0.5;
          const py3 = cy - z1 * scale3d * 0.75 - y2 * scale3d * 0.3;

          ctx3D.fillStyle = getColorForZ((z1 + z2) / 2);
          ctx3D.strokeStyle = 'rgba(0,0,0,0.18)';
          ctx3D.lineWidth = 0.5;
          ctx3D.beginPath();
          ctx3D.moveTo(px1, py1); ctx3D.lineTo(px2, py2); ctx3D.stroke();
          ctx3D.beginPath();
          ctx3D.moveTo(px1, py1); ctx3D.lineTo(px3, py3); ctx3D.stroke();
        }
      }

      // Constraint circle on surface
      ctx3D.strokeStyle = SOL.base03;
      ctx3D.lineWidth = 3;
      ctx3D.beginPath();
      for (let i = 0; i <= 80; i++) {
        const theta = (i / 80) * 2 * Math.PI;
        const x = Math.cos(theta), y = Math.sin(theta), z = x * x - y * y;
        const px = cx + x * scale3d - y * scale3d * 0.5;
        const py = cy - z * scale3d * 0.75 - y * scale3d * 0.3;
        if (i === 0) ctx3D.moveTo(px, py);
        else ctx3D.lineTo(px, py);
      }
      ctx3D.stroke();

      // Current point on 3D surface
      const pointX = Math.cos(ang), pointY = Math.sin(ang), pointZ = pointX * pointX - pointY * pointY;
      const px = cx + pointX * scale3d - pointY * scale3d * 0.5;
      const py = cy - pointZ * scale3d * 0.75 - pointY * scale3d * 0.3;

      ctx3D.fillStyle = SOL.green;
      ctx3D.strokeStyle = '#fff';
      ctx3D.lineWidth = 3;
      ctx3D.beginPath();
      ctx3D.arc(px, py, 8, 0, 2 * Math.PI);
      ctx3D.fill();
      ctx3D.stroke();
    }

    function draw() {
      ctx2D.clearRect(0, 0, width, height);
      ctx2D.drawImage(contourCanvas, 0, 0);

      // Axes
      const origin = toPixel(0, 0);
      ctx2D.strokeStyle = 'rgba(0, 0, 0, 0.3)';
      ctx2D.lineWidth = 1;
      ctx2D.beginPath();
      ctx2D.moveTo(0, origin.y); ctx2D.lineTo(width, origin.y);
      ctx2D.moveTo(origin.x, 0); ctx2D.lineTo(origin.x, height);
      ctx2D.stroke();

      // Constraint circle: x^2 + y^2 = 1
      const center = toPixel(0, 0);
      ctx2D.strokeStyle = SOL.base03;
      ctx2D.lineWidth = 2.5;
      ctx2D.beginPath();
      ctx2D.arc(center.x, center.y, 1.0 * scale, 0, 2 * Math.PI);
      ctx2D.stroke();

      const pointX = Math.cos(angle), pointY = Math.sin(angle);
      const pixelPoint = toPixel(pointX, pointY);

      // Gradients:
      // f(x, y) = x^2 - y^2  => \nabla f = (2x, -2y)
      // g(x, y) = x^2 + y^2 - 1 => \nabla g = (2x, 2y)
      const gradF = { x: 2 * pointX, y: -2 * pointY };
      const gradG = { x: 2 * pointX, y: 2 * pointY };

      const gradScale = 0.35;
      const gradFPixelEnd = toPixel(pointX + gradF.x * gradScale, pointY + gradF.y * gradScale);
      const gradGPixelEnd = toPixel(pointX + gradG.x * gradScale, pointY + gradG.y * gradScale);

      // Draw vectors
      drawArrow(ctx2D, pixelPoint.x, pixelPoint.y, gradGPixelEnd.x, gradGPixelEnd.y, SOL.red, 3);
      drawArrow(ctx2D, pixelPoint.x, pixelPoint.y, gradFPixelEnd.x, gradFPixelEnd.y, SOL.blue, 3);

      // Draggable point
      ctx2D.fillStyle = SOL.green;
      ctx2D.strokeStyle = '#fff';
      ctx2D.lineWidth = 3;
      ctx2D.beginPath();
      ctx2D.arc(pixelPoint.x, pixelPoint.y, 9, 0, 2 * Math.PI);
      ctx2D.fill();
      ctx2D.stroke();

      draw3DView(angle);

      const zVal = pointX * pointX - pointY * pointY;
      if (zValueLabel) zValueLabel.textContent = zVal.toFixed(3);

      // Collinearity check: cross product of gradF and gradG
      const cross = gradF.x * gradG.y - gradF.y * gradG.x; // (2x)(2y) - (-2y)(2x) = 8xy
      const isOptimal = Math.abs(cross) < 0.25;

      if (statusLabel) {
        if (isOptimal) {
          const lambda = (Math.abs(pointY) < 0.1) ? '+1.0' : '-1.0';
          statusLabel.style.background = '#859900';
          statusLabel.style.color = '#fff';
          statusLabel.textContent = `★ STATIONARY / OPTIMUM: ∇f || ∇g (λ = ${lambda})`;
        } else {
          statusLabel.style.background = '#eee8d5';
          statusLabel.style.color = '#586e75';
          statusLabel.textContent = `Drag point along circle: ∇f and ∇g not parallel`;
        }
      }
    }

    function handleDrag(e) {
      const rect = canvas2D.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      const px = (clientX - rect.left) * (width / rect.width);
      const py = (clientY - rect.top) * (height / rect.height);
      const plotMouse = toPlot(px, py);
      angle = Math.atan2(plotMouse.y, plotMouse.x);
      requestAnimationFrame(draw);
    }

    canvas2D.addEventListener('mousedown', (e) => { isDragging = true; canvas2D.style.cursor = 'grabbing'; handleDrag(e); });
    window.addEventListener('mouseup', () => { isDragging = false; canvas2D.style.cursor = 'grab'; });
    window.addEventListener('mousemove', (e) => { if (isDragging) handleDrag(e); });

    canvas2D.addEventListener('touchstart', (e) => { isDragging = true; handleDrag(e); e.preventDefault(); }, { passive: false });
    window.addEventListener('touchmove', (e) => { if (isDragging) handleDrag(e); }, { passive: false });
    window.addEventListener('touchend', () => { isDragging = false; });

    drawContourPlot();
    registerSlideListener(canvas2D, draw);
    draw();
  }

  // =========================================================================
  // 4. DEMO 3: PRIMAL-DUAL EXPLORER (min x^2 s.t. x >= b)
  // =========================================================================
  function initDualityExplorerDemo() {
    const canvasPrimal = document.getElementById('svm-dual-primal-canvas');
    const canvasDual = document.getElementById('svm-dual-dual-canvas');
    if (!canvasPrimal || !canvasDual) return;

    let bVal = 0.8;

    const bSlider = document.getElementById('svm-dual-b-slider');
    const bDisplay = document.getElementById('svm-dual-b-val');
    const primalOptSpan = document.getElementById('svm-dual-primal-opt');
    const dualOptSpan = document.getElementById('svm-dual-dual-opt');
    const gapSpan = document.getElementById('svm-dual-gap-val');
    const constraintStatus = document.getElementById('svm-dual-status');

    function drawPrimal() {
      const { ctx, width: W, height: H } = setupHiDPI(canvasPrimal);
      ctx.clearRect(0, 0, W, H);

      const xMin = -2.2, xMax = 2.2;
      const yMin = -0.5, yMax = 4.0;
      const toX = x => ((x - xMin) / (xMax - xMin)) * W;
      const toY = y => H - ((y - yMin) / (yMax - yMin)) * H;

      // Axes
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, toY(0)); ctx.lineTo(W, toY(0));
      ctx.moveTo(toX(0), 0); ctx.lineTo(toX(0), H);
      ctx.stroke();

      // Shaded Feasible Region x >= b
      ctx.fillStyle = 'rgba(133, 153, 0, 0.15)';
      ctx.fillRect(toX(bVal), 0, W - toX(bVal), H);

      // Constraint line x = b
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(toX(bVal), 0);
      ctx.lineTo(toX(bVal), H);
      ctx.stroke();
      ctx.setLineDash([]);

      // Objective f(x) = x^2
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let px = 0; px <= W; px += 2) {
        const x = xMin + (px / W) * (xMax - xMin);
        const y = x * x;
        if (px === 0) ctx.moveTo(px, toY(y));
        else ctx.lineTo(px, toY(y));
      }
      ctx.stroke();

      // Primal optimum: x* = max(0, b)
      const xStar = Math.max(0, bVal);
      const pStar = xStar * xStar;

      ctx.fillStyle = SOL.green;
      ctx.strokeStyle = SOL.base03;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(toX(xStar), toY(pStar), 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Labels
      ctx.fillStyle = SOL.blue;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('f(x) = x²', toX(1.4), toY(2.2));

      ctx.fillStyle = SOL.orange;
      ctx.fillText(`x ≥ b (b = ${bVal.toFixed(2)})`, toX(bVal) + 6, 25);

      ctx.fillStyle = SOL.base03;
      ctx.fillText(`Primal x* = ${xStar.toFixed(2)}, p* = ${pStar.toFixed(2)}`, toX(xStar) - 50, toY(pStar) - 15);
    }

    function drawDual() {
      const { ctx, width: W, height: H } = setupHiDPI(canvasDual);
      ctx.clearRect(0, 0, W, H);

      const aMin = 0.0, aMax = 4.0;
      const dMin = -1.0, dMax = 3.5;
      const toX = a => ((a - aMin) / (aMax - aMin)) * W;
      const toY = d => H - ((d - dMin) / (dMax - dMin)) * H;

      // Axes
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, toY(0)); ctx.lineTo(W, toY(0));
      ctx.moveTo(toX(0), 0); ctx.lineTo(toX(0), H);
      ctx.stroke();

      // Dual function: g(alpha) = b * alpha - alpha^2 / 4 for alpha >= 0
      ctx.strokeStyle = SOL.magenta;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let px = 0; px <= W; px += 2) {
        const alpha = aMin + (px / W) * (aMax - aMin);
        const dualVal = bVal * alpha - (alpha * alpha) / 4.0;
        if (px === 0) ctx.moveTo(px, toY(dualVal));
        else ctx.lineTo(px, toY(dualVal));
      }
      ctx.stroke();

      // Dual optimum: alpha* = max(0, 2*b), d* = g(alpha*)
      const alphaStar = Math.max(0, 2 * bVal);
      const dStar = bVal * alphaStar - (alphaStar * alphaStar) / 4.0;

      ctx.fillStyle = SOL.yellow;
      ctx.strokeStyle = SOL.base03;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(toX(alphaStar), toY(dStar), 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Labels
      ctx.fillStyle = SOL.magenta;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Dual g(α) = b·α - α²/4', toX(1.8), toY(0.4));

      ctx.fillStyle = SOL.base03;
      ctx.fillText(`Dual α* = ${alphaStar.toFixed(2)}, d* = ${dStar.toFixed(2)}`, Math.max(10, toX(alphaStar) - 40), toY(dStar) - 15);
    }

    function updateAll() {
      drawPrimal();
      drawDual();

      const xStar = Math.max(0, bVal);
      const pStar = xStar * xStar;
      const alphaStar = Math.max(0, 2 * bVal);
      const dStar = bVal * alphaStar - (alphaStar * alphaStar) / 4.0;
      const gap = Math.abs(pStar - dStar);

      if (bDisplay) bDisplay.textContent = bVal.toFixed(2);
      if (primalOptSpan) primalOptSpan.textContent = pStar.toFixed(3);
      if (dualOptSpan) dualOptSpan.textContent = dStar.toFixed(3);
      if (gapSpan) gapSpan.textContent = gap.toFixed(4) + ' (Zero Gap!)';

      if (constraintStatus) {
        if (bVal <= 0) {
          constraintStatus.style.color = SOL.blue;
          constraintStatus.textContent = `Constraint Inactive (x* = 0 > b): Multiplier α* = 0. Unconstrained optimum holds.`;
        } else {
          constraintStatus.style.color = SOL.orange;
          constraintStatus.textContent = `Constraint Active (x* = b): Multiplier α* = 2b = ${alphaStar.toFixed(2)} > 0. Complementary slackness: α*(x* - b) = 0.`;
        }
      }
    }

    if (bSlider) {
      bSlider.addEventListener('input', (e) => {
        bVal = parseFloat(e.target.value);
        updateAll();
      });
    }

    registerSlideListener(canvasPrimal, updateAll);
    registerSlideListener(canvasDual, updateAll);
    updateAll();
  }

  // =========================================================================
  // 5. DEMO 5: INTERACTIVE 2D KERNEL SVM PLAYGROUND (LIVE SMO IN JS)
  // =========================================================================
  function initKernelPlaygroundDemo() {
    const canvas = document.getElementById('svm-kernel-canvas');
    if (!canvas) return;

    let kernelType = 'rbf'; // 'linear', 'poly', 'rbf'
    let C_val = 1.0;
    let gamma_val = 0.8;
    let polyDegree = 2;
    let addClass = 1; // Class of new point on click

    // Seeded initial dataset (Concentric non-linear rings)
    function generatePreset(type) {
      const pts = [];
      const rng = pseudoRandom(101);

      if (type === 'circles') {
        // Inner circle (+1)
        for (let i = 0; i < 14; i++) {
          const theta = rng() * Math.PI * 2;
          const r = rng() * 0.7;
          pts.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), label: 1 });
        }
        // Outer ring (-1)
        for (let i = 0; i < 20; i++) {
          const theta = rng() * Math.PI * 2;
          const r = 1.3 + rng() * 0.7;
          pts.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), label: -1 });
        }
      } else if (type === 'linear') {
        for (let i = 0; i < 15; i++) {
          pts.push({ x: sampleGaussian(rng, -1.1, 0.4), y: sampleGaussian(rng, -0.6, 0.4), label: 1 });
        }
        for (let i = 0; i < 15; i++) {
          pts.push({ x: sampleGaussian(rng, 1.1, 0.4), y: sampleGaussian(rng, 0.6, 0.4), label: -1 });
        }
      } else if (type === 'xor') {
        // Quadrants 1 & 3: +1, Quadrants 2 & 4: -1
        for (let i = 0; i < 7; i++) {
          pts.push({ x: sampleGaussian(rng, 1.1, 0.3), y: sampleGaussian(rng, 1.1, 0.3), label: 1 });
          pts.push({ x: sampleGaussian(rng, -1.1, 0.3), y: sampleGaussian(rng, -1.1, 0.3), label: 1 });
          pts.push({ x: sampleGaussian(rng, -1.1, 0.3), y: sampleGaussian(rng, 1.1, 0.3), label: -1 });
          pts.push({ x: sampleGaussian(rng, 1.1, 0.3), y: sampleGaussian(rng, -1.1, 0.3), label: -1 });
        }
      }
      return pts;
    }

    let dataPoints = generatePreset('circles');

    // Kernel evaluation
    function kernelFunc(p1, p2) {
      if (kernelType === 'linear') {
        return p1.x * p2.x + p1.y * p2.y;
      } else if (kernelType === 'poly') {
        const dot = p1.x * p2.x + p1.y * p2.y;
        return Math.pow(dot + 1.0, polyDegree);
      } else { // RBF
        const d2 = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
        return Math.exp(-gamma_val * d2);
      }
    }

    /**
     * Simplified Sequential Minimal Optimization (SMO) Solver
     */
    function trainSVM() {
      const N = dataPoints.length;
      if (N < 2) return { alphas: [], b: 0, supportVectors: [] };

      const alphas = new Float64Array(N);
      let b = 0.0;
      const tol = 1e-3;
      const eps = 1e-3;
      const maxPasses = 15;
      let passes = 0;

      // Precompute Gram matrix
      const K = [];
      for (let i = 0; i < N; i++) {
        K[i] = new Float64Array(N);
        for (let j = 0; j < N; j++) {
          K[i][j] = kernelFunc(dataPoints[i], dataPoints[j]);
        }
      }

      function predictRaw(idx) {
        let sum = b;
        for (let j = 0; j < N; j++) {
          if (alphas[j] > 1e-7) {
            sum += alphas[j] * dataPoints[j].label * K[j][idx];
          }
        }
        return sum;
      }

      while (passes < maxPasses) {
        let numChanged = 0;
        for (let i = 0; i < N; i++) {
          const Ei = predictRaw(i) - dataPoints[i].label;
          const yi = dataPoints[i].label;

          if ((yi * Ei < -tol && alphas[i] < C_val) || (yi * Ei > tol && alphas[i] > 0)) {
            // Select j != i randomly
            let j = Math.floor(Math.random() * (N - 1));
            if (j >= i) j++;

            const Ej = predictRaw(j) - dataPoints[j].label;
            const yj = dataPoints[j].label;

            const aiOld = alphas[i];
            const ajOld = alphas[j];

            // Compute bounds L and H
            let L = 0, H = 0;
            if (yi !== yj) {
              L = Math.max(0, ajOld - aiOld);
              H = Math.min(C_val, C_val + ajOld - aiOld);
            } else {
              L = Math.max(0, aiOld + ajOld - C_val);
              H = Math.min(C_val, aiOld + ajOld);
            }

            if (Math.abs(L - H) < 1e-5) continue;

            const eta = 2.0 * K[i][j] - K[i][i] - K[j][j];
            if (eta >= 0) continue;

            let ajNew = ajOld - (yj * (Ei - Ej)) / eta;
            ajNew = Math.max(L, Math.min(H, ajNew));

            if (Math.abs(ajNew - ajOld) < 1e-5) continue;

            const aiNew = aiOld + yi * yj * (ajOld - ajNew);

            // Compute threshold b
            const b1 = b - Ei - yi * (aiNew - aiOld) * K[i][i] - yj * (ajNew - ajOld) * K[i][j];
            const b2 = b - Ej - yi * (aiNew - aiOld) * K[i][j] - yj * (ajNew - ajOld) * K[j][j];

            if (aiNew > 0 && aiNew < C_val) b = b1;
            else if (ajNew > 0 && ajNew < C_val) b = b2;
            else b = (b1 + b2) * 0.5;

            alphas[i] = aiNew;
            alphas[j] = ajNew;
            numChanged++;
          }
        }
        if (numChanged === 0) passes++;
        else passes = 0;
      }

      const supportVectors = [];
      for (let i = 0; i < N; i++) {
        if (alphas[i] > 1e-4) {
          supportVectors.push({ point: dataPoints[i], alpha: alphas[i], index: i });
        }
      }

      return { alphas, b, supportVectors };
    }

    // DOM Controls
    const kernelSelect = document.getElementById('svm-kernel-select');
    const cSlider = document.getElementById('svm-c-slider');
    const cValDisp = document.getElementById('svm-c-val');
    const gammaSlider = document.getElementById('svm-gamma-slider');
    const gammaValDisp = document.getElementById('svm-gamma-val');
    const gammaControlRow = document.getElementById('svm-gamma-ctrl');
    const classBtn = document.getElementById('svm-btn-toggle-class');
    const clearBtn = document.getElementById('svm-btn-clear');
    const presetCirclesBtn = document.getElementById('svm-preset-circles');
    const presetLinearBtn = document.getElementById('svm-preset-linear');
    const presetXorBtn = document.getElementById('svm-preset-xor');

    const svCountSpan = document.getElementById('svm-sv-count');
    const accuracySpan = document.getElementById('svm-accuracy-val');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const xMin = -2.5, xMax = 2.5;
      const yMin = -1.8, yMax = 1.8;
      const toX = x => ((x - xMin) / (xMax - xMin)) * W;
      const toY = y => H - ((y - yMin) / (yMax - yMin)) * H;
      const fromX = px => xMin + (px / W) * (xMax - xMin);
      const fromY = py => yMin + ((H - py) / H) * (yMax - yMin);

      const model = trainSVM();

      // 1. Render Decision Field / Contour Grid
      const gridW = 75;
      const gridH = 50;
      const cellW = W / gridW;
      const cellH = H / gridH;

      const fGrid = [];
      for (let gy = 0; gy <= gridH; gy++) {
        fGrid[gy] = [];
        const py = (gy / gridH) * H;
        const y = fromY(py);
        for (let gx = 0; gx <= gridW; gx++) {
          const px = (gx / gridW) * W;
          const x = fromX(px);

          let fVal = model.b;
          for (let i = 0; i < dataPoints.length; i++) {
            if (model.alphas[i] > 1e-4) {
              fVal += model.alphas[i] * dataPoints[i].label * kernelFunc({ x, y }, dataPoints[i]);
            }
          }
          fGrid[gy][gx] = fVal;
        }
      }

      // Draw tinted background based on sign of f(x)
      for (let gy = 0; gy < gridH; gy++) {
        for (let gx = 0; gx < gridW; gx++) {
          const f = fGrid[gy][gx];
          const px = gx * cellW;
          const py = gy * cellH;

          if (f >= 0) {
            const intensity = Math.min(0.25, Math.abs(f) * 0.12);
            ctx.fillStyle = `rgba(133, 153, 0, ${intensity})`;
          } else {
            const intensity = Math.min(0.25, Math.abs(f) * 0.12);
            ctx.fillStyle = `rgba(220, 50, 47, ${intensity})`;
          }
          ctx.fillRect(px, py, cellW + 1, cellH + 1);
        }
      }

      // 2. Draw Decision Boundary f(x) = 0 and Margins f(x) = ±1
      function marchContours(threshold, strokeColor, lineWidth, isDashed) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        if (isDashed) ctx.setLineDash([5, 5]);
        else ctx.setLineDash([]);
        ctx.beginPath();

        for (let gy = 0; gy < gridH; gy++) {
          for (let gx = 0; gx < gridW; gx++) {
            const fTL = fGrid[gy][gx] - threshold;
            const fTR = fGrid[gy][gx + 1] - threshold;
            const fBL = fGrid[gy + 1][gx] - threshold;
            const fBR = fGrid[gy + 1][gx + 1] - threshold;

            // Simple zero-crossing segment
            if ((fTL * fTR) < 0) {
              const t = fTL / (fTL - fTR);
              const x1 = (gx + t) * cellW;
              const y1 = gy * cellH;
              ctx.moveTo(x1, y1);
              if ((fTL * fBL) < 0) {
                const s = fTL / (fTL - fBL);
                ctx.lineTo(gx * cellW, (gy + s) * cellH);
              } else if ((fTR * fBR) < 0) {
                const s = fTR / (fTR - fBR);
                ctx.lineTo((gx + 1) * cellW, (gy + s) * cellH);
              }
            }
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      marchContours(0.0, SOL.base03, 3.0, false);  // f(x) = 0
      marchContours(1.0, SOL.yellow, 1.5, true);   // f(x) = +1
      marchContours(-1.0, SOL.yellow, 1.5, true);  // f(x) = -1

      // 3. Draw Data Points and Support Vectors
      let correctCount = 0;
      dataPoints.forEach((p, idx) => {
        let fVal = model.b;
        for (let j = 0; j < dataPoints.length; j++) {
          if (model.alphas[j] > 1e-4) {
            fVal += model.alphas[j] * dataPoints[j].label * kernelFunc(p, dataPoints[j]);
          }
        }
        if (p.label * fVal > 0) correctCount++;

        const isSV = model.alphas[idx] > 1e-4;
        const isViolator = model.alphas[idx] >= C_val * 0.95;

        // Support Vector Ring
        if (isSV) {
          ctx.strokeStyle = isViolator ? SOL.orange : SOL.yellow;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(toX(p.x), toY(p.y), 13, 0, Math.PI * 2);
          ctx.stroke();

          // Outer halo
          ctx.strokeStyle = 'rgba(0, 43, 54, 0.4)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(toX(p.x), toY(p.y), 16, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.fillStyle = p.label === 1 ? SOL.green : SOL.red;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(toX(p.x), toY(p.y), 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });

      // Update Displays
      if (svCountSpan) svCountSpan.textContent = `${model.supportVectors.length} / ${dataPoints.length}`;
      if (accuracySpan) {
        const acc = dataPoints.length > 0 ? ((correctCount / dataPoints.length) * 100).toFixed(0) : 100;
        accuracySpan.textContent = `${acc}%`;
      }
    }

    // Canvas click: Add new data point
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const fracX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const fracY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      const xMin = -2.5, xMax = 2.5;
      const yMin = -1.8, yMax = 1.8;

      const x = xMin + fracX * (xMax - xMin);
      const y = yMin + (1 - fracY) * (yMax - yMin);

      dataPoints.push({ x, y, label: addClass });
      requestAnimationFrame(draw);
    });

    if (kernelSelect) {
      kernelSelect.addEventListener('change', (e) => {
        kernelType = e.target.value;
        if (gammaControlRow) {
          gammaControlRow.style.display = kernelType === 'rbf' ? 'flex' : 'none';
        }
        requestAnimationFrame(draw);
      });
    }

    if (cSlider) {
      cSlider.addEventListener('input', (e) => {
        C_val = parseFloat(e.target.value);
        if (cValDisp) cValDisp.textContent = C_val.toFixed(2);
        requestAnimationFrame(draw);
      });
    }

    if (gammaSlider) {
      gammaSlider.addEventListener('input', (e) => {
        gamma_val = parseFloat(e.target.value);
        if (gammaValDisp) gammaValDisp.textContent = gamma_val.toFixed(2);
        requestAnimationFrame(draw);
      });
    }

    if (classBtn) {
      classBtn.addEventListener('click', () => {
        addClass = -addClass;
        classBtn.style.color = addClass === 1 ? SOL.green : SOL.red;
        classBtn.textContent = `Click Adds: Class ${addClass === 1 ? '+1 (Green)' : '-1 (Red)'}`;
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        dataPoints = [];
        requestAnimationFrame(draw);
      });
    }

    if (presetCirclesBtn) {
      presetCirclesBtn.addEventListener('click', () => {
        dataPoints = generatePreset('circles');
        requestAnimationFrame(draw);
      });
    }
    if (presetLinearBtn) {
      presetLinearBtn.addEventListener('click', () => {
        dataPoints = generatePreset('linear');
        requestAnimationFrame(draw);
      });
    }
    if (presetXorBtn) {
      presetXorBtn.addEventListener('click', () => {
        dataPoints = generatePreset('xor');
        requestAnimationFrame(draw);
      });
    }

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // 6. DEMO 6: LOSS FUNCTION ZOO (0-1 vs Hinge vs Logistic vs Squared)
  // =========================================================================
  function initLossZooDemo() {
    const canvas = document.getElementById('svm-loss-zoo-canvas');
    if (!canvas) return;

    let marginZ = 0.5;

    const zSlider = document.getElementById('svm-loss-z-slider');
    const zValDisp = document.getElementById('svm-loss-z-val');
    const hingeValDisp = document.getElementById('svm-loss-hinge-val');
    const logValDisp = document.getElementById('svm-loss-log-val');
    const zeroOneDisp = document.getElementById('svm-loss-01-val');
    const regimeStatus = document.getElementById('svm-loss-regime-status');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const zMin = -2.5, zMax = 3.0;
      const lMin = -0.3, lMax = 3.5;
      const toX = z => ((z - zMin) / (zMax - zMin)) * W;
      const toY = l => H - ((l - lMin) / (lMax - lMin)) * H;

      // Grid and Axes
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, toY(0)); ctx.lineTo(W, toY(0));
      ctx.moveTo(toX(0), 0); ctx.lineTo(toX(0), H);
      ctx.stroke();

      // Margin threshold z = 1 (Margin slab)
      ctx.strokeStyle = 'rgba(181, 137, 0, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(toX(1.0), 0);
      ctx.lineTo(toX(1.0), H);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = SOL.yellow;
      ctx.font = '11px monospace';
      ctx.fillText('Margin z=1', toX(1.0) + 4, toY(3.2));

      // 1. Zero-One Loss: I(z < 0)
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(toX(zMin), toY(1.0));
      ctx.lineTo(toX(0.0), toY(1.0));
      ctx.lineTo(toX(0.0), toY(0.0));
      ctx.lineTo(toX(zMax), toY(0.0));
      ctx.stroke();

      // 2. Logistic Loss: log2(1 + exp(-z))
      ctx.strokeStyle = SOL.green;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let px = 0; px <= W; px += 2) {
        const z = zMin + (px / W) * (zMax - zMin);
        const l = Math.log(1 + Math.exp(-z)) / Math.LN2;
        if (px === 0) ctx.moveTo(px, toY(l));
        else ctx.lineTo(px, toY(l));
      }
      ctx.stroke();

      // 3. Hinge Loss: max(0, 1 - z)
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(toX(zMin), toY(1 - zMin));
      ctx.lineTo(toX(1.0), toY(0.0));
      ctx.lineTo(toX(zMax), toY(0.0));
      ctx.stroke();

      // Markers at current marginZ
      const hingeVal = Math.max(0, 1 - marginZ);
      const logVal = Math.log(1 + Math.exp(-marginZ)) / Math.LN2;
      const zeroOneVal = marginZ < 0 ? 1 : 0;

      // Vertical line at current z
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(toX(marginZ), 0);
      ctx.lineTo(toX(marginZ), H);
      ctx.stroke();
      ctx.setLineDash([]);

      // Points on curves
      ctx.fillStyle = SOL.red;
      ctx.beginPath(); ctx.arc(toX(marginZ), toY(hingeVal), 6, 0, Math.PI * 2); ctx.fill();

      ctx.fillStyle = SOL.green;
      ctx.beginPath(); ctx.arc(toX(marginZ), toY(logVal), 5, 0, Math.PI * 2); ctx.fill();

      ctx.fillStyle = SOL.base01;
      ctx.beginPath(); ctx.arc(toX(marginZ), toY(zeroOneVal), 5, 0, Math.PI * 2); ctx.fill();

      // Legend in top-right
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = SOL.red; ctx.fillText('■ Hinge: max(0, 1-z)', W - 210, 25);
      ctx.fillStyle = SOL.green; ctx.fillText('■ Logistic: log₂(1+e⁻ᶻ)', W - 210, 45);
      ctx.fillStyle = SOL.base01; ctx.fillText('■ 0-1 Loss: I(z < 0)', W - 210, 65);

      // Value readouts
      if (zValDisp) zValDisp.textContent = marginZ.toFixed(2);
      if (hingeValDisp) hingeValDisp.textContent = hingeVal.toFixed(2);
      if (logValDisp) logValDisp.textContent = logVal.toFixed(2);
      if (zeroOneDisp) zeroOneDisp.textContent = zeroOneVal.toString();

      if (regimeStatus) {
        if (marginZ > 1.0) {
          regimeStatus.style.color = SOL.green;
          regimeStatus.textContent = `z > 1 (Beyond Margin): Hinge loss = 0, subgradient = 0 ⇒ Multiplier αᵢ = 0 (Non-support vector, zero model impact).`;
        } else if (Math.abs(marginZ - 1.0) < 0.05) {
          regimeStatus.style.color = SOL.yellow;
          regimeStatus.textContent = `z = 1 (On Margin): Non-differentiable knee. Subgradient ∈ [-1, 0] ⇒ 0 < αᵢ < C (True Support Vector).`;
        } else {
          regimeStatus.style.color = SOL.red;
          regimeStatus.textContent = `z < 1 (Margin Violator / Misclassified): Hinge loss = 1 - z = ${hingeVal.toFixed(2)} > 0, gradient = -1 ⇒ Multiplier αᵢ = C, slack ξᵢ > 0.`;
        }
      }
    }

    if (zSlider) {
      zSlider.addEventListener('input', (e) => {
        marginZ = parseFloat(e.target.value);
        requestAnimationFrame(draw);
      });
    }

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // INITIALIZATION ON SCRIPT LOAD
  // =========================================================================
  function initAll() {
    initBridgeConnectors();
    initDecisionSurfaceDemo();
    initLagrangeDemo();
    initDualityExplorerDemo();
    initKernelPlaygroundDemo();
    initLossZooDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

})();
