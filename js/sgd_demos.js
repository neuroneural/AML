/**
 * CS8850: Advanced Machine Learning - Lecture 15 (Stochastic Gradient Descent)
 * Interactive Pedagogical Demonstrations Suite
 *
 * 0. Dynamic Bridge Slide Connectors (Precision SVG alignment)
 * 1. Batch vs. Mini-Batch vs. Stochastic Gradient Descent Explorer (1:1 Contour + Sample Gradient Fan + Noise Ball)
 * 2. Ravines, Condition Number, Heavy-Ball Momentum & Nesterov Lookahead Laboratory (1:1 Canyon + Vector Magnifier)
 * 3. Adaptive Optimizers Arena: SGD, Momentum, AdaGrad, RMSprop, Adam & Muon (Saddle Points, Banana Valleys & Sparse Coordinates)
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
   */
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

  // =========================================================================
  // 0. DYNAMIC BRIDGE SLIDE CONNECTORS (Pixel-perfect alignment)
  // =========================================================================
  function updateBridgeConnectors() {
    const container = document.getElementById('bridge-container');
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    if (cRect.width < 50) return;
    const baseW = container.offsetWidth || 940;
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

    setCurve('conn-0-2', c0, getRowLeftCenter('bridge-row-2'));
    setCurve('conn-0-6', c0, getRowLeftCenter('bridge-row-6'));
    setCurve('conn-1-9', c1, getRowLeftCenter('bridge-row-9'));
    setCurve('conn-1-11', c1, getRowLeftCenter('bridge-row-11'));
    setCurve('conn-1-13', c1, getRowLeftCenter('bridge-row-13'));
    setCurve('conn-2-14', c2, getRowLeftCenter('bridge-row-14'));
    setCurve('conn-2-16', c2, getRowLeftCenter('bridge-row-16'));
    setCurve('conn-2-20', c2, getRowLeftCenter('bridge-row-20'));
    setCurve('conn-3-7', c3, getRowLeftCenter('bridge-row-7'));
    setCurve('conn-3-20', c3, getRowLeftCenter('bridge-row-20'));
    setCurve('conn-4-21', c4, getRowLeftCenter('bridge-row-21'));
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
        if (e.currentSlide && (e.currentSlide === slide || e.currentSlide.contains(slide))) {
          requestAnimationFrame(updateBridgeConnectors);
          setTimeout(updateBridgeConnectors, 60);
          setTimeout(updateBridgeConnectors, 220);
        }
      });
      Reveal.addEventListener('ready', () => {
        setTimeout(updateBridgeConnectors, 120);
      });
    }
    setTimeout(updateBridgeConnectors, 150);
  }

  // =========================================================================
  // 1. DEMO 1: BATCH VS. MINI-BATCH VS. STOCHASTIC GRADIENT DESCENT
  // =========================================================================
  function initVariantsDemo() {
    const canvas = document.getElementById('sgd-variants-canvas');
    if (!canvas) return;

    const btnStep = document.getElementById('var-btn-step');
    const btn10Step = document.getElementById('var-btn-10step');
    const btnAuto = document.getElementById('var-btn-auto');
    const btnReset = document.getElementById('var-btn-reset');
    const chkAnneal = document.getElementById('var-chk-anneal');
    const chkFan = document.getElementById('var-chk-fan');
    const sliderB = document.getElementById('var-b-slider');
    const sliderEta = document.getElementById('var-eta-slider');
    const sliderNoise = document.getElementById('var-noise-slider');

    const valStep = document.getElementById('var-step-val');
    const valLossBatch = document.getElementById('var-loss-batch');
    const valLossMB = document.getElementById('var-loss-mb');
    const valLossSGD = document.getElementById('var-loss-sgd');
    const valB = document.getElementById('var-b-val');
    const valEta = document.getElementById('var-eta-val');
    const valNoise = document.getElementById('var-noise-val');

    const N = 64;
    // Quadratic curvature of empirical risk J(theta) = 0.5 * (lam1 * theta1^2 + lam2 * theta2^2)
    const lam1 = 1.15;
    const lam2 = 2.15;

    let sampleOffsets = [];
    let rng = pseudoRandom(158850);
    let startPos = { x: -3.15, y: -1.85 };
    let pathBatch = [];
    let pathMB = [];
    let pathSGD = [];
    let stepCount = 0;
    let autoTimer = null;

    // Generate N=64 zero-mean sample gradient perturbations so sum_{i=1}^N grad_i(theta) == N * grad J(theta)
    function buildSamples() {
      const r = pseudoRandom(42015);
      const raw = [];
      let sumX = 0;
      let sumY = 0;
      for (let i = 0; i < N; i++) {
        const gx = sampleGaussian(r, 0, 1.0);
        const gy = sampleGaussian(r, 0, 1.0);
        raw.push({ gx, gy });
        sumX += gx;
        sumY += gy;
      }
      const meanX = sumX / N;
      const meanY = sumY / N;
      sampleOffsets = raw.map((p) => ({
        gx: p.gx - meanX,
        gy: p.gy - meanY
      }));
    }

    function evalJ(p) {
      return 0.5 * (lam1 * p.x * p.x + lam2 * p.y * p.y);
    }

    function gradJ(p) {
      return { gx: lam1 * p.x, gy: lam2 * p.y };
    }

    function sampleGrad(p, idx, sigmaG) {
      const g = gradJ(p);
      const off = sampleOffsets[idx % N];
      return {
        gx: g.gx + sigmaG * off.gx,
        gy: g.gy + sigmaG * off.gy
      };
    }

    function stopAuto() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
      }
      if (btnAuto) btnAuto.textContent = '▶ Auto-Run';
    }

    function resetSim() {
      stopAuto();
      rng = pseudoRandom(98765);
      stepCount = 0;
      const j0 = evalJ(startPos);
      pathBatch = [{ x: startPos.x, y: startPos.y, loss: j0, evals: 0 }];
      pathMB = [{ x: startPos.x, y: startPos.y, loss: j0, evals: 0 }];
      pathSGD = [{ x: startPos.x, y: startPos.y, loss: j0, evals: 0 }];
      draw();
    }

    function doOneStep() {
      if (stepCount >= 180) {
        stopAuto();
        return;
      }
      const B = parseInt(sliderB.value, 10) || 8;
      const eta0 = parseFloat(sliderEta.value) || 0.18;
      const sigmaG = parseFloat(sliderNoise.value) || 1.2;
      const anneal = chkAnneal && chkAnneal.checked;

      stepCount++;
      const etaT = anneal ? eta0 / (1 + 0.045 * stepCount) : eta0;

      // 1. Batch GD (uses exact gradient over all N=64 samples)
      const prevB = pathBatch[pathBatch.length - 1];
      const gB = gradJ(prevB);
      const nextB = {
        x: prevB.x - eta0 * gB.gx,
        y: prevB.y - eta0 * gB.gy
      };
      nextB.loss = evalJ(nextB);
      nextB.evals = prevB.evals + N;
      pathBatch.push(nextB);

      // 2. Mini-Batch GD (averages B random sample gradients)
      const prevMB = pathMB[pathMB.length - 1];
      let sumMBx = 0;
      let sumMBy = 0;
      for (let k = 0; k < B; k++) {
        const idx = Math.floor(rng() * N);
        const gi = sampleGrad(prevMB, idx, sigmaG);
        sumMBx += gi.gx;
        sumMBy += gi.gy;
      }
      const nextMB = {
        x: prevMB.x - etaT * (sumMBx / B),
        y: prevMB.y - etaT * (sumMBy / B)
      };
      nextMB.loss = evalJ(nextMB);
      nextMB.evals = prevMB.evals + B;
      pathMB.push(nextMB);

      // 3. Stochastic GD (uses 1 random sample gradient)
      const prevS = pathSGD[pathSGD.length - 1];
      const idxS = Math.floor(rng() * N);
      const gS = sampleGrad(prevS, idxS, sigmaG);
      const nextS = {
        x: prevS.x - etaT * gS.gx,
        y: prevS.y - etaT * gS.gy
      };
      nextS.loss = evalJ(nextS);
      nextS.evals = prevS.evals + 1;
      pathSGD.push(nextS);
    }

    // Left 2D contour plot geometry: EXACT 1:1 aspect ratio (pxPerUnitX === pxPerUnitY = 64)
    const plotL = {
      x: 46,
      y: 34,
      w: 512,
      h: 384,
      cx: 46 + 256,
      cy: 34 + 192,
      pxPerUnit: 64 // 512 / 64 = 8 units wide [-4, 4]; 384 / 64 = 6 units tall [-3, 3]
    };

    function dataToPx(wx, wy) {
      return {
        x: plotL.cx + wx * plotL.pxPerUnit,
        y: plotL.cy - wy * plotL.pxPerUnit
      };
    }

    function pxToData(px, py) {
      return {
        x: (px - plotL.cx) / plotL.pxPerUnit,
        y: (plotL.cy - py) / plotL.pxPerUnit
      };
    }

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const B = parseInt(sliderB.value, 10) || 8;
      const eta0 = parseFloat(sliderEta.value) || 0.18;
      const sigmaG = parseFloat(sliderNoise.value) || 1.2;
      const anneal = chkAnneal && chkAnneal.checked;
      const showFan = chkFan && chkFan.checked;
      const curEta = anneal ? eta0 / (1 + 0.045 * stepCount) : eta0;

      if (valB) valB.textContent = String(B);
      if (valEta) valEta.textContent = eta0.toFixed(2);
      if (valNoise) valNoise.textContent = sigmaG.toFixed(2);
      if (valStep) valStep.textContent = String(stepCount);

      const curB = pathBatch[pathBatch.length - 1];
      const curMB = pathMB[pathMB.length - 1];
      const curS = pathSGD[pathSGD.length - 1];

      if (valLossBatch) valLossBatch.textContent = curB.loss.toFixed(4);
      if (valLossMB) valLossMB.textContent = curMB.loss.toFixed(4);
      if (valLossSGD) valLossSGD.textContent = curS.loss.toFixed(4);

      // ---------------------------------------------------------------------
      // LEFT PANEL: 2D Parameter Space (Exact 1:1 Aspect Ratio)
      // ---------------------------------------------------------------------
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.fillRect(plotL.x, plotL.y, plotL.w, plotL.h);
      ctx.strokeRect(plotL.x, plotL.y, plotL.w, plotL.h);

      ctx.beginPath();
      ctx.rect(plotL.x, plotL.y, plotL.w, plotL.h);
      ctx.clip();

      // Grid lines (every 1 unit in 1:1 scale)
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.22)';
      ctx.lineWidth = 1;
      for (let gx = -4; gx <= 4; gx++) {
        const p = dataToPx(gx, 0);
        ctx.beginPath();
        ctx.moveTo(p.x, plotL.y);
        ctx.lineTo(p.x, plotL.y + plotL.h);
        ctx.stroke();
      }
      for (let gy = -3; gy <= 3; gy++) {
        const p = dataToPx(0, gy);
        ctx.beginPath();
        ctx.moveTo(plotL.x, p.y);
        ctx.lineTo(plotL.x + plotL.w, p.y);
        ctx.stroke();
      }

      // Coordinate axes through (0,0)
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.45)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(plotL.x, plotL.cy);
      ctx.lineTo(plotL.x + plotL.w, plotL.cy);
      ctx.moveTo(plotL.cx, plotL.y);
      ctx.lineTo(plotL.cx, plotL.y + plotL.h);
      ctx.stroke();

      // Iso-loss contours of J(theta) = 0.5*(lam1*x^2 + lam2*y^2) = c
      const levels = [0.15, 0.45, 1.0, 1.8, 3.0, 4.6, 6.8, 9.5, 13.0];
      levels.forEach((c, idx) => {
        const rx = Math.sqrt((2 * c) / lam1) * plotL.pxPerUnit;
        const ry = Math.sqrt((2 * c) / lam2) * plotL.pxPerUnit;
        ctx.beginPath();
        ctx.ellipse(plotL.cx, plotL.cy, rx, ry, 0, 0, 2 * Math.PI);
        ctx.strokeStyle = idx % 2 === 0 ? 'rgba(38, 139, 210, 0.28)' : 'rgba(42, 161, 152, 0.22)';
        ctx.lineWidth = 1.3;
        ctx.stroke();
      });

      // Draw Stationary Noise Ball around optimum (0,0) for SGD: radius ~ curEta * sigmaG * 0.72
      const noiseRadiusUnits = Math.sqrt(curEta) * sigmaG * 0.62;
      const noiseRadiusPx = noiseRadiusUnits * plotL.pxPerUnit;
      ctx.save();
      ctx.beginPath();
      ctx.arc(plotL.cx, plotL.cy, noiseRadiusPx, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(220, 50, 47, 0.08)';
      ctx.fill();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = 'rgba(220, 50, 47, 0.65)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      // Optimum star at (0,0)
      ctx.fillStyle = SOL.base03;
      ctx.beginPath();
      ctx.arc(plotL.cx, plotL.cy, 4.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('θ* (Minimum)', plotL.cx + 7, plotL.cy - 6);

      // Optional: Draw fan of individual sample negative gradients -eta * grad l_i(theta) at current SGD point
      if (showFan) {
        const pCenter = dataToPx(curS.x, curS.y);
        const scaleVec = 0.22;
        for (let i = 0; i < 18; i++) {
          const gi = sampleGrad(curS, i * 3, sigmaG);
          const tip = dataToPx(curS.x - scaleVec * gi.gx, curS.y - scaleVec * gi.gy);
          drawArrow(ctx, pCenter.x, pCenter.y, tip.x, tip.y, 'rgba(108, 113, 196, 0.35)', 1.2, 5);
        }
        // True mean negative gradient at curS
        const gTrue = gradJ(curS);
        const tipMean = dataToPx(curS.x - scaleVec * gTrue.gx, curS.y - scaleVec * gTrue.gy);
        drawArrow(ctx, pCenter.x, pCenter.y, tipMean.x, tipMean.y, SOL.violet, 2.5, 7);
      }

      // Helper to draw optimization trajectory
      function drawTrajectory(path, color, lineWidth) {
        if (path.length < 2) return;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < path.length; i++) {
          const pt = dataToPx(path[i].x, path[i].y);
          if (i === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        }
        ctx.stroke();

        const last = dataToPx(path[path.length - 1].x, path[path.length - 1].y);
        ctx.fillStyle = color;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(last.x, last.y, 5.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }

      drawTrajectory(pathSGD, SOL.red, 1.6);
      drawTrajectory(pathMB, SOL.green, 2.2);
      drawTrajectory(pathBatch, SOL.blue, 2.6);

      // Start marker theta_0
      const p0 = dataToPx(startPos.x, startPos.y);
      ctx.fillStyle = SOL.base02;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(p0.x, p0.y, 5.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('θ₀', p0.x + 8, p0.y - 6);

      ctx.restore();

      // Left Panel Titles & Axis Labels
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.fillText('2D Parameter Space (Exact 1:1 Aspect Ratio)', plotL.x, 22);

      // Legend inside top-left of contour plot
      ctx.save();
      ctx.fillStyle = 'rgba(253, 246, 227, 0.92)';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.fillRect(plotL.x + 8, plotL.y + 8, 226, 68);
      ctx.strokeRect(plotL.x + 8, plotL.y + 8, 226, 68);
      ctx.font = 'bold 11.5px sans-serif';
      ctx.fillStyle = SOL.blue;
      ctx.fillText('━● Batch GD (B = 64, Var = 0)', plotL.x + 15, plotL.y + 25);
      ctx.fillStyle = SOL.green;
      ctx.fillText(`━● Mini-Batch (B = ${B}, Var = Σ/B)`, plotL.x + 15, plotL.y + 43);
      ctx.fillStyle = SOL.red;
      ctx.fillText('━● Stochastic GD (B = 1, Var = Σ)', plotL.x + 15, plotL.y + 61);
      ctx.restore();

      ctx.fillStyle = SOL.base01;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Parameter θ₁', plotL.cx, plotL.y + plotL.h + 22);
      ctx.save();
      ctx.translate(18, plotL.cy);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Parameter θ₂', 0, 0);
      ctx.restore();

      // ---------------------------------------------------------------------
      // RIGHT PANEL TOP: Objective J(theta) vs. Compute (Gradient Evaluations)
      // ---------------------------------------------------------------------
      const boxTop = { x: 604, y: 34, w: 554, h: 174 };
      drawChartBox(
        ctx,
        boxTop,
        'Objective J(θ) vs. Compute Budget (Sample Gradient Evaluations)',
        'Gradient Evaluations (Cost = B per step)',
        'log₁₀ J(θ)',
        [
          { path: pathBatch, color: SOL.blue, xKey: 'evals' },
          { path: pathMB, color: SOL.green, xKey: 'evals' },
          { path: pathSGD, color: SOL.red, xKey: 'evals' }
        ],
        Math.max(256, pathMB[pathMB.length - 1].evals, pathSGD[pathSGD.length - 1].evals * 2)
      );

      // ---------------------------------------------------------------------
      // RIGHT PANEL BOTTOM: Objective J(theta) vs. Update Steps t
      // ---------------------------------------------------------------------
      const boxBot = { x: 604, y: 244, w: 554, h: 174 };
      drawChartBox(
        ctx,
        boxBot,
        `Objective J(θ) vs. Update Steps t  (Current ηₜ = ${curEta.toFixed(3)})`,
        'Parameter Update Step t',
        'log₁₀ J(θ)',
        [
          { path: pathBatch, color: SOL.blue, xKey: 'step' },
          { path: pathMB, color: SOL.green, xKey: 'step' },
          { path: pathSGD, color: SOL.red, xKey: 'step' }
        ],
        Math.max(30, stepCount)
      );
    }

    function drawChartBox(ctx, box, title, xLabel, yLabel, seriesList, maxX) {
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.fillRect(box.x, box.y, box.w, box.h);
      ctx.strokeRect(box.x, box.y, box.w, box.h);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(title, box.x, box.y - 8);

      // Log10 range: [-4.0, 1.2]
      const minLog = -4.0;
      const maxLog = 1.2;

      // Horizontal grid lines
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.25)';
      ctx.fillStyle = SOL.base01;
      ctx.font = '10.5px monospace';
      ctx.textAlign = 'right';
      [-3, -2, -1, 0, 1].forEach((lg) => {
        const py = box.y + box.h - ((lg - minLog) / (maxLog - minLog)) * box.h;
        ctx.beginPath();
        ctx.moveTo(box.x, py);
        ctx.lineTo(box.x + box.w, py);
        ctx.stroke();
        ctx.fillText(`10^${lg}`, box.x - 4, py + 3);
      });

      // Plot each series
      ctx.save();
      ctx.beginPath();
      ctx.rect(box.x, box.y, box.w, box.h);
      ctx.clip();

      seriesList.forEach((s) => {
        if (s.path.length < 2) return;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < s.path.length; i++) {
          const xVal = s.xKey === 'evals' ? s.path[i].evals : i;
          if (xVal > maxX) break;
          const px = box.x + (xVal / maxX) * box.w;
          const lg = Math.max(minLog, Math.min(maxLog, Math.log10(Math.max(1e-5, s.path[i].loss))));
          const py = box.y + box.h - ((lg - minLog) / (maxLog - minLog)) * box.h;
          if (!started) {
            ctx.moveTo(px, py);
            started = true;
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.stroke();
      });
      ctx.restore();

      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${xLabel} (0 to ${maxX})`, box.x + box.w / 2, box.y + box.h + 16);
      ctx.restore();
    }

    // Event Listeners
    if (btnStep) {
      btnStep.addEventListener('click', () => {
        stopAuto();
        doOneStep();
        draw();
      });
    }
    if (btn10Step) {
      btn10Step.addEventListener('click', () => {
        stopAuto();
        for (let k = 0; k < 10; k++) doOneStep();
        draw();
      });
    }
    if (btnAuto) {
      btnAuto.addEventListener('click', () => {
        if (autoTimer) {
          stopAuto();
        } else {
          if (stepCount >= 180) resetSim();
          btnAuto.textContent = '⏸ Pause';
          autoTimer = setInterval(() => {
            doOneStep();
            draw();
            if (stepCount >= 180) stopAuto();
          }, 70);
        }
      });
    }
    if (btnReset) btnReset.addEventListener('click', resetSim);
    if (chkAnneal) chkAnneal.addEventListener('change', draw);
    if (chkFan) chkFan.addEventListener('change', draw);
    if (sliderB) sliderB.addEventListener('input', resetSim);
    if (sliderEta) sliderEta.addEventListener('input', resetSim);
    if (sliderNoise) sliderNoise.addEventListener('input', resetSim);

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = (canvas._logicalW || 1180) / rect.width;
      const scaleY = (canvas._logicalH || 450) / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;
      if (mx >= plotL.x && mx <= plotL.x + plotL.w && my >= plotL.y && my <= plotL.y + plotL.h) {
        const d = pxToData(mx, my);
        startPos = {
          x: Math.max(-3.8, Math.min(3.8, d.x)),
          y: Math.max(-2.8, Math.min(2.8, d.y))
        };
        resetSim();
      }
    });

    buildSamples();
    resetSim();
    for (let k = 0; k < 6; k++) doOneStep();
    draw();
    registerSlideListener(canvas, draw);
  }

  // =========================================================================
  // 2. DEMO 2: RAVINES, CONDITION NUMBER, MOMENTUM & NESTEROV LOOKAHEAD
  // =========================================================================
  function initMomentumDemo() {
    const canvas = document.getElementById('sgd-momentum-canvas');
    if (!canvas) return;

    const btnStep = document.getElementById('mom-btn-step');
    const btnAuto = document.getElementById('mom-btn-auto');
    const btnReset = document.getElementById('mom-btn-reset');
    const preIso = document.getElementById('mom-pre-isotropic');
    const preRavine = document.getElementById('mom-pre-ravine');
    const preUnstable = document.getElementById('mom-pre-unstable');

    const sliderKappa = document.getElementById('mom-kappa-slider');
    const sliderEta = document.getElementById('mom-eta-slider');
    const sliderGamma = document.getElementById('mom-gamma-slider');

    const valStep = document.getElementById('mom-step-val');
    const valGD = document.getElementById('mom-gd-val');
    const valHB = document.getElementById('mom-hb-val');
    const valNAG = document.getElementById('mom-nag-val');
    const valLimit = document.getElementById('mom-limit-val');
    const valKappa = document.getElementById('mom-kappa-val');
    const valEta = document.getElementById('mom-eta-val');
    const valGamma = document.getElementById('mom-gamma-val');

    let startPt = { x: -3.45, y: -1.65 };
    let pathGD = [];
    let pathHB = [];
    let pathNAG = [];
    let stepCount = 0;
    let autoTimer = null;

    // Left 2D contour plot: EXACT 1:1 aspect ratio (pxPerUnitX === pxPerUnitY = 74)
    const plotL = {
      x: 44,
      y: 34,
      w: 592, // 8 units wide [-4, 4] * 74 = 592
      h: 370, // 5 units tall [-2.5, 2.5] * 74 = 370
      cx: 44 + 296,
      cy: 34 + 185,
      pxPerUnit: 74
    };

    function dataToPx(wx, wy) {
      return {
        x: plotL.cx + wx * plotL.pxPerUnit,
        y: plotL.cy - wy * plotL.pxPerUnit
      };
    }

    function pxToData(px, py) {
      return {
        x: (px - plotL.cx) / plotL.pxPerUnit,
        y: (plotL.cy - py) / plotL.pxPerUnit
      };
    }

    function evalJ(x, y, kappa) {
      return 0.5 * (x * x + kappa * y * y);
    }

    function stopAuto() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
      }
      if (btnAuto) btnAuto.textContent = '▶ Auto-Run';
    }

    function resetSim() {
      stopAuto();
      stepCount = 0;
      const kappa = parseFloat(sliderKappa.value) || 20;
      const j0 = evalJ(startPt.x, startPt.y, kappa);
      pathGD = [{ x: startPt.x, y: startPt.y, vx: 0, vy: 0, loss: j0 }];
      pathHB = [{ x: startPt.x, y: startPt.y, vx: 0, vy: 0, loss: j0 }];
      pathNAG = [
        {
          x: startPt.x,
          y: startPt.y,
          vx: 0,
          vy: 0,
          lookX: startPt.x,
          lookY: startPt.y,
          loss: j0
        }
      ];
      draw();
    }

    function doOneStep() {
      if (stepCount >= 80) {
        stopAuto();
        return;
      }
      const kappa = parseFloat(sliderKappa.value) || 20;
      const eta = parseFloat(sliderEta.value) || 0.065;
      const gamma = parseFloat(sliderGamma.value) || 0.78;

      stepCount++;

      // 1. Vanilla GD
      const pGD = pathGD[pathGD.length - 1];
      const gxGD = pGD.x;
      const gyGD = kappa * pGD.y;
      const nxGD = Math.max(-12, Math.min(12, pGD.x - eta * gxGD));
      const nyGD = Math.max(-12, Math.min(12, pGD.y - eta * gyGD));
      pathGD.push({ x: nxGD, y: nyGD, vx: eta * gxGD, vy: eta * gyGD, loss: evalJ(nxGD, nyGD, kappa) });

      // 2. Heavy-Ball Momentum: v_t = gamma * v_{t-1} + eta * grad J(theta_{t-1})
      const pHB = pathHB[pathHB.length - 1];
      const gxHB = pHB.x;
      const gyHB = kappa * pHB.y;
      const vxHB = gamma * pHB.vx + eta * gxHB;
      const vyHB = gamma * pHB.vy + eta * gyHB;
      const nxHB = pHB.x - vxHB;
      const nyHB = pHB.y - vyHB;
      pathHB.push({ x: nxHB, y: nyHB, vx: vxHB, vy: vyHB, loss: evalJ(nxHB, nyHB, kappa) });

      // 3. Nesterov Accelerated Gradient: lookahead tilde_theta = theta_{t-1} - gamma * v_{t-1}
      const pNAG = pathNAG[pathNAG.length - 1];
      const lookX = pNAG.x - gamma * pNAG.vx;
      const lookY = pNAG.y - gamma * pNAG.vy;
      const gxNAG = lookX;
      const gyNAG = kappa * lookY;
      const vxNAG = gamma * pNAG.vx + eta * gxNAG;
      const vyNAG = gamma * pNAG.vy + eta * gyNAG;
      const nxNAG = pNAG.x - vxNAG;
      const nyNAG = pNAG.y - vyNAG;
      pathNAG.push({
        x: nxNAG,
        y: nyNAG,
        vx: vxNAG,
        vy: vyNAG,
        lookX,
        lookY,
        gxLook: gxNAG,
        gyLook: gyNAG,
        gxOld: pNAG.x,
        gyOld: kappa * pNAG.y,
        loss: evalJ(nxNAG, nyNAG, kappa)
      });
    }

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const kappa = parseFloat(sliderKappa.value) || 20;
      const eta = parseFloat(sliderEta.value) || 0.065;
      const gamma = parseFloat(sliderGamma.value) || 0.78;
      const limitEta = 2.0 / kappa;

      if (valKappa) valKappa.textContent = String(kappa);
      if (valEta) valEta.textContent = eta.toFixed(3);
      if (valGamma) valGamma.textContent = gamma.toFixed(2);
      if (valLimit) valLimit.textContent = limitEta.toFixed(3);
      if (valStep) valStep.textContent = String(stepCount);

      const curGD = pathGD[pathGD.length - 1];
      const curHB = pathHB[pathHB.length - 1];
      const curNAG = pathNAG[pathNAG.length - 1];

      if (valGD) valGD.textContent = curGD.loss > 999 ? 'DIVERGED' : curGD.loss.toFixed(4);
      if (valHB) valHB.textContent = curHB.loss.toFixed(4);
      if (valNAG) valNAG.textContent = curNAG.loss.toFixed(4);

      // ---------------------------------------------------------------------
      // LEFT PANEL: 2D Canyon Contour Plot (Exact 1:1 Aspect Ratio)
      // ---------------------------------------------------------------------
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.fillRect(plotL.x, plotL.y, plotL.w, plotL.h);
      ctx.strokeRect(plotL.x, plotL.y, plotL.w, plotL.h);

      ctx.beginPath();
      ctx.rect(plotL.x, plotL.y, plotL.w, plotL.h);
      ctx.clip();

      // 1:1 Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.22)';
      ctx.lineWidth = 1;
      for (let gx = -4; gx <= 4; gx++) {
        const p = dataToPx(gx, 0);
        ctx.beginPath();
        ctx.moveTo(p.x, plotL.y);
        ctx.lineTo(p.x, plotL.y + plotL.h);
        ctx.stroke();
      }
      for (let gy = -2; gy <= 2; gy++) {
        const p = dataToPx(0, gy);
        ctx.beginPath();
        ctx.moveTo(plotL.x, p.y);
        ctx.lineTo(plotL.x + plotL.w, p.y);
        ctx.stroke();
      }

      // Valley floor axis
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.5)';
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(plotL.x, plotL.cy);
      ctx.lineTo(plotL.x + plotL.w, plotL.cy);
      ctx.moveTo(plotL.cx, plotL.y);
      ctx.lineTo(plotL.cx, plotL.y + plotL.h);
      ctx.stroke();

      // Exact geometric contour ellipses: 0.5*(x^2 + kappa*y^2) = c => rx = sqrt(2c), ry = sqrt(2c/kappa)
      const cLevels = [0.1, 0.4, 1.0, 2.2, 4.0, 7.0, 11.0, 17.0, 25.0, 36.0];
      cLevels.forEach((c, idx) => {
        const rx = Math.sqrt(2 * c) * plotL.pxPerUnit;
        const ry = Math.sqrt((2 * c) / kappa) * plotL.pxPerUnit;
        ctx.beginPath();
        ctx.ellipse(plotL.cx, plotL.cy, rx, ry, 0, 0, 2 * Math.PI);
        ctx.strokeStyle = idx % 2 === 0 ? 'rgba(38, 139, 210, 0.28)' : 'rgba(42, 161, 152, 0.22)';
        ctx.lineWidth = 1.3;
        ctx.stroke();
      });

      // Draw trajectories
      function drawPath(path, color, lw) {
        if (path.length < 2) return;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < path.length; i++) {
          const p = dataToPx(path[i].x, path[i].y);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();

        // Dots at vertices
        for (let i = 1; i < path.length; i++) {
          const p = dataToPx(path[i].x, path[i].y);
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, i === path.length - 1 ? 5.5 : 2.6, 0, 2 * Math.PI);
          ctx.fill();
        }
        ctx.restore();
      }

      drawPath(pathGD, SOL.red, 1.8);
      drawPath(pathHB, SOL.yellow, 2.2);
      drawPath(pathNAG, SOL.green, 2.5);

      // Optimum & Start markers
      ctx.fillStyle = SOL.base03;
      ctx.beginPath();
      ctx.arc(plotL.cx, plotL.cy, 4.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('θ* (0,0)', plotL.cx + 7, plotL.cy - 6);

      const p0 = dataToPx(startPt.x, startPt.y);
      ctx.fillStyle = SOL.base02;
      ctx.beginPath();
      ctx.arc(p0.x, p0.y, 5.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillText('θ₀', p0.x + 8, p0.y - 6);

      ctx.restore();

      // Legend inside left plot
      ctx.save();
      ctx.fillStyle = 'rgba(253, 246, 227, 0.92)';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.fillRect(plotL.x + 8, plotL.y + 8, 228, 66);
      ctx.strokeRect(plotL.x + 8, plotL.y + 8, 228, 66);
      ctx.font = 'bold 11.5px sans-serif';
      ctx.fillStyle = SOL.red;
      ctx.fillText('━● Vanilla GD (No Momentum)', plotL.x + 15, plotL.y + 25);
      ctx.fillStyle = SOL.yellow;
      ctx.fillText('━● Heavy-Ball Momentum (Polyak)', plotL.x + 15, plotL.y + 43);
      ctx.fillStyle = SOL.green;
      ctx.fillText('━● Nesterov Lookahead (NAG)', plotL.x + 15, plotL.y + 61);
      ctx.restore();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.fillText(`2D Ravine Contour Plot (1:1 Scale, Condition Number κ = ${kappa})`, plotL.x, 22);

      ctx.fillStyle = SOL.base01;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Valley Floor Direction θ₁ (λ₁ = 1)', plotL.cx, plotL.y + plotL.h + 22);
      ctx.save();
      ctx.translate(18, plotL.cy);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(`Steep Wall θ₂ (λ₂ = ${kappa})`, 0, 0);
      ctx.restore();

      // ---------------------------------------------------------------------
      // RIGHT TOP: Live Vector Lookahead Magnifier (NAG vs. Momentum)
      // ---------------------------------------------------------------------
      const magBox = { x: 658, y: 34, w: 500, h: 204 };
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.fillRect(magBox.x, magBox.y, magBox.w, magBox.h);
      ctx.strokeRect(magBox.x, magBox.y, magBox.w, magBox.h);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Vector Lookahead Magnifier at Current Step', magBox.x, magBox.y - 8);

      // Compute vectors for current or next step from curNAG
      const prevIdx = Math.max(0, pathNAG.length - 2);
      const basePt = pathNAG[prevIdx];
      const momVecX = -gamma * basePt.vx;
      const momVecY = -gamma * basePt.vy;
      const lookX = basePt.x + momVecX;
      const lookY = basePt.y + momVecY;

      const gradOldX = -eta * basePt.x;
      const gradOldY = -eta * kappa * basePt.y;
      const gradLookX = -eta * lookX;
      const gradLookY = -eta * kappa * lookY;

      // Auto-scale vectors to fit nicely inside magBox
      const maxSpan = Math.max(
        0.35,
        Math.hypot(momVecX + gradLookX, momVecY + gradLookY),
        Math.hypot(momVecX + gradOldX, momVecY + gradOldY),
        Math.hypot(gradOldX, gradOldY)
      );
      const scaleM = Math.min(130, 72 / maxSpan);
      const originX = magBox.x + 125;
      const originY = magBox.y + 115;

      // Valley horizontal reference line inside magnifier
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.35)';
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(magBox.x + 12, originY);
      ctx.lineTo(magBox.x + magBox.w - 12, originY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Origin theta_{t-1}
      ctx.fillStyle = SOL.base02;
      ctx.beginPath();
      ctx.arc(originX, originY, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('θₜ', originX - 20, originY + 4);

      // 1. Lookahead vector -gamma * v_{t-1} (Blue)
      const lx = originX + momVecX * scaleM;
      const ly = originY - momVecY * scaleM;
      drawArrow(ctx, originX, originY, lx, ly, SOL.blue, 2.6, 8);
      ctx.fillStyle = SOL.blue;
      ctx.font = 'bold 11.5px sans-serif';
      ctx.fillText('1. Lookahead −γvₜ₋₁ (θ̃ₜ)', lx - 30, ly - 10);

      // 2. Corrective gradient at lookahead -eta * grad J(tilde_theta) (Red)
      const nagTipX = lx + gradLookX * scaleM;
      const nagTipY = ly - gradLookY * scaleM;
      drawArrow(ctx, lx, ly, nagTipX, nagTipY, SOL.red, 2.6, 8);
      ctx.fillStyle = SOL.red;
      ctx.fillText('2. −η∇J(θ̃ₜ) (at lookahead)', nagTipX + 8, nagTipY + 4);

      // 3. Combined NAG step (Green)
      drawArrow(ctx, originX, originY, nagTipX, nagTipY, SOL.green, 2.8, 9);
      ctx.fillStyle = SOL.green;
      ctx.fillText('NAG θₜ₊₁', nagTipX + 8, nagTipY - 12);

      // 4. Compare Heavy-Ball gradient evaluated at old point (Yellow dashed)
      const hbTipX = lx + gradOldX * scaleM;
      const hbTipY = ly - gradOldY * scaleM;
      ctx.save();
      ctx.setLineDash([4, 3]);
      drawArrow(ctx, lx, ly, hbTipX, hbTipY, SOL.yellow, 2.0, 7);
      ctx.restore();
      ctx.fillStyle = SOL.yellow;
      ctx.font = '11px sans-serif';
      ctx.fillText('−η∇J(θₜ) (old gradient)', hbTipX + 8, hbTipY + 12);

      ctx.restore();

      // ---------------------------------------------------------------------
      // RIGHT BOTTOM: Convergence Plot log10 J(theta) vs. Step t
      // ---------------------------------------------------------------------
      const convBox = { x: 658, y: 270, w: 500, h: 148 };
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.fillRect(convBox.x, convBox.y, convBox.w, convBox.h);
      ctx.strokeRect(convBox.x, convBox.y, convBox.w, convBox.h);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Convergence Rate: log₁₀ J(θₜ) vs. Step t', convBox.x, convBox.y - 8);

      const minLg = -4.0;
      const maxLg = 2.0;
      const maxT = Math.max(35, stepCount);

      ctx.strokeStyle = 'rgba(147, 161, 161, 0.25)';
      ctx.fillStyle = SOL.base01;
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      [-3, -1, 1].forEach((lg) => {
        const py = convBox.y + convBox.h - ((lg - minLg) / (maxLg - minLg)) * convBox.h;
        ctx.beginPath();
        ctx.moveTo(convBox.x, py);
        ctx.lineTo(convBox.x + convBox.w, py);
        ctx.stroke();
        ctx.fillText(`10^${lg}`, convBox.x - 4, py + 3);
      });

      ctx.save();
      ctx.beginPath();
      ctx.rect(convBox.x, convBox.y, convBox.w, convBox.h);
      ctx.clip();

      [
        { path: pathGD, color: SOL.red },
        { path: pathHB, color: SOL.yellow },
        { path: pathNAG, color: SOL.green }
      ].forEach((s) => {
        if (s.path.length < 2) return;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        for (let i = 0; i < s.path.length; i++) {
          const px = convBox.x + (i / maxT) * convBox.w;
          const lg = Math.max(minLg, Math.min(maxLg, Math.log10(Math.max(1e-5, s.path[i].loss))));
          const py = convBox.y + convBox.h - ((lg - minLg) / (maxLg - minLg)) * convBox.h;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      });
      ctx.restore();

      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Iteration Step t (0 to ${maxT})`, convBox.x + convBox.w / 2, convBox.y + convBox.h + 16);
      ctx.restore();
    }

    // Event Listeners
    if (btnStep) {
      btnStep.addEventListener('click', () => {
        stopAuto();
        doOneStep();
        draw();
      });
    }
    if (btnAuto) {
      btnAuto.addEventListener('click', () => {
        if (autoTimer) {
          stopAuto();
        } else {
          if (stepCount >= 80) resetSim();
          btnAuto.textContent = '⏸ Pause';
          autoTimer = setInterval(() => {
            doOneStep();
            draw();
            if (stepCount >= 80) stopAuto();
          }, 80);
        }
      });
    }
    if (btnReset) btnReset.addEventListener('click', resetSim);

    if (preIso) {
      preIso.addEventListener('click', () => {
        sliderKappa.value = '1';
        sliderEta.value = '0.120';
        sliderGamma.value = '0.80';
        resetSim();
      });
    }
    if (preRavine) {
      preRavine.addEventListener('click', () => {
        sliderKappa.value = '20';
        sliderEta.value = '0.065';
        sliderGamma.value = '0.78';
        resetSim();
      });
    }
    if (preUnstable) {
      preUnstable.addEventListener('click', () => {
        sliderKappa.value = '20';
        sliderEta.value = '0.105'; // > 2/20 = 0.100 so Vanilla GD diverges!
        sliderGamma.value = '0.78';
        resetSim();
      });
    }

    if (sliderKappa) sliderKappa.addEventListener('input', resetSim);
    if (sliderEta) sliderEta.addEventListener('input', resetSim);
    if (sliderGamma) sliderGamma.addEventListener('input', resetSim);

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = (canvas._logicalW || 1180) / rect.width;
      const scaleY = (canvas._logicalH || 450) / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;
      if (mx >= plotL.x && mx <= plotL.x + plotL.w && my >= plotL.y && my <= plotL.y + plotL.h) {
        const d = pxToData(mx, my);
        startPt = {
          x: Math.max(-3.8, Math.min(3.8, d.x)),
          y: Math.max(-2.3, Math.min(2.3, d.y))
        };
        resetSim();
      }
    });

    resetSim();
    for (let k = 0; k < 3; k++) doOneStep();
    draw();
    registerSlideListener(canvas, draw);
  }

  // =========================================================================
  // 3. DEMO 3: ADAPTIVE OPTIMIZERS ARENA (SGD, MOM, ADAGRAD, RMSPROP, ADAM, MUON)
  // =========================================================================
  function initAdaptiveDemo() {
    const canvas = document.getElementById('sgd-adaptive-canvas');
    if (!canvas) return;

    const selSurface = document.getElementById('ada-sel-surface');
    const btnStep = document.getElementById('ada-btn-step');
    const btnAuto = document.getElementById('ada-btn-auto');
    const btnReset = document.getElementById('ada-btn-reset');
    const chkBias = document.getElementById('ada-chk-bias');

    const sliderEta = document.getElementById('ada-eta-slider');
    const sliderB1 = document.getElementById('ada-b1-slider');
    const sliderB2 = document.getElementById('ada-b2-slider');

    const valStep = document.getElementById('ada-step-val');
    const valLossSGD = document.getElementById('ada-loss-sgd');
    const valLossMom = document.getElementById('ada-loss-mom');
    const valLossAdaGrad = document.getElementById('ada-loss-adagrad');
    const valLossRMS = document.getElementById('ada-loss-rms');
    const valLossAdam = document.getElementById('ada-loss-adam');
    const valLossMuon = document.getElementById('ada-loss-muon');

    const valEta = document.getElementById('ada-eta-val');
    const valB1 = document.getElementById('ada-b1-val');
    const valB2 = document.getElementById('ada-b2-val');

    // Left 2D landscape box: EXACT 1:1 aspect ratio (pxPerUnitX === pxPerUnitY = 82)
    const plotL = {
      x: 44,
      y: 34,
      w: 574, // 7 units wide [-3.5, 3.5] * 82 = 574
      h: 378, // 4.60975 units tall [-2.305, 2.305] * 82 = 378
      cx: 44 + 287,
      cy: 34 + 189,
      pxPerUnit: 82
    };

    function dataToPx(wx, wy) {
      return {
        x: plotL.cx + wx * plotL.pxPerUnit,
        y: plotL.cy - wy * plotL.pxPerUnit
      };
    }

    function pxToData(px, py) {
      return {
        x: (px - plotL.cx) / plotL.pxPerUnit,
        y: (plotL.cy - py) / plotL.pxPerUnit
      };
    }

    // 3 Surfaces:
    // 1. Saddle Point Plateau: J(x, y) = 0.55*x^2 - 0.65*y^2 + 0.16*y^4 + 0.660156 (min = 0 at (0, +-1.425))
    // 2. Curved Banana Valley: J(x, y) = 0.2*(x - 1.2)^2 + 1.6*(y - 0.32*x^2 + 0.35)^2
    // 3. Sparse / Ill-Scaled Ravine: J(x, y) = 0.08*(x - 1.8)^2 + 2.4*(y + 0.4)^2 + early steep pulse
    function evalSurface(mode, x, y) {
      if (mode === 'saddle') {
        return Math.max(1e-5, 0.55 * x * x - 0.65 * y * y + 0.16 * y * y * y * y + 0.66015625);
      } else if (mode === 'rosenbrock') {
        const dx = x - 1.2;
        const dy = y - (0.32 * x * x - 0.55);
        return Math.max(1e-5, 0.22 * dx * dx + 1.45 * dy * dy);
      } else {
        const dx = x - 1.85;
        const dy = y + 0.35;
        return Math.max(1e-5, 0.07 * dx * dx + 2.8 * dy * dy);
      }
    }

    function gradSurface(mode, x, y, step) {
      if (mode === 'saddle') {
        return {
          gx: 1.1 * x,
          gy: -1.3 * y + 0.64 * y * y * y
        };
      } else if (mode === 'rosenbrock') {
        const dx = x - 1.2;
        const dy = y - (0.32 * x * x - 0.55);
        return {
          gx: 0.44 * dx + 2.9 * dy * (-0.64 * x),
          gy: 2.9 * dy
        };
      } else {
        const dx = x - 1.85;
        const dy = y + 0.35;
        // In sparse/multi-scale mode, early steps have high gradient bursts that trap AdaGrad's accumulator
        const burst = step <= 5 ? 4.5 : 1.0;
        return {
          gx: 0.14 * dx * burst,
          gy: 5.6 * dy
        };
      }
    }

    // 5-Step Newton-Schulz Orthogonalization of a 2x2 embedded momentum matrix M
    // Embeds a 2D momentum vector (mx, my) + local curvature coupling into a 2x2 weight block and applies 5-step Newton-Schulz
    function newtonSchulz2x2(m00, m01, m10, m11) {
      const frob = Math.hypot(m00, m01, m10, m11) + 1e-8;
      let x00 = m00 / frob;
      let x01 = m01 / frob;
      let x10 = m10 / frob;
      let x11 = m11 / frob;

      // Standard Newton-Schulz iteration X_{k+1} = 1.5 * X_k - 0.5 * (X_k X_k^T) X_k
      for (let s = 0; s < 5; s++) {
        // A = X * X^T
        const a00 = x00 * x00 + x01 * x01;
        const a01 = x00 * x10 + x01 * x11;
        const a10 = a01;
        const a11 = x10 * x10 + x11 * x11;

        // B = A * X
        const b00 = a00 * x00 + a01 * x10;
        const b01 = a00 * x01 + a01 * x11;
        const b10 = a10 * x00 + a11 * x10;
        const b11 = a10 * x01 + a11 * x11;

        x00 = 1.5 * x00 - 0.5 * b00;
        x01 = 1.5 * x01 - 0.5 * b01;
        x10 = 1.5 * x10 - 0.5 * b10;
        x11 = 1.5 * x11 - 0.5 * b11;
      }
      return { x00, x01, x10, x11 };
    }

    let startPt = { x: -2.45, y: 0.04 };
    let optimizers = {};
    let stepCount = 0;
    let autoTimer = null;
    let contourCache = null;
    let cachedSurfaceMode = '';

    function defaultStartForSurface(mode) {
      if (mode === 'saddle') return { x: -2.45, y: 0.035 };
      if (mode === 'rosenbrock') return { x: -2.35, y: 1.55 };
      return { x: -2.65, y: 1.65 };
    }

    function stopAuto() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
      }
      if (btnAuto) btnAuto.textContent = '▶ Auto-Race';
    }

    function resetSim() {
      stopAuto();
      stepCount = 0;
      const mode = selSurface ? selSurface.value : 'saddle';
      const j0 = evalSurface(mode, startPt.x, startPt.y);

      const makeInit = (color, label) => ({
        label,
        color,
        x: startPt.x,
        y: startPt.y,
        mx: 0,
        my: 0,
        mCross: 0,
        vx: 0,
        vy: 0,
        effX: 1,
        effY: 1,
        path: [{ x: startPt.x, y: startPt.y, loss: j0 }]
      });

      optimizers = {
        sgd: makeInit(SOL.red, 'SGD'),
        mom: makeInit(SOL.yellow, 'Momentum'),
        adagrad: makeInit(SOL.violet, 'AdaGrad'),
        rms: makeInit(SOL.cyan, 'RMSprop'),
        adam: makeInit(SOL.blue, 'Adam'),
        muon: makeInit(SOL.green, 'Muon')
      };

      draw();
    }

    function doOneStep() {
      if (stepCount >= 110) {
        stopAuto();
        return;
      }
      const mode = selSurface ? selSurface.value : 'saddle';
      const eta = parseFloat(sliderEta.value) || 0.08;
      const beta1 = parseFloat(sliderB1.value) || 0.9;
      const beta2 = parseFloat(sliderB2.value) || 0.95;
      const useBias = chkBias && chkBias.checked;
      const eps = 1e-6;

      stepCount++;

      // 1. Vanilla SGD
      {
        const o = optimizers.sgd;
        const g = gradSurface(mode, o.x, o.y, stepCount);
        o.x = Math.max(-3.45, Math.min(3.45, o.x - eta * g.gx));
        o.y = Math.max(-2.25, Math.min(2.25, o.y - eta * g.gy));
        o.effX = eta;
        o.effY = eta;
        o.path.push({ x: o.x, y: o.y, loss: evalSurface(mode, o.x, o.y) });
      }

      // 2. Heavy-Ball Momentum
      {
        const o = optimizers.mom;
        const g = gradSurface(mode, o.x, o.y, stepCount);
        o.mx = beta1 * o.mx + (1 - beta1) * g.gx;
        o.my = beta1 * o.my + (1 - beta1) * g.gy;
        const scaleM = eta * 2.2;
        o.x = Math.max(-3.45, Math.min(3.45, o.x - scaleM * o.mx));
        o.y = Math.max(-2.25, Math.min(2.25, o.y - scaleM * o.my));
        o.effX = scaleM;
        o.effY = scaleM;
        o.path.push({ x: o.x, y: o.y, loss: evalSurface(mode, o.x, o.y) });
      }

      // 3. AdaGrad (cumulative sum of squared gradients)
      {
        const o = optimizers.adagrad;
        const g = gradSurface(mode, o.x, o.y, stepCount);
        o.vx += g.gx * g.gx;
        o.vy += g.gy * g.gy;
        const etaAda = eta * 3.2;
        o.effX = etaAda / (Math.sqrt(o.vx) + eps);
        o.effY = etaAda / (Math.sqrt(o.vy) + eps);
        o.x = Math.max(-3.45, Math.min(3.45, o.x - o.effX * g.gx));
        o.y = Math.max(-2.25, Math.min(2.25, o.y - o.effY * g.gy));
        o.path.push({ x: o.x, y: o.y, loss: evalSurface(mode, o.x, o.y) });
      }

      // 4. RMSprop (EMA of squared gradients)
      {
        const o = optimizers.rms;
        const g = gradSurface(mode, o.x, o.y, stepCount);
        o.vx = beta2 * o.vx + (1 - beta2) * g.gx * g.gx;
        o.vy = beta2 * o.vy + (1 - beta2) * g.gy * g.gy;
        const etaRMS = eta * 1.15;
        o.effX = etaRMS / (Math.sqrt(o.vx) + eps);
        o.effY = etaRMS / (Math.sqrt(o.vy) + eps);
        o.x = Math.max(-3.45, Math.min(3.45, o.x - o.effX * g.gx));
        o.y = Math.max(-2.25, Math.min(2.25, o.y - o.effY * g.gy));
        o.path.push({ x: o.x, y: o.y, loss: evalSurface(mode, o.x, o.y) });
      }

      // 5. Adam (1st + 2nd moment + optional bias correction)
      {
        const o = optimizers.adam;
        const g = gradSurface(mode, o.x, o.y, stepCount);
        o.mx = beta1 * o.mx + (1 - beta1) * g.gx;
        o.my = beta1 * o.my + (1 - beta1) * g.gy;
        o.vx = beta2 * o.vx + (1 - beta2) * g.gx * g.gx;
        o.vy = beta2 * o.vy + (1 - beta2) * g.gy * g.gy;

        const corr1 = useBias ? 1 - Math.pow(beta1, stepCount) : 1.0;
        const corr2 = useBias ? 1 - Math.pow(beta2, stepCount) : 1.0;

        const mHatX = o.mx / corr1;
        const mHatY = o.my / corr1;
        const vHatX = o.vx / corr2;
        const vHatY = o.vy / corr2;

        const etaAdam = eta * 1.25;
        o.effX = etaAdam / (Math.sqrt(vHatX) + eps);
        o.effY = etaAdam / (Math.sqrt(vHatY) + eps);

        o.x = Math.max(-3.45, Math.min(3.45, o.x - o.effX * mHatX));
        o.y = Math.max(-2.25, Math.min(2.25, o.y - o.effY * mHatY));
        o.path.push({ x: o.x, y: o.y, loss: evalSurface(mode, o.x, o.y) });
      }

      // 6. Muon (Nesterov Momentum + 5-Step Newton-Schulz Orthogonalization of 2x2 Matrix Update)
      {
        const o = optimizers.muon;
        const g = gradSurface(mode, o.x, o.y, stepCount);
        o.mx = beta1 * o.mx + g.gx;
        o.my = beta1 * o.my + g.gy;
        // Nesterov lookahead momentum
        const nmx = beta1 * o.mx + g.gx;
        const nmy = beta1 * o.my + g.gy;
        o.mCross = beta1 * o.mCross + 0.25 * (g.gx * o.y - g.gy * o.x);

        // Orthogonalize 2x2 momentum matrix via 5-step Newton-Schulz so both singular values -> 1
        const ortho = newtonSchulz2x2(nmx, o.mCross, o.mCross, nmy);
        const etaMuon = eta * 1.15;
        o.effX = etaMuon / (Math.abs(nmx) + 0.05);
        o.effY = etaMuon / (Math.abs(nmy) + 0.05);

        o.x = Math.max(-3.45, Math.min(3.45, o.x - etaMuon * ortho.x00));
        o.y = Math.max(-2.25, Math.min(2.25, o.y - etaMuon * ortho.x11));
        o.path.push({ x: o.x, y: o.y, loss: evalSurface(mode, o.x, o.y) });
      }
    }

    function ensureContourHeatmap(mode) {
      if (contourCache && cachedSurfaceMode === mode) return contourCache;
      const off = document.createElement('canvas');
      off.width = plotL.w;
      off.height = plotL.h;
      const octx = off.getContext('2d');
      const img = octx.createImageData(plotL.w, plotL.h);
      const data = img.data;

      for (let py = 0; py < plotL.h; py++) {
        for (let px = 0; px < plotL.w; px++) {
          const wx = (px - plotL.w / 2) / plotL.pxPerUnit;
          const wy = (plotL.h / 2 - py) / plotL.pxPerUnit;
          const val = evalSurface(mode, wx, wy);
          const norm = Math.min(1, Math.log1p(val) / 2.6);

          // Subtle contour bands
          const band = Math.abs((( Math.sqrt(val) * 5.5 ) % 1) - 0.5) < 0.055 ? 0.88 : 1.0;
          const r = Math.round((253 - norm * 32) * band);
          const g = Math.round((246 - norm * 28) * band);
          const b = Math.round((227 - norm * 14) * band);

          const idx = (py * plotL.w + px) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      contourCache = off;
      cachedSurfaceMode = mode;
      return off;
    }

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const mode = selSurface ? selSurface.value : 'saddle';
      const eta = parseFloat(sliderEta.value) || 0.08;
      const beta1 = parseFloat(sliderB1.value) || 0.9;
      const beta2 = parseFloat(sliderB2.value) || 0.95;

      if (valEta) valEta.textContent = eta.toFixed(2);
      if (valB1) valB1.textContent = beta1.toFixed(2);
      if (valB2) valB2.textContent = beta2.toFixed(2);
      if (valStep) valStep.textContent = String(stepCount);

      const lastLoss = (k) => {
        const p = optimizers[k].path;
        return p[p.length - 1].loss.toFixed(3);
      };
      if (valLossSGD) valLossSGD.textContent = lastLoss('sgd');
      if (valLossMom) valLossMom.textContent = lastLoss('mom');
      if (valLossAdaGrad) valLossAdaGrad.textContent = lastLoss('adagrad');
      if (valLossRMS) valLossRMS.textContent = lastLoss('rms');
      if (valLossAdam) valLossAdam.textContent = lastLoss('adam');
      if (valLossMuon) valLossMuon.textContent = lastLoss('muon');

      // ---------------------------------------------------------------------
      // LEFT PANEL: 2D Landscape (Exact 1:1 Aspect Ratio)
      // ---------------------------------------------------------------------
      ctx.save();
      const bgImg = ensureContourHeatmap(mode);
      ctx.drawImage(bgImg, plotL.x, plotL.y);
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(plotL.x, plotL.y, plotL.w, plotL.h);

      ctx.beginPath();
      ctx.rect(plotL.x, plotL.y, plotL.w, plotL.h);
      ctx.clip();

      // Coordinate axes
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.35)';
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(plotL.x, plotL.cy);
      ctx.lineTo(plotL.x + plotL.w, plotL.cy);
      ctx.moveTo(plotL.cx, plotL.y);
      ctx.lineTo(plotL.cx, plotL.y + plotL.h);
      ctx.stroke();

      // Mark saddle / optima targets
      if (mode === 'saddle') {
        const ps = dataToPx(0, 0);
        ctx.fillStyle = SOL.red;
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText('✕ Saddle (0,0)', ps.x + 6, ps.y - 5);

        [1.425, -1.425].forEach((ym) => {
          const pm = dataToPx(0, ym);
          ctx.fillStyle = SOL.base03;
          ctx.beginPath();
          ctx.arc(pm.x, pm.y, 4.5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.fillText('★ Min', pm.x + 7, pm.y + 4);
        });
      } else if (mode === 'rosenbrock') {
        const pm = dataToPx(1.2, 0.32 * 1.44 - 0.55);
        ctx.fillStyle = SOL.base03;
        ctx.beginPath();
        ctx.arc(pm.x, pm.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText('★ Global Min', pm.x + 7, pm.y + 4);
      } else {
        const pm = dataToPx(1.85, -0.35);
        ctx.fillStyle = SOL.base03;
        ctx.beginPath();
        ctx.arc(pm.x, pm.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText('★ Global Min', pm.x + 7, pm.y + 4);
      }

      // Draw each optimizer trajectory
      Object.keys(optimizers).forEach((key) => {
        const o = optimizers[key];
        if (o.path.length < 2) return;
        ctx.save();
        ctx.strokeStyle = o.color;
        ctx.lineWidth = key === 'muon' || key === 'adam' ? 2.6 : 2.0;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < o.path.length; i++) {
          const pt = dataToPx(o.path[i].x, o.path[i].y);
          if (i === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        }
        ctx.stroke();

        const last = dataToPx(o.x, o.y);
        ctx.fillStyle = o.color;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(last.x, last.y, 5.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      });

      // Start marker
      const p0 = dataToPx(startPt.x, startPt.y);
      ctx.fillStyle = SOL.base02;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(p0.x, p0.y, 5.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('θ₀', p0.x - 18, p0.y - 8);

      ctx.restore();

      // Legend overlay
      ctx.save();
      ctx.fillStyle = 'rgba(253, 246, 227, 0.92)';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.fillRect(plotL.x + 8, plotL.y + 8, 175, 108);
      ctx.strokeRect(plotL.x + 8, plotL.y + 8, 175, 108);
      ctx.font = 'bold 11px sans-serif';
      const legItems = [
        ['sgd', '━● Vanilla SGD'],
        ['mom', '━● Momentum'],
        ['adagrad', '━● AdaGrad (Σ g²)'],
        ['rms', '━● RMSprop (EMA g²)'],
        ['adam', '━● Adam (1st + 2nd)'],
        ['muon', '━● Muon (Ortho UVᵀ)']
      ];
      legItems.forEach((item, idx) => {
        ctx.fillStyle = optimizers[item[0]].color;
        ctx.fillText(item[1], plotL.x + 14, plotL.y + 23 + idx * 16);
      });
      ctx.restore();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.fillText('2D Optimization Landscape (1:1 Geometric Aspect Ratio)', plotL.x, 22);

      ctx.fillStyle = SOL.base01;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Parameter θ₁', plotL.cx, plotL.y + plotL.h + 22);
      ctx.save();
      ctx.translate(18, plotL.cy);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Parameter θ₂', 0, 0);
      ctx.restore();

      // ---------------------------------------------------------------------
      // RIGHT TOP: Effective Coordinate Step Multiplier Bars
      // ---------------------------------------------------------------------
      const barBox = { x: 648, y: 34, w: 510, h: 168 };
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.fillRect(barBox.x, barBox.y, barBox.w, barBox.h);
      ctx.strokeRect(barBox.x, barBox.y, barBox.w, barBox.h);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Effective Per-Coordinate Step Gain (θ₁ vs. θ₂) at Step t', barBox.x, barBox.y - 8);

      const keys = ['sgd', 'mom', 'adagrad', 'rms', 'adam', 'muon'];
      const colW = barBox.w / keys.length;
      keys.forEach((k, idx) => {
        const o = optimizers[k];
        const cx = barBox.x + (idx + 0.5) * colW;
        const maxBarH = 115;
        const h1 = Math.min(maxBarH, Math.max(4, Math.log1p(o.effX * 14) * 38));
        const h2 = Math.min(maxBarH, Math.max(4, Math.log1p(o.effY * 14) * 38));

        // Bar 1 (theta_1)
        ctx.fillStyle = o.color;
        ctx.fillRect(cx - 17, barBox.y + 138 - h1, 15, h1);
        // Bar 2 (theta_2)
        ctx.globalAlpha = 0.65;
        ctx.fillRect(cx + 2, barBox.y + 138 - h2, 15, h2);
        ctx.globalAlpha = 1.0;

        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(o.label, cx, barBox.y + 155);
      });

      ctx.fillStyle = SOL.base01;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('Solid: θ₁ gain  |  Tinted: θ₂ gain', barBox.x + barBox.w - 10, barBox.y + 18);
      ctx.restore();

      // ---------------------------------------------------------------------
      // RIGHT BOTTOM: Objective J(theta) vs. Step t
      // ---------------------------------------------------------------------
      const lossBox = { x: 648, y: 238, w: 510, h: 174 };
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.2;
      ctx.fillRect(lossBox.x, lossBox.y, lossBox.w, lossBox.h);
      ctx.strokeRect(lossBox.x, lossBox.y, lossBox.w, lossBox.h);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Objective J(θₜ) vs. Iteration Step t (Log Scale)', lossBox.x, lossBox.y - 8);

      const minLg = -3.5;
      const maxLg = 1.2;
      const maxT = Math.max(40, stepCount);

      ctx.strokeStyle = 'rgba(147, 161, 161, 0.25)';
      ctx.fillStyle = SOL.base01;
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      [-3, -2, -1, 0, 1].forEach((lg) => {
        const py = lossBox.y + lossBox.h - ((lg - minLg) / (maxLg - minLg)) * lossBox.h;
        ctx.beginPath();
        ctx.moveTo(lossBox.x, py);
        ctx.lineTo(lossBox.x + lossBox.w, py);
        ctx.stroke();
        ctx.fillText(`10^${lg}`, lossBox.x - 4, py + 3);
      });

      ctx.save();
      ctx.beginPath();
      ctx.rect(lossBox.x, lossBox.y, lossBox.w, lossBox.h);
      ctx.clip();

      keys.forEach((k) => {
        const o = optimizers[k];
        if (o.path.length < 2) return;
        ctx.strokeStyle = o.color;
        ctx.lineWidth = k === 'muon' || k === 'adam' ? 2.4 : 1.8;
        ctx.beginPath();
        for (let i = 0; i < o.path.length; i++) {
          const px = lossBox.x + (i / maxT) * lossBox.w;
          const lg = Math.max(minLg, Math.min(maxLg, Math.log10(Math.max(1e-4, o.path[i].loss))));
          const py = lossBox.y + lossBox.h - ((lg - minLg) / (maxLg - minLg)) * lossBox.h;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      });
      ctx.restore();

      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Iteration Step t (0 to ${maxT})`, lossBox.x + lossBox.w / 2, lossBox.y + lossBox.h + 16);
      ctx.restore();
    }

    // Event Listeners
    if (selSurface) {
      selSurface.addEventListener('change', () => {
        startPt = defaultStartForSurface(selSurface.value);
        resetSim();
      });
    }
    if (btnStep) {
      btnStep.addEventListener('click', () => {
        stopAuto();
        doOneStep();
        draw();
      });
    }
    if (btnAuto) {
      btnAuto.addEventListener('click', () => {
        if (autoTimer) {
          stopAuto();
        } else {
          if (stepCount >= 110) resetSim();
          btnAuto.textContent = '⏸ Pause';
          autoTimer = setInterval(() => {
            doOneStep();
            draw();
            if (stepCount >= 110) stopAuto();
          }, 70);
        }
      });
    }
    if (btnReset) btnReset.addEventListener('click', resetSim);
    if (chkBias) chkBias.addEventListener('change', resetSim);
    if (sliderEta) sliderEta.addEventListener('input', resetSim);
    if (sliderB1) sliderB1.addEventListener('input', resetSim);
    if (sliderB2) sliderB2.addEventListener('input', resetSim);

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = (canvas._logicalW || 1180) / rect.width;
      const scaleY = (canvas._logicalH || 450) / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;
      if (mx >= plotL.x && mx <= plotL.x + plotL.w && my >= plotL.y && my <= plotL.y + plotL.h) {
        const d = pxToData(mx, my);
        startPt = {
          x: Math.max(-3.3, Math.min(3.3, d.x)),
          y: Math.max(-2.1, Math.min(2.1, d.y))
        };
        resetSim();
      }
    });

    startPt = defaultStartForSurface('saddle');
    resetSim();
    for (let k = 0; k < 18; k++) doOneStep();
    draw();
    registerSlideListener(canvas, draw);
  }

  // Initialize all demos on DOM ready
  function initAll() {
    initBridgeSlide();
    initVariantsDemo();
    initMomentumDemo();
    initAdaptiveDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
