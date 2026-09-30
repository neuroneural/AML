/**
 * CS8850: Advanced Machine Learning - Lecture 12 (Kernel Density Estimation & Non-parametric Methods)
 * Interactive Pedagogical Demonstrations
 * 
 * 1. Dynamic Bridge Slide Connectors (Precision SVG alignment)
 * 2. Histogram Sensitivity: Bin Width vs. Origin Shift Explorer
 * 3. Parzen Window & Box Kernel Accumulator (with Lecture Exercise calculation)
 * 4. Smooth KDE: The "Sum of Bumps" Explorer (Gaussian, Epanechnikov, Box, Triangular)
 * 5. Bandwidth Selection, Bias-Variance Tradeoff & Silverman's Rule of Thumb Explorer
 * 6. 2D Multivariate KDE: Isotropic vs. Pre-Whitened / Product Kernels Explorer
 * 7. Non-parametric Bayes Classifier: Two-Class Decision Boundary & Outlier Vulnerability
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
   * High-DPI canvas setup with robust fallback when canvas is in a hidden Reveal slide.
   */
  function setupHiDPI(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const declaredW = parseInt(canvas.getAttribute('width'), 10) || 940;
    const declaredH = parseInt(canvas.getAttribute('height'), 10) || 340;
    const w = (rect.width > 50) ? Math.round(rect.width) : declaredW;
    const h = (rect.height > 50) ? Math.round(rect.height) : declaredH;

    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, width: w, height: h };
  }

  /**
   * Helper to ensure canvases render immediately when navigated to without requiring interaction.
   */
  function registerSlideListener(canvas, drawFn) {
    window.addEventListener('resize', drawFn);

    if (window.ResizeObserver) {
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.contentRect.width > 50) {
            drawFn();
          }
        }
      });
      ro.observe(canvas);
    }

    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) {
          requestAnimationFrame(drawFn);
          setTimeout(drawFn, 50);
          setTimeout(drawFn, 200);
          setTimeout(drawFn, 400);
        }
      });
      Reveal.addEventListener('ready', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) {
          setTimeout(drawFn, 100);
          setTimeout(drawFn, 300);
        }
      });
    }
  }

  // Common random generator with seed
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
  // DYNAMIC BRIDGE SLIDE CONNECTORS (Pixel-perfect alignment)
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
      path.setAttribute('d', `M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} C ${(p1.x + dx).toFixed(1)} ${p1.y.toFixed(1)}, ${(p2.x - dx).toFixed(1)} ${p2.y.toFixed(1)}, ${(p2.x - 4).toFixed(1)} ${p2.y.toFixed(1)}`);
    }

    const c0 = getCardRightCenter('bridge-card-0');
    const c1 = getCardRightCenter('bridge-card-1');
    const c2 = getCardRightCenter('bridge-card-2');
    const c3 = getCardRightCenter('bridge-card-3');
    const c4 = getCardRightCenter('bridge-card-4');

    setCurve('conn-0-2', c0, getRowLeftCenter('bridge-row-2'));
    setCurve('conn-0-3', c0, getRowLeftCenter('bridge-row-3'));
    setCurve('conn-1-4', c1, getRowLeftCenter('bridge-row-4'));
    setCurve('conn-1-6', c1, getRowLeftCenter('bridge-row-6'));
    setCurve('conn-2-8', c2, getRowLeftCenter('bridge-row-8'));
    setCurve('conn-3-9', c3, getRowLeftCenter('bridge-row-9'));
    setCurve('conn-3-10', c3, getRowLeftCenter('bridge-row-10'));
    setCurve('conn-3-11', c3, getRowLeftCenter('bridge-row-11'));
    setCurve('conn-4-12', c4, getRowLeftCenter('bridge-row-12'));
  }

  function initBridgeSlide() {
    window.addEventListener('resize', updateBridgeConnectors);

    const container = document.getElementById('bridge-container');
    if (container && window.ResizeObserver) {
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.contentRect.width > 50) {
            updateBridgeConnectors();
          }
        }
      });
      ro.observe(container);
    }

    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        const slide = document.getElementById('bridge-slide');
        if (e.currentSlide && (e.currentSlide === slide || e.currentSlide.contains(slide))) {
          requestAnimationFrame(updateBridgeConnectors);
          setTimeout(updateBridgeConnectors, 50);
          setTimeout(updateBridgeConnectors, 200);
          setTimeout(updateBridgeConnectors, 400);
        }
      });
      Reveal.addEventListener('fragmentshown', updateBridgeConnectors);
      Reveal.addEventListener('fragmenthidden', updateBridgeConnectors);
      Reveal.addEventListener('ready', () => {
        setTimeout(updateBridgeConnectors, 100);
        setTimeout(updateBridgeConnectors, 300);
      });
    }
    setTimeout(updateBridgeConnectors, 100);
  }

  // =========================================================================
  // DEMO 1: Histogram Sensitivity (Bin Width vs. Origin Shifting)
  // =========================================================================
  function initHistogramDemo() {
    const canvas = document.getElementById('hist-demo-canvas');
    if (!canvas) return;

    let binWidth = 1.6;
    let binOrigin = 0.0;
    let showSliding = false;
    let currentDataset = 'bimodal';

    const datasets = {
      bimodal: [
        1.2, 1.5, 1.7, 1.8, 2.1, 2.3, 2.4, 2.5, 2.7, 2.8, 3.1, 3.2,
        6.2, 6.4, 6.5, 6.8, 7.0, 7.1, 7.2, 7.5, 7.7, 8.0, 8.2, 8.5
      ],
      skewed: [
        0.8, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 1.9, 2.1, 2.4, 2.8, 3.3, 3.9, 4.6, 5.5, 6.8, 8.5
      ],
      separated: [
        1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 5.5, 5.8, 6.1, 6.4, 6.7, 7.0, 9.5
      ]
    };

    let data = [...datasets[currentDataset]];

    const sliderWidth = document.getElementById('hist-width-slider');
    const sliderOrigin = document.getElementById('hist-origin-slider');
    const valWidth = document.getElementById('hist-width-val');
    const valOrigin = document.getElementById('hist-origin-val');
    const valBins = document.getElementById('hist-bins-count');
    const btnBimodal = document.getElementById('btn-hist-bimodal');
    const btnSkewed = document.getElementById('btn-hist-skewed');
    const btnSeparated = document.getElementById('btn-hist-separated');
    const btnSlidingToggle = document.getElementById('btn-hist-sliding-toggle');

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const padL = 60, padR = 40, padT = 30, padB = 55;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;

      const xMin = -1.0, xMax = 11.0;
      function toScreenX(x) { return padL + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toScreenY(y, maxVal) { return padT + plotH - (y / maxVal) * plotH; }

      const startX = binOrigin;
      const minData = Math.min(...data);
      const maxData = Math.max(...data);

      const firstBinIdx = Math.floor((minData - startX) / binWidth) - 1;
      const lastBinIdx = Math.ceil((maxData - startX) / binWidth) + 1;

      const bins = [];
      let maxDensity = 0.05;

      for (let i = firstBinIdx; i <= lastBinIdx; i++) {
        const b0 = startX + i * binWidth;
        const b1 = b0 + binWidth;
        let count = 0;
        for (let d of data) {
          if (d >= b0 && d < b1) count++;
        }
        const density = count / (data.length * binWidth);
        if (density > maxDensity) maxDensity = density;
        bins.push({ b0, b1, count, density });
      }

      maxDensity = Math.max(maxDensity * 1.15, 0.35);

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);

      for (let yVal = 0.1; yVal <= maxDensity; yVal += 0.1) {
        const sy = toScreenY(yVal, maxDensity);
        ctx.beginPath();
        ctx.moveTo(padL, sy);
        ctx.lineTo(padL + plotW, sy);
        ctx.stroke();

        ctx.fillStyle = SOL.base01;
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(yVal.toFixed(1), padL - 8, sy + 3);
      }
      ctx.setLineDash([]);

      // X axis
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base01;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      for (let xVal = 0; xVal <= 10; xVal += 2) {
        const sx = toScreenX(xVal);
        ctx.beginPath();
        ctx.moveTo(sx, padT + plotH);
        ctx.lineTo(sx, padT + plotH + 5);
        ctx.stroke();
        ctx.fillText(xVal.toString(), sx, padT + plotH + 18);
      }

      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.base02;
      ctx.textAlign = 'center';
      ctx.fillText('Sample Value (x)', padL + plotW / 2, height - 12);

      // Draw Bars
      let occupiedBins = 0;
      bins.forEach(b => {
        if (b.count > 0) occupiedBins++;
        const x0 = toScreenX(b.b0);
        const x1 = toScreenX(b.b1);
        const yTop = toScreenY(b.density, maxDensity);
        const yBot = toScreenY(0, maxDensity);
        const barW = Math.max(0, x1 - x0);
        const barH = yBot - yTop;

        ctx.fillStyle = b.count > 0 ? 'rgba(38, 139, 210, 0.45)' : 'rgba(238, 232, 213, 0.3)';
        ctx.fillRect(x0, yTop, barW, barH);

        ctx.strokeStyle = SOL.blue;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x0, yTop, barW, barH);

        if (b.count > 0 && barW > 14) {
          ctx.fillStyle = SOL.base02;
          ctx.font = 'bold 11px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(b.count.toString(), x0 + barW / 2, Math.min(yTop + 16, yBot - 6));
        }
      });

      // Origin Line
      const originX = toScreenX(binOrigin);
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(originX, padT);
      ctx.lineTo(originX, padT + plotH + 10);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = SOL.red;
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Origin x₀=' + binOrigin.toFixed(2), originX, padT - 8);

      // Sliding Window Idea
      if (showSliding) {
        const slideCenterX = 4.0;
        const sL = toScreenX(slideCenterX - binWidth / 2);
        const sR = toScreenX(slideCenterX + binWidth / 2);
        const sTop = toScreenY(maxDensity * 0.85, maxDensity);
        const sBot = toScreenY(0, maxDensity);

        ctx.fillStyle = 'rgba(203, 75, 22, 0.25)';
        ctx.fillRect(sL, sTop, sR - sL, sBot - sTop);
        ctx.strokeStyle = SOL.orange;
        ctx.lineWidth = 2;
        ctx.strokeRect(sL, sTop, sR - sL, sBot - sTop);

        ctx.fillStyle = SOL.orange;
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText('Sliding Window (Parzen Idea)', (sL + sR) / 2, sTop - 6);
      }

      // Rug plot
      data.forEach(xVal => {
        const sx = toScreenX(xVal);
        ctx.beginPath();
        ctx.arc(sx, padT + plotH + 4, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = SOL.magenta;
        ctx.fill();
        ctx.strokeStyle = SOL.base03;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      });

      if (valBins) valBins.textContent = occupiedBins.toString();
    }

    if (sliderWidth) {
      sliderWidth.addEventListener('input', (e) => {
        binWidth = parseFloat(e.target.value);
        if (valWidth) valWidth.textContent = binWidth.toFixed(2);
        draw();
      });
    }

    if (sliderOrigin) {
      sliderOrigin.addEventListener('input', (e) => {
        binOrigin = parseFloat(e.target.value);
        if (valOrigin) valOrigin.textContent = binOrigin.toFixed(2);
        draw();
      });
    }

    function setActiveDataBtn(btn) {
      [btnBimodal, btnSkewed, btnSeparated].forEach(b => {
        if (b) { b.style.background = '#fdf6e3'; b.style.fontWeight = 'normal'; }
      });
      if (btn) { btn.style.background = '#eee8d5'; btn.style.fontWeight = 'bold'; }
    }

    if (btnBimodal) {
      btnBimodal.addEventListener('click', () => {
        currentDataset = 'bimodal';
        data = [...datasets.bimodal];
        setActiveDataBtn(btnBimodal);
        draw();
      });
    }

    if (btnSkewed) {
      btnSkewed.addEventListener('click', () => {
        currentDataset = 'skewed';
        data = [...datasets.skewed];
        setActiveDataBtn(btnSkewed);
        draw();
      });
    }

    if (btnSeparated) {
      btnSeparated.addEventListener('click', () => {
        currentDataset = 'separated';
        data = [...datasets.separated];
        setActiveDataBtn(btnSeparated);
        draw();
      });
    }

    if (btnSlidingToggle) {
      btnSlidingToggle.addEventListener('click', () => {
        showSliding = !showSliding;
        btnSlidingToggle.style.background = showSliding ? '#eee8d5' : '#fdf6e3';
        btnSlidingToggle.style.color = showSliding ? SOL.orange : SOL.base01;
        btnSlidingToggle.textContent = showSliding ? 'Hide Sliding Window' : 'Compare Sliding Window';
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // DEMO 2: Parzen Window & Box Kernel Accumulator (with Lecture Exercise)
  // =========================================================================
  function initParzenDemo() {
    const canvas = document.getElementById('parzen-demo-canvas');
    if (!canvas) return;

    const data = [4, 5, 5, 6, 12, 14, 15, 15, 16, 17];
    let h = 4.0;
    let queryY = 15.0;

    const sliderH = document.getElementById('parzen-h-slider');
    const valH = document.getElementById('parzen-h-val');
    const sliderY = document.getElementById('parzen-y-slider');
    const valY = document.getElementById('parzen-y-val');
    const queryResultDiv = document.getElementById('parzen-calc-result');
    const btnY3 = document.getElementById('btn-parzen-y3');
    const btnY10 = document.getElementById('btn-parzen-y10');
    const btnY15 = document.getElementById('btn-parzen-y15');

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const padL = 60, padR = 40, padT = 35, padB = 55;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;

      const xMin = 0.0, xMax = 22.0;
      function toScreenX(x) { return padL + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toScreenY(dens, maxDens) { return padT + plotH - (dens / maxDens) * plotH; }

      const N = data.length;
      const boxHeight = 1.0 / (N * h);

      const steps = 300;
      const xs = [];
      const ys = [];
      let maxDens = 0.15;

      for (let s = 0; s <= steps; s++) {
        const x = xMin + (s / steps) * (xMax - xMin);
        let k = 0;
        for (let pt of data) {
          if (Math.abs(x - pt) < (h / 2.0)) k++;
        }
        const density = k / (N * h);
        if (density > maxDens) maxDens = density;
        xs.push(x);
        ys.push(density);
      }
      maxDens = Math.max(maxDens * 1.25, 0.15);

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (let yVal = 0.025; yVal <= maxDens; yVal += 0.025) {
        const sy = toScreenY(yVal, maxDens);
        ctx.beginPath();
        ctx.moveTo(padL, sy);
        ctx.lineTo(padL + plotW, sy);
        ctx.stroke();

        ctx.fillStyle = SOL.base01;
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(yVal.toFixed(3), padL - 8, sy + 3);
      }
      ctx.setLineDash([]);

      // Axis
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base01;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      for (let xVal = 0; xVal <= 20; xVal += 2) {
        const sx = toScreenX(xVal);
        ctx.beginPath();
        ctx.moveTo(sx, padT + plotH);
        ctx.lineTo(sx, padT + plotH + 5);
        ctx.stroke();
        ctx.fillText(xVal.toString(), sx, padT + plotH + 18);
      }

      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.base02;
      ctx.fillText('Feature x / y', padL + plotW / 2, height - 12);

      // 1. Boxes
      data.forEach(pt => {
        const xL = toScreenX(pt - h / 2);
        const xR = toScreenX(pt + h / 2);
        const yTop = toScreenY(boxHeight, maxDens);
        const yBot = toScreenY(0, maxDens);

        ctx.fillStyle = 'rgba(42, 161, 152, 0.12)';
        ctx.fillRect(xL, yTop, xR - xL, yBot - yTop);
        ctx.strokeStyle = 'rgba(42, 161, 152, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(xL, yTop, xR - xL, yBot - yTop);
      });

      // 2. Aggregate Step Curve
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const sx = toScreenX(xs[s]);
        const sy = toScreenY(ys[s], maxDens);
        if (s === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      ctx.lineTo(toScreenX(xs[steps]), toScreenY(0, maxDens));
      ctx.lineTo(toScreenX(xs[0]), toScreenY(0, maxDens));
      ctx.closePath();
      ctx.fillStyle = 'rgba(38, 139, 210, 0.2)';
      ctx.fill();

      // 3. Highlight Query Point y
      let kQuery = 0;
      const insidePoints = [];
      data.forEach(pt => {
        if (Math.abs(queryY - pt) < (h / 2.0)) {
          kQuery++;
          insidePoints.push(pt);
        }
      });
      const queryDens = kQuery / (N * h);

      const syLine = toScreenX(queryY);
      const wL = toScreenX(queryY - h / 2);
      const wR = toScreenX(queryY + h / 2);

      ctx.fillStyle = 'rgba(203, 75, 22, 0.15)';
      ctx.fillRect(wL, padT, wR - wL, plotH);
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 2]);
      ctx.strokeRect(wL, padT, wR - wL, plotH);
      ctx.setLineDash([]);

      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(syLine, padT);
      ctx.lineTo(syLine, padT + plotH);
      ctx.stroke();

      const curveY = toScreenY(queryDens, maxDens);
      ctx.beginPath();
      ctx.arc(syLine, curveY, 6, 0, 2 * Math.PI);
      ctx.fillStyle = SOL.red;
      ctx.fill();
      ctx.strokeStyle = SOL.base3;
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = SOL.red;
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`P(${queryY}) = ${queryDens.toFixed(4)}`, syLine + 10, curveY - 8);

      data.forEach(pt => {
        const sx = toScreenX(pt);
        const isInside = Math.abs(queryY - pt) < (h / 2.0);
        ctx.beginPath();
        ctx.arc(sx, padT + plotH + 5, 4, 0, 2 * Math.PI);
        ctx.fillStyle = isInside ? SOL.red : SOL.cyan;
        ctx.fill();
        ctx.strokeStyle = SOL.base03;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      if (queryResultDiv) {
        queryResultDiv.innerHTML = `<b>P<sub>KDE</sub>(y = ${queryY})</b> = <sup>1</sup>/<sub>(10 &times; ${h})</sub> &times; [${kQuery} points inside] = <b>${kQuery} / ${N * h} = ${queryDens.toFixed(4)}</b> &emsp; <span style="color:#586e75;">Points in [${(queryY - h/2).toFixed(1)}, ${(queryY + h/2).toFixed(1)}]: {${insidePoints.join(', ') || 'none'}}</span>`;
      }
    }

    if (sliderH) {
      sliderH.addEventListener('input', (e) => {
        h = parseFloat(e.target.value);
        if (valH) valH.textContent = h.toFixed(1);
        draw();
      });
    }

    if (sliderY) {
      sliderY.addEventListener('input', (e) => {
        queryY = parseFloat(e.target.value);
        if (valY) valY.textContent = queryY.toFixed(1);
        draw();
      });
    }

    function setQueryY(val) {
      queryY = val;
      if (sliderY) sliderY.value = val;
      if (valY) valY.textContent = val.toFixed(1);
      draw();
    }

    if (btnY3) btnY3.addEventListener('click', () => setQueryY(3.0));
    if (btnY10) btnY10.addEventListener('click', () => setQueryY(10.0));
    if (btnY15) btnY15.addEventListener('click', () => setQueryY(15.0));

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // DEMO 3: Smooth KDE - The "Sum of Bumps" Explorer
  // =========================================================================
  function initSmoothKdeDemo() {
    const canvas = document.getElementById('kde-smooth-canvas');
    if (!canvas) return;

    let kernelType = 'gaussian';
    let h = 0.9;
    let showBumps = true;
    let showTrue = true;

    function truePdf(x) {
      const g1 = (1.0 / (0.7 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 2.5) / 0.7, 2));
      const g2 = (1.0 / (0.9 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 6.5) / 0.9, 2));
      return 0.5 * g1 + 0.5 * g2;
    }

    const defaultData = [
      1.5, 1.8, 2.1, 2.3, 2.4, 2.5, 2.7, 2.9, 3.1, 3.4,
      5.2, 5.5, 5.8, 6.0, 6.2, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3, 7.6, 7.9, 8.2, 8.5
    ];
    let data = [...defaultData];

    function K(u, type) {
      switch (type) {
        case 'gaussian':
          return (1.0 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * u * u);
        case 'epanechnikov':
          return Math.abs(u) <= 1.0 ? 0.75 * (1.0 - u * u) : 0.0;
        case 'box':
          return Math.abs(u) <= 0.5 ? 1.0 : 0.0;
        case 'triangular':
          return Math.abs(u) <= 1.0 ? (1.0 - Math.abs(u)) : 0.0;
        default:
          return (1.0 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * u * u);
      }
    }

    const sliderH = document.getElementById('smooth-h-slider');
    const valH = document.getElementById('smooth-h-val');
    const btnGaussian = document.getElementById('btn-kernel-gaussian');
    const btnEpan = document.getElementById('btn-kernel-epan');
    const btnBox = document.getElementById('btn-kernel-box');
    const btnTri = document.getElementById('btn-kernel-tri');
    const chkBumps = document.getElementById('chk-smooth-bumps');
    const chkTrue = document.getElementById('chk-smooth-true');
    const btnResetData = document.getElementById('btn-smooth-reset');

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const padL = 60, padR = 40, padT = 30, padB = 55;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;

      const xMin = 0.0, xMax = 10.0;
      function toScreenX(x) { return padL + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toScreenY(dens, maxDens) { return padT + plotH - (dens / maxDens) * plotH; }

      const N = data.length;
      const steps = 300;
      const xs = [];
      const ys = [];
      let maxDens = 0.45;

      for (let s = 0; s <= steps; s++) {
        const x = xMin + (s / steps) * (xMax - xMin);
        let sum = 0;
        for (let pt of data) sum += K((x - pt) / h, kernelType);
        const density = sum / (N * h);
        if (density > maxDens) maxDens = density;
        xs.push(x);
        ys.push(density);
      }
      maxDens = Math.max(maxDens * 1.15, 0.45);

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (let yVal = 0.1; yVal <= maxDens; yVal += 0.1) {
        const sy = toScreenY(yVal, maxDens);
        ctx.beginPath();
        ctx.moveTo(padL, sy);
        ctx.lineTo(padL + plotW, sy);
        ctx.stroke();

        ctx.fillStyle = SOL.base01;
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(yVal.toFixed(1), padL - 8, sy + 3);
      }
      ctx.setLineDash([]);

      // Axes
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base01;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      for (let xVal = 0; xVal <= 10; xVal += 2) {
        const sx = toScreenX(xVal);
        ctx.beginPath();
        ctx.moveTo(sx, padT + plotH);
        ctx.lineTo(sx, padT + plotH + 5);
        ctx.stroke();
        ctx.fillText(xVal.toString(), sx, padT + plotH + 18);
      }

      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.base02;
      ctx.fillText('Sample Value (x) — Click canvas to add points!', padL + plotW / 2, height - 12);

      // True PDF
      if (showTrue) {
        ctx.strokeStyle = SOL.base01;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (let s = 0; s <= steps; s++) {
          const x = xs[s];
          const ty = truePdf(x);
          const sx = toScreenX(x);
          const sy = toScreenY(ty, maxDens);
          if (s === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Individual Bumps
      if (showBumps) {
        data.forEach(pt => {
          ctx.beginPath();
          let started = false;
          for (let s = 0; s <= steps; s++) {
            const x = xs[s];
            if (Math.abs(x - pt) > (kernelType === 'gaussian' ? 3.5 * h : (kernelType === 'box' ? 0.6 * h : 1.1 * h))) {
              continue;
            }
            const bumpVal = K((x - pt) / h, kernelType) / (N * h);
            const sx = toScreenX(x);
            const sy = toScreenY(bumpVal, maxDens);
            if (!started) { ctx.moveTo(sx, sy); started = true; }
            else { ctx.lineTo(sx, sy); }
          }
          ctx.strokeStyle = 'rgba(42, 161, 152, 0.45)';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        });
      }

      // Combined KDE Curve
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const sx = toScreenX(xs[s]);
        const sy = toScreenY(ys[s], maxDens);
        if (s === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      ctx.lineTo(toScreenX(xs[steps]), toScreenY(0, maxDens));
      ctx.lineTo(toScreenX(xs[0]), toScreenY(0, maxDens));
      ctx.closePath();
      ctx.fillStyle = 'rgba(38, 139, 210, 0.18)';
      ctx.fill();

      // Rug plot
      data.forEach(pt => {
        const sx = toScreenX(pt);
        ctx.beginPath();
        ctx.arc(sx, padT + plotH + 5, 4, 0, 2 * Math.PI);
        ctx.fillStyle = SOL.magenta;
        ctx.fill();
        ctx.strokeStyle = SOL.base03;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Legend
      const legX = padL + plotW - 190;
      const legY = padT + 10;
      ctx.fillStyle = 'rgba(253, 246, 227, 0.9)';
      ctx.fillRect(legX - 8, legY - 8, 195, showTrue ? 64 : 46);
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.strokeRect(legX - 8, legY - 8, 195, showTrue ? 64 : 46);

      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(legX, legY + 6); ctx.lineTo(legX + 22, legY + 6); ctx.stroke();
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`KDE P(x) [${kernelType}]`, legX + 30, legY + 9);

      ctx.strokeStyle = 'rgba(42, 161, 152, 0.8)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(legX, legY + 22); ctx.lineTo(legX + 22, legY + 22); ctx.stroke();
      ctx.fillStyle = SOL.base01;
      ctx.font = '11px sans-serif';
      ctx.fillText('Individual Bumps', legX + 30, legY + 25);

      if (showTrue) {
        ctx.strokeStyle = SOL.base01;
        ctx.lineWidth = 1.8;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(legX, legY + 38); ctx.lineTo(legX + 22, legY + 38); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = SOL.base01;
        ctx.fillText('True Mixture PDF', legX + 30, legY + 41);
      }
    }

    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const padL = 60, padR = 40;
      const plotW = canvas.width / (window.devicePixelRatio || 1) - padL - padR;
      const xMin = 0.0, xMax = 10.0;
      const newX = xMin + ((clickX - padL) / plotW) * (xMax - xMin);
      if (newX >= xMin && newX <= xMax) {
        data.push(parseFloat(newX.toFixed(2)));
        draw();
      }
    });

    if (sliderH) {
      sliderH.addEventListener('input', (e) => {
        h = parseFloat(e.target.value);
        if (valH) valH.textContent = h.toFixed(2);
        draw();
      });
    }

    function setActiveKernelBtn(btn) {
      [btnGaussian, btnEpan, btnBox, btnTri].forEach(b => {
        if (b) { b.style.background = '#fdf6e3'; b.style.fontWeight = 'normal'; }
      });
      if (btn) { btn.style.background = '#eee8d5'; btn.style.fontWeight = 'bold'; }
    }

    if (btnGaussian) btnGaussian.addEventListener('click', () => { kernelType = 'gaussian'; setActiveKernelBtn(btnGaussian); draw(); });
    if (btnEpan) btnEpan.addEventListener('click', () => { kernelType = 'epanechnikov'; setActiveKernelBtn(btnEpan); draw(); });
    if (btnBox) btnBox.addEventListener('click', () => { kernelType = 'box'; setActiveKernelBtn(btnBox); draw(); });
    if (btnTri) btnTri.addEventListener('click', () => { kernelType = 'triangular'; setActiveKernelBtn(btnTri); draw(); });

    if (chkBumps) chkBumps.addEventListener('change', (e) => { showBumps = e.target.checked; draw(); });
    if (chkTrue) chkTrue.addEventListener('change', (e) => { showTrue = e.target.checked; draw(); });
    if (btnResetData) btnResetData.addEventListener('click', () => { data = [...defaultData]; draw(); });

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // DEMO 4: Bandwidth Selection, Bias-Variance & Silverman's Rule
  // =========================================================================
  function initBandwidthDemo() {
    const canvas = document.getElementById('bandwidth-biasvar-canvas');
    if (!canvas) return;

    let h = 0.55;
    let sampleSize = 100;
    let currentScenario = 'silverman';

    const rng = pseudoRandom(42);
    const pool = [];
    for (let i = 0; i < 500; i++) {
      pool.push(rng() < 0.5 ? sampleGaussian(rng, 2.5, 0.7) : sampleGaussian(rng, 6.5, 0.9));
    }

    function getSample(N) {
      return pool.slice(0, N);
    }

    function truePdf(x) {
      const g1 = (1.0 / (0.7 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 2.5) / 0.7, 2));
      const g2 = (1.0 / (0.9 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - 6.5) / 0.9, 2));
      return 0.5 * g1 + 0.5 * g2;
    }

    function K(u) {
      return (1.0 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * u * u);
    }

    function computeSilverman(data) {
      const N = data.length;
      const mean = data.reduce((a, b) => a + b, 0) / N;
      const std = Math.sqrt(data.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / (N - 1));
      const sorted = [...data].sort((a, b) => a - b);
      const q75 = sorted[Math.floor(N * 0.75)];
      const q25 = sorted[Math.floor(N * 0.25)];
      const iqr = q75 - q25;
      const A = Math.min(std, iqr / 1.34);

      const hStd = 1.06 * std * Math.pow(N, -0.2);
      const hRobust = 0.9 * A * Math.pow(N, -0.2);
      return { std, iqr, A, hStd, hRobust };
    }

    const sliderH = document.getElementById('bv-h-slider');
    const valH = document.getElementById('bv-h-val');
    const valMse = document.getElementById('bv-mse-val');
    const btnUnder = document.getElementById('btn-bv-under');
    const btnOptimal = document.getElementById('btn-bv-optimal');
    const btnRobust = document.getElementById('btn-bv-robust');
    const btnOver = document.getElementById('btn-bv-over');
    const btnN50 = document.getElementById('btn-bv-n50');
    const btnN100 = document.getElementById('btn-bv-n100');
    const btnN300 = document.getElementById('btn-bv-n300');

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const data = getSample(sampleSize);
      const silverman = computeSilverman(data);

      const padL = 60, padR = 40;
      const topT = 25, topH = 175;
      const botT = 230, botH = 95;
      const plotW = width - padL - padR;

      // TOP PANEL
      const xMin = 0.0, xMax = 10.0;
      function toScreenX(x) { return padL + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toScreenTopY(dens, maxDens) { return topT + topH - (dens / maxDens) * topH; }

      const steps = 250;
      const xs = [];
      const ysKde = [];
      let maxDens = 0.45;
      let totalMse = 0;

      for (let s = 0; s <= steps; s++) {
        const x = xMin + (s / steps) * (xMax - xMin);
        let sum = 0;
        for (let pt of data) sum += K((x - pt) / h);
        const dens = sum / (sampleSize * h);
        const trueVal = truePdf(x);
        totalMse += Math.pow(dens - trueVal, 2);
        if (dens > maxDens) maxDens = dens;
        xs.push(x);
        ysKde.push(dens);
      }
      totalMse = (totalMse / steps) * (xMax - xMin);
      maxDens = Math.max(maxDens * 1.15, 0.45);

      // Top grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (let yVal = 0.1; yVal <= maxDens; yVal += 0.1) {
        const sy = toScreenTopY(yVal, maxDens);
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(padL + plotW, sy); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(yVal.toFixed(1), padL - 8, sy + 3);
      }
      ctx.setLineDash([]);

      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(padL, topT + topH); ctx.lineTo(padL + plotW, topT + topH); ctx.stroke();

      // True Density
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const sx = toScreenX(xs[s]);
        const sy = toScreenTopY(truePdf(xs[s]), maxDens);
        if (s === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // KDE Curve
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const sx = toScreenX(xs[s]);
        const sy = toScreenTopY(ysKde[s], maxDens);
        if (s === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`Top: True Density (dashed) vs. KDE (blue) with h = ${h.toFixed(2)} [N=${sampleSize}]`, padL, topT - 8);

      data.forEach(pt => {
        const sx = toScreenX(pt);
        ctx.beginPath();
        ctx.arc(sx, topT + topH + 3, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = SOL.magenta;
        ctx.fill();
      });

      // BOTTOM PANEL
      const hMin = 0.05, hMax = 2.2;
      function toScreenH(val) { return padL + ((val - hMin) / (hMax - hMin)) * plotW; }
      function toScreenBotY(val, maxVal) { return botT + botH - (val / maxVal) * botH; }

      const c1 = 0.008;
      const c2 = 0.65 / sampleSize;

      const hSteps = 100;
      let maxError = 0.035;
      const hVals = [];
      const biasVals = [];
      const varVals = [];
      const mseVals = [];

      for (let i = 0; i <= hSteps; i++) {
        const hv = hMin + (i / hSteps) * (hMax - hMin);
        const b2 = c1 * Math.pow(hv, 4);
        const vr = c2 / hv;
        const ms = b2 + vr;
        if (ms > maxError) maxError = ms;
        hVals.push(hv);
        biasVals.push(b2);
        varVals.push(vr);
        mseVals.push(ms);
      }
      maxError = Math.min(maxError, 0.06);

      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(padL, botT + botH); ctx.lineTo(padL + plotW, botT + botH); ctx.stroke();

      ctx.fillStyle = SOL.base01;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      for (let hv = 0.2; hv <= 2.0; hv += 0.4) {
        const sh = toScreenH(hv);
        ctx.beginPath(); ctx.moveTo(sh, botT + botH); ctx.lineTo(sh, botT + botH + 4); ctx.stroke();
        ctx.fillText(hv.toFixed(1), sh, botT + botH + 14);
      }

      // Bias²
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i <= hSteps; i++) {
        const sx = toScreenH(hVals[i]);
        const sy = toScreenBotY(biasVals[i], maxError);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Variance
      ctx.strokeStyle = SOL.cyan;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i <= hSteps; i++) {
        const sx = toScreenH(hVals[i]);
        const sy = toScreenBotY(varVals[i], maxError);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Total MSE
      ctx.strokeStyle = SOL.green;
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      for (let i = 0; i <= hSteps; i++) {
        const sx = toScreenH(hVals[i]);
        const sy = toScreenBotY(mseVals[i], maxError);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Cursor
      const curHScreen = toScreenH(h);
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 2]);
      ctx.beginPath();
      ctx.moveTo(curHScreen, botT);
      ctx.lineTo(curHScreen, botT + botH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = SOL.orange;
      ctx.beginPath();
      ctx.arc(curHScreen, toScreenBotY(c1 * Math.pow(h, 4) + c2 / h, maxError), 5, 0, 2 * Math.PI);
      ctx.fill();

      // Silverman Marker
      const shOpt = toScreenH(silverman.hRobust);
      ctx.strokeStyle = SOL.violet;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(shOpt, botT);
      ctx.lineTo(shOpt, botT + botH);
      ctx.stroke();

      ctx.fillStyle = SOL.violet;
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`h*=${silverman.hRobust.toFixed(2)} (Silverman)`, Math.min(padL + plotW - 60, Math.max(padL + 70, shOpt)), botT - 6);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Bottom: Bias-Variance Decomposition vs Bandwidth h', padL, botT - 6);

      ctx.font = '10px sans-serif';
      ctx.fillStyle = SOL.red; ctx.fillText('■ Bias²', padL + 340, botT - 6);
      ctx.fillStyle = SOL.cyan; ctx.fillText('■ Variance', padL + 400, botT - 6);
      ctx.fillStyle = SOL.green; ctx.fillText('■ Total MSE', padL + 480, botT - 6);

      if (valMse) valMse.textContent = totalMse.toFixed(4);
    }

    if (sliderH) {
      sliderH.addEventListener('input', (e) => {
        h = parseFloat(e.target.value);
        if (valH) valH.textContent = h.toFixed(2);
        draw();
      });
    }

    function setH(newH, scenario) {
      h = newH;
      currentScenario = scenario;
      if (sliderH) sliderH.value = newH;
      if (valH) valH.textContent = newH.toFixed(2);
      [btnUnder, btnOptimal, btnRobust, btnOver].forEach(b => {
        if (b) { b.style.background = '#fdf6e3'; b.style.fontWeight = 'normal'; }
      });
      if (scenario === 'under' && btnUnder) { btnUnder.style.background = '#eee8d5'; btnUnder.style.fontWeight = 'bold'; }
      if (scenario === 'silverman' && btnOptimal) { btnOptimal.style.background = '#eee8d5'; btnOptimal.style.fontWeight = 'bold'; }
      if (scenario === 'robust' && btnRobust) { btnRobust.style.background = '#eee8d5'; btnRobust.style.fontWeight = 'bold'; }
      if (scenario === 'over' && btnOver) { btnOver.style.background = '#eee8d5'; btnOver.style.fontWeight = 'bold'; }
      draw();
    }

    if (btnUnder) btnUnder.addEventListener('click', () => setH(0.12, 'under'));
    if (btnOptimal) {
      btnOptimal.addEventListener('click', () => {
        const silv = computeSilverman(getSample(sampleSize));
        setH(parseFloat(silv.hStd.toFixed(2)), 'silverman');
      });
    }
    if (btnRobust) {
      btnRobust.addEventListener('click', () => {
        const silv = computeSilverman(getSample(sampleSize));
        setH(parseFloat(silv.hRobust.toFixed(2)), 'robust');
      });
    }
    if (btnOver) btnOver.addEventListener('click', () => setH(1.65, 'over'));

    function setSampleSize(N, btn) {
      sampleSize = N;
      [btnN50, btnN100, btnN300].forEach(b => {
        if (b) { b.style.background = '#fdf6e3'; b.style.fontWeight = 'normal'; }
      });
      if (btn) { btn.style.background = '#eee8d5'; btn.style.fontWeight = 'bold'; }
      const silv = computeSilverman(getSample(sampleSize));
      setH(parseFloat(silv.hRobust.toFixed(2)), 'robust');
    }

    if (btnN50) btnN50.addEventListener('click', () => setSampleSize(50, btnN50));
    if (btnN100) btnN100.addEventListener('click', () => setSampleSize(100, btnN100));
    if (btnN300) btnN300.addEventListener('click', () => setSampleSize(300, btnN300));

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // DEMO 5: 2D Multivariate KDE: Isotropic vs. Whitened vs. Product Kernels
  // =========================================================================
  function initMultivariateKdeDemo() {
    const canvas = document.getElementById('kde-2d-canvas');
    if (!canvas) return;

    let mode = 'whitened';
    let hGlobal = 0.8;
    let corrAngle = 40;

    function generate2dData(angleDeg) {
      const theta = (angleDeg * Math.PI) / 180.0;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);
      const s1 = 2.4, s2 = 0.7;

      const rng = pseudoRandom(1234);
      const pts = [];
      for (let i = 0; i < 55; i++) {
        const u1 = sampleGaussian(rng, 0, s1);
        const u2 = sampleGaussian(rng, 0, s2);
        const x = 5.0 + u1 * cosT - u2 * sinT;
        const y = 5.0 + u1 * sinT + u2 * cosT;
        pts.push({ x, y });
      }
      return pts;
    }

    let points = generate2dData(corrAngle);

    function computeCovariance(pts) {
      const N = pts.length;
      let mx = 0, my = 0;
      pts.forEach(p => { mx += p.x; my += p.y; });
      mx /= N; my /= N;

      let cxx = 0, cyy = 0, cxy = 0;
      pts.forEach(p => {
        const dx = p.x - mx;
        const dy = p.y - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
      });
      cxx /= (N - 1); cyy /= (N - 1); cxy /= (N - 1);

      const trace = cxx + cyy;
      const det = cxx * cyy - cxy * cxy;
      const disc = Math.max(0, trace * trace / 4 - det);
      const l1 = trace / 2 + Math.sqrt(disc);
      const l2 = trace / 2 - Math.sqrt(disc);

      return { mx, my, cxx, cyy, cxy, det, l1, l2 };
    }

    const btnIso = document.getElementById('btn-2d-iso');
    const btnProd = document.getElementById('btn-2d-prod');
    const btnWhite = document.getElementById('btn-2d-white');
    const sliderAngle = document.getElementById('corr-angle-slider');
    const valAngle = document.getElementById('corr-angle-val');
    const sliderH = document.getElementById('kde-2d-h-slider');
    const valH = document.getElementById('kde-2d-h-val');

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const pad = 35;
      const size = Math.min(width - 2 * pad, height - 2 * pad);
      const plotL = (width - size) / 2;
      const plotT = (height - size) / 2;

      const xMin = 0.0, xMax = 10.0;
      function toScreenX(x) { return plotL + ((x - xMin) / (xMax - xMin)) * size; }
      function toScreenY(y) { return plotT + size - ((y - xMin) / (xMax - xMin)) * size; }

      const cov = computeCovariance(points);
      const N = points.length;

      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(plotL, plotT, size, size);

      for (let v = 2; v <= 8; v += 2) {
        const sx = toScreenX(v);
        const sy = toScreenY(v);
        ctx.beginPath(); ctx.moveTo(sx, plotT); ctx.lineTo(sx, plotT + size); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(plotL, sy); ctx.lineTo(plotL + size, sy); ctx.stroke();

        ctx.fillStyle = SOL.base01;
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(v.toString(), sx, plotT + size + 14);
        ctx.textAlign = 'right';
        ctx.fillText(v.toString(), plotL - 6, sy + 3);
      }

      const gridN = 40;
      const grid = [];
      let maxD = 0;

      const invDet = 1.0 / Math.max(1e-5, cov.det);
      const iCxx = cov.cyy * invDet;
      const iCyy = cov.cxx * invDet;
      const iCxy = -cov.cxy * invDet;

      const hx = hGlobal * Math.sqrt(cov.cxx);
      const hy = hGlobal * Math.sqrt(cov.cyy);

      for (let gx = 0; gx < gridN; gx++) {
        grid[gx] = [];
        const x = xMin + (gx / (gridN - 1)) * (xMax - xMin);
        for (let gy = 0; gy < gridN; gy++) {
          const y = xMin + (gy / (gridN - 1)) * (xMax - xMin);
          let sum = 0;

          if (mode === 'isotropic') {
            const h2 = hGlobal * hGlobal;
            for (let p of points) {
              const d2 = Math.pow(x - p.x, 2) + Math.pow(y - p.y, 2);
              sum += Math.exp(-0.5 * d2 / h2);
            }
            sum /= (N * 2 * Math.PI * h2);
          } else if (mode === 'product') {
            for (let p of points) {
              const u1 = (x - p.x) / hx;
              const u2 = (y - p.y) / hy;
              sum += Math.exp(-0.5 * (u1 * u1 + u2 * u2));
            }
            sum /= (N * 2 * Math.PI * hx * hy);
          } else {
            const h2 = hGlobal * hGlobal;
            for (let p of points) {
              const dx = x - p.x;
              const dy = y - p.y;
              const mahalanobis = (dx * dx * iCxx + 2 * dx * dy * iCxy + dy * dy * iCyy);
              sum += Math.exp(-0.5 * mahalanobis / h2);
            }
            sum /= (N * 2 * Math.PI * h2 * Math.sqrt(cov.det));
          }

          if (sum > maxD) maxD = sum;
          grid[gx][gy] = sum;
        }
      }

      const levels = [0.15, 0.35, 0.6, 0.85];
      levels.forEach((lvl, lvlIdx) => {
        const threshold = lvl * maxD;
        ctx.strokeStyle = lvlIdx === 3 ? SOL.orange : (lvlIdx === 2 ? SOL.yellow : SOL.cyan);
        ctx.lineWidth = 1.8;
        ctx.beginPath();

        for (let gx = 0; gx < gridN - 1; gx++) {
          for (let gy = 0; gy < gridN - 1; gy++) {
            const v00 = grid[gx][gy] >= threshold;
            const v10 = grid[gx + 1][gy] >= threshold;
            const v11 = grid[gx + 1][gy + 1] >= threshold;
            const v01 = grid[gx][gy + 1] >= threshold;

            const c = (v00 ? 1 : 0) | (v10 ? 2 : 0) | (v11 ? 4 : 0) | (v01 ? 8 : 0);
            if (c === 0 || c === 15) continue;

            const x0 = toScreenX(xMin + (gx / (gridN - 1)) * (xMax - xMin));
            const x1 = toScreenX(xMin + ((gx + 1) / (gridN - 1)) * (xMax - xMin));
            const y0 = toScreenY(xMin + (gy / (gridN - 1)) * (xMax - xMin));
            const y1 = toScreenY(xMin + ((gy + 1) / (gridN - 1)) * (xMax - xMin));
            const xm = (x0 + x1) / 2;
            const ym = (y0 + y1) / 2;

            if (c === 1 || c === 14) { ctx.moveTo(x0, ym); ctx.lineTo(xm, y0); }
            else if (c === 2 || c === 13) { ctx.moveTo(xm, y0); ctx.lineTo(x1, ym); }
            else if (c === 3 || c === 12) { ctx.moveTo(x0, ym); ctx.lineTo(x1, ym); }
            else if (c === 4 || c === 11) { ctx.moveTo(x1, ym); ctx.lineTo(xm, y1); }
            else if (c === 6 || c === 9) { ctx.moveTo(xm, y0); ctx.lineTo(xm, y1); }
            else if (c === 7 || c === 8) { ctx.moveTo(x0, ym); ctx.lineTo(xm, y1); }
          }
        }
        ctx.stroke();
      });

      // Principal Axes
      const theta = Math.atan2(cov.l1 - cov.cxx, cov.cxy);
      const eigLen1 = Math.sqrt(cov.l1) * 25;
      const eigLen2 = Math.sqrt(cov.l2) * 25;
      const cx = toScreenX(cov.mx);
      const cy = toScreenY(cov.my);

      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx - Math.cos(theta) * eigLen1, cy + Math.sin(theta) * eigLen1);
      ctx.lineTo(cx + Math.cos(theta) * eigLen1, cy - Math.sin(theta) * eigLen1);
      ctx.stroke();

      ctx.strokeStyle = SOL.violet;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - Math.sin(theta) * eigLen2, cy - Math.cos(theta) * eigLen2);
      ctx.lineTo(cx + Math.sin(theta) * eigLen2, cy + Math.cos(theta) * eigLen2);
      ctx.stroke();

      // Points
      points.forEach(p => {
        const sx = toScreenX(p.x);
        const sy = toScreenY(p.y);
        ctx.beginPath();
        ctx.arc(sx, sy, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = SOL.blue;
        ctx.fill();
        ctx.strokeStyle = SOL.base03;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      });

      // Overlay explanation badge
      ctx.fillStyle = 'rgba(253, 246, 227, 0.92)';
      ctx.fillRect(plotL + 8, plotT + 8, 260, 52);
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.strokeRect(plotL + 8, plotT + 8, 260, 52);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'left';
      const modeTitle = mode === 'isotropic' ? 'Isotropic (Circular / Equal h)' : (mode === 'product' ? 'Product Kernel (Axis-Aligned h₁, h₂)' : 'Pre-whitened (Rotated Elliptical)');
      ctx.fillText(modeTitle, plotL + 14, plotT + 24);

      ctx.fillStyle = SOL.base01;
      ctx.font = '10px sans-serif';
      const modeDesc = mode === 'isotropic'
        ? 'Drawback: Fails to capture feature correlation.'
        : (mode === 'product' ? 'Scales axes independently, but cannot rotate.' : 'Optimal: Aligns contours with data covariance!');
      ctx.fillText(modeDesc, plotL + 14, plotT + 44);
    }

    if (sliderAngle) {
      sliderAngle.addEventListener('input', (e) => {
        corrAngle = parseInt(e.target.value, 10);
        if (valAngle) valAngle.textContent = corrAngle.toString();
        points = generate2dData(corrAngle);
        draw();
      });
    }

    if (sliderH) {
      sliderH.addEventListener('input', (e) => {
        hGlobal = parseFloat(e.target.value);
        if (valH) valH.textContent = hGlobal.toFixed(2);
        draw();
      });
    }

    function setActiveMode(newMode, btn) {
      mode = newMode;
      [btnIso, btnProd, btnWhite].forEach(b => {
        if (b) { b.style.background = '#fdf6e3'; b.style.fontWeight = 'normal'; }
      });
      if (btn) { btn.style.background = '#eee8d5'; btn.style.fontWeight = 'bold'; }
      draw();
    }

    if (btnIso) btnIso.addEventListener('click', () => setActiveMode('isotropic', btnIso));
    if (btnProd) btnProd.addEventListener('click', () => setActiveMode('product', btnProd));
    if (btnWhite) btnWhite.addEventListener('click', () => setActiveMode('whitened', btnWhite));

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // DEMO 6: Non-parametric Bayes Classifier & Outlier Vulnerability
  // =========================================================================
  function initClassifierDemo() {
    const canvas = document.getElementById('kde-classifier-canvas');
    if (!canvas) return;

    let h = 0.45;
    let hasOutlier = false;

    // Two well-separated clusters with 6 points each
    const baseClass1 = [1.2, 1.6, 2.0, 2.4, 2.8, 3.2]; // Red / ω₁ (Center ~ 2.2)
    const baseClass2 = [5.2, 5.6, 6.0, 6.4, 6.8, 7.2]; // Blue / ω₂ (Center ~ 6.2)

    let class1 = [...baseClass1];
    let class2 = [...baseClass2];

    function K(u) {
      return (1.0 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * u * u);
    }

    const sliderH = document.getElementById('clf-h-slider');
    const valH = document.getElementById('clf-h-val');
    const btnOutlierToggle = document.getElementById('btn-clf-outlier');
    const btnReset = document.getElementById('btn-clf-reset');
    const alertBox = document.getElementById('clf-alert-box');

    function draw() {
      const { ctx, width, height } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, width, height);

      const padL = 60, padR = 40, padT = 30, padB = 60;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;

      const xMin = 0.0, xMax = 9.0;
      function toScreenX(x) { return padL + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toScreenY(d, maxD) { return padT + plotH - (d / maxD) * plotH; }

      const steps = 400;
      const xs = [];
      const p1s = [];
      const p2s = [];
      const boundaries = [];
      let maxD = 0.45;

      const N1 = class1.length;
      const N2 = class2.length;

      for (let s = 0; s <= steps; s++) {
        const x = xMin + (s / steps) * (xMax - xMin);
        let s1 = 0, s2 = 0;
        for (let pt of class1) s1 += K((x - pt) / h);
        for (let pt of class2) s2 += K((x - pt) / h);

        const d1 = s1 / (N1 * h);
        const d2 = s2 / (N2 * h);

        if (d1 > maxD) maxD = d1;
        if (d2 > maxD) maxD = d2;

        xs.push(x);
        p1s.push(d1);
        p2s.push(d2);

        // Detect crossings
        if (s > 0) {
          const prevDiff = p1s[s - 1] - p2s[s - 1];
          const curDiff = d1 - d2;
          if (prevDiff * curDiff < 0) {
            const t = Math.abs(prevDiff) / (Math.abs(prevDiff) + Math.abs(curDiff));
            const bx = xs[s - 1] + t * (x - xs[s - 1]);
            boundaries.push(bx);
          }
        }
      }
      maxD = Math.max(maxD * 1.15, 0.45);

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (let yVal = 0.1; yVal <= maxD; yVal += 0.1) {
        const sy = toScreenY(yVal, maxD);
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(padL + plotW, sy); ctx.stroke();
      }
      ctx.setLineDash([]);

      // Decision Regions Ribbon
      const ribH = 14;
      const ribY = padT + plotH + 18;
      for (let s = 0; s < steps; s++) {
        const x0 = toScreenX(xs[s]);
        const x1 = toScreenX(xs[s + 1]);
        const isClass1 = p1s[s] >= p2s[s];
        ctx.fillStyle = isClass1 ? 'rgba(220, 50, 47, 0.4)' : 'rgba(38, 139, 210, 0.4)';
        ctx.fillRect(x0, ribY, x1 - x0, ribH);
      }
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, ribY, plotW, ribH);

      // Axis
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH); ctx.stroke();

      for (let xVal = 0; xVal <= 9; xVal += 1) {
        const sx = toScreenX(xVal);
        ctx.beginPath(); ctx.moveTo(sx, padT + plotH); ctx.lineTo(sx, padT + plotH + 5); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(xVal.toString(), sx, padT + plotH + 15);
      }

      // Class 1 (Red)
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const sx = toScreenX(xs[s]);
        const sy = toScreenY(p1s[s], maxD);
        if (s === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Class 2 (Blue)
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const sx = toScreenX(xs[s]);
        const sy = toScreenY(p2s[s], maxD);
        if (s === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Boundaries (Vertical dotted lines)
      boundaries.forEach((bx, bIdx) => {
        const sx = toScreenX(bx);
        ctx.strokeStyle = SOL.orange;
        ctx.lineWidth = 2.4;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(sx, padT);
        ctx.lineTo(sx, ribY + ribH);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = SOL.orange;
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`Boundary: ${bx.toFixed(2)}`, sx, padT - 8);
      });

      // Data Points
      class1.forEach((pt, idx) => {
        const sx = toScreenX(pt);
        const isOutlier = (hasOutlier && idx === class1.length - 1);
        ctx.beginPath();
        ctx.arc(sx, padT + plotH + 5, isOutlier ? 5.5 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = isOutlier ? '#b58900' : SOL.red;
        ctx.fill();
        ctx.strokeStyle = isOutlier ? SOL.red : SOL.base03;
        ctx.lineWidth = isOutlier ? 2 : 1;
        ctx.stroke();
      });

      class2.forEach(pt => {
        const sx = toScreenX(pt);
        ctx.beginPath();
        ctx.arc(sx, padT + plotH + 5, 4, 0, 2 * Math.PI);
        ctx.fillStyle = SOL.blue;
        ctx.fill();
        ctx.strokeStyle = SOL.base03;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Alert Box
      if (alertBox) {
        if (hasOutlier) {
          const shiftDesc = boundaries.length === 1
            ? `Shifted from <b>4.20 &rarr; ${boundaries[0].toFixed(2)}</b>!`
            : `Multiple boundaries at: <b>${boundaries.map(b => b.toFixed(2)).join(', ')}</b> (island formed)!`;
          alertBox.innerHTML = `<span style="color:#dc322f; font-weight:bold;">⚠️ HIGH SENSITIVITY DETECTED:</span> Single outlier at x=4.9 ${shiftDesc} KDE decision boundaries are extremely vulnerable to boundary noise.`;
        } else {
          const curB = boundaries.length > 0 ? boundaries[0].toFixed(2) : '4.20';
          alertBox.innerHTML = `<span style="color:#859900; font-weight:bold;">✓ Clean Bayes Boundary</span> at x = <b>${curB}</b>. Decision: Decide &omega;₁ if x &lt; ${curB}, else &omega;₂.`;
        }
      }
    }

    if (sliderH) {
      sliderH.addEventListener('input', (e) => {
        h = parseFloat(e.target.value);
        if (valH) valH.textContent = h.toFixed(2);
        draw();
      });
    }

    if (btnOutlierToggle) {
      btnOutlierToggle.addEventListener('click', () => {
        hasOutlier = !hasOutlier;
        if (hasOutlier) {
          class1.push(4.9); // Outlier inside Class 2 territory
          btnOutlierToggle.style.background = '#eee8d5';
          btnOutlierToggle.style.color = SOL.red;
          btnOutlierToggle.textContent = 'Remove Outlier';
        } else {
          class1 = [...baseClass1];
          btnOutlierToggle.style.background = '#fdf6e3';
          btnOutlierToggle.style.color = SOL.base01;
          btnOutlierToggle.textContent = 'Add Boundary Outlier (x=4.9)';
        }
        draw();
      });
    }

    if (btnReset) {
      btnReset.addEventListener('click', () => {
        hasOutlier = false;
        class1 = [...baseClass1];
        class2 = [...baseClass2];
        h = 0.45;
        if (sliderH) sliderH.value = 0.45;
        if (valH) valH.textContent = '0.45';
        if (btnOutlierToggle) {
          btnOutlierToggle.style.background = '#fdf6e3';
          btnOutlierToggle.style.color = SOL.base01;
          btnOutlierToggle.textContent = 'Add Boundary Outlier (x=4.9)';
        }
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    draw();
  }

  // =========================================================================
  // Initialize all demonstrations when DOM is ready
  // =========================================================================
  function initAll() {
    initBridgeSlide();
    initHistogramDemo();
    initParzenDemo();
    initSmoothKdeDemo();
    initBandwidthDemo();
    initMultivariateKdeDemo();
    initClassifierDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

})();
