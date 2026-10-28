/**
 * CS8850: Advanced Machine Learning - Lecture 17 (Expectation Maximization)
 * Interactive Pedagogical Demonstrations Suite
 *
 * 0. Dynamic Bridge Slide Connectors (Precision SVG alignment)
 * 1. GMM MLE Singularity & Variance-Collapse Explorer
 * 2. Jensen's Inequality & Concavity of ln(x) Visualizer
 * 3. Iterative Surrogate Lower-Bound (ELBO / Minorize-Maximize) Stepper
 * 4. Forward vs. Reverse KL Divergence Explorer (Mass-Covering vs. Mode-Seeking)
 * 5. Interactive 2D Full-Covariance GMM EM Playground (Responsibility Pie-Charts & Ellipses)
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
      canvas._logicalH = parseInt(canvas.getAttribute('height'), 10) || 330;
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

  function gaussian1D(x, mu, sigma) {
    const s = Math.max(1e-4, sigma);
    const z = (x - mu) / s;
    return (1.0 / (Math.sqrt(2.0 * Math.PI) * s)) * Math.exp(-0.5 * z * z);
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
    setCurve('conn-1-8', c1, getRowLeftCenter('bridge-row-8'));
    setCurve('conn-1-9', c1, getRowLeftCenter('bridge-row-9'));
    setCurve('conn-1-11', c1, getRowLeftCenter('bridge-row-11'));
    setCurve('conn-2-12', c2, getRowLeftCenter('bridge-row-12'));
    setCurve('conn-2-13', c2, getRowLeftCenter('bridge-row-13'));
    setCurve('conn-3-14', c3, getRowLeftCenter('bridge-row-14'));
    setCurve('conn-3-16', c3, getRowLeftCenter('bridge-row-16'));
    setCurve('conn-4-20', c4, getRowLeftCenter('bridge-row-20'));
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
      Reveal.addEventListener('fragmentshown', updateBridgeConnectors);
      Reveal.addEventListener('fragmenthidden', updateBridgeConnectors);
      Reveal.addEventListener('ready', () => setTimeout(updateBridgeConnectors, 150));
    }
    setTimeout(updateBridgeConnectors, 100);
  }

  // =========================================================================
  // 1. DEMO 1: GMM MLE SINGULARITY & VARIANCE-COLLAPSE EXPLORER
  // =========================================================================
  function initSingularityDemo() {
    const canvas = document.getElementById('em-singularity-canvas');
    if (!canvas) return;

    // 8 points matching the classic MLE_collapse.svg geometry
    const data = [1.5, 2.0, 2.9, 4.2, 5.4, 7.0, 8.2];
    const targetPoint = 8.2;

    // Component 1 fixed over the main bulk
    const mu1 = 4.2;
    const sigma1 = 1.45;
    const pi1 = 0.5;
    const pi2 = 0.5;

    let mu2 = 6.5;
    let sigma2 = 0.75;
    let useFloor = false;
    const floorSigma = 0.12;
    let animId = null;

    const mu2Slider = document.getElementById('sing-mu2-slider');
    const sig2Slider = document.getElementById('sing-sig2-slider');
    const mu2Val = document.getElementById('sing-mu2-val');
    const sig2Val = document.getElementById('sing-sig2-val');
    const llVal = document.getElementById('sing-ll-val');
    const peakVal = document.getElementById('sing-peak-val');
    const statusBanner = document.getElementById('sing-status-banner');
    const chkFloor = document.getElementById('sing-chk-floor');
    const btnCollapse = document.getElementById('sing-btn-collapse');
    const btnNormal = document.getElementById('sing-btn-normal');

    function getEffectiveSigma2() {
      return useFloor ? Math.sqrt(sigma2 * sigma2 + floorSigma * floorSigma) : sigma2;
    }

    function computeLogLikelihood() {
      const s2 = getEffectiveSigma2();
      let ll = 0;
      for (let i = 0; i < data.length; i++) {
        const p = pi1 * gaussian1D(data[i], mu1, sigma1) + pi2 * gaussian1D(data[i], mu2, s2);
        ll += Math.log(Math.max(1e-30, p));
      }
      return ll;
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const s2 = getEffectiveSigma2();
      const padL = 58, padR = 30, padT = 28, padB = 42;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const xMin = 0.0, xMax = 10.0;
      // Dynamic yMax that scales gracefully up to display the singularity spike
      const spikePeak = pi2 * gaussian1D(mu2, mu2, s2) + pi1 * gaussian1D(mu2, mu1, sigma1);
      const yMax = Math.min(4.0, Math.max(0.65, spikePeak * 1.12));

      function toX(x) { return padL + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toY(y) { return padT + plotH - (Math.min(y, yMax * 1.04) / yMax) * plotH; }

      // Grid lines
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.22)';
      ctx.lineWidth = 1;
      for (let gx = 1; gx <= 9; gx++) {
        const sx = toX(gx);
        ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, padT + plotH); ctx.stroke();
      }
      const ySteps = 4;
      for (let i = 1; i <= ySteps; i++) {
        const gy = (yMax / ySteps) * i;
        const sy = toY(gy);
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(padL + plotW, sy); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(gy.toFixed(2), padL - 6, sy + 4);
      }

      // Axes
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Data Space x', padL + plotW / 2, H - 8);
      ctx.save();
      ctx.translate(15, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Density p(x)', 0, 0);
      ctx.restore();

      // Draw Component 1 (bulk Gaussian)
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 2) {
        const x = xMin + (px / plotW) * (xMax - xMin);
        const y = pi1 * gaussian1D(x, mu1, sigma1);
        if (px === 0) ctx.moveTo(padL + px, toY(y));
        else ctx.lineTo(padL + px, toY(y));
      }
      ctx.stroke();

      // Draw Component 2 (collapsible Gaussian)
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 1) {
        const x = xMin + (px / plotW) * (xMax - xMin);
        const y = pi2 * gaussian1D(x, mu2, s2);
        if (px === 0) ctx.moveTo(padL + px, toY(y));
        else ctx.lineTo(padL + px, toY(y));
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Total Mixture p(x)
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 1) {
        const x = xMin + (px / plotW) * (xMax - xMin);
        const y = pi1 * gaussian1D(x, mu1, sigma1) + pi2 * gaussian1D(x, mu2, s2);
        if (px === 0) ctx.moveTo(padL + px, toY(y));
        else ctx.lineTo(padL + px, toY(y));
      }
      ctx.stroke();

      // Draw vertical green likelihood stems and data points (like MLE_collapse.svg)
      for (let i = 0; i < data.length; i++) {
        const xi = data[i];
        const py = pi1 * gaussian1D(xi, mu1, sigma1) + pi2 * gaussian1D(xi, mu2, s2);
        const sx = toX(xi);
        const sy = toY(py);
        const baseSy = padT + plotH;

        // Green likelihood stem
        ctx.strokeStyle = SOL.green;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(sx, baseSy);
        ctx.lineTo(sx, sy);
        ctx.stroke();

        // Point on curve
        ctx.fillStyle = SOL.blue;
        ctx.beginPath();
        ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
        ctx.fill();

        // Data point on x-axis
        ctx.fillStyle = (i === data.length - 1) ? SOL.red : SOL.base03;
        ctx.beginPath();
        ctx.arc(sx, baseSy, 5.5, 0, 2 * Math.PI);
        ctx.fill();

        if (i === data.length - 1) {
          ctx.fillStyle = SOL.red;
          ctx.font = 'bold 11px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('xᵢ = 8.20', sx, baseSy + 16);
        }
      }

      // Legend box
      const legX = padL + 16, legY = padT + 10;
      ctx.fillStyle = 'rgba(253, 246, 227, 0.92)';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.fillRect(legX, legY, 245, 68);
      ctx.strokeRect(legX, legY, 245, 68);

      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      // Mixture
      ctx.strokeStyle = SOL.red; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 16); ctx.lineTo(legX + 34, legY + 16); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText('Mixture p(x) = π₁N₁ + π₂N₂', legX + 42, legY + 20);
      // Comp 1
      ctx.strokeStyle = SOL.blue; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 35); ctx.lineTo(legX + 34, legY + 35); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText('Component 1 (bulk, σ₁=1.45)', legX + 42, legY + 39);
      // Comp 2
      ctx.strokeStyle = SOL.orange;
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 54); ctx.lineTo(legX + 34, legY + 54); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = SOL.base02; ctx.fillText(`Component 2 (μ₂=${mu2.toFixed(2)}, σ₂=${s2.toFixed(3)})`, legX + 42, legY + 58);

      // Update readouts
      const ll = computeLogLikelihood();
      const peakDensity = pi2 * gaussian1D(mu2, mu2, s2);
      if (mu2Val) mu2Val.textContent = mu2.toFixed(2);
      if (sig2Val) sig2Val.textContent = s2.toFixed(3);
      if (llVal) llVal.textContent = (ll > 0 ? '+' : '') + ll.toFixed(2);
      if (peakVal) peakVal.textContent = peakDensity.toFixed(2);

      if (statusBanner) {
        const isCollapsed = (Math.abs(mu2 - targetPoint) < 0.12 && s2 < 0.05);
        if (isCollapsed) {
          statusBanner.style.color = SOL.red;
          statusBanner.innerHTML = `⚠️ <b>SINGULARITY COLLAPSE!</b> μ₂ = xᵢ (8.20) and σ₂ → 0: Dirac spike height = ${peakDensity.toFixed(1)} → ∞, Log-Likelihood = ${ll.toFixed(2)} → +∞!`;
        } else if (useFloor && sigma2 < 0.08) {
          statusBanner.style.color = SOL.green;
          statusBanner.innerHTML = `✓ <b>Regularized (Σₖ + εI):</b> Variance floor bounds σ₂ ≥ ${floorSigma.toFixed(2)}, preventing singularity blow-up!`;
        } else {
          statusBanner.style.color = SOL.base02;
          statusBanner.innerHTML = `Drag <b>μ₂</b> onto <b>xᵢ = 8.20</b> and shrink <b>σ₂ → 0</b> (or click <em>Trigger Singularity Collapse</em>) to see unconstrained MLE blow up.`;
        }
      }
    }

    if (mu2Slider) {
      mu2Slider.addEventListener('input', () => {
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        mu2 = parseFloat(mu2Slider.value);
        draw();
      });
    }
    if (sig2Slider) {
      sig2Slider.addEventListener('input', () => {
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        sigma2 = parseFloat(sig2Slider.value);
        draw();
      });
    }
    if (chkFloor) {
      chkFloor.addEventListener('change', () => {
        useFloor = chkFloor.checked;
        draw();
      });
    }
    if (btnNormal) {
      btnNormal.addEventListener('click', () => {
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        mu2 = 6.5;
        sigma2 = 0.75;
        if (mu2Slider) mu2Slider.value = mu2;
        if (sig2Slider) sig2Slider.value = sigma2;
        draw();
      });
    }
    if (btnCollapse) {
      btnCollapse.addEventListener('click', () => {
        if (animId) cancelAnimationFrame(animId);
        const startMu = mu2;
        const startSig = sigma2;
        const startTime = performance.now();
        const duration = 900;

        function step(now) {
          const t = Math.min(1, (now - startTime) / duration);
          const ease = 1 - Math.pow(1 - t, 3);
          mu2 = startMu + (targetPoint - startMu) * ease;
          sigma2 = startSig * Math.pow(0.008 / Math.max(0.008, startSig), ease);
          if (mu2Slider) mu2Slider.value = mu2.toFixed(2);
          if (sig2Slider) sig2Slider.value = sigma2.toFixed(3);
          draw();
          if (t < 1) animId = requestAnimationFrame(step);
        }
        animId = requestAnimationFrame(step);
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // =========================================================================
  // 2. DEMO 2: JENSEN'S INEQUALITY & CONCAVITY OF ln(x) VISUALIZER
  // =========================================================================
  function initJensenDemo() {
    const canvas = document.getElementById('em-jensen-canvas');
    if (!canvas) return;

    let x1 = 0.65;
    let x2 = 3.80;
    let lambda = 0.45;
    let mode = 'ln'; // 'ln' (concave, >=) or 'negln' (convex, <=)

    const x1Slider = document.getElementById('jensen-x1-slider');
    const x2Slider = document.getElementById('jensen-x2-slider');
    const lamSlider = document.getElementById('jensen-lam-slider');
    const x1Val = document.getElementById('jensen-x1-val');
    const x2Val = document.getElementById('jensen-x2-val');
    const lamVal = document.getElementById('jensen-lam-val');
    const lhsVal = document.getElementById('jensen-lhs-val');
    const rhsVal = document.getElementById('jensen-rhs-val');
    const gapVal = document.getElementById('jensen-gap-val');
    const btnLn = document.getElementById('jensen-btn-ln');
    const btnNegLn = document.getElementById('jensen-btn-negln');
    const btnTight = document.getElementById('jensen-btn-tight');
    const btnWide = document.getElementById('jensen-btn-wide');

    function f(x) {
      return mode === 'ln' ? Math.log(x) : -Math.log(x);
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const padL = 62, padR = 35, padT = 26, padB = 42;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const xMin = 0.2, xMax = 4.5;
      const yMin = -1.6, yMax = 1.6;

      function toX(x) { return padL + ((x - xMin) / (xMax - xMin)) * plotW; }
      function toY(y) { return padT + plotH - ((y - yMin) / (yMax - yMin)) * plotH; }

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.22)';
      ctx.lineWidth = 1;
      for (let gx = 0.5; gx <= 4.0; gx += 0.5) {
        const sx = toX(gx);
        ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, padT + plotH); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(gx.toFixed(1), sx, padT + plotH + 15);
      }
      for (let gy = -1.5; gy <= 1.5; gy += 0.5) {
        const sy = toY(gy);
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(padL + plotW, sy); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(gy.toFixed(1), padL - 6, sy + 4);
      }

      // Zero line y = 0
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.45)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(padL, toY(0));
      ctx.lineTo(padL + plotW, toY(0));
      ctx.stroke();

      // Axes
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      // Curve f(x)
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 2) {
        const x = xMin + (px / plotW) * (xMax - xMin);
        const y = f(x);
        if (px === 0) ctx.moveTo(padL + px, toY(y));
        else ctx.lineTo(padL + px, toY(y));
      }
      ctx.stroke();

      // Helper to draw a text badge with a readable pill background
      function drawBadge(text, x, y, textColor, borderColor, align = 'center') {
        ctx.save();
        ctx.font = 'bold 11.5px sans-serif';
        const padX = 6, padY = 4, h = 19;
        const w = ctx.measureText(text).width + padX * 2;
        let bx = x - w / 2;
        if (align === 'left') bx = x;
        if (align === 'right') bx = x - w;
        // Keep inside plot area horizontally
        bx = Math.max(padL + 4, Math.min(padL + plotW - w - 4, bx));
        const by = y - h / 2;
        ctx.fillStyle = 'rgba(253, 246, 227, 0.94)';
        ctx.strokeStyle = borderColor || SOL.base1;
        ctx.lineWidth = 1;
        ctx.fillRect(bx, by, w, h);
        ctx.strokeRect(bx, by, w, h);
        ctx.fillStyle = textColor;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, bx + padX, by + h / 2 + 0.5);
        ctx.restore();
      }

      // Points x1, x2 and convex combination xBar
      const y1 = f(x1);
      const y2 = f(x2);
      const xBar = lambda * x1 + (1 - lambda) * x2;
      const fOfExp = f(xBar); // f(E[X])
      const expOfF = lambda * y1 + (1 - lambda) * y2; // E[f(X)]
      const fnName = (mode === 'ln') ? 'ln' : '-ln';

      // Secant chord between (x1, y1) and (x2, y2)
      ctx.strokeStyle = SOL.orange;
      ctx.lineWidth = 2.4;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(toX(x1), toY(y1));
      ctx.lineTo(toX(x2), toY(y2));
      ctx.stroke();
      ctx.setLineDash([]);

      // Dashed vertical guides at x1, x2
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      [ [x1, y1], [x2, y2] ].forEach(([xv, yv]) => {
        ctx.beginPath();
        ctx.moveTo(toX(xv), padT + plotH);
        ctx.lineTo(toX(xv), toY(yv));
        ctx.stroke();
      });
      ctx.setLineDash([]);

      // Highlight the Jensen Gap between f(E[X]) and E[f(X)] at xBar
      const sxBar = toX(xBar);
      const syCurve = toY(fOfExp);
      const syChord = toY(expOfF);

      // Vertical guide from axis to xBar
      ctx.strokeStyle = 'rgba(220, 50, 47, 0.45)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sxBar, padT + plotH);
      ctx.lineTo(sxBar, Math.max(syCurve, syChord));
      ctx.stroke();
      ctx.setLineDash([]);

      // Shaded vertical bar for the Jensen gap
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 4.5;
      ctx.beginPath();
      ctx.moveTo(sxBar, syCurve);
      ctx.lineTo(sxBar, syChord);
      ctx.stroke();

      // Draw endpoint circles (x1, y1) and (x2, y2) with explicit coordinate labels
      [
        [x1, y1, 'x₁', `(x₁, ${fnName} x₁)`],
        [x2, y2, 'x₂', `(x₂, ${fnName} x₂)`]
      ].forEach(([xv, yv, axisLbl, ptLbl], idx) => {
        const sx = toX(xv), sy = toY(yv);
        ctx.fillStyle = SOL.base03;
        ctx.beginPath();
        ctx.arc(sx, sy, 5.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(axisLbl, sx, padT + plotH - 6);
        if (Math.abs(x2 - x1) > 0.25 || idx === 0) {
          const badgeY = (mode === 'ln')
            ? (idx === 0 ? sy - 16 : sy - 16)
            : (idx === 0 ? sy + 18 : sy + 18);
          drawBadge(ptLbl, sx, badgeY, SOL.base03, SOL.base1, 'center');
        }
      });

      // X-axis badge for E[X] = λx₁ + (1-λ)x₂
      drawBadge(`E[X] = λx₁+(1−λ)x₂ = ${xBar.toFixed(2)}`, sxBar, padT + plotH - 12, SOL.red, SOL.red, 'center');

      // Draw Curve Point f(E[X]) (Green) and Chord Point E[f(X)] (Orange)
      ctx.fillStyle = SOL.green;
      ctx.beginPath();
      ctx.arc(sxBar, syCurve, 6.5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = SOL.orange;
      ctx.beginPath();
      ctx.arc(sxBar, syChord, 6.5, 0, 2 * Math.PI);
      ctx.fill();

      // Non-colliding pill badges for Curve Point (Log of Sum) and Chord Point (Sum of Logs)
      const gap = Math.abs(fOfExp - expOfF);
      const curveBadgeY = (mode === 'ln') ? (syCurve - 18) : (syCurve + 18);
      const chordBadgeY = (mode === 'ln') ? (syChord + 19) : (syChord - 19);

      drawBadge(
        `Curve (Log of Sum): ${fnName}(λx₁+(1−λ)x₂) = ${fOfExp.toFixed(3)}`,
        sxBar,
        curveBadgeY,
        SOL.green,
        SOL.green,
        'center'
      );

      if (gap > 0.015) {
        drawBadge(
          `Secant Chord (Sum of Logs): λ${fnName}(x₁)+(1−λ)${fnName}(x₂) = ${expOfF.toFixed(3)}`,
          sxBar,
          chordBadgeY,
          SOL.orange,
          SOL.orange,
          'center'
        );
      }

      // Formula callout card
      const cardX = W - 345, cardY = (mode === 'ln') ? (padT + plotH - 86) : (padT + 10);
      ctx.fillStyle = 'rgba(253, 246, 227, 0.95)';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.fillRect(cardX, cardY, 305, 72);
      ctx.strokeRect(cardX, cardY, 305, 72);

      ctx.textAlign = 'left';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.blue;
      ctx.fillText(mode === 'ln' ? 'Concave f(x)=ln(x): Curve ≥ Secant Chord' : 'Convex f(x)=-ln(x): Curve ≤ Secant Chord', cardX + 10, cardY + 18);
      ctx.font = '11.5px sans-serif';
      ctx.fillStyle = SOL.base02;
      ctx.fillText(mode === 'ln' ? 'EM Lower Bound: ln(E[X]) ≥ E[ln(X)]' : 'Jensen Inequality: -ln(E[X]) ≤ E[-ln(X)]', cardX + 10, cardY + 38);
      ctx.fillStyle = gap < 0.005 ? SOL.green : SOL.red;
      ctx.font = 'bold 12px monospace';
      ctx.fillText(gap < 0.005 ? 'Gap = 0.000 (Tight Bound: x₁ = x₂!)' : `Jensen Gap (Red Bar) = ${gap.toFixed(3)} > 0`, cardX + 10, cardY + 58);

      if (x1Val) x1Val.textContent = x1.toFixed(2);
      if (x2Val) x2Val.textContent = x2.toFixed(2);
      if (lamVal) lamVal.textContent = lambda.toFixed(2);
      if (lhsVal) lhsVal.textContent = fOfExp.toFixed(3);
      if (rhsVal) rhsVal.textContent = expOfF.toFixed(3);
      if (gapVal) gapVal.textContent = gap.toFixed(3);
    }

    if (x1Slider) x1Slider.addEventListener('input', () => { x1 = parseFloat(x1Slider.value); draw(); });
    if (x2Slider) x2Slider.addEventListener('input', () => { x2 = parseFloat(x2Slider.value); draw(); });
    if (lamSlider) lamSlider.addEventListener('input', () => { lambda = parseFloat(lamSlider.value); draw(); });

    if (btnLn) {
      btnLn.addEventListener('click', () => {
        mode = 'ln';
        btnLn.style.background = SOL.base2;
        btnLn.style.fontWeight = 'bold';
        if (btnNegLn) { btnNegLn.style.background = SOL.base3; btnNegLn.style.fontWeight = 'normal'; }
        draw();
      });
    }
    if (btnNegLn) {
      btnNegLn.addEventListener('click', () => {
        mode = 'negln';
        btnNegLn.style.background = SOL.base2;
        btnNegLn.style.fontWeight = 'bold';
        if (btnLn) { btnLn.style.background = SOL.base3; btnLn.style.fontWeight = 'normal'; }
        draw();
      });
    }
    if (btnTight) {
      btnTight.addEventListener('click', () => {
        x1 = 2.20; x2 = 2.20;
        if (x1Slider) x1Slider.value = x1;
        if (x2Slider) x2Slider.value = x2;
        draw();
      });
    }
    if (btnWide) {
      btnWide.addEventListener('click', () => {
        x1 = 0.65; x2 = 3.80;
        if (x1Slider) x1Slider.value = x1;
        if (x2Slider) x2Slider.value = x2;
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // =========================================================================
  // 3. DEMO 3: ITERATIVE SURROGATE LOWER-BOUND (ELBO) STEPPER
  // =========================================================================
  function initLowerBoundDemo() {
    const canvas = document.getElementById('em-lowerbound-canvas');
    if (!canvas) return;

    // Non-convex 1D log-likelihood with 2 local maxima (around theta=2.45 and theta=6.85)
    function trueLogLikelihood(th) {
      const m1 = 1.55 * Math.exp(-0.5 * Math.pow((th - 2.45) / 1.15, 2));
      const m2 = 2.45 * Math.exp(-0.5 * Math.pow((th - 6.85) / 1.35, 2));
      return -2.2 + Math.log(m1 + m2 + 0.12);
    }

    function trueDerivative(th) {
      const eps = 1e-4;
      return (trueLogLikelihood(th + eps) - trueLogLikelihood(th - eps)) / (2 * eps);
    }

    // Concave quadratic surrogate tangent to L(theta) at theta_t:
    // l(theta | theta_t) = L(theta_t) + L'(theta_t)*(theta - theta_t) - 0.5 * curv * (theta - theta_t)^2
    const curvature = 0.72;
    function surrogateBound(th, th_t) {
      const L_t = trueLogLikelihood(th_t);
      const dL_t = trueDerivative(th_t);
      const diff = th - th_t;
      return L_t + dL_t * diff - 0.5 * curvature * diff * diff;
    }

    function nextTheta(th_t) {
      return th_t + trueDerivative(th_t) / curvature;
    }

    let thetaCurr = 4.35;
    let phase = 'e_done'; // 'init', 'e_done' (bound built), 'm_done' (jumped to argmax)
    let history = [ { th: 4.35 } ];

    const thSlider = document.getElementById('lb-theta-slider');
    const thVal = document.getElementById('lb-theta-val');
    const iterVal = document.getElementById('lb-iter-val');
    const lVal = document.getElementById('lb-l-val');
    const statusText = document.getElementById('lb-status-text');
    const btnEStep = document.getElementById('lb-btn-estep');
    const btnMStep = document.getElementById('lb-btn-mstep');
    const btnFullStep = document.getElementById('lb-btn-fullstep');
    const btnResetLeft = document.getElementById('lb-btn-left');
    const btnResetMid = document.getElementById('lb-btn-mid');
    const btnResetRight = document.getElementById('lb-btn-right');

    function resetTo(th0) {
      thetaCurr = th0;
      history = [ { th: th0 } ];
      phase = 'e_done';
      if (thSlider) thSlider.value = th0.toFixed(2);
      draw();
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const padL = 62, padR = 30, padT = 26, padB = 42;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const xMin = 0.0, xMax = 10.0;
      const yMin = -4.2, yMax = -0.8;

      function toX(th) { return padL + ((th - xMin) / (xMax - xMin)) * plotW; }
      function toY(y) { return padT + plotH - ((y - yMin) / (yMax - yMin)) * plotH; }

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      for (let gx = 1; gx <= 9; gx++) {
        const sx = toX(gx);
        ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, padT + plotH); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(gx.toString(), sx, padT + plotH + 15);
      }
      for (let gy = -4.0; gy <= -1.0; gy += 0.5) {
        const sy = toY(gy);
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(padL + plotW, sy); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(gy.toFixed(1), padL - 6, sy + 4);
      }

      // Axes
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Parameter Space θ', padL + plotW / 2, H - 6);

      // Draw previous surrogate bounds in history (faint)
      for (let i = 0; i < history.length - 1; i++) {
        const th_i = history[i].th;
        ctx.strokeStyle = 'rgba(38, 139, 210, 0.28)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        for (let px = 0; px <= plotW; px += 2) {
          const th = xMin + (px / plotW) * (xMax - xMin);
          const y = surrogateBound(th, th_i);
          if (y >= yMin && y <= yMax) {
            if (!started) { ctx.moveTo(padL + px, toY(y)); started = true; }
            else ctx.lineTo(padL + px, toY(y));
          }
        }
        ctx.stroke();
      }

      // Draw True Marginal Log-Likelihood L(theta)
      ctx.strokeStyle = SOL.base03;
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 2) {
        const th = xMin + (px / plotW) * (xMax - xMin);
        const y = trueLogLikelihood(th);
        if (px === 0) ctx.moveTo(padL + px, toY(y));
        else ctx.lineTo(padL + px, toY(y));
      }
      ctx.stroke();

      // Current active lower bound l(theta | theta_t)
      const activeAnchor = (phase === 'm_done' && history.length >= 2)
        ? history[history.length - 2].th
        : thetaCurr;
      const thNext = nextTheta(activeAnchor);

      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      let started = false;
      for (let px = 0; px <= plotW; px += 2) {
        const th = xMin + (px / plotW) * (xMax - xMin);
        const y = surrogateBound(th, activeAnchor);
        if (y >= yMin - 0.2) {
          const cy = Math.max(yMin, y);
          if (!started) { ctx.moveTo(padL + px, toY(cy)); started = true; }
          else ctx.lineTo(padL + px, toY(cy));
        }
      }
      ctx.stroke();

      // Tangent point at activeAnchor: L(theta_t) = l(theta_t | theta_t)
      const yAnchor = trueLogLikelihood(activeAnchor);
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.5)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(toX(activeAnchor), padT + plotH);
      ctx.lineTo(toX(activeAnchor), toY(yAnchor));
      ctx.stroke();

      // Show M-step peak on the surrogate curve and true L(theta_{t+1})
      const ySurrPeak = surrogateBound(thNext, activeAnchor);
      const yTrueNext = trueLogLikelihood(thNext);

      if (Math.abs(thNext - activeAnchor) > 0.03) {
        ctx.strokeStyle = SOL.orange;
        ctx.beginPath();
        ctx.moveTo(toX(thNext), padT + plotH);
        ctx.lineTo(toX(thNext), toY(yTrueNext));
        ctx.stroke();
        ctx.setLineDash([]);

        // Highlight the KL gap between L(thNext) and l(thNext | th_t)
        ctx.strokeStyle = SOL.green;
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.moveTo(toX(thNext), toY(ySurrPeak));
        ctx.lineTo(toX(thNext), toY(yTrueNext));
        ctx.stroke();

        // Surrogate peak dot
        ctx.fillStyle = SOL.blue;
        ctx.beginPath();
        ctx.arc(toX(thNext), toY(ySurrPeak), 5.5, 0, 2 * Math.PI);
        ctx.fill();

        // True L(theta_{t+1}) dot
        if (phase === 'm_done') {
          ctx.fillStyle = SOL.red;
          ctx.beginPath();
          ctx.arc(toX(thNext), toY(yTrueNext), 6.5, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      ctx.setLineDash([]);

      // Draw history trajectory dots on L(theta)
      for (let i = 0; i < history.length; i++) {
        const th_i = history[i].th;
        const yi = trueLogLikelihood(th_i);
        ctx.fillStyle = (i === history.length - 1) ? SOL.red : SOL.base01;
        ctx.beginPath();
        ctx.arc(toX(th_i), toY(yi), (i === history.length - 1) ? 6.5 : 4.5, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Labels for Local Maxima
      ctx.font = '11px sans-serif';
      ctx.fillStyle = SOL.base01;
      ctx.textAlign = 'center';
      ctx.fillText('Local Max (θ≈2.45)', toX(2.45), toY(trueLogLikelihood(2.45)) - 10);
      ctx.fillStyle = SOL.green;
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText('Global Max (θ≈6.85)', toX(6.85), toY(trueLogLikelihood(6.85)) - 10);

      // Legend
      const legX = padL + 14, legY = padT + 10;
      ctx.fillStyle = 'rgba(253, 246, 227, 0.93)';
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1;
      ctx.fillRect(legX, legY, 275, 64);
      ctx.strokeRect(legX, legY, 275, 64);

      ctx.textAlign = 'left';
      ctx.font = '12px sans-serif';
      ctx.strokeStyle = SOL.base03; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 16); ctx.lineTo(legX + 34, legY + 16); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText('Marginal Log-Likelihood L(θ)', legX + 42, legY + 20);

      ctx.strokeStyle = SOL.blue; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 35); ctx.lineTo(legX + 34, legY + 35); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText('ELBO Surrogate l(θ | θₜ) (tight at θₜ)', legX + 42, legY + 39);

      ctx.strokeStyle = SOL.green; ctx.lineWidth = 3.5;
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 52); ctx.lineTo(legX + 34, legY + 52); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText('KL Gap D_KL(q || p) ≥ 0', legX + 42, legY + 56);

      // Update status & readouts
      const curL = trueLogLikelihood(thetaCurr);
      if (thVal) thVal.textContent = thetaCurr.toFixed(2);
      if (iterVal) iterVal.textContent = (history.length - 1).toString();
      if (lVal) lVal.textContent = curL.toFixed(3);

      if (statusText) {
        const grad = Math.abs(trueDerivative(thetaCurr));
        if (grad < 0.015) {
          statusText.style.color = SOL.green;
          statusText.innerHTML = `✓ <b>Converged to Stationary Point (θ* = ${thetaCurr.toFixed(2)})!</b> Surrogate is tangent at the peak: l'(θ*|θ*) = L'(θ*) = 0.`;
        } else if (phase === 'e_done') {
          statusText.style.color = SOL.blue;
          statusText.innerHTML = `<b>E-Step Active:</b> Surrogate l(θ | θₜ=${thetaCurr.toFixed(2)}) touches L(θ) at θₜ (KL = 0). Click <b>M-Step</b> to jump to its peak θₜ₊₁ = ${thNext.toFixed(2)}!`;
        } else {
          statusText.style.color = SOL.orange;
          statusText.innerHTML = `<b>M-Step Complete:</b> Moved to θₜ₊₁ = ${thetaCurr.toFixed(2)}. Notice L(θₜ₊₁) ≥ l(θₜ₊₁|θₜ) ≥ L(θₜ)! Click <b>E-Step</b> to tighten the bound.`;
        }
      }
    }

    if (btnEStep) {
      btnEStep.addEventListener('click', () => {
        phase = 'e_done';
        draw();
      });
    }
    if (btnMStep) {
      btnMStep.addEventListener('click', () => {
        const thNext = Math.max(0.2, Math.min(9.8, nextTheta(thetaCurr)));
        if (Math.abs(thNext - thetaCurr) > 1e-4) {
          thetaCurr = thNext;
          history.push({ th: thetaCurr });
          if (thSlider) thSlider.value = thetaCurr.toFixed(2);
        }
        phase = 'm_done';
        draw();
      });
    }
    if (btnFullStep) {
      btnFullStep.addEventListener('click', () => {
        const thNext = Math.max(0.2, Math.min(9.8, nextTheta(thetaCurr)));
        if (Math.abs(thNext - thetaCurr) > 1e-4) {
          thetaCurr = thNext;
          history.push({ th: thetaCurr });
          if (thSlider) thSlider.value = thetaCurr.toFixed(2);
        }
        phase = 'e_done';
        draw();
      });
    }
    if (thSlider) {
      thSlider.addEventListener('input', () => {
        resetTo(parseFloat(thSlider.value));
      });
    }
    if (btnResetLeft) btnResetLeft.addEventListener('click', () => resetTo(0.85));
    if (btnResetMid) btnResetMid.addEventListener('click', () => resetTo(4.35));
    if (btnResetRight) btnResetRight.addEventListener('click', () => resetTo(8.90));

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // =========================================================================
  // 4. DEMO 4: FORWARD VS. REVERSE KL DIVERGENCE EXPLORER
  // =========================================================================
  function initKLDemo() {
    const canvas = document.getElementById('em-kl-canvas');
    if (!canvas) return;

    let sep = 1.85; // Modes at -sep and +sep
    const sigP = 0.65;
    let muQ = 0.0;
    let sigQ = Math.sqrt(sep * sep + sigP * sigP); // Optimal forward KL std
    let activeMode = 'forward'; // 'forward' or 'reverse'

    const muSlider = document.getElementById('kl-mu-slider');
    const sigSlider = document.getElementById('kl-sig-slider');
    const sepSlider = document.getElementById('kl-sep-slider');
    const muVal = document.getElementById('kl-mu-val');
    const sigVal = document.getElementById('kl-sig-val');
    const fwdVal = document.getElementById('kl-fwd-val');
    const revVal = document.getElementById('kl-rev-val');
    const btnForward = document.getElementById('kl-btn-forward');
    const btnRevLeft = document.getElementById('kl-btn-rev-left');
    const btnRevRight = document.getElementById('kl-btn-rev-right');

    function pPDF(x) {
      return 0.5 * gaussian1D(x, -sep, sigP) + 0.5 * gaussian1D(x, sep, sigP);
    }

    function qPDF(x) {
      return gaussian1D(x, muQ, sigQ);
    }

    function computeKLs() {
      const xMin = -6.0, xMax = 6.0, N = 400;
      const dx = (xMax - xMin) / N;
      let klFwd = 0, klRev = 0;
      for (let i = 0; i <= N; i++) {
        const x = xMin + i * dx;
        const p = Math.max(1e-15, pPDF(x));
        const q = Math.max(1e-15, qPDF(x));
        klFwd += p * Math.log(p / q) * dx;
        klRev += q * Math.log(q / p) * dx;
      }
      return { klFwd: Math.max(0, klFwd), klRev: Math.max(0, klRev) };
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const padT = 28, padB = 40;
      const plotH = H - padT - padB;

      // Left Panel: Densities p(x) and q(x)
      const lPadL = 52, lW = 500;
      // Right Panel: Pointwise Integrand
      const rPadL = 610, rW = W - rPadL - 24;

      const xMin = -4.8, xMax = 4.8;
      const yMax = 0.65;

      function toLX(x) { return lPadL + ((x - xMin) / (xMax - xMin)) * lW; }
      function toLY(y) { return padT + plotH - (Math.min(y, yMax) / yMax) * plotH; }

      // Left Panel Grid & Axes
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      for (let gx = -4; gx <= 4; gx += 2) {
        const sx = toLX(gx);
        ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, padT + plotH); ctx.stroke();
        ctx.fillStyle = SOL.base01; ctx.font = '11px monospace'; ctx.textAlign = 'center';
        ctx.fillText(gx.toString(), sx, padT + plotH + 15);
      }
      ctx.strokeStyle = SOL.base01; ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(lPadL, padT); ctx.lineTo(lPadL, padT + plotH); ctx.lineTo(lPadL + lW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base02; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Bimodal Target p(x) vs. Gaussian Approximation q(x)', lPadL + lW / 2, 16);

      // Draw p(x) (Blue solid with light fill)
      ctx.fillStyle = 'rgba(38, 139, 210, 0.12)';
      ctx.beginPath();
      ctx.moveTo(lPadL, padT + plotH);
      for (let px = 0; px <= lW; px += 2) {
        const x = xMin + (px / lW) * (xMax - xMin);
        ctx.lineTo(lPadL + px, toLY(pPDF(x)));
      }
      ctx.lineTo(lPadL + lW, padT + plotH);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = SOL.blue; ctx.lineWidth = 3;
      ctx.beginPath();
      for (let px = 0; px <= lW; px += 2) {
        const x = xMin + (px / lW) * (xMax - xMin);
        if (px === 0) ctx.moveTo(lPadL + px, toLY(pPDF(x)));
        else ctx.lineTo(lPadL + px, toLY(pPDF(x)));
      }
      ctx.stroke();

      // Draw q(x) (Green dashed)
      ctx.strokeStyle = SOL.green; ctx.lineWidth = 3;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      for (let px = 0; px <= lW; px += 2) {
        const x = xMin + (px / lW) * (xMax - xMin);
        if (px === 0) ctx.moveTo(lPadL + px, toLY(qPDF(x)));
        else ctx.lineTo(lPadL + px, toLY(qPDF(x)));
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Right Panel: Pointwise Integrand
      const intMin = -0.25, intMax = 0.65;
      function toRX(x) { return rPadL + ((x - xMin) / (xMax - xMin)) * rW; }
      function toRY(y) {
        const clamped = Math.max(intMin, Math.min(intMax, y));
        return padT + plotH - ((clamped - intMin) / (intMax - intMin)) * plotH;
      }

      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      for (let gx = -4; gx <= 4; gx += 2) {
        const sx = toRX(gx);
        ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, padT + plotH); ctx.stroke();
        ctx.fillStyle = SOL.base01; ctx.font = '11px monospace'; ctx.textAlign = 'center';
        ctx.fillText(gx.toString(), sx, padT + plotH + 15);
      }
      // Zero baseline for integrand
      const yZero = toRY(0);
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.45)';
      ctx.beginPath(); ctx.moveTo(rPadL, yZero); ctx.lineTo(rPadL + rW, yZero); ctx.stroke();

      ctx.strokeStyle = SOL.base01; ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(rPadL, padT); ctx.lineTo(rPadL, padT + plotH); ctx.lineTo(rPadL + rW, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base02; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(
        activeMode === 'forward' ? 'Integrand: p(x) ln[p(x)/q(x)]' : 'Integrand: q(x) ln[q(x)/p(x)] (EM / VI)',
        rPadL + rW / 2, 16
      );

      // Plot Integrand Curve
      ctx.fillStyle = activeMode === 'forward' ? 'rgba(203, 75, 22, 0.18)' : 'rgba(108, 113, 196, 0.18)';
      ctx.beginPath();
      ctx.moveTo(rPadL, yZero);
      for (let px = 0; px <= rW; px += 2) {
        const x = xMin + (px / rW) * (xMax - xMin);
        const p = Math.max(1e-12, pPDF(x));
        const q = Math.max(1e-12, qPDF(x));
        const val = activeMode === 'forward' ? p * Math.log(p / q) : q * Math.log(q / p);
        ctx.lineTo(rPadL + px, toRY(val));
      }
      ctx.lineTo(rPadL + rW, yZero);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = activeMode === 'forward' ? SOL.orange : SOL.violet;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let px = 0; px <= rW; px += 2) {
        const x = xMin + (px / rW) * (xMax - xMin);
        const p = Math.max(1e-12, pPDF(x));
        const q = Math.max(1e-12, qPDF(x));
        const val = activeMode === 'forward' ? p * Math.log(p / q) : q * Math.log(q / p);
        if (px === 0) ctx.moveTo(rPadL + px, toRY(val));
        else ctx.lineTo(rPadL + px, toRY(val));
      }
      ctx.stroke();

      // Explanation callout inside right panel
      ctx.fillStyle = SOL.base01;
      ctx.font = '11.5px sans-serif';
      ctx.textAlign = 'left';
      if (activeMode === 'forward') {
        ctx.fillText('• Mass-Covering: p(x)>0 forces q(x)>0 everywhere', rPadL + 10, padT + 20);
        ctx.fillText('• Averages across modes (places mass in valley!)', rPadL + 10, padT + 36);
      } else {
        ctx.fillText('• Mode-Seeking (Zero-Forcing): p(x)≈0 forces q(x)≈0', rPadL + 10, padT + 20);
        ctx.fillText('• Locks cleanly onto a single mode (used in EM & VI!)', rPadL + 10, padT + 36);
      }

      const { klFwd, klRev } = computeKLs();
      if (muVal) muVal.textContent = muQ.toFixed(2);
      if (sigVal) sigVal.textContent = sigQ.toFixed(2);
      if (fwdVal) fwdVal.textContent = klFwd.toFixed(3);
      if (revVal) revVal.textContent = klRev.toFixed(3);
    }

    if (muSlider) muSlider.addEventListener('input', () => { muQ = parseFloat(muSlider.value); draw(); });
    if (sigSlider) sigSlider.addEventListener('input', () => { sigQ = parseFloat(sigSlider.value); draw(); });
    if (sepSlider) sepSlider.addEventListener('input', () => { sep = parseFloat(sepSlider.value); draw(); });

    if (btnForward) {
      btnForward.addEventListener('click', () => {
        activeMode = 'forward';
        muQ = 0.0;
        sigQ = Math.sqrt(sep * sep + sigP * sigP);
        if (muSlider) muSlider.value = muQ.toFixed(2);
        if (sigSlider) sigSlider.value = sigQ.toFixed(2);
        draw();
      });
    }
    if (btnRevLeft) {
      btnRevLeft.addEventListener('click', () => {
        activeMode = 'reverse';
        muQ = -sep;
        sigQ = sigP;
        if (muSlider) muSlider.value = muQ.toFixed(2);
        if (sigSlider) sigSlider.value = sigQ.toFixed(2);
        draw();
      });
    }
    if (btnRevRight) {
      btnRevRight.addEventListener('click', () => {
        activeMode = 'reverse';
        muQ = sep;
        sigQ = sigP;
        if (muSlider) muSlider.value = muQ.toFixed(2);
        if (sigSlider) sigSlider.value = sigQ.toFixed(2);
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // =========================================================================
  // 5. DEMO 5: INTERACTIVE 2D FULL-COVARIANCE GMM EM PLAYGROUND
  // =========================================================================
  function initGMM2DDemo() {
    const canvas = document.getElementById('em-gmm2d-canvas');
    if (!canvas) return;

    const clusterColors = [SOL.green, SOL.red, SOL.blue, SOL.violet];
    let K = 3;
    let covType = 'full'; // 'full', 'diag', 'spherical', 'kmeans'
    let presetName = 'tilted';
    let points = [];
    let params = []; // [{ mu: [x,y], cov: [[a,b],[b,d]], pi: p }]
    let responsibilities = []; // N x K
    let llHistory = [];
    let iterCount = 0;
    let stepStage = 'init'; // 'init', 'e_done', 'm_done'
    let autoTimer = null;

    const btnEStep = document.getElementById('gmm-btn-estep');
    const btnMStep = document.getElementById('gmm-btn-mstep');
    const btnFullStep = document.getElementById('gmm-btn-fullstep');
    const btnAuto = document.getElementById('gmm-btn-auto');
    const btnReinit = document.getElementById('gmm-btn-reinit');
    const selDataset = document.getElementById('gmm-sel-dataset');
    const selCov = document.getElementById('gmm-sel-cov');
    const selK = document.getElementById('gmm-sel-k');
    const iterSpan = document.getElementById('gmm-iter-val');
    const llSpan = document.getElementById('gmm-ll-val');
    const stageSpan = document.getElementById('gmm-stage-val');

    function stopAuto() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
        if (btnAuto) btnAuto.textContent = '▶ Auto-Run';
      }
    }

    function generateDataset(name) {
      const rng = pseudoRandom(name === 'tilted' ? 137 : (name === 'unequal' ? 271 : 42));
      points = [];

      function addGaussianCluster(n, mx, my, sxx, sxy, syy) {
        // Cholesky of [[sxx, sxy], [sxy, syy]]
        const l11 = Math.sqrt(Math.max(1e-4, sxx));
        const l21 = sxy / l11;
        const l22 = Math.sqrt(Math.max(1e-4, syy - l21 * l21));
        for (let i = 0; i < n; i++) {
          const z1 = sampleGaussian(rng, 0, 1);
          const z2 = sampleGaussian(rng, 0, 1);
          points.push({
            x: mx + l11 * z1,
            y: my + l21 * z1 + l22 * z2
          });
        }
      }

      if (name === 'tilted') {
        // Two strongly elongated parallel/tilted ellipses + one compact cluster (where K-means fails)
        addGaussianCluster(24, -1.5, 1.1, 1.35, -1.05, 1.05);
        addGaussianCluster(24, 1.1, 1.3, 1.25, -0.95, 0.95);
        addGaussianCluster(22, 0.6, -1.6, 0.35, 0.08, 0.28);
      } else if (name === 'unequal') {
        // One huge diffuse cluster next to two tight small clusters
        addGaussianCluster(30, -1.2, 0.2, 1.65, 0.2, 1.55);
        addGaussianCluster(18, 2.1, 1.5, 0.14, 0.0, 0.14);
        addGaussianCluster(18, 2.0, -1.5, 0.14, 0.0, 0.14);
      } else {
        // Classic Barnabas 3 clusters
        addGaussianCluster(18, -1.7, 0.8, 0.65, -0.25, 0.45);
        addGaussianCluster(20, 0.9, 1.1, 0.60, -0.35, 0.55);
        addGaussianCluster(22, 1.3, -1.4, 0.35, 0.05, 0.30);
      }
    }

    function initParameters(seedOffset) {
      stopAuto();
      const rng = pseudoRandom(901 + (seedOffset || 0) * 73);
      params = [];
      for (let k = 0; k < K; k++) {
        // Deliberately broad initial Gaussians (like Barnabas Slide 61)
        const angle = (2 * Math.PI * k) / K + rng() * 0.8;
        const rad = 0.6 + rng() * 0.6;
        params.push({
          mu: [ rad * Math.cos(angle), rad * Math.sin(angle) ],
          cov: [ [1.6, 0.2 * (k - 1)], [0.2 * (k - 1), 1.6] ],
          pi: 1.0 / K
        });
      }
      responsibilities = points.map(() => {
        const r = new Array(K).fill(1.0 / K);
        return r;
      });
      iterCount = 0;
      llHistory = [];
      stepStage = 'init';
      // Compute initial responsibilities and log-likelihood
      runEStep(false);
    }

    function evalGaussian2D(pt, mu, cov) {
      const a = cov[0][0] + 1e-4;
      const b = cov[0][1];
      const d = cov[1][1] + 1e-4;
      const det = Math.max(1e-6, a * d - b * b);
      const invA = d / det;
      const invB = -b / det;
      const invD = a / det;
      const dx = pt.x - mu[0];
      const dy = pt.y - mu[1];
      const mahal = dx * (invA * dx + invB * dy) + dy * (invB * dx + invD * dy);
      return (1.0 / (2.0 * Math.PI * Math.sqrt(det))) * Math.exp(-0.5 * mahal);
    }

    function computeCurrentLL() {
      let ll = 0;
      for (let n = 0; n < points.length; n++) {
        let sumP = 0;
        for (let k = 0; k < K; k++) {
          sumP += params[k].pi * evalGaussian2D(points[n], params[k].mu, params[k].cov);
        }
        ll += Math.log(Math.max(1e-30, sumP));
      }
      return ll;
    }

    function runEStep(recordStage = true) {
      for (let n = 0; n < points.length; n++) {
        const pt = points[n];
        if (covType === 'kmeans') {
          let bestK = 0, bestDist = Infinity;
          for (let k = 0; k < K; k++) {
            const dx = pt.x - params[k].mu[0];
            const dy = pt.y - params[k].mu[1];
            const dist = dx * dx + dy * dy;
            if (dist < bestDist) { bestDist = dist; bestK = k; }
          }
          for (let k = 0; k < K; k++) responsibilities[n][k] = (k === bestK) ? 1.0 : 0.0;
        } else {
          let sum = 0;
          const probs = new Array(K);
          for (let k = 0; k < K; k++) {
            probs[k] = params[k].pi * evalGaussian2D(pt, params[k].mu, params[k].cov);
            sum += probs[k];
          }
          for (let k = 0; k < K; k++) {
            responsibilities[n][k] = sum > 1e-30 ? probs[k] / sum : (1.0 / K);
          }
        }
      }
      const ll = computeCurrentLL();
      if (llHistory.length === 0) llHistory.push(ll);
      if (recordStage) stepStage = 'e_done';
    }

    function runMStep() {
      const N = points.length;
      const reg = 0.04; // Regularization floor

      for (let k = 0; k < K; k++) {
        let Nk = 0;
        let mx = 0, my = 0;
        for (let n = 0; n < N; n++) {
          const r = responsibilities[n][k];
          Nk += r;
          mx += r * points[n].x;
          my += r * points[n].y;
        }
        if (Nk < 1e-4) continue;
        mx /= Nk;
        my /= Nk;
        params[k].mu = [mx, my];
        params[k].pi = (covType === 'kmeans') ? (1.0 / K) : (Nk / N);

        let sxx = 0, sxy = 0, syy = 0;
        for (let n = 0; n < N; n++) {
          const r = responsibilities[n][k];
          const dx = points[n].x - mx;
          const dy = points[n].y - my;
          sxx += r * dx * dx;
          sxy += r * dx * dy;
          syy += r * dy * dy;
        }
        sxx = sxx / Nk + reg;
        sxy = sxy / Nk;
        syy = syy / Nk + reg;

        if (covType === 'full') {
          params[k].cov = [ [sxx, sxy], [sxy, syy] ];
        } else if (covType === 'diag') {
          params[k].cov = [ [sxx, 0], [0, syy] ];
        } else if (covType === 'spherical') {
          const iso = 0.5 * (sxx + syy);
          params[k].cov = [ [iso, 0], [0, iso] ];
        } else if (covType === 'kmeans') {
          params[k].cov = [ [0.45, 0], [0, 0.45] ];
        }
      }

      iterCount++;
      llHistory.push(computeCurrentLL());
      stepStage = 'm_done';
    }

    function drawCovarianceEllipse(ctx, toX, toY, scaleX, scaleY, mu, cov, color, nStd, dashed) {
      // Eigendecomposition of 2x2 symmetric matrix [[a,b],[b,d]]
      const a = cov[0][0], b = cov[0][1], d = cov[1][1];
      const trace = a + d;
      const det = a * d - b * b;
      const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
      const lam1 = Math.max(1e-4, trace / 2 + disc);
      const lam2 = Math.max(1e-4, trace / 2 - disc);
      const angle = (Math.abs(b) > 1e-6) ? Math.atan2(lam1 - a, b) : (a >= d ? 0 : Math.PI / 2);

      const rx = nStd * Math.sqrt(lam1);
      const ry = nStd * Math.sqrt(lam2);

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 1.4 : 2.2;
      if (dashed) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      const steps = 64;
      for (let i = 0; i <= steps; i++) {
        const t = (2 * Math.PI * i) / steps;
        const ex = rx * Math.cos(t);
        const ey = ry * Math.sin(t);
        const wx = mu[0] + ex * Math.cos(angle) - ey * Math.sin(angle);
        const wy = mu[1] + ex * Math.sin(angle) + ey * Math.cos(angle);
        const sx = toX(wx), sy = toY(wy);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      // Left Scatter Panel
      const sPadL = 42, sPadT = 26, sPadB = 34;
      const sW = 530, sH = H - sPadT - sPadB;
      const xMin = -3.8, xMax = 3.8;
      const yMin = -3.2, yMax = 3.2;

      function toSX(x) { return sPadL + ((x - xMin) / (xMax - xMin)) * sW; }
      function toSY(y) { return sPadT + sH - ((y - yMin) / (yMax - yMin)) * sH; }

      // Scatter Grid & Frame
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.2)';
      ctx.lineWidth = 1;
      for (let gx = -3; gx <= 3; gx++) {
        const sx = toSX(gx);
        ctx.beginPath(); ctx.moveTo(sx, sPadT); ctx.lineTo(sx, sPadT + sH); ctx.stroke();
      }
      for (let gy = -3; gy <= 3; gy++) {
        const sy = toSY(gy);
        ctx.beginPath(); ctx.moveTo(sPadL, sy); ctx.lineTo(sPadL + sW, sy); ctx.stroke();
      }
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sPadL, sPadT, sW, sH);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('2D Feature Space (Pie-Charts = Latent Responsibilities γₙₖ; Click to Add Point)', sPadL + sW / 2, 16);

      // Draw Covariance Ellipses (1-sigma solid, 2-sigma dashed)
      for (let k = 0; k < K; k++) {
        drawCovarianceEllipse(ctx, toSX, toSY, 1, 1, params[k].mu, params[k].cov, clusterColors[k], 1.2, false);
        drawCovarianceEllipse(ctx, toSX, toSY, 1, 1, params[k].mu, params[k].cov, clusterColors[k], 2.1, true);
      }

      // Draw Data Points as Mini Responsibility Pie-Charts (matching Barnabas slides 61-68!)
      const radius = 7.5;
      for (let n = 0; n < points.length; n++) {
        const sx = toSX(points[n].x);
        const sy = toSY(points[n].y);
        if (sx < sPadL || sx > sPadL + sW || sy < sPadT || sy > sPadT + sH) continue;

        let startAngle = -Math.PI / 2;
        for (let k = 0; k < K; k++) {
          const frac = responsibilities[n] ? responsibilities[n][k] : (1.0 / K);
          if (frac <= 0.002) continue;
          const sliceAngle = frac * 2 * Math.PI;
          ctx.fillStyle = clusterColors[k];
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.arc(sx, sy, radius, startAngle, startAngle + sliceAngle);
          ctx.closePath();
          ctx.fill();
          startAngle += sliceAngle;
        }
        ctx.strokeStyle = SOL.base03;
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.arc(sx, sy, radius, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // Draw Centroids mu_k and mixing proportions pi_k
      for (let k = 0; k < K; k++) {
        const cx = toSX(params[k].mu[0]);
        const cy = toSY(params[k].mu[1]);
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(cx, cy, 7, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = clusterColors[k];
        ctx.strokeStyle = SOL.base03;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

        // Mixing weight badge
        ctx.fillStyle = 'rgba(253, 246, 227, 0.9)';
        ctx.fillRect(cx + 8, cy - 10, 56, 16);
        ctx.fillStyle = SOL.base03;
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`π=${params[k].pi.toFixed(2)}`, cx + 10, cy + 2);
      }

      // Right Panel: Log-Likelihood Convergence Curve
      const rPadL = 620, rW = W - rPadL - 22;
      ctx.strokeStyle = SOL.base1;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rPadL, sPadT, rW, sH);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Monotonic Log-Likelihood L(θ⁽ᵗ⁾)', rPadL + rW / 2, 16);

      if (llHistory.length > 0) {
        const minLL = Math.min(...llHistory) - 5;
        const maxLL = Math.max(...llHistory) + 5;
        const maxT = Math.max(8, llHistory.length - 1);

        function toRX(t) { return rPadL + 36 + (t / maxT) * (rW - 50); }
        function toRY(ll) {
          const span = Math.max(1, maxLL - minLL);
          return sPadT + sH - 28 - ((ll - minLL) / span) * (sH - 52);
        }

        // Horizontal grid lines
        ctx.strokeStyle = 'rgba(147, 161, 161, 0.22)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 3; i++) {
          const v = minLL + ((maxLL - minLL) * i) / 3;
          const sy = toRY(v);
          ctx.beginPath(); ctx.moveTo(rPadL + 34, sy); ctx.lineTo(rPadL + rW - 10, sy); ctx.stroke();
          ctx.fillStyle = SOL.base01;
          ctx.font = '10px monospace';
          ctx.textAlign = 'right';
          ctx.fillText(Math.round(v).toString(), rPadL + 30, sy + 3);
        }

        // Curve
        ctx.strokeStyle = SOL.blue;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let t = 0; t < llHistory.length; t++) {
          const sx = toRX(t), sy = toRY(llHistory[t]);
          if (t === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();

        // Dots
        for (let t = 0; t < llHistory.length; t++) {
          const sx = toRX(t), sy = toRY(llHistory[t]);
          ctx.fillStyle = (t === llHistory.length - 1) ? SOL.red : SOL.blue;
          ctx.beginPath();
          ctx.arc(sx, sy, (t === llHistory.length - 1) ? 5 : 3.5, 0, 2 * Math.PI);
          ctx.fill();
        }

        ctx.fillStyle = SOL.base01;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Iteration t', rPadL + rW / 2, sPadT + sH - 6);
      }

      const curLL = llHistory.length > 0 ? llHistory[llHistory.length - 1] : 0;
      if (iterSpan) iterSpan.textContent = iterCount.toString();
      if (llSpan) llSpan.textContent = curLL.toFixed(1);
      if (stageSpan) {
        if (stepStage === 'init') stageSpan.textContent = 'Initialized';
        else if (stepStage === 'e_done') stageSpan.textContent = 'E-Step (Updated γₙₖ)';
        else stageSpan.textContent = 'M-Step (Updated μₖ, Σₖ, πₖ)';
      }
    }

    // Click on left plot to add custom data point
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const logW = canvas._logicalW || 940;
      const logH = canvas._logicalH || 350;
      const clickX = ((e.clientX - rect.left) / rect.width) * logW;
      const clickY = ((e.clientY - rect.top) / rect.height) * logH;

      const sPadL = 42, sPadT = 26, sPadB = 34;
      const sW = 530, sH = logH - sPadT - sPadB;
      if (clickX >= sPadL && clickX <= sPadL + sW && clickY >= sPadT && clickY <= sPadT + sH) {
        const wx = -3.8 + ((clickX - sPadL) / sW) * 7.6;
        const wy = 3.2 - ((clickY - sPadT) / sH) * 6.4;
        points.push({ x: wx, y: wy });
        responsibilities.push(new Array(K).fill(1.0 / K));
        runEStep(false);
        draw();
      }
    });

    if (btnEStep) {
      btnEStep.addEventListener('click', () => {
        stopAuto();
        runEStep(true);
        draw();
      });
    }
    if (btnMStep) {
      btnMStep.addEventListener('click', () => {
        stopAuto();
        runMStep();
        draw();
      });
    }
    if (btnFullStep) {
      btnFullStep.addEventListener('click', () => {
        stopAuto();
        runEStep(false);
        runMStep();
        runEStep(false);
        stepStage = 'm_done';
        draw();
      });
    }
    if (btnAuto) {
      btnAuto.addEventListener('click', () => {
        if (autoTimer) {
          stopAuto();
        } else {
          btnAuto.textContent = '⏸ Pause';
          autoTimer = setInterval(() => {
            runEStep(false);
            runMStep();
            runEStep(false);
            draw();
            if (iterCount >= 25) stopAuto();
          }, 420);
        }
      });
    }
    let seedCounter = 0;
    if (btnReinit) {
      btnReinit.addEventListener('click', () => {
        seedCounter++;
        initParameters(seedCounter);
        draw();
      });
    }
    if (selDataset) {
      selDataset.addEventListener('change', () => {
        presetName = selDataset.value;
        generateDataset(presetName);
        initParameters(seedCounter);
        draw();
      });
    }
    if (selCov) {
      selCov.addEventListener('change', () => {
        covType = selCov.value;
        initParameters(seedCounter);
        draw();
      });
    }
    if (selK) {
      selK.addEventListener('change', () => {
        K = parseInt(selK.value, 10) || 3;
        initParameters(seedCounter);
        draw();
      });
    }

    generateDataset(presetName);
    initParameters(0);
    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // Initialize all demos on DOMContentLoaded
  function initAll() {
    initBridgeSlide();
    initSingularityDemo();
    initJensenDemo();
    initLowerBoundDemo();
    initKLDemo();
    initGMM2DDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
