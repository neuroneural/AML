/**
 * CS8850: Advanced Machine Learning - Lecture 18 (Automatic Differentiation)
 * Interactive Pedagogical Demonstrations Suite
 *
 * 1. Numerical Differentiation Trap: Truncation vs. Catastrophic Cancellation Explorer
 * 2. Forward-Mode AD & Dual Numbers Graph Tracer (y = ln(x1) + x1*x2 - sin(x2))
 * 3. Reverse-Mode AD (Backpropagation): Forward Tape & Backward Adjoint Tracer
 * 4. Forward JVP vs. Reverse VJP Jacobian & Complexity Explorer
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
    return { ctx, width: logW, height: logH };
  }

  /**
   * Helper to ensure canvases render immediately when navigated to.
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

  function drawRoundedRect(ctx, x, y, w, h, r, fill, stroke, lineWidth) {
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
      ctx.lineWidth = lineWidth || 1.5;
      ctx.stroke();
    }
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, lineWidth, headLen, dashed) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const angle = Math.atan2(dy, dx);
    const hl = headLen || 9;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth || 2;
    if (dashed) ctx.setLineDash(dashed);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - hl * Math.cos(angle - Math.PI / 6), y2 - hl * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - hl * Math.cos(angle + Math.PI / 6), y2 - hl * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // =========================================================================
  // 1. DEMO 1: NUMERICAL DIFFERENTIATION (TRUNCATION VS. CANCELLATION)
  // =========================================================================
  function initNumDiffDemo() {
    const canvas = document.getElementById('ad-numdiff-canvas');
    if (!canvas) return;

    let logH = -8.0; // h = 10^logH
    let x0 = 0.28;
    let fnType = 'logistic'; // 'logistic' | 'trig'

    const hSlider = document.getElementById('nd-h-slider');
    const x0Slider = document.getElementById('nd-x0-slider');
    const selFn = document.getElementById('nd-sel-fn');
    const btnTrunc = document.getElementById('nd-btn-trunc');
    const btnSweet = document.getElementById('nd-btn-sweet');
    const btnCancel = document.getElementById('nd-btn-cancel');

    const hValEl = document.getElementById('nd-h-val');
    const x0ValEl = document.getElementById('nd-x0-val');
    const exactValEl = document.getElementById('nd-exact-val');
    const fwdErrEl = document.getElementById('nd-fwd-err');
    const cenErrEl = document.getElementById('nd-cen-err');
    const statusBanner = document.getElementById('nd-status-banner');

    // Dual number evaluation of f(x) -> { v, dv }
    function evalDual(x, dx) {
      if (fnType === 'logistic') {
        // l_1 = x; l_{n+1} = 4 * l_n * (1 - l_n) for 3 iterations -> l_4
        let v = x;
        let dv = dx;
        for (let i = 0; i < 3; i++) {
          dv = 4.0 * dv * (1.0 - 2.0 * v);
          v = 4.0 * v * (1.0 - v);
        }
        return { v, dv };
      } else {
        // y = sin(4x) + x^2
        const v = Math.sin(4.0 * x) + x * x;
        const dv = (4.0 * Math.cos(4.0 * x) + 2.0 * x) * dx;
        return { v, dv };
      }
    }

    function evalPrimal(x) {
      return evalDual(x, 0.0).v;
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const h = Math.pow(10, logH);
      const exact = evalDual(x0, 1.0);
      const fx0 = exact.v;
      const dfExact = exact.dv;

      const fxPlus = evalPrimal(x0 + h);
      const fxMinus = evalPrimal(x0 - h);
      const dfFwd = (fxPlus - fx0) / h;
      const dfCen = (fxPlus - fxMinus) / (2.0 * h);

      const errFwd = Math.max(1e-17, Math.abs(dfFwd - dfExact));
      const errCen = Math.max(1e-17, Math.abs(dfCen - dfExact));

      // Layout: Left panel (Function + Secant vs Tangent + Float64 cancellation box), Right panel (Log-Log Error curve)
      const leftX = 52, leftY = 26, leftW = 490, leftH = 265;
      const rightX = 615, rightY = 26, rightW = 535, rightH = 380;

      // ---------------- LEFT SUBPLOT: f(x), Exact AD Tangent vs Finite-Diff Secant ----------------
      const xMin = 0.0, xMax = 1.0;
      const yMin = fnType === 'logistic' ? -0.25 : -1.2;
      const yMax = fnType === 'logistic' ? 1.25 : 2.2;

      function toLX(x) { return leftX + ((x - xMin) / (xMax - xMin)) * leftW; }
      function toLY(y) { return leftY + leftH - ((y - yMin) / (yMax - yMin)) * leftH; }

      drawRoundedRect(ctx, leftX, leftY, leftW, leftH, 4, '#ffffff', SOL.base1, 1);

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.25)';
      ctx.lineWidth = 1;
      for (let gx = 0.2; gx <= 0.85; gx += 0.2) {
        const sx = toLX(gx);
        ctx.beginPath(); ctx.moveTo(sx, leftY); ctx.lineTo(sx, leftY + leftH); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11.5px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(gx.toFixed(1), sx, leftY + leftH + 15);
      }
      const zeroY = toLY(0);
      ctx.strokeStyle = 'rgba(88, 110, 117, 0.4)';
      ctx.beginPath(); ctx.moveTo(leftX, zeroY); ctx.lineTo(leftX + leftW, zeroY); ctx.stroke();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        fnType === 'logistic'
          ? 'Iterated Logistic Map f(x) = l₄(x)  &  Derivative Slope at x₀'
          : 'Function f(x) = sin(4x) + x²  &  Derivative Slope at x₀',
        leftX + leftW / 2,
        leftY - 7
      );

      // Clip to left plot box
      ctx.save();
      ctx.beginPath();
      ctx.rect(leftX, leftY, leftW, leftH);
      ctx.clip();

      // Curve f(x)
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      const Npts = 350;
      for (let i = 0; i <= Npts; i++) {
        const xv = xMin + (i / Npts) * (xMax - xMin);
        const yv = evalPrimal(xv);
        const sx = toLX(xv), sy = toLY(yv);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Exact AD tangent line (Green)
      const dxSpan = 0.22;
      ctx.strokeStyle = SOL.green;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(toLX(x0 - dxSpan), toLY(fx0 - dfExact * dxSpan));
      ctx.lineTo(toLX(x0 + dxSpan), toLY(fx0 + dfExact * dxSpan));
      ctx.stroke();

      // Finite-difference secant line (Red dashed)
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2.4;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(toLX(x0 - dxSpan), toLY(fx0 - dfFwd * dxSpan));
      ctx.lineTo(toLX(x0 + dxSpan), toLY(fx0 + dfFwd * dxSpan));
      ctx.stroke();
      ctx.setLineDash([]);

      // Show (x0 + h, f(x0 + h)) if h is visually distinguishable
      if (h >= 0.005) {
        const sxH = toLX(x0 + h), syH = toLY(fxPlus);
        ctx.fillStyle = SOL.orange;
        ctx.beginPath(); ctx.arc(sxH, syH, 5.5, 0, 2 * Math.PI); ctx.fill();
      }

      // Point (x0, f(x0))
      const sx0 = toLX(x0), sy0 = toLY(fx0);
      ctx.fillStyle = SOL.base03;
      ctx.beginPath(); ctx.arc(sx0, sy0, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();

      // Legend inside left plot
      drawRoundedRect(ctx, leftX + 10, leftY + 8, 255, 58, 4, 'rgba(253,246,227,0.92)', SOL.base1, 1);
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.strokeStyle = SOL.green; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(leftX + 18, leftY + 23); ctx.lineTo(leftX + 42, leftY + 23); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText(`Exact AD Slope: ${dfExact.toFixed(4)}`, leftX + 48, leftY + 27);

      ctx.strokeStyle = SOL.red; ctx.lineWidth = 2.4; ctx.setLineDash([5, 3]);
      ctx.beginPath(); ctx.moveTo(leftX + 18, leftY + 46); ctx.lineTo(leftX + 42, leftY + 46); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = SOL.base02; ctx.fillText(`Fwd Diff Slope:  ${dfFwd.toFixed(4)}`, leftX + 48, leftY + 50);

      // ---------------- IEEE-754 FLOAT64 CANCELLATION INSPECTOR BOX ----------------
      const boxX = leftX, boxY = leftY + leftH + 28, boxW = leftW, boxH = 112;
      drawRoundedRect(ctx, boxX, boxY, boxW, boxH, 6, SOL.base2, SOL.base1, 1.2);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12.5px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('IEEE-754 64-bit Float Subtraction:  f(x₀ + h) − f(x₀)', boxX + 12, boxY + 20);

      const s1 = fxPlus.toFixed(16);
      const s0 = fx0.toFixed(16);
      let matchLen = 0;
      while (matchLen < Math.min(s1.length, s0.length) && s1[matchLen] === s0[matchLen]) {
        matchLen++;
      }

      ctx.font = '13px monospace';
      const labelW = 92;
      ctx.fillStyle = SOL.base01;
      ctx.fillText('f(x₀ + h) =', boxX + 12, boxY + 44);
      ctx.fillText('f(x₀)     =', boxX + 12, boxY + 66);
      ctx.fillText('Δf / h    =', boxX + 12, boxY + 92);

      // Draw matching prefix in green (cancels out!), differing suffix in red/orange
      const prefix1 = s1.slice(0, matchLen);
      const suffix1 = s1.slice(matchLen);
      const prefix0 = s0.slice(0, matchLen);
      const suffix0 = s0.slice(matchLen);

      const numX = boxX + 12 + labelW;
      ctx.fillStyle = SOL.green;
      ctx.fillText(prefix1, numX, boxY + 44);
      ctx.fillText(prefix0, numX, boxY + 66);

      const prefPx = ctx.measureText(prefix1).width;
      ctx.fillStyle = SOL.red;
      ctx.fillText(suffix1, numX + prefPx, boxY + 44);
      ctx.fillText(suffix0, numX + prefPx, boxY + 66);

      const canceledDigits = Math.max(0, prefix1.replace('.', '').replace('-', '').length);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 12.5px monospace';
      ctx.fillText(
        `${dfFwd.toFixed(6)}   (${canceledDigits}/16 digits canceled!)`,
        numX,
        boxY + 92
      );

      // ---------------- RIGHT SUBPLOT: Log-Log Error vs Step Size h ----------------
      drawRoundedRect(ctx, rightX, rightY, rightW, rightH, 4, '#ffffff', SOL.base1, 1);

      const logHMin = -16.0, logHMax = -1.0;
      const logEMin = -16.5, logEMax = 2.5;

      function toRX(lh) { return rightX + ((lh - logHMin) / (logHMax - logHMin)) * rightW; }
      function toRY(le) {
        const clamped = Math.max(logEMin, Math.min(logEMax, le));
        return rightY + rightH - ((clamped - logEMin) / (logEMax - logEMin)) * rightH;
      }

      // Grid lines
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.22)';
      ctx.lineWidth = 1;
      for (let lh = -15; lh <= -1; lh += 2) {
        const sx = toRX(lh);
        ctx.beginPath(); ctx.moveTo(sx, rightY); ctx.lineTo(sx, rightY + rightH); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11.5px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`10^${lh}`, sx, rightY + rightH + 16);
      }
      for (let le = -16; le <= 2; le += 3) {
        const sy = toRY(le);
        ctx.beginPath(); ctx.moveTo(rightX, sy); ctx.lineTo(rightX + rightW, sy); ctx.stroke();
        ctx.fillStyle = SOL.base01;
        ctx.font = '11.5px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`10^${le}`, rightX - 6, sy + 4);
      }

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Absolute Derivative Error |Dₕf(x₀) − f′(x₀)| vs. Step Size h', rightX + rightW / 2, rightY - 7);
      ctx.fillText('Step Size h (log scale)', rightX + rightW / 2, rightY + rightH + 34);

      ctx.save();
      ctx.translate(rightX - 48, rightY + rightH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Absolute Error (log scale)', 0, 0);
      ctx.restore();

      // Shade regimes
      ctx.fillStyle = 'rgba(220, 50, 47, 0.05)';
      ctx.fillRect(rightX, rightY, toRX(-8.0) - rightX, rightH);
      ctx.fillStyle = 'rgba(203, 75, 22, 0.05)';
      ctx.fillRect(toRX(-8.0), rightY, rightX + rightW - toRX(-8.0), rightH);

      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.red;
      ctx.textAlign = 'center';
      ctx.fillText('Round-off / Cancellation O(ε/h)', rightX + 125, rightY + 22);
      ctx.fillStyle = SOL.orange;
      ctx.fillText('Truncation Error O(h) / O(h²)', rightX + rightW - 115, rightY + 22);

      // Clip to right plot
      ctx.save();
      ctx.beginPath();
      ctx.rect(rightX, rightY, rightW, rightH);
      ctx.clip();

      // Exact AD baseline at machine precision (~10^-16)
      const adY = toRY(-15.8);
      ctx.strokeStyle = SOL.green;
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(rightX, adY);
      ctx.lineTo(rightX + rightW, adY);
      ctx.stroke();

      // Forward difference curve (Red)
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      const Nsteps = 180;
      for (let i = 0; i <= Nsteps; i++) {
        const lh = logHMin + (i / Nsteps) * (logHMax - logHMin);
        const hi = Math.pow(10, lh);
        const dfi = (evalPrimal(x0 + hi) - fx0) / hi;
        const ei = Math.max(1e-17, Math.abs(dfi - dfExact));
        const sx = toRX(lh), sy = toRY(Math.log10(ei));
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();

      // Central difference curve (Blue dashed)
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i <= Nsteps; i++) {
        const lh = logHMin + (i / Nsteps) * (logHMax - logHMin);
        const hi = Math.pow(10, lh);
        const dfi = (evalPrimal(x0 + hi) - evalPrimal(x0 - hi)) / (2.0 * hi);
        const ei = Math.max(1e-17, Math.abs(dfi - dfExact));
        const sx = toRX(lh), sy = toRY(Math.log10(ei));
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Current h vertical marker line
      const curX = toRX(logH);
      ctx.strokeStyle = SOL.base02;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(curX, rightY);
      ctx.lineTo(curX, rightY + rightH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Dots on curves at current h
      const curFwdY = toRY(Math.log10(errFwd));
      const curCenY = toRY(Math.log10(errCen));
      ctx.fillStyle = SOL.red;
      ctx.beginPath(); ctx.arc(curX, curFwdY, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = SOL.blue;
      ctx.beginPath(); ctx.arc(curX, curCenY, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = SOL.green;
      ctx.beginPath(); ctx.arc(curX, adY, 6, 0, 2 * Math.PI); ctx.fill();

      ctx.restore();

      // Legend inside right plot (bottom left)
      const rLegX = rightX + 14, rLegY = rightY + rightH - 82;
      drawRoundedRect(ctx, rLegX, rLegY, 275, 70, 4, 'rgba(253,246,227,0.94)', SOL.base1, 1);
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';

      ctx.strokeStyle = SOL.red; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(rLegX + 10, rLegY + 16); ctx.lineTo(rLegX + 34, rLegY + 16); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText('Forward Diff: (f(x+h)−f(x))/h', rLegX + 42, rLegY + 20);

      ctx.strokeStyle = SOL.blue; ctx.lineWidth = 2.2; ctx.setLineDash([5, 3]);
      ctx.beginPath(); ctx.moveTo(rLegX + 10, rLegY + 36); ctx.lineTo(rLegX + 34, rLegY + 36); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = SOL.base02; ctx.fillText('Central Diff: (f(x+h)−f(x−h))/(2h)', rLegX + 42, rLegY + 40);

      ctx.strokeStyle = SOL.green; ctx.lineWidth = 2.8;
      ctx.beginPath(); ctx.moveTo(rLegX + 10, rLegY + 56); ctx.lineTo(rLegX + 34, rLegY + 56); ctx.stroke();
      ctx.fillStyle = SOL.green; ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Automatic Differentiation (Exact!)', rLegX + 42, rLegY + 60);

      // Update DOM readouts
      if (hValEl) hValEl.textContent = `10^(${logH.toFixed(1)})`;
      if (x0ValEl) x0ValEl.textContent = x0.toFixed(2);
      if (exactValEl) exactValEl.textContent = dfExact.toFixed(6);
      if (fwdErrEl) fwdErrEl.textContent = errFwd.toExponential(2);
      if (cenErrEl) cenErrEl.textContent = errCen.toExponential(2);

      if (statusBanner) {
        if (logH <= -12.5) {
          statusBanner.style.color = SOL.red;
          statusBanner.innerHTML = `⚠️ <b>Catastrophic Cancellation (h = 10<sup>${logH.toFixed(1)}</sup>):</b> ${canceledDigits}/16 float digits cancel! Finite-difference error explodes to <b>${errFwd.toExponential(2)}</b> while AD stays exact.`;
        } else if (logH >= -4.0) {
          statusBanner.style.color = SOL.orange;
          statusBanner.innerHTML = `📐 <b>Truncation Error Regime (h = 10<sup>${logH.toFixed(1)}</sup>):</b> Step size h is too coarse; higher-order Taylor terms O(h) skew the secant slope (error = <b>${errFwd.toExponential(2)}</b>).`;
        } else {
          statusBanner.style.color = SOL.green;
          statusBanner.innerHTML = `✓ <b>Finite-Difference Compromise (h = 10<sup>${logH.toFixed(1)}</sup>):</b> Best possible finite-diff accuracy (~10<sup>−8</sup> forward, ~10<sup>−11</sup> central), still 10<sup>5</sup>× worse than <b>Exact AD (~10<sup>−16</sup>)</b>!`;
        }
      }
    }

    if (hSlider) {
      hSlider.addEventListener('input', () => {
        logH = parseFloat(hSlider.value);
        draw();
      });
    }
    if (x0Slider) {
      x0Slider.addEventListener('input', () => {
        x0 = parseFloat(x0Slider.value);
        draw();
      });
    }
    if (selFn) {
      selFn.addEventListener('change', () => {
        fnType = selFn.value;
        draw();
      });
    }
    if (btnTrunc) {
      btnTrunc.addEventListener('click', () => {
        logH = -1.8;
        if (hSlider) hSlider.value = logH;
        draw();
      });
    }
    if (btnSweet) {
      btnSweet.addEventListener('click', () => {
        logH = -8.0;
        if (hSlider) hSlider.value = logH;
        draw();
      });
    }
    if (btnCancel) {
      btnCancel.addEventListener('click', () => {
        logH = -15.2;
        if (hSlider) hSlider.value = logH;
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // =========================================================================
  // 2. DEMO 2: FORWARD-MODE AD & DUAL NUMBERS GRAPH TRACER
  // =========================================================================
  function initForwardADDemo() {
    const canvas = document.getElementById('ad-forward-canvas');
    if (!canvas) return;

    let x1 = 2.0;
    let x2 = 5.0;
    let dx1 = 1.0;
    let dx2 = 0.0;
    let curStep = 7; // 0..7 (default shows complete trace like forwardAD_example.svg)
    let autoTimer = null;

    const btnReset = document.getElementById('fwd-btn-reset');
    const btnStep = document.getElementById('fwd-btn-step');
    const btnAll = document.getElementById('fwd-btn-all');
    const btnAuto = document.getElementById('fwd-btn-auto');
    const btnSeed1 = document.getElementById('fwd-btn-seed1');
    const btnSeed2 = document.getElementById('fwd-btn-seed2');
    const btnSeedDir = document.getElementById('fwd-btn-seeddir');
    const x1Slider = document.getElementById('fwd-x1-slider');
    const x2Slider = document.getElementById('fwd-x2-slider');
    const x1ValEl = document.getElementById('fwd-x1-val');
    const x2ValEl = document.getElementById('fwd-x2-val');
    const statusBanner = document.getElementById('fwd-status-banner');

    function stopAuto() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
      }
      if (btnAuto) btnAuto.textContent = '▶ Auto-Trace';
    }

    function computeTrace() {
      const v_m1 = x1, dv_m1 = dx1;
      const v_0 = x2, dv_0 = dx2;
      const v_1 = Math.log(v_m1), dv_1 = dv_m1 / v_m1;
      const v_2 = v_m1 * v_0, dv_2 = dv_m1 * v_0 + v_m1 * dv_0;
      const v_3 = Math.sin(v_0), dv_3 = dv_0 * Math.cos(v_0);
      const v_4 = v_1 + v_2, dv_4 = dv_1 + dv_2;
      const v_5 = v_4 - v_3, dv_5 = dv_4 - dv_3;

      return [
        {
          id: 'v_{-1}', label: 'v₋₁', op: 'x₁',
          primalFormula: 'v₋₁ = x₁',
          primalEval: `= ${v_m1.toFixed(2)}`,
          tangentFormula: 'v̇₋₁ = ẋ₁',
          tangentEval: `= ${dv_m1.toFixed(2)}`,
          dualRule: `Seed input x₁ + ẋ₁ε = ${v_m1.toFixed(2)} + ${dv_m1.toFixed(2)}ε`,
          v: v_m1, dv: dv_m1, x: 130, y: 120
        },
        {
          id: 'v_0', label: 'v₀', op: 'x₂',
          primalFormula: 'v₀  = x₂',
          primalEval: `= ${v_0.toFixed(2)}`,
          tangentFormula: 'v̇₀  = ẋ₂',
          tangentEval: `= ${dv_0.toFixed(2)}`,
          dualRule: `Seed input x₂ + ẋ₂ε = ${v_0.toFixed(2)} + ${dv_0.toFixed(2)}ε`,
          v: v_0, dv: dv_0, x: 130, y: 345
        },
        {
          id: 'v_1', label: 'v₁', op: 'ln(v₋₁)',
          primalFormula: 'v₁  = ln v₋₁',
          primalEval: `= ${v_1.toFixed(3)}`,
          tangentFormula: 'v̇₁  = v̇₋₁ / v₋₁',
          tangentEval: `= ${dv_1.toFixed(3)}`,
          dualRule: `ln(v₋₁ + v̇₋₁ε) = ln(v₋₁) + (v̇₋₁/v₋₁)ε = ${v_1.toFixed(3)} + ${dv_1.toFixed(3)}ε`,
          v: v_1, dv: dv_1, x: 280, y: 120
        },
        {
          id: 'v_2', label: 'v₂', op: 'v₋₁ · v₀',
          primalFormula: 'v₂  = v₋₁ × v₀',
          primalEval: `= ${v_2.toFixed(3)}`,
          tangentFormula: 'v̇₂  = v̇₋₁v₀ + v₋₁v̇₀',
          tangentEval: `= ${dv_2.toFixed(3)}`,
          dualRule: `(v₋₁+v̇₋₁ε)(v₀+v̇₀ε) = v₋₁v₀ + (v̇₋₁v₀ + v₋₁v̇₀)ε = ${v_2.toFixed(3)} + ${dv_2.toFixed(3)}ε`,
          v: v_2, dv: dv_2, x: 280, y: 235
        },
        {
          id: 'v_3', label: 'v₃', op: 'sin(v₀)',
          primalFormula: 'v₃  = sin v₀',
          primalEval: `= ${v_3.toFixed(3)}`,
          tangentFormula: 'v̇₃  = v̇₀ cos v₀',
          tangentEval: `= ${dv_3.toFixed(3)}`,
          dualRule: `sin(v₀ + v̇₀ε) = sin(v₀) + (v̇₀ cos v₀)ε = ${v_3.toFixed(3)} + ${dv_3.toFixed(3)}ε`,
          v: v_3, dv: dv_3, x: 375, y: 345
        },
        {
          id: 'v_4', label: 'v₄', op: 'v₁ + v₂',
          primalFormula: 'v₄  = v₁ + v₂',
          primalEval: `= ${v_4.toFixed(3)}`,
          tangentFormula: 'v̇₄  = v̇₁ + v̇₂',
          tangentEval: `= ${dv_4.toFixed(3)}`,
          dualRule: `(v₁+v̇₁ε) + (v₂+v̇₂ε) = (v₁+v₂) + (v̇₁+v̇₂)ε = ${v_4.toFixed(3)} + ${dv_4.toFixed(3)}ε`,
          v: v_4, dv: dv_4, x: 430, y: 120
        },
        {
          id: 'v_5', label: 'v₅ = y', op: 'v₄ − v₃',
          primalFormula: 'y = v₅ = v₄ − v₃',
          primalEval: `= ${v_5.toFixed(3)}`,
          tangentFormula: 'ẏ = v̇₅ = v̇₄ − v̇₃',
          tangentEval: `= ${dv_5.toFixed(3)}`,
          dualRule: `Output y + ẏε = (v₄−v₃) + (v̇₄−v̇₃)ε = ${v_5.toFixed(3)} + ${dv_5.toFixed(3)}ε`,
          v: v_5, dv: dv_5, x: 545, y: 235
        }
      ];
    }

    const edges = [
      { from: 0, to: 2 }, // v_-1 -> v_1
      { from: 0, to: 3 }, // v_-1 -> v_2
      { from: 1, to: 3 }, // v_0 -> v_2
      { from: 1, to: 4 }, // v_0 -> v_3
      { from: 2, to: 5 }, // v_1 -> v_4
      { from: 3, to: 5 }, // v_2 -> v_4
      { from: 5, to: 6 }, // v_4 -> v_5
      { from: 4, to: 6 }  // v_3 -> v_5
    ];

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const nodes = computeTrace();
      const nodeR = 30;

      // Left DAG panel background
      drawRoundedRect(ctx, 14, 14, 608, H - 28, 6, '#ffffff', SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Computation Graph:  y = f(x₁, x₂) = ln(x₁) + x₁x₂ − sin(x₂)', 318, 36);

      // Input arrows into v_-1 and v_0
      drawArrow(ctx, 48, nodes[0].y, nodes[0].x - nodeR - 2, nodes[0].y, SOL.base01, 2, 8);
      ctx.fillStyle = SOL.base02; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'right';
      ctx.fillText('x₁', 42, nodes[0].y + 5);

      drawArrow(ctx, 48, nodes[1].y, nodes[1].x - nodeR - 2, nodes[1].y, SOL.base01, 2, 8);
      ctx.fillText('x₂', 42, nodes[1].y + 5);

      // Output arrow from v_5
      drawArrow(ctx, nodes[6].x + nodeR, nodes[6].y, 605, nodes[6].y, SOL.base01, 2, 8);

      // Draw DAG edges
      for (const e of edges) {
        const n1 = nodes[e.from];
        const n2 = nodes[e.to];
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.hypot(dx, dy);
        const sx = n1.x + (dx / dist) * nodeR;
        const sy = n1.y + (dy / dist) * nodeR;
        const ex = n2.x - (dx / dist) * (nodeR + 3);
        const ey = n2.y - (dy / dist) * (nodeR + 3);

        const isActiveEdge = (curStep - 1 === e.to);
        const isDoneEdge = (curStep - 1 >= e.to);
        const col = isActiveEdge ? SOL.orange : (isDoneEdge ? SOL.blue : SOL.base1);
        const lw = isActiveEdge ? 3.2 : (isDoneEdge ? 2.2 : 1.5);
        drawArrow(ctx, sx, sy, ex, ey, col, lw, 8);
      }

      // Draw DAG nodes
      for (let i = 0; i < nodes.length; i++) {
        const nd = nodes[i];
        const isDone = i < curStep;
        const isCurrent = (i === curStep - 1);

        ctx.beginPath();
        ctx.arc(nd.x, nd.y, nodeR, 0, 2 * Math.PI);
        ctx.fillStyle = isCurrent ? '#fff3e0' : (isDone ? '#eef7fc' : SOL.base3);
        ctx.fill();
        ctx.lineWidth = isCurrent ? 3.5 : (isDone ? 2.4 : 1.5);
        ctx.strokeStyle = isCurrent ? SOL.orange : (isDone ? SOL.blue : SOL.base1);
        ctx.stroke();

        // Node name & op inside circle
        ctx.fillStyle = SOL.base03;
        ctx.font = 'bold 13.5px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(nd.label, nd.x, nd.y - 4);
        ctx.fillStyle = SOL.base01;
        ctx.font = '11px sans-serif';
        ctx.fillText(nd.op, nd.x, nd.y + 12);

        // Dual number badge above/below node when evaluated
        if (isDone) {
          const badgeY = (nd.y < 200) ? (nd.y - nodeR - 36) : (nd.y + nodeR + 8);
          const badgeW = 132, badgeH = 30;
          drawRoundedRect(
            ctx,
            nd.x - badgeW / 2,
            badgeY,
            badgeW,
            badgeH,
            5,
            isCurrent ? '#fdf6e3' : 'rgba(253,246,227,0.95)',
            isCurrent ? SOL.orange : SOL.base1,
            isCurrent ? 1.8 : 1
          );
          ctx.font = 'bold 11.5px monospace';
          ctx.textAlign = 'center';
          ctx.fillStyle = SOL.blue;
          const vStr = `${nd.v.toFixed(2)}`;
          const dvStr = `${nd.dv >= 0 ? '+' : ''}${nd.dv.toFixed(2)}ε`;
          ctx.fillText(`${vStr} ${dvStr}`, nd.x, badgeY + 19);
        }
      }

      // ---------------- RIGHT PANEL: SYNCHRONIZED PRIMAL & TANGENT TRACE TABLES ----------------
      const tblX = 636, tblY = 14, tblW = W - tblX - 14, tblH = H - 28;
      drawRoundedRect(ctx, tblX, tblY, tblW, tblH, 6, '#ffffff', SOL.base1, 1);

      const colW = (tblW - 24) / 2;
      const pX = tblX + 8;
      const tX = tblX + 16 + colW;

      // Headers matching forwardAD_example.svg
      drawRoundedRect(ctx, pX, tblY + 10, colW, 32, 4, SOL.base2, SOL.base1, 1);
      drawRoundedRect(ctx, tX, tblY + 10, colW, 32, 4, '#e6ece8', SOL.base1, 1);

      ctx.font = 'bold 13px sans-serif';
      ctx.fillStyle = SOL.blue;
      ctx.textAlign = 'center';
      ctx.fillText('Forward Primal Trace (vᵢ)', pX + colW / 2, tblY + 31);
      ctx.fillStyle = SOL.orange;
      ctx.fillText('Forward Tangent Trace (v̇ᵢ)', tX + colW / 2, tblY + 31);

      const rowStart = tblY + 52;
      const rowH = 44;

      for (let i = 0; i < nodes.length; i++) {
        const nd = nodes[i];
        const ry = rowStart + i * rowH;
        const isDone = i < curStep;
        const isCurrent = (i === curStep - 1);

        if (isCurrent) {
          drawRoundedRect(ctx, pX, ry, colW, rowH - 4, 4, 'rgba(203, 75, 22, 0.12)', SOL.orange, 1.5);
          drawRoundedRect(ctx, tX, ry, colW, rowH - 4, 4, 'rgba(203, 75, 22, 0.12)', SOL.orange, 1.5);
        } else if (i === 6 && isDone) {
          drawRoundedRect(ctx, pX, ry, colW, rowH - 4, 4, 'rgba(133, 153, 0, 0.12)', SOL.green, 1.5);
          drawRoundedRect(ctx, tX, ry, colW, rowH - 4, 4, 'rgba(133, 153, 0, 0.12)', SOL.green, 1.5);
        }

        // Divider lines after inputs (i=1) and before output (i=5)
        if (i === 2 || i === 6) {
          ctx.strokeStyle = SOL.base1;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(pX + 4, ry - 2); ctx.lineTo(pX + colW - 4, ry - 2);
          ctx.moveTo(tX + 4, ry - 2); ctx.lineTo(tX + colW - 4, ry - 2);
          ctx.stroke();
        }

        ctx.font = (i === 6) ? 'bold 12.5px monospace' : '12.5px monospace';
        ctx.textAlign = 'left';

        // Primal column
        ctx.fillStyle = isDone ? SOL.base03 : SOL.base1;
        ctx.fillText(nd.primalFormula, pX + 10, ry + 24);
        ctx.textAlign = 'right';
        ctx.fillStyle = isDone ? SOL.blue : SOL.base1;
        ctx.fillText(isDone ? nd.primalEval : '= ...', pX + colW - 10, ry + 24);

        // Tangent column
        ctx.textAlign = 'left';
        ctx.fillStyle = isDone ? SOL.base03 : SOL.base1;
        ctx.fillText(nd.tangentFormula, tX + 10, ry + 24);
        ctx.textAlign = 'right';
        ctx.fillStyle = isDone ? (i === 6 ? SOL.green : SOL.orange) : SOL.base1;
        ctx.fillText(isDone ? nd.tangentEval : '= ...', tX + colW - 10, ry + 24);
      }

      // Bottom summary inside right panel
      const sumY = rowStart + 7 * rowH + 6;
      drawRoundedRect(ctx, pX, sumY, tblW - 16, 48, 5, SOL.base3, SOL.base1, 1);
      ctx.font = 'bold 12.5px sans-serif';
      ctx.fillStyle = SOL.base02;
      ctx.textAlign = 'center';
      const seedDesc = (dx1 === 1 && dx2 === 0)
        ? 'Seed ẋ = (1, 0)  ⇒  ẏ = ∂y/∂x₁'
        : (dx1 === 0 && dx2 === 1)
          ? 'Seed ẋ = (0, 1)  ⇒  ẏ = ∂y/∂x₂'
          : `Seed ẋ = (${dx1.toFixed(1)}, ${dx2.toFixed(1)})  ⇒  Directional JVP J·ẋ`;
      ctx.fillText(
        `${seedDesc}:   y = ${nodes[6].v.toFixed(3)},   ẏ = ${nodes[6].dv.toFixed(3)}`,
        tblX + tblW / 2,
        sumY + 29
      );

      // Update DOM readouts
      if (x1ValEl) x1ValEl.textContent = x1.toFixed(2);
      if (x2ValEl) x2ValEl.textContent = x2.toFixed(2);
      if (statusBanner) {
        if (curStep === 0) {
          statusBanner.style.color = SOL.base02;
          statusBanner.innerHTML = `Click <b>Step Forward</b> to propagate dual numbers <b>vᵢ + v̇ᵢε</b> (where ε² = 0) through the Wengert list.`;
        } else {
          const active = nodes[curStep - 1];
          statusBanner.style.color = (curStep === 7) ? SOL.green : SOL.orange;
          statusBanner.innerHTML = `<b>Step ${curStep}/7 (${active.label}):</b> &nbsp; <span style="font-family: monospace;">${active.dualRule}</span>`;
        }
      }
    }

    if (btnReset) {
      btnReset.addEventListener('click', () => {
        stopAuto();
        curStep = 0;
        draw();
      });
    }
    if (btnStep) {
      btnStep.addEventListener('click', () => {
        stopAuto();
        curStep = (curStep < 7) ? curStep + 1 : 1;
        draw();
      });
    }
    if (btnAll) {
      btnAll.addEventListener('click', () => {
        stopAuto();
        curStep = 7;
        draw();
      });
    }
    if (btnAuto) {
      btnAuto.addEventListener('click', () => {
        if (autoTimer) {
          stopAuto();
          return;
        }
        if (curStep >= 7) curStep = 0;
        btnAuto.textContent = '⏸ Pause';
        autoTimer = setInterval(() => {
          if (curStep < 7) {
            curStep++;
            draw();
          } else {
            stopAuto();
          }
        }, 700);
      });
    }
    if (btnSeed1) {
      btnSeed1.addEventListener('click', () => {
        dx1 = 1.0; dx2 = 0.0;
        if (curStep === 0) curStep = 7;
        draw();
      });
    }
    if (btnSeed2) {
      btnSeed2.addEventListener('click', () => {
        dx1 = 0.0; dx2 = 1.0;
        if (curStep === 0) curStep = 7;
        draw();
      });
    }
    if (btnSeedDir) {
      btnSeedDir.addEventListener('click', () => {
        dx1 = 0.6; dx2 = 0.8;
        if (curStep === 0) curStep = 7;
        draw();
      });
    }
    if (x1Slider) {
      x1Slider.addEventListener('input', () => {
        x1 = parseFloat(x1Slider.value);
        draw();
      });
    }
    if (x2Slider) {
      x2Slider.addEventListener('input', () => {
        x2 = parseFloat(x2Slider.value);
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // =========================================================================
  // 3. DEMO 3: REVERSE-MODE AD (FORWARD TAPE & BACKWARD ADJOINT TRACER)
  // =========================================================================
  function initReverseADDemo() {
    const canvas = document.getElementById('ad-reverse-canvas');
    if (!canvas) return;

    let graphMode = 'dag2d'; // 'dag2d' (y = ln(x1) + x1*x2 - sin(x2)) | 'chain1d' (y = ln(sin(x^2)))
    let x1 = 2.0;
    let x2 = 5.0;
    let stage = 14; // for dag2d: 0..7 forward, 8..14 reverse; for chain1d: 0..4 forward, 5..8 reverse
    let autoTimer = null;

    const selGraph = document.getElementById('rev-sel-graph');
    const btnReset = document.getElementById('rev-btn-reset');
    const btnStepFwd = document.getElementById('rev-btn-fwd');
    const btnStepRev = document.getElementById('rev-btn-rev');
    const btnAuto = document.getElementById('rev-btn-auto');
    const x1Slider = document.getElementById('rev-x1-slider');
    const x2Slider = document.getElementById('rev-x2-slider');
    const x1ValEl = document.getElementById('rev-x1-val');
    const x2ValEl = document.getElementById('rev-x2-val');
    const statusBanner = document.getElementById('rev-status-banner');

    function getMaxStage() {
      return graphMode === 'dag2d' ? 14 : 8;
    }
    function getFwdStages() {
      return graphMode === 'dag2d' ? 7 : 4;
    }

    function stopAuto() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
      }
      if (btnAuto) btnAuto.textContent = '▶ Auto-Play Both Passes';
    }

    function buildDAG2D() {
      const v_m1 = x1;
      const v_0 = x2;
      const v_1 = Math.log(v_m1);
      const v_2 = v_m1 * v_0;
      const v_3 = Math.sin(v_0);
      const v_4 = v_1 + v_2;
      const v_5 = v_4 - v_3;

      // Adjoints
      const bv_5 = 1.0;
      const bv_4 = bv_5 * 1.0;
      const bv_3 = bv_5 * (-1.0);
      const bv_2 = bv_4 * 1.0;
      const bv_1 = bv_4 * 1.0;
      const bv_0_from3 = bv_3 * Math.cos(v_0);
      const bv_0_from2 = bv_2 * v_m1;
      const bv_0 = bv_0_from2 + bv_0_from3;
      const bv_m1_from2 = bv_2 * v_0;
      const bv_m1_from1 = bv_1 * (1.0 / v_m1);
      const bv_m1 = bv_m1_from1 + bv_m1_from2;

      const nodes = [
        { label: 'v₋₁', op: 'x₁', v: v_m1, bv: bv_m1, x: 115, y: 108, fwdIdx: 1, revIdx: 14,
          fwdText: `v₋₁ = x₁ = ${v_m1.toFixed(2)}`,
          revText: `v̄₋₁ = v̄₁(1/v₋₁) + v̄₂(v₀) = ${bv_m1_from1.toFixed(2)} + ${bv_m1_from2.toFixed(2)} = ${bv_m1.toFixed(3)}  [FAN-OUT SUM!]` },
        { label: 'v₀', op: 'x₂', v: v_0, bv: bv_0, x: 115, y: 342, fwdIdx: 2, revIdx: 13,
          fwdText: `v₀ = x₂ = ${v_0.toFixed(2)}`,
          revText: `v̄₀ = v̄₂(v₋₁) + v̄₃(cos v₀) = ${bv_0_from2.toFixed(2)} + (${bv_0_from3.toFixed(3)}) = ${bv_0.toFixed(3)}  [FAN-OUT SUM!]` },
        { label: 'v₁', op: 'ln v₋₁', v: v_1, bv: bv_1, x: 270, y: 108, fwdIdx: 3, revIdx: 12,
          fwdText: `v₁ = ln(v₋₁) = ${v_1.toFixed(3)}`,
          revText: `v̄₁ = v̄₄ · (∂v₄/∂v₁) = ${bv_4.toFixed(1)} × 1 = ${bv_1.toFixed(3)}` },
        { label: 'v₂', op: 'v₋₁·v₀', v: v_2, bv: bv_2, x: 270, y: 242, fwdIdx: 4, revIdx: 11,
          fwdText: `v₂ = v₋₁ × v₀ = ${v_2.toFixed(3)}`,
          revText: `v̄₂ = v̄₄ · (∂v₄/∂v₂) = ${bv_4.toFixed(1)} × 1 = ${bv_2.toFixed(3)}` },
        { label: 'v₃', op: 'sin v₀', v: v_3, bv: bv_3, x: 380, y: 342, fwdIdx: 5, revIdx: 10,
          fwdText: `v₃ = sin(v₀) = ${v_3.toFixed(3)}`,
          revText: `v̄₃ = v̄₅ · (∂v₅/∂v₃) = 1 × (−1) = ${bv_3.toFixed(3)}` },
        { label: 'v₄', op: 'v₁ + v₂', v: v_4, bv: bv_4, x: 425, y: 108, fwdIdx: 6, revIdx: 9,
          fwdText: `v₄ = v₁ + v₂ = ${v_4.toFixed(3)}`,
          revText: `v̄₄ = v̄₅ · (∂v₅/∂v₄) = 1 × 1 = ${bv_4.toFixed(3)}` },
        { label: 'v₅ = y', op: 'v₄ − v₃', v: v_5, bv: bv_5, x: 545, y: 228, fwdIdx: 7, revIdx: 8,
          fwdText: `y = v₅ = v₄ − v₃ = ${v_5.toFixed(3)}`,
          revText: `Seed output adjoint:  ȳ = v̄₅ = ∂y/∂y = 1.000` }
      ];

      const edges = [
        { from: 0, to: 2, localDeriv: `1/v₋₁ = ${(1 / v_m1).toFixed(2)}` },
        { from: 0, to: 3, localDeriv: `v₀ = ${v_0.toFixed(1)}` },
        { from: 1, to: 3, localDeriv: `v₋₁ = ${v_m1.toFixed(1)}` },
        { from: 1, to: 4, localDeriv: `cos v₀ = ${Math.cos(v_0).toFixed(2)}` },
        { from: 2, to: 5, localDeriv: '+1' },
        { from: 3, to: 5, localDeriv: '+1' },
        { from: 5, to: 6, localDeriv: '+1' },
        { from: 4, to: 6, localDeriv: '−1' }
      ];

      return { nodes, edges };
    }

    function buildChain1D() {
      // Ensure sin(x^2) > 0 by taking |sin(x^2)| in log if needed, default x1 = 1.2 -> x^2 = 1.44 rad -> sin(1.44) = 0.991
      const xv = Math.max(0.4, Math.min(1.7, x1));
      const z1 = xv * xv;
      const z2 = Math.max(1e-4, Math.sin(z1));
      const z3 = Math.log(z2);

      const bz3 = 1.0;
      const bz2 = bz3 * (1.0 / z2);
      const bz1 = bz2 * Math.cos(z1);
      const bx = bz1 * (2.0 * xv);

      const nodes = [
        { label: 'x', op: 'input', v: xv, bv: bx, x: 105, y: 230, fwdIdx: 1, revIdx: 8,
          fwdText: `x = ${xv.toFixed(3)}`,
          revText: `x̄ = z̄₁ · (2x) = ${bz1.toFixed(3)} × ${(2 * xv).toFixed(2)} = ${bx.toFixed(3)} = 2x cot(x²)` },
        { label: 'z₁', op: 'x²', v: z1, bv: bz1, x: 250, y: 230, fwdIdx: 2, revIdx: 7,
          fwdText: `z₁ = x² = ${z1.toFixed(3)} (saved on tape)`,
          revText: `z̄₁ = z̄₂ · cos(z₁) = ${bz2.toFixed(3)} × ${Math.cos(z1).toFixed(3)} = ${bz1.toFixed(3)}` },
        { label: 'z₂', op: 'sin(z₁)', v: z2, bv: bz2, x: 395, y: 230, fwdIdx: 3, revIdx: 6,
          fwdText: `z₂ = sin(z₁) = ${z2.toFixed(3)} (saved on tape)`,
          revText: `z̄₂ = z̄₃ · (1/z₂) = 1 / ${z2.toFixed(3)} = ${bz2.toFixed(3)}` },
        { label: 'z₃ = y', op: 'log(z₂)', v: z3, bv: bz3, x: 540, y: 230, fwdIdx: 4, revIdx: 5,
          fwdText: `y = z₃ = log(z₂) = ${z3.toFixed(3)}`,
          revText: `Seed output adjoint:  ȳ = z̄₃ = ∂y/∂z₃ = 1.000` }
      ];

      const edges = [
        { from: 0, to: 1, localDeriv: `2x = ${(2 * xv).toFixed(2)}` },
        { from: 1, to: 2, localDeriv: `cos z₁ = ${Math.cos(z1).toFixed(2)}` },
        { from: 2, to: 3, localDeriv: `1/z₂ = ${(1 / z2).toFixed(2)}` }
      ];

      return { nodes, edges };
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const { nodes, edges } = (graphMode === 'dag2d') ? buildDAG2D() : buildChain1D();
      const fwdCount = getFwdStages();
      const nodeR = 30;

      // Left DAG panel
      drawRoundedRect(ctx, 14, 14, 608, H - 28, 6, '#ffffff', SOL.base1, 1);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        graphMode === 'dag2d'
          ? 'Two-Phase Reverse AD:  y = ln(x₁) + x₁x₂ − sin(x₂)  (Fan-Out Accumulation)'
          : 'Two-Phase Reverse AD on Lecture Chain:  y = log(sin(x²))',
        318,
        36
      );

      // Legend for Forward Primal vs Reverse Adjoint inside left panel
      drawRoundedRect(ctx, 26, H - 52, 584, 30, 4, SOL.base3, SOL.base1, 1);
      ctx.font = 'bold 11.5px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillStyle = SOL.blue;
      ctx.fillText('→ Phase 1 (Forward Tape): saves vᵢ', 38, H - 33);
      ctx.fillStyle = SOL.red;
      ctx.fillText('← Phase 2 (Reverse Pass): propagates v̄ᵢ = ∂y/∂vᵢ', 298, H - 33);

      // Draw edges (Forward blue solid + Reverse red dashed when active)
      for (const e of edges) {
        const n1 = nodes[e.from];
        const n2 = nodes[e.to];
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.hypot(dx, dy);
        const sx = n1.x + (dx / dist) * nodeR;
        const sy = n1.y + (dy / dist) * nodeR;
        const ex = n2.x - (dx / dist) * (nodeR + 3);
        const ey = n2.y - (dy / dist) * (nodeR + 3);

        const fwdDone = stage >= n2.fwdIdx;
        const revDone = stage >= n1.revIdx;

        // Forward arrow slightly offset up
        const nx = -dy / dist * 4;
        const ny = dx / dist * 4;
        drawArrow(ctx, sx + nx, sy + ny, ex + nx, ey + ny, fwdDone ? SOL.blue : SOL.base1, fwdDone ? 2.2 : 1.4, 7);

        // Reverse adjoint arrow slightly offset down
        if (stage > fwdCount) {
          drawArrow(
            ctx,
            ex - nx,
            ey - ny,
            sx - nx,
            sy - ny,
            revDone ? SOL.red : 'rgba(220,50,47,0.3)',
            revDone ? 2.6 : 1.4,
            7,
            [5, 3]
          );
          // Local partial derivative label on edge (positioned in open space clear of node badges)
          const isDownDiag = dy > 30;
          const isUpDiag = dy < -30;
          const tEdge = isDownDiag ? 0.62 : (isUpDiag ? 0.48 : 0.5);
          const normShift = isDownDiag ? -2.2 : (isUpDiag ? 3.2 : -3.0);
          const mx = sx + (ex - sx) * tEdge + nx * normShift;
          const my = sy + (ey - sy) * tEdge + ny * normShift;
          ctx.font = 'bold 10.5px monospace';
          const tw = ctx.measureText(e.localDeriv).width + 6;
          drawRoundedRect(ctx, mx - tw / 2, my - 7, tw, 15, 3, 'rgba(253,246,227,0.92)', 'transparent', 0);
          ctx.fillStyle = revDone ? SOL.red : SOL.base01;
          ctx.textAlign = 'center';
          ctx.fillText(e.localDeriv, mx, my + 4);
        }
      }

      // Draw nodes
      for (const nd of nodes) {
        const hasFwd = stage >= nd.fwdIdx;
        const hasRev = stage >= nd.revIdx;
        const isCurrent = (stage === nd.fwdIdx) || (stage === nd.revIdx);

        ctx.beginPath();
        ctx.arc(nd.x, nd.y, nodeR, 0, 2 * Math.PI);
        ctx.fillStyle = isCurrent ? '#fff3e0' : (hasRev ? '#fdeeed' : (hasFwd ? '#eef7fc' : SOL.base3));
        ctx.fill();
        ctx.lineWidth = isCurrent ? 3.5 : 2.2;
        ctx.strokeStyle = isCurrent ? SOL.orange : (hasRev ? SOL.red : (hasFwd ? SOL.blue : SOL.base1));
        ctx.stroke();

        ctx.fillStyle = SOL.base03;
        ctx.font = 'bold 13.5px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(nd.label, nd.x, nd.y - 4);
        ctx.fillStyle = SOL.base01;
        ctx.font = '11px sans-serif';
        ctx.fillText(nd.op, nd.x, nd.y + 12);

        // Primal tape badge (top)
        if (hasFwd) {
          const bW = 92, bH = 22;
          const bY = nd.y - nodeR - 26;
          drawRoundedRect(ctx, nd.x - bW / 2, bY, bW, bH, 4, '#eef7fc', SOL.blue, 1.2);
          ctx.fillStyle = SOL.blue;
          ctx.font = 'bold 11px monospace';
          ctx.fillText(`v=${nd.v.toFixed(2)}`, nd.x, bY + 15);
        }
        // Adjoint badge (bottom)
        if (hasRev) {
          const bW = 104, bH = 22;
          const bY = nd.y + nodeR + 5;
          drawRoundedRect(ctx, nd.x - bW / 2, bY, bW, bH, 4, '#fdeeed', SOL.red, 1.4);
          ctx.fillStyle = SOL.red;
          ctx.font = 'bold 11px monospace';
          ctx.fillText(`v̄=${nd.bv.toFixed(3)}`, nd.x, bY + 15);
        }
      }

      // ---------------- RIGHT PANEL: SAVED TAPE + REVERSE ADJOINT ACCUMULATION TABLE ----------------
      const tblX = 636, tblY = 14, tblW = W - tblX - 14, tblH = H - 28;
      drawRoundedRect(ctx, tblX, tblY, tblW, tblH, 6, '#ffffff', SOL.base1, 1);

      const colW = (tblW - 24) / 2;
      const pX = tblX + 8;
      const rX = tblX + 16 + colW;

      drawRoundedRect(ctx, pX, tblY + 10, colW, 32, 4, '#eef7fc', SOL.blue, 1);
      drawRoundedRect(ctx, rX, tblY + 10, colW, 32, 4, '#fdeeed', SOL.red, 1);

      ctx.font = 'bold 12.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = SOL.blue;
      ctx.fillText('1. Forward Primal Tape (↓)', pX + colW / 2, tblY + 31);
      ctx.fillStyle = SOL.red;
      ctx.fillText('2. Reverse Adjoint Pass (↑)', rX + colW / 2, tblY + 31);

      const rowStart = tblY + 52;
      const rowH = graphMode === 'dag2d' ? 44 : 68;

      for (let i = 0; i < nodes.length; i++) {
        const nd = nodes[i];
        const ry = rowStart + i * rowH;
        const hasFwd = stage >= nd.fwdIdx;
        const hasRev = stage >= nd.revIdx;
        const isCurFwd = stage === nd.fwdIdx;
        const isCurRev = stage === nd.revIdx;

        drawRoundedRect(
          ctx, pX, ry, colW, rowH - 5, 4,
          isCurFwd ? '#fff3e0' : (hasFwd ? SOL.base3 : '#f8f8f8'),
          isCurFwd ? SOL.orange : SOL.base1,
          isCurFwd ? 1.8 : 0.8
        );
        drawRoundedRect(
          ctx, rX, ry, colW, rowH - 5, 4,
          isCurRev ? '#fff3e0' : (hasRev ? '#fff8f6' : '#f8f8f8'),
          isCurRev ? SOL.orange : (hasRev ? SOL.red : SOL.base1),
          isCurRev ? 1.8 : 0.8
        );

        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillStyle = hasFwd ? SOL.base03 : SOL.base1;
        ctx.fillText(`${nd.label} (${nd.op})`, pX + 8, ry + 23);
        ctx.textAlign = 'right';
        ctx.fillStyle = hasFwd ? SOL.blue : SOL.base1;
        ctx.font = 'bold 12px monospace';
        ctx.fillText(hasFwd ? nd.v.toFixed(3) : '...', pX + colW - 8, ry + 23);

        // Reverse column
        ctx.textAlign = 'left';
        ctx.font = '11.5px monospace';
        ctx.fillStyle = hasRev ? SOL.base03 : SOL.base1;
        const revShort = (graphMode === 'dag2d')
          ? (i === 0 ? 'v̄₋₁ = v̄₁/v₋₁ + v̄₂v₀' :
             i === 1 ? 'v̄₀ = v̄₂v₋₁ + v̄₃cos v₀' :
             i === 2 ? 'v̄₁ = v̄₄ · 1' :
             i === 3 ? 'v̄₂ = v̄₄ · 1' :
             i === 4 ? 'v̄₃ = v̄₅ · (−1)' :
             i === 5 ? 'v̄₄ = v̄₅ · 1' : 'ȳ = v̄₅ = 1 (seed)')
          : (i === 0 ? 'x̄ = z̄₁ · 2x' :
             i === 1 ? 'z̄₁ = z̄₂ · cos z₁' :
             i === 2 ? 'z̄₂ = z̄₃ / z₂' : 'ȳ = z̄₃ = 1 (seed)');
        ctx.fillText(revShort, rX + 8, ry + 23);
        ctx.textAlign = 'right';
        ctx.font = 'bold 12px monospace';
        ctx.fillStyle = hasRev ? SOL.red : SOL.base1;
        ctx.fillText(hasRev ? nd.bv.toFixed(3) : '...', rX + colW - 8, ry + 23);
      }

      // Bottom punchline box
      const sumY = tblY + tblH - 54;
      drawRoundedRect(ctx, pX, sumY, tblW - 16, 44, 5, SOL.base2, SOL.base1, 1);
      ctx.font = 'bold 12.5px sans-serif';
      ctx.fillStyle = SOL.base02;
      ctx.textAlign = 'center';
      if (graphMode === 'dag2d') {
        ctx.fillText(
          `1 Reverse Pass computes ALL gradients:  ∂y/∂x₁ = v̄₋₁ = ${nodes[0].bv.toFixed(3)},   ∂y/∂x₂ = v̄₀ = ${nodes[1].bv.toFixed(3)}`,
          tblX + tblW / 2,
          sumY + 27
        );
      } else {
        ctx.fillText(
          `Exact Adjoint Gradient at x = ${nodes[0].v.toFixed(2)}:   dy/dx = x̄ = 2x cot(x²) = ${nodes[0].bv.toFixed(3)}`,
          tblX + tblW / 2,
          sumY + 27
        );
      }

      if (x1ValEl) x1ValEl.textContent = x1.toFixed(2);
      if (x2ValEl) x2ValEl.textContent = x2.toFixed(2);

      if (statusBanner) {
        if (stage === 0) {
          statusBanner.style.color = SOL.base02;
          statusBanner.innerHTML = `Click <b>Step Forward Tape</b> to evaluate primal values vᵢ, then <b>Step Reverse Adjoint</b> to propagate v̄ᵢ = ∂y/∂vᵢ backward!`;
        } else if (stage <= fwdCount) {
          const active = nodes.find(n => n.fwdIdx === stage);
          statusBanner.style.color = SOL.blue;
          statusBanner.innerHTML = `<b>Forward Tape (${stage}/${fwdCount}):</b> &nbsp; <span style="font-family: monospace;">${active ? active.fwdText : ''}</span>`;
        } else {
          const active = nodes.find(n => n.revIdx === stage);
          statusBanner.style.color = SOL.red;
          statusBanner.innerHTML = `<b>Reverse Adjoint (${stage - fwdCount}/${fwdCount}):</b> &nbsp; <span style="font-family: monospace;">${active ? active.revText : ''}</span>`;
        }
      }
    }

    if (selGraph) {
      selGraph.addEventListener('change', () => {
        stopAuto();
        graphMode = selGraph.value;
        if (graphMode === 'chain1d' && x1 > 1.65) {
          x1 = 1.2;
          if (x1Slider) x1Slider.value = x1;
        }
        stage = getMaxStage();
        draw();
      });
    }
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        stopAuto();
        stage = 0;
        draw();
      });
    }
    if (btnStepFwd) {
      btnStepFwd.addEventListener('click', () => {
        stopAuto();
        const fwdMax = getFwdStages();
        stage = (stage < fwdMax) ? stage + 1 : fwdMax;
        draw();
      });
    }
    if (btnStepRev) {
      btnStepRev.addEventListener('click', () => {
        stopAuto();
        const fwdMax = getFwdStages();
        const maxS = getMaxStage();
        if (stage < fwdMax) stage = fwdMax + 1;
        else if (stage < maxS) stage++;
        else stage = fwdMax + 1;
        draw();
      });
    }
    if (btnAuto) {
      btnAuto.addEventListener('click', () => {
        if (autoTimer) {
          stopAuto();
          return;
        }
        const maxS = getMaxStage();
        if (stage >= maxS) stage = 0;
        btnAuto.textContent = '⏸ Pause';
        autoTimer = setInterval(() => {
          if (stage < maxS) {
            stage++;
            draw();
          } else {
            stopAuto();
          }
        }, 650);
      });
    }
    if (x1Slider) {
      x1Slider.addEventListener('input', () => {
        x1 = parseFloat(x1Slider.value);
        draw();
      });
    }
    if (x2Slider) {
      x2Slider.addEventListener('input', () => {
        x2 = parseFloat(x2Slider.value);
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // =========================================================================
  // 4. DEMO 4: JVP VS. VJP COMPLEXITY & SPEELPENNING PRODUCT BENCHMARK
  // =========================================================================
  function initComplexityDemo() {
    const canvas = document.getElementById('ad-complexity-canvas');
    if (!canvas) return;

    let nDim = 120; // inputs n
    let mDim = 1;   // outputs m

    const nSlider = document.getElementById('cmp-n-slider');
    const mSlider = document.getElementById('cmp-m-slider');
    const btnML = document.getElementById('cmp-btn-ml');
    const btnSquare = document.getElementById('cmp-btn-square');
    const btnCurve = document.getElementById('cmp-btn-curve');
    const nValEl = document.getElementById('cmp-n-val');
    const mValEl = document.getElementById('cmp-m-val');
    const fwdCostEl = document.getElementById('cmp-fwd-cost');
    const revCostEl = document.getElementById('cmp-rev-cost');
    const speedupEl = document.getElementById('cmp-speedup');
    const statusBanner = document.getElementById('cmp-status-banner');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      // Left Panel: Jacobian Matrix M x N Structure (Column-by-Column JVP vs Row-by-Row VJP) + Method Comparison Bars
      const leftX = 20, leftY = 20, leftW = 530, leftH = H - 40;
      drawRoundedRect(ctx, leftX, leftY, leftW, leftH, 6, '#ffffff', SOL.base1, 1);

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Jacobian J ∈ ℝ^(${mDim} × ${nDim}):  Column-wise JVP vs. Row-wise VJP`, leftX + leftW / 2, leftY + 24);

      // Visual grid representation of m x n Jacobian (capped to 10 rows x 16 cols for visual clarity)
      const visCols = Math.min(16, nDim);
      const visRows = Math.min(8, mDim);
      const cellW = Math.min(26, Math.floor(380 / visCols));
      const cellH = Math.min(24, Math.floor(155 / visRows));
      const gridW = visCols * cellW;
      const gridH = visRows * cellH;
      const gridX = leftX + (leftW - gridW) / 2;
      const gridY = leftY + 58;

      // Highlight 1st column (Forward JVP) and 1st row (Reverse VJP)
      for (let r = 0; r < visRows; r++) {
        for (let c = 0; c < visCols; c++) {
          const cx = gridX + c * cellW;
          const cy = gridY + r * cellH;
          let fill = SOL.base3;
          if (r === 0 && c === 0) fill = '#e6efc2';
          else if (r === 0) fill = 'rgba(133, 153, 0, 0.22)'; // Reverse VJP row
          else if (c === 0) fill = 'rgba(38, 139, 210, 0.22)'; // Forward JVP col
          ctx.fillStyle = fill;
          ctx.fillRect(cx, cy, cellW - 2, cellH - 2);
          ctx.strokeStyle = SOL.base1;
          ctx.lineWidth = 1;
          ctx.strokeRect(cx, cy, cellW - 2, cellH - 2);
        }
      }

      // Annotations around Jacobian grid
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.blue;
      ctx.textAlign = 'center';
      ctx.fillText(`↓ 1 Forward Pass (J·e₁) = 1 Column (${nDim} passes for full J)`, leftX + leftW / 2, gridY - 10);

      ctx.fillStyle = SOL.green;
      ctx.fillText(`→ 1 Reverse Pass (e₁ᵀ·J) = 1 Full Row (${mDim} pass${mDim > 1 ? 'es' : ''} for full J!)`, leftX + leftW / 2, gridY + gridH + 18);

      // Horizontal Bar Comparison of Operation Cost for f: R^n -> R^m
      const barTop = leftY + 255;
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Cost to Compute Full Jacobian (in Equivalent Forward Passes)`, leftX + leftW / 2, barTop);

      const symCost = Math.round(0.5 * nDim * nDim); // Speelpenning naive product O(n^2)
      const numCost = nDim + 1;
      const fwdCost = 2 * nDim; // primal + tangent per input
      const revCost = 1 + 3 * mDim; // 1 forward tape + ~3x per output row

      const methods = [
        { name: 'Unfactored Symbolic (∏ xᵢ)', cost: symCost, note: 'O(n²) swell', col: SOL.violet },
        { name: 'Numerical Finite Diff', cost: numCost, note: `${numCost} evals (Approx!)`, col: SOL.red },
        { name: 'Forward-Mode AD (JVP)', cost: fwdCost, note: `${nDim} passes (${fwdCost}× T_f)`, col: SOL.blue },
        { name: 'Reverse-Mode AD (VJP)', cost: revCost, note: `${mDim} rev pass (${revCost}× T_f)`, col: SOL.green }
      ];

      const maxCost = Math.max(10, symCost, fwdCost, revCost);
      const bStartX = leftX + 185;
      const bMaxW = leftW - 295;

      methods.forEach((m, idx) => {
        const by = barTop + 18 + idx * 34;
        ctx.font = '12.5px sans-serif';
        ctx.fillStyle = SOL.base02;
        ctx.textAlign = 'right';
        ctx.fillText(m.name, bStartX - 10, by + 15);

        const logRatio = Math.log10(Math.max(1, m.cost)) / Math.log10(Math.max(10, maxCost));
        const bw = Math.max(8, logRatio * bMaxW);
        drawRoundedRect(ctx, bStartX, by, bw, 22, 4, m.col, null, 0);

        ctx.font = 'bold 12px monospace';
        ctx.fillStyle = m.col;
        ctx.textAlign = 'left';
        ctx.fillText(m.note, bStartX + bw + 8, by + 15);
      });

      // ---------------- RIGHT PANEL: SCALING CURVE vs INPUT DIMENSION n ----------------
      const rightX = 570, rightY = 20, rightW = W - rightX - 20, rightH = H - 40;
      drawRoundedRect(ctx, rightX, rightY, rightW, rightH, 6, '#ffffff', SOL.base1, 1);

      const padL = 62, padR = 24, padT = 38, padB = 46;
      const plotX = rightX + padL, plotY = rightY + padT;
      const plotW = rightW - padL - padR, plotH = rightH - padT - padB;

      const nMax = 500;
      const yMax = 1050;

      function toPX(n) { return plotX + (n / nMax) * plotW; }
      function toPY(c) { return plotY + plotH - (Math.min(c, yMax) / yMax) * plotH; }

      // Grid
      ctx.strokeStyle = 'rgba(147, 161, 161, 0.22)';
      ctx.lineWidth = 1;
      for (let gn = 100; gn <= 500; gn += 100) {
        const sx = toPX(gn);
        ctx.beginPath(); ctx.moveTo(sx, plotY); ctx.lineTo(sx, plotY + plotH); ctx.stroke();
        ctx.fillStyle = SOL.base01; ctx.font = '11.5px monospace'; ctx.textAlign = 'center';
        ctx.fillText(String(gn), sx, plotY + plotH + 16);
      }
      for (let gc = 200; gc <= 1000; gc += 200) {
        const sy = toPY(gc);
        ctx.beginPath(); ctx.moveTo(plotX, sy); ctx.lineTo(plotX + plotW, sy); ctx.stroke();
        ctx.fillStyle = SOL.base01; ctx.font = '11.5px monospace'; ctx.textAlign = 'right';
        ctx.fillText(`${gc}×`, plotX - 6, sy + 4);
      }

      // Axes
      ctx.strokeStyle = SOL.base01;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(plotX, plotY);
      ctx.lineTo(plotX, plotY + plotH);
      ctx.lineTo(plotX + plotW, plotY + plotH);
      ctx.stroke();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Total Compute Cost vs. Number of Inputs n  (at Fixed m = ${mDim})`, rightX + rightW / 2, rightY + 22);
      ctx.fillText('Number of Input Parameters n', plotX + plotW / 2, rightY + rightH - 10);

      ctx.save();
      ctx.translate(rightX + 18, plotY + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Relative Cost (× Forward Eval)', 0, 0);
      ctx.restore();

      // Forward-Mode AD line: cost = 2*n
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(toPX(1), toPY(2));
      ctx.lineTo(toPX(nMax), toPY(2 * nMax));
      ctx.stroke();

      // Numerical Finite Diff line: cost = n + 1
      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(toPX(1), toPY(2));
      ctx.lineTo(toPX(nMax), toPY(nMax + 1));
      ctx.stroke();
      ctx.setLineDash([]);

      // Reverse-Mode AD line: cost = 1 + 3*mDim (independent of n!)
      const revY = toPY(revCost);
      ctx.strokeStyle = SOL.green;
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      ctx.moveTo(toPX(1), revY);
      ctx.lineTo(toPX(nMax), revY);
      ctx.stroke();

      // Vertical marker at current nDim
      const curX = toPX(nDim);
      ctx.strokeStyle = SOL.base02;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(curX, plotY);
      ctx.lineTo(curX, plotY + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = SOL.blue;
      ctx.beginPath(); ctx.arc(curX, toPY(fwdCost), 6, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = SOL.green;
      ctx.beginPath(); ctx.arc(curX, revY, 6, 0, 2 * Math.PI); ctx.fill();

      // Legend on right plot
      const legX = plotX + 16, legY = plotY + 12;
      drawRoundedRect(ctx, legX, legY, 265, 68, 4, 'rgba(253,246,227,0.94)', SOL.base1, 1);
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';

      ctx.strokeStyle = SOL.blue; ctx.lineWidth = 2.8;
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 16); ctx.lineTo(legX + 34, legY + 16); ctx.stroke();
      ctx.fillStyle = SOL.base02; ctx.fillText('Forward AD: O(n) passes', legX + 42, legY + 20);

      ctx.strokeStyle = SOL.red; ctx.lineWidth = 2.2; ctx.setLineDash([5, 3]);
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 36); ctx.lineTo(legX + 34, legY + 36); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = SOL.base02; ctx.fillText('Finite Diff: n + 1 evals (inexact)', legX + 42, legY + 40);

      ctx.strokeStyle = SOL.green; ctx.lineWidth = 3.2;
      ctx.beginPath(); ctx.moveTo(legX + 10, legY + 56); ctx.lineTo(legX + 34, legY + 56); ctx.stroke();
      ctx.fillStyle = SOL.green; ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Reverse AD (Backprop): O(m) passes!', legX + 42, legY + 60);

      // Update DOM readouts
      if (nValEl) nValEl.textContent = String(nDim);
      if (mValEl) nValEl && (mValEl.textContent = String(mDim));
      if (fwdCostEl) fwdCostEl.textContent = `${nDim} passes`;
      if (revCostEl) revCostEl.textContent = `${mDim} pass${mDim > 1 ? 'es' : ''}`;
      if (speedupEl) {
        const ratio = (fwdCost / revCost);
        speedupEl.textContent = ratio >= 1 ? `${ratio.toFixed(1)}× faster (Reverse)` : `${(1 / ratio).toFixed(1)}× faster (Forward)`;
      }

      if (statusBanner) {
        if (nDim > mDim * 3) {
          statusBanner.style.color = SOL.green;
          statusBanner.innerHTML = `🚀 <b>Many Inputs, Few Outputs (n = ${nDim} ≫ m = ${mDim}):</b> Reverse-Mode VJP computes the entire gradient in <b>${mDim} backward pass${mDim > 1 ? 'es' : ''}</b> (independent of n — Speelpenning's 1980 theorem!).`;
        } else if (mDim > nDim * 2) {
          statusBanner.style.color = SOL.blue;
          statusBanner.innerHTML = `📐 <b>Few Inputs, Many Outputs (n = ${nDim} ≪ m = ${mDim}):</b> Forward-Mode JVP wins — only <b>${nDim} forward pass${nDim > 1 ? 'es' : ''}</b> vs. ${mDim} reverse passes!`;
        } else {
          statusBanner.style.color = SOL.base02;
          statusBanner.innerHTML = `⚖️ <b>Comparable Dimensions (n = ${nDim}, m = ${mDim}):</b> Forward JVP (${nDim} cols) and Reverse VJP (${mDim} rows) have similar cost.`;
        }
      }
    }

    if (nSlider) {
      nSlider.addEventListener('input', () => {
        nDim = parseInt(nSlider.value, 10);
        draw();
      });
    }
    if (mSlider) {
      mSlider.addEventListener('input', () => {
        mDim = parseInt(mSlider.value, 10);
        draw();
      });
    }
    if (btnML) {
      btnML.addEventListener('click', () => {
        nDim = 450; mDim = 1;
        if (nSlider) nSlider.value = nDim;
        if (mSlider) mSlider.value = mDim;
        draw();
      });
    }
    if (btnSquare) {
      btnSquare.addEventListener('click', () => {
        nDim = 50; mDim = 50;
        if (nSlider) nSlider.value = nDim;
        if (mSlider) mSlider.value = mDim;
        draw();
      });
    }
    if (btnCurve) {
      btnCurve.addEventListener('click', () => {
        nDim = 2; mDim = 150;
        if (nSlider) nSlider.value = nDim;
        if (mSlider) mSlider.value = mDim;
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // ===========================================================================
  // DEMO 3B: MATRIX REVERSE-MODE AD VS. SCALAR BACKPROP MESS (#ad-matrix-canvas)
  // ===========================================================================
  function initMatrixADDemo() {
    const canvas = document.getElementById('ad-matrix-canvas');
    if (!canvas) return;

    const selProb = document.getElementById('mat-sel-prob');
    const btnPrev = document.getElementById('mat-btn-prev');
    const btnNext = document.getElementById('mat-btn-next');
    const btnAll = document.getElementById('mat-btn-all');
    const btnSpaghetti = document.getElementById('mat-btn-spaghetti');
    const statusBanner = document.getElementById('mat-status-banner');

    let probKey = 'mlp';
    let revStep = 4; // number of backward steps revealed (1..maxSteps)
    let showSpaghetti = false;

    const PROBLEMS = {
      mlp: {
        title: '2-Layer Neural Net: L = ½‖W₂σ(W₁x) − y‖²',
        scalarCount: '26 scalar nodes, 42 edges, nested ∑ᵢ∑ₖ index sums',
        matrixCount: '4 Matrix Steps — zero scalar indices!',
        finalFormula: 'W̄₁ = [ (W₂ᵀ(W₂Z₂ − y)) ⊙ σ′(Z₁) ] xᵀ   [4×3]',
        scalarMessFormula: '∂L/∂W₁,ab = ∑ᵢ (∑ₖ W₂,ᵢₖ σ(∑ⱼ W₁,ₖⱼ xⱼ) − yᵢ) W₂,ᵢa σ′(z₁,a) x_b',
        nodes: [
          { name: 'W₁, x', shape: '4×3, 3×1', x: 68, y: 74 },
          { name: 'Z₁ = W₁x', shape: '4×1', x: 185, y: 74 },
          { name: 'Z₂ = σ(Z₁)', shape: '4×1', x: 302, y: 74 },
          { name: 'Z₃ = W₂Z₂−y', shape: '2×1', x: 419, y: 74 },
          { name: 'L = ½Z₃ᵀZ₃', shape: '1×1', x: 522, y: 74 }
        ],
        steps: [
          {
            fwd: 'L = ½ Z₃ᵀ Z₃',
            fwdShape: '1×1',
            rev: 'L̄ = 1  ⇒  Z̄₃ = Z₃',
            revRule: 'Seed L̄ = 1 & quadratic form ½Z₃ᵀZ₃',
            lhsName: 'Z̄₃', lhsR: 2, lhsC: 1,
            opType: 'identity',
            rhs1Name: 'Z₃', rhs1R: 2, rhs1C: 1,
            rhs2Name: 'L̄', rhs2R: 1, rhs2C: 1,
            note: 'Adjoint Z̄₃ has exact same shape [2×1] as primal Z₃!'
          },
          {
            fwd: 'Z₃ = W₂ Z₂ − y',
            fwdShape: '2×1',
            rev: 'W̄₂ = Z̄₃ Z₂ᵀ,   Z̄₂ = W₂ᵀ Z̄₃',
            revRule: 'Matrix Product: Z = A B ⇒ Ā = Z̄ Bᵀ, B̄ = Aᵀ Z̄',
            lhsName: 'Z̄₂', lhsR: 4, lhsC: 1,
            opType: 'matmul',
            rhs1Name: 'W₂ᵀ', rhs1R: 4, rhs1C: 2,
            rhs2Name: 'Z̄₃', rhs2R: 2, rhs2C: 1,
            note: 'Dimensions force the transpose: [4×2] · [2×1] = [4×1]!'
          },
          {
            fwd: 'Z₂ = σ(Z₁)',
            fwdShape: '4×1',
            rev: 'Z̄₁ = Z̄₂ ⊙ σ′(Z₁)',
            revRule: 'Element-wise φ: Z = φ(A) ⇒ Ā = Z̄ ⊙ φ′(A)',
            lhsName: 'Z̄₁', lhsR: 4, lhsC: 1,
            opType: 'hadamard',
            rhs1Name: 'Z̄₂', rhs1R: 4, rhs1C: 1,
            rhs2Name: 'σ′(Z₁)', rhs2R: 4, rhs2C: 1,
            note: 'Element-wise activation ⇒ Hadamard product ⊙ [4×1]!'
          },
          {
            fwd: 'Z₁ = W₁ x',
            fwdShape: '4×1',
            rev: 'W̄₁ = Z̄₁ xᵀ,   x̄ = W₁ᵀ Z̄₁',
            revRule: 'Matrix Product: Z = W₁ x ⇒ W̄₁ = Z̄₁ xᵀ',
            lhsName: 'W̄₁', lhsR: 4, lhsC: 3,
            opType: 'matmul',
            rhs1Name: 'Z̄₁', rhs1R: 4, rhs1C: 1,
            rhs2Name: 'xᵀ', rhs2R: 1, rhs2C: 3,
            note: 'Outer product [4×1] · [1×3] = [4×3] matches W₁ shape!'
          }
        ]
      },
      silva: {
        title: 'Silva et al. Appendix III: y = 1ᵀ log(W X + B) 1',
        scalarCount: '38 scalar nodes, double sums ∑ᵢ∑ⱼ∑ₖ',
        matrixCount: '5 Matrix Steps — mechanical & index-free!',
        finalFormula: 'W̄ = ( (1 1ᵀ) ⊘ (W X + B) ) Xᵀ,    X̄ = Wᵀ ( (1 1ᵀ) ⊘ (W X + B) )',
        scalarMessFormula: '∂y/∂W_ab = ∑ᵢ ∑ⱼ (1 / (∑ₖ W_ik X_kj + B_ij)) X_bj',
        nodes: [
          { name: 'W, X', shape: '3×4, 4×3', x: 64, y: 74 },
          { name: 'Z₁ = WX', shape: '3×3', x: 178, y: 74 },
          { name: 'Z₂ = Z₁+B', shape: '3×3', x: 292, y: 74 },
          { name: 'Z₃ = log Z₂', shape: '3×3', x: 406, y: 74 },
          { name: 'y = 1ᵀZ₃1', shape: '1×1', x: 520, y: 74 }
        ],
        steps: [
          {
            fwd: 'y = 1ᵀ Z₄   (Z₄ = Z₃ 1)',
            fwdShape: '1×1',
            rev: 'ȳ = 1  ⇒  Z̄₄ = 1,  Z̄₃ = Z̄₄ 1ᵀ = 1 1ᵀ',
            revRule: 'Seed ȳ = 1 & Linear Product Z₄ = Z₃ 1',
            lhsName: 'Z̄₃', lhsR: 3, lhsC: 3,
            opType: 'matmul',
            rhs1Name: 'Z̄₄ (1)', rhs1R: 3, rhs1C: 1,
            rhs2Name: '1ᵀ', rhs2R: 1, rhs2C: 3,
            note: '1ᵀ Z₃ 1 seeds all-ones matrix Z̄₃ = 1 1ᵀ [3×3]!'
          },
          {
            fwd: 'Z₃ = log(Z₂)',
            fwdShape: '3×3',
            rev: 'Z̄₂ = Z̄₃ ⊙ (1 ⊘ Z₂) = (1 1ᵀ) ⊘ Z₂',
            revRule: 'Element-wise log: Z = log(A) ⇒ Ā = Z̄ ⊘ A',
            lhsName: 'Z̄₂', lhsR: 3, lhsC: 3,
            opType: 'hadamard',
            rhs1Name: 'Z̄₃', rhs1R: 3, rhs1C: 3,
            rhs2Name: '1 ⊘ Z₂', rhs2R: 3, rhs2C: 3,
            note: 'Element-wise derivative of log(Z₂) is 1 ⊘ Z₂ [3×3]!'
          },
          {
            fwd: 'Z₂ = Z₁ + B',
            fwdShape: '3×3',
            rev: 'Z̄₁ = Z̄₂,   B̄ = Z̄₂',
            revRule: 'Matrix Addition: Z = A + B ⇒ Ā = Z̄, B̄ = Z̄',
            lhsName: 'Z̄₁', lhsR: 3, lhsC: 3,
            opType: 'identity',
            rhs1Name: 'Z̄₂', rhs1R: 3, rhs1C: 3,
            rhs2Name: 'I', rhs2R: 3, rhs2C: 3,
            note: 'Addition passes the adjoint matrix Z̄₂ straight through!'
          },
          {
            fwd: 'Z₁ = W X',
            fwdShape: '3×3',
            rev: 'W̄ = Z̄₁ Xᵀ,   X̄ = Wᵀ Z̄₁',
            revRule: 'Matrix Product: Z₁ = W X ⇒ W̄ = Z̄₁ Xᵀ, X̄ = Wᵀ Z̄₁',
            lhsName: 'W̄', lhsR: 3, lhsC: 4,
            opType: 'matmul',
            rhs1Name: 'Z̄₁', rhs1R: 3, rhs1C: 3,
            rhs2Name: 'Xᵀ', rhs2R: 3, rhs2C: 4,
            note: 'Shape lock: Z̄₁ [3×3] · Xᵀ [3×4] = W̄ [3×4]!'
          }
        ]
      },
      hw5: {
        title: 'Pairwise Distance Log-Sum: 1ᵀ log((x⊙x)1ᵀ + 1(x⊙x)ᵀ − 2xxᵀ + I)1',
        scalarCount: 'n² pairwise scalar nodes + 6 fan-out branches per xᵢ',
        matrixCount: '5 Matrix Steps + Fan-Out Sum (from HW5)!',
        finalFormula: 'x̄ = 4 ( ((1 ⊘ Z₃)1) ⊙ x − (1 ⊘ Z₃) x ) = 4 L_aff x   [n×1]',
        scalarMessFormula: '∂L/∂xₖ = ∑ᵢ ∑ⱼ [ ((xᵢ−xⱼ)² + δᵢⱼ)⁻¹ (2xᵢδᵢₖ + 2xⱼδⱼₖ − 2xⱼδᵢₖ − 2xᵢδⱼₖ) ]',
        nodes: [
          { name: 'x, Z₇=x⊙x', shape: 'n×1', x: 64, y: 74 },
          { name: 'Z₄, Z₅, Z₆', shape: 'n×n', x: 185, y: 74 },
          { name: 'Z₃=Z₄+Z₅−2Z₆+I', shape: 'n×n', x: 320, y: 74 },
          { name: 'Z₂ = log Z₃', shape: 'n×n', x: 438, y: 74 },
          { name: 'L = 1ᵀZ₂1', shape: '1×1', x: 528, y: 74 }
        ],
        steps: [
          {
            fwd: 'L = 1ᵀ Z₂ 1,   Z₂ = log(Z₃)',
            fwdShape: 'n×n',
            rev: 'Z̄₂ = 1 1ᵀ  ⇒  Z̄₃ = (1 1ᵀ) ⊘ Z₃',
            revRule: 'Outer sum + element-wise log (Z₃ is symmetric!)',
            lhsName: 'Z̄₃', lhsR: 4, lhsC: 4,
            opType: 'hadamard',
            rhs1Name: '1 1ᵀ', rhs1R: 4, rhs1C: 4,
            rhs2Name: '1 ⊘ Z₃', rhs2R: 4, rhs2C: 4,
            note: 'Since Z₃ is symmetric, Z̄₃ = (1 1ᵀ) ⊘ Z₃ is also symmetric!'
          },
          {
            fwd: 'Z₃ = Z₄ + Z₅ − 2 Z₆ + I',
            fwdShape: 'n×n',
            rev: 'Z̄₄ = Z̄₃,   Z̄₅ = Z̄₃,   Z̄₆ = −2 Z̄₃',
            revRule: 'Linear Combination: distribute Z̄₃ to each term',
            lhsName: 'Z̄₄, Z̄₅, Z̄₆', lhsR: 4, lhsC: 4,
            opType: 'identity',
            rhs1Name: 'Z̄₃', rhs1R: 4, rhs1C: 4,
            rhs2Name: '±1, −2', rhs2R: 1, rhs2C: 1,
            note: 'Constant I has zero adjoint; Z̄₄=Z̄₃, Z̄₅=Z̄₃, Z̄₆=−2Z̄₃!'
          },
          {
            fwd: 'Z₄ = Z₇ 1ᵀ,   Z₅ = 1 Z₇ᵀ',
            fwdShape: 'n×n',
            rev: 'Z̄₇ = Z̄₄ 1 + Z̄₅ᵀ 1 = 2 Z̄₃ 1',
            revRule: 'Fan-out sum on Z₇ = x ⊙ x from Z₄ and Z₅',
            lhsName: 'Z̄₇', lhsR: 4, lhsC: 1,
            opType: 'matmul',
            rhs1Name: '2 Z̄₃', rhs1R: 4, rhs1C: 4,
            rhs2Name: '1', rhs2R: 4, rhs2C: 1,
            note: 'Both broadcast terms collapse cleanly into 2 Z̄₃ 1 [n×1]!'
          },
          {
            fwd: 'Z₇ = x ⊙ x,   Z₆ = x xᵀ',
            fwdShape: 'n×1',
            rev: 'x̄ = 2(Z̄₇ ⊙ x) + (Z̄₆ + Z̄₆ᵀ)x = 4((Z̄₃1)⊙x − Z̄₃x)',
            revRule: 'Sum all fan-out paths into x (Hadamard + Outer Product)',
            lhsName: 'x̄', lhsR: 4, lhsC: 1,
            opType: 'hadamard',
            rhs1Name: '2 Z̄₇', rhs1R: 4, rhs1C: 1,
            rhs2Name: 'x', rhs2R: 4, rhs2C: 1,
            note: 'Collects all 6 paths from x into 1 clean Laplacian formula!'
          }
        ]
      }
    };

    function drawOverbarText(ctx, text, x, y) {
      let cleanStr = '';
      const barIndices = [];
      for (let i = 0; i < text.length; i++) {
        if (text[i] === '\u0304') {
          if (cleanStr.length > 0) barIndices.push(cleanStr.length - 1);
        } else {
          cleanStr += text[i];
        }
      }
      ctx.fillText(cleanStr, x, y);
      if (barIndices.length === 0) return;
      const totalW = ctx.measureText(cleanStr).width;
      let startX = x;
      if (ctx.textAlign === 'center') startX = x - totalW / 2;
      else if (ctx.textAlign === 'right') startX = x - totalW;
      const m = ctx.font.match(/(\d+(?:\.\d+)?)px/);
      const fSize = m ? parseFloat(m[1]) : 12;
      ctx.save();
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 1.6;
      for (const idx of barIndices) {
        const preW = ctx.measureText(cleanStr.slice(0, idx)).width;
        const ch = cleanStr[idx];
        const chW = ctx.measureText(ch).width;
        const isLower = ch >= 'a' && ch <= 'z';
        const barY = y - fSize * (isLower ? 0.72 : 0.90);
        ctx.beginPath();
        ctx.moveTo(startX + preW + 0.5, barY);
        ctx.lineTo(startX + preW + chW - 0.5, barY);
        ctx.stroke();
      }
      ctx.restore();
    }

    function formatHtmlOverbar(str) {
      return str.replace(/([A-Za-z])\u0304/g, '<span style="text-decoration: overline;">$1</span>');
    }

    function drawMatrixGrid(ctx, x, y, rows, cols, cellSz, colorHex, fillAlpha, label, shapeStr) {
      const w = cols * cellSz;
      const h = rows * cellSz;
      drawRoundedRect(ctx, x, y, w, h, 3, fillAlpha, colorHex, 1.8);
      ctx.strokeStyle = colorHex;
      ctx.lineWidth = 0.8;
      for (let r = 1; r < rows; r++) {
        ctx.beginPath();
        ctx.moveTo(x, y + r * cellSz);
        ctx.lineTo(x + w, y + r * cellSz);
        ctx.stroke();
      }
      for (let c = 1; c < cols; c++) {
        ctx.beginPath();
        ctx.moveTo(x + c * cellSz, y);
        ctx.lineTo(x + c * cellSz, y + h);
        ctx.stroke();
      }
      ctx.font = 'bold 12.5px monospace';
      ctx.fillStyle = SOL.base03;
      ctx.textAlign = 'center';
      drawOverbarText(ctx, label, x + w / 2, y - 8);
      ctx.font = 'bold 11px monospace';
      ctx.fillStyle = colorHex;
      ctx.fillText(`[${shapeStr}]`, x + w / 2, y + h + 14);
      return { w, h };
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const prob = PROBLEMS[probKey];
      const maxSteps = prob.steps.length;
      if (revStep > maxSteps) revStep = maxSteps;
      if (revStep < 1) revStep = 1;
      const curStepObj = prob.steps[revStep - 1];

      // ---------------- LEFT PANEL: MATRIX CHAIN & SHAPE INSPECTOR vs SCALAR MESS ----------------
      const leftX = 14, leftY = 14, leftW = 556, leftH = H - 28;
      drawRoundedRect(ctx, leftX, leftY, leftW, leftH, 6, '#ffffff', SOL.base1, 1);

      ctx.font = 'bold 14px sans-serif';
      ctx.fillStyle = showSpaghetti ? SOL.red : SOL.base03;
      ctx.textAlign = 'center';
      ctx.fillText(
        showSpaghetti
          ? '⚠️ Classical Scalar Backprop: Neuron-by-Neuron Index Mess'
          : `✓ Matrix Reverse-Mode AD: ${prob.matrixCount}`,
        leftX + leftW / 2,
        leftY + 22
      );

      if (!showSpaghetti) {
        // 1. Top chain of Matrix Blocks
        const nds = prob.nodes;
        for (let i = 0; i < nds.length; i++) {
          const nd = nds[i];
          const isReached = (nds.length - 1 - i) <= revStep;
          const isActive = (nds.length - 1 - i) === revStep;

          if (i < nds.length - 1) {
            const nx = nds[i + 1];
            // Forward blue arrow
            drawArrow(ctx, nd.x + 42, nd.y - 8, nx.x - 42, nx.y - 8, SOL.blue, 2.0, 6);
            // Backward red adjoint arrow
            const edgeActive = (nds.length - 1 - i) <= revStep;
            drawArrow(
              ctx,
              nx.x - 42,
              nx.y + 10,
              nd.x + 42,
              nd.y + 10,
              edgeActive ? SOL.red : 'rgba(220,50,47,0.28)',
              edgeActive ? 2.4 : 1.3,
              6,
              [4, 3]
            );
          }

          const bw = 84, bh = 44;
          drawRoundedRect(
            ctx,
            nd.x - bw / 2,
            nd.y - bh / 2,
            bw,
            bh,
            6,
            isActive ? '#fff3e0' : (isReached ? '#fdeeed' : '#eef7fc'),
            isActive ? SOL.orange : (isReached ? SOL.red : SOL.blue),
            isActive ? 2.8 : 1.8
          );
          ctx.font = 'bold 11.5px monospace';
          ctx.fillStyle = SOL.base03;
          ctx.textAlign = 'center';
          ctx.fillText(nd.name, nd.x, nd.y - 3);
          ctx.font = '10.5px monospace';
          ctx.fillStyle = isReached ? SOL.red : SOL.blue;
          ctx.fillText(`[${nd.shape}]`, nd.x, nd.y + 13);
        }

        // 2. Middle Box: Live Matrix Shape-Lock Inspector for Current Step
        const inspX = leftX + 14, inspY = 108, inspW = leftW - 28, inspH = 182;
        drawRoundedRect(ctx, inspX, inspY, inspW, inspH, 6, SOL.base3, SOL.base1, 1.2);

        ctx.font = 'bold 13px sans-serif';
        ctx.fillStyle = SOL.orange;
        ctx.textAlign = 'left';
        drawOverbarText(ctx, `Step ${revStep}/${maxSteps} Rule:  ${curStepObj.revRule}`, inspX + 14, inspY + 20);

        // Draw visual matrix grids locking shapes together
        const cellSz = 15;
        const baseGridY = inspY + 48;
        const lhsDims = `${curStepObj.lhsR}×${curStepObj.lhsC}`;
        const r1Dims = `${curStepObj.rhs1R}×${curStepObj.rhs1C}`;
        const r2Dims = `${curStepObj.rhs2R}×${curStepObj.rhs2C}`;

        const g1X = inspX + 54;
        const g1 = drawMatrixGrid(
          ctx, g1X, baseGridY,
          curStepObj.lhsR, curStepObj.lhsC, cellSz,
          SOL.red, 'rgba(220,50,47,0.14)',
          curStepObj.lhsName, lhsDims
        );

        ctx.font = 'bold 20px monospace';
        ctx.fillStyle = SOL.base02;
        ctx.textAlign = 'center';
        const eqX = g1X + g1.w + 34;
        ctx.fillText('=', eqX, baseGridY + g1.h / 2 + 6);

        const g2X = eqX + 34;
        const g2 = drawMatrixGrid(
          ctx, g2X, baseGridY,
          curStepObj.rhs1R, curStepObj.rhs1C, cellSz,
          SOL.red, 'rgba(220,50,47,0.12)',
          curStepObj.rhs1Name, r1Dims
        );

        const opSym = curStepObj.opType === 'matmul' ? '·' : (curStepObj.opType === 'hadamard' ? '⊙' : '×');
        const dotX = g2X + g2.w + 32;
        ctx.font = 'bold 20px monospace';
        ctx.fillStyle = curStepObj.opType === 'matmul' ? SOL.green : SOL.blue;
        ctx.fillText(opSym, dotX, baseGridY + Math.max(g1.h, g2.h) / 2 + 6);

        const g3X = dotX + 32;
        drawMatrixGrid(
          ctx, g3X, baseGridY,
          curStepObj.rhs2R, curStepObj.rhs2C, cellSz,
          SOL.blue, 'rgba(38,139,210,0.14)',
          curStepObj.rhs2Name, r2Dims
        );

        // Shape-matching explanation pill inside inspector
        drawRoundedRect(ctx, inspX + 12, inspY + inspH - 32, inspW - 24, 24, 4, '#e8f5e9', SOL.green, 1);
        ctx.font = 'bold 12px sans-serif';
        ctx.fillStyle = SOL.green;
        ctx.textAlign = 'center';
        drawOverbarText(ctx, `✓ ${curStepObj.note}`, inspX + inspW / 2, inspY + inspH - 15);

        // 3. Bottom Comparison Card: Clean Matrix Formula vs Scalar Index Soup
        const cmpX = leftX + 14, cmpY = 298, cmpW = leftW - 28, cmpH = 78;
        drawRoundedRect(ctx, cmpX, cmpY, cmpW, cmpH, 6, SOL.base2, SOL.base1, 1);
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillStyle = SOL.green;
        ctx.fillText('Matrix AD Result (1 line, 0 indices):', cmpX + 12, cmpY + 18);
        ctx.font = 'bold 12px monospace';
        ctx.fillStyle = SOL.base03;
        drawOverbarText(ctx, prob.finalFormula, cmpX + 12, cmpY + 36);

        ctx.font = 'bold 11.5px sans-serif';
        ctx.fillStyle = SOL.red;
        ctx.fillText('Classical Scalar Backprop Equivalent:', cmpX + 12, cmpY + 54);
        ctx.font = '11px monospace';
        ctx.fillStyle = SOL.base01;
        ctx.fillText(prob.scalarMessFormula, cmpX + 12, cmpY + 70);

      } else {
        // SCALAR SPAGHETTI VIEW: Draw 4 layers of scalar nodes with all-to-all crossing edges
        const spX = leftX + 14, spY = 36, spW = leftW - 28, spH = 254;
        drawRoundedRect(ctx, spX, spY, spW, spH, 6, '#fff8f6', SOL.red, 1.5);

        const layers = [
          { label: 'Inputs xⱼ', count: 3, x: spX + 55 },
          { label: 'Pre-act z₁,ₖ', count: 4, x: spX + 175 },
          { label: 'Hidden aₖ', count: 4, x: spX + 295 },
          { label: 'Outputs z₃,ᵢ', count: 3, x: spX + 415 },
          { label: 'Loss L', count: 1, x: spX + 495 }
        ];

        const layerPts = layers.map(ly => {
          const pts = [];
          const span = (ly.count - 1) * 44;
          const startY = spY + 120 - span / 2;
          for (let k = 0; k < ly.count; k++) {
            pts.push({ x: ly.x, y: startY + k * 44 });
          }
          return pts;
        });

        // Draw dense crossing lines + backward red dashed arrows
        for (let l = 0; l < layerPts.length - 1; l++) {
          const src = layerPts[l];
          const dst = layerPts[l + 1];
          for (let i = 0; i < src.length; i++) {
            for (let j = 0; j < dst.length; j++) {
              ctx.strokeStyle = 'rgba(38,139,210,0.28)';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(src[i].x + 14, src[i].y);
              ctx.lineTo(dst[j].x - 14, dst[j].y);
              ctx.stroke();

              ctx.strokeStyle = 'rgba(220,50,47,0.55)';
              ctx.lineWidth = 1.2;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(dst[j].x - 14, dst[j].y + 3);
              ctx.lineTo(src[i].x + 14, src[i].y + 3);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        }

        // Draw scalar nodes
        for (let l = 0; l < layerPts.length; l++) {
          ctx.font = 'bold 11px sans-serif';
          ctx.fillStyle = SOL.base02;
          ctx.textAlign = 'center';
          ctx.fillText(layers[l].label, layers[l].x, spY + 20);
          for (let k = 0; k < layerPts[l].length; k++) {
            const pt = layerPts[l][k];
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 13, 0, 2 * Math.PI);
            ctx.fillStyle = '#fdeeed';
            ctx.fill();
            ctx.lineWidth = 1.8;
            ctx.strokeStyle = SOL.red;
            ctx.stroke();
            ctx.font = 'bold 10px monospace';
            ctx.fillStyle = SOL.base03;
            ctx.fillText(`δ_${l}${k + 1}`, pt.x, pt.y + 3);
          }
        }

        ctx.font = 'bold 12px monospace';
        ctx.fillStyle = SOL.red;
        ctx.textAlign = 'center';
        ctx.fillText(
          `Scalar Backprop: ${prob.scalarCount}`,
          spX + spW / 2,
          spY + spH - 10
        );

        const cmpX = leftX + 14, cmpY = 298, cmpW = leftW - 28, cmpH = 78;
        drawRoundedRect(ctx, cmpX, cmpY, cmpW, cmpH, 6, SOL.base2, SOL.red, 1.2);
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillStyle = SOL.red;
        ctx.fillText('Why Scalar Backprop Hurts Your Brain:', cmpX + 12, cmpY + 18);
        ctx.font = '11px monospace';
        ctx.fillStyle = SOL.base03;
        ctx.fillText(prob.scalarMessFormula, cmpX + 12, cmpY + 38);
        ctx.font = 'bold 12px sans-serif';
        ctx.fillStyle = SOL.green;
        drawOverbarText(ctx, `vs. Matrix Reverse-Mode AD:  ${prob.finalFormula}`, cmpX + 12, cmpY + 62);
      }

      // ---------------- RIGHT PANEL: STEP-BY-STEP MATRIX TRACE & ADJOINT TABLE ----------------
      const rtX = 582, rtY = 14, rtW = W - rtX - 14, rtH = H - 28;
      drawRoundedRect(ctx, rtX, rtY, rtW, rtH, 6, '#ffffff', SOL.base1, 1);

      const c1W = 212;
      const c2W = rtW - c1W - 24;
      const c1X = rtX + 8;
      const c2X = rtX + 16 + c1W;

      drawRoundedRect(ctx, c1X, rtY + 8, c1W, 28, 4, '#eef7fc', SOL.blue, 1);
      drawRoundedRect(ctx, c2X, rtY + 8, c2W, 28, 4, '#fdeeed', SOL.red, 1);

      ctx.font = 'bold 12.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = SOL.blue;
      ctx.fillText('Forward Matrix Tape (Zᵢ)', c1X + c1W / 2, rtY + 27);
      ctx.fillStyle = SOL.red;
      drawOverbarText(ctx, 'Backward Matrix Adjoints (Z̄ᵢ = ∂y/∂Zᵢ)', c2X + c2W / 2, rtY + 27);

      const rowStart = rtY + 44;
      const rowH = 66;

      // Display steps from top (input-nearest) to bottom (output seed) or in reverse-derivation order
      for (let s = 0; s < maxSteps; s++) {
        const st = prob.steps[s];
        const ry = rowStart + s * rowH;
        const isDone = s < revStep;
        const isCurrent = s === (revStep - 1);

        drawRoundedRect(
          ctx, c1X, ry, c1W, rowH - 6, 5,
          isCurrent ? '#fff3e0' : SOL.base3,
          isCurrent ? SOL.orange : SOL.base1,
          isCurrent ? 2.0 : 1.0
        );
        drawRoundedRect(
          ctx, c2X, ry, c2W, rowH - 6, 5,
          isCurrent ? '#fff3e0' : (isDone ? '#fff8f6' : '#f7f7f7'),
          isCurrent ? SOL.orange : (isDone ? SOL.red : SOL.base1),
          isCurrent ? 2.0 : 1.0
        );

        // Forward column text
        ctx.font = 'bold 12px monospace';
        ctx.fillStyle = SOL.base03;
        ctx.textAlign = 'left';
        ctx.fillText(st.fwd, c1X + 10, ry + 23);
        ctx.font = '11px monospace';
        ctx.fillStyle = SOL.blue;
        ctx.fillText(`Shape: [${st.fwdShape}]`, c1X + 10, ry + 44);

        // Reverse adjoint column text
        if (isDone) {
          ctx.font = 'bold 12px monospace';
          ctx.fillStyle = isCurrent ? SOL.orange : SOL.red;
          ctx.textAlign = 'left';
          drawOverbarText(ctx, `Step ${s + 1}: ${st.rev}`, c2X + 10, ry + 23);
          ctx.font = '11px sans-serif';
          ctx.fillStyle = SOL.base01;
          drawOverbarText(ctx, st.revRule, c2X + 10, ry + 44);
        } else {
          ctx.font = 'italic 12px sans-serif';
          ctx.fillStyle = SOL.base1;
          ctx.textAlign = 'left';
          ctx.fillText(`Step ${s + 1}: click "Step Backward" to derive...`, c2X + 10, ry + 34);
        }
      }

      // Bottom summary box in right panel
      const sumY = rtY + rtH - 58;
      drawRoundedRect(ctx, c1X, sumY, rtW - 16, 50, 5, SOL.base2, SOL.green, 1.5);
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = SOL.green;
      ctx.textAlign = 'left';
      ctx.fillText('Collected Analytical Matrix Gradient (No Scalar Indices!):', c1X + 12, sumY + 19);
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = SOL.base03;
      drawOverbarText(ctx, prob.finalFormula, c1X + 12, sumY + 38);

      // Toggle live MathJax expression at bottom of slide
      ['mlp', 'silva', 'hw5'].forEach(k => {
        const el = document.getElementById(`mat-latex-${k}`);
        if (el) el.style.display = (k === probKey) ? 'block' : 'none';
      });

      if (statusBanner) {
        statusBanner.style.color = showSpaghetti ? SOL.red : SOL.base03;
        statusBanner.innerHTML = showSpaghetti
          ? `🍝 <b>Scalar Backprop View:</b> ${prob.scalarCount} — vs. <b>${prob.matrixCount}</b> in Matrix Reverse-Mode AD!`
          : `📐 <b>Matrix Reverse-Mode Step ${revStep}/${maxSteps}:</b> <code>${formatHtmlOverbar(curStepObj.rev)}</code> — ${formatHtmlOverbar(curStepObj.note)}`;
      }
    }

    if (selProb) {
      selProb.addEventListener('change', () => {
        probKey = selProb.value;
        revStep = PROBLEMS[probKey].steps.length;
        draw();
      });
    }
    if (btnPrev) {
      btnPrev.addEventListener('click', () => {
        revStep = Math.max(1, revStep - 1);
        draw();
      });
    }
    if (btnNext) {
      btnNext.addEventListener('click', () => {
        const maxSteps = PROBLEMS[probKey].steps.length;
        revStep = (revStep < maxSteps) ? (revStep + 1) : 1;
        draw();
      });
    }
    if (btnAll) {
      btnAll.addEventListener('click', () => {
        revStep = PROBLEMS[probKey].steps.length;
        draw();
      });
    }
    if (btnSpaghetti) {
      btnSpaghetti.addEventListener('click', () => {
        showSpaghetti = !showSpaghetti;
        btnSpaghetti.textContent = showSpaghetti
          ? '📐 Back to Clean Matrix AD'
          : '🍝 Compare Scalar Backprop Mess';
        draw();
      });
    }

    registerSlideListener(canvas, draw);
    setTimeout(draw, 100);
  }

  // Initialize all demos
  function initAll() {
    initNumDiffDemo();
    initForwardADDemo();
    initReverseADDemo();
    initMatrixADDemo();
    initComplexityDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
