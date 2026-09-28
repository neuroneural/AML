/**
 * CS8850: Advanced Machine Learning - Lecture 11 (Logistic Regression)
 * Interactive Pedagogical Demonstrations
 * 
 * 1. GNB vs. Logistic Regression (Generative vs Discriminative Boundary Explorer)
 * 2. 1D Classification: Why Not Linear Regression? (Outlier Vulnerability)
 * 3. Geometry of Logistic Regression (Linear Hyperplane Boundary vs Nonlinear Sigmoid Surface)
 * 4. Interactive Softmax & Temperature Explorer (Multi-Sample Signal Segmentation & Detection)
 * 5. Interactive Taylor Expansion & Newton-Raphson Explorer
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

  function setupHiDPI(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.round(rect.width) || canvas.width || 940;
    const h = Math.round(rect.height) || canvas.height || 350;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, width: w, height: h };
  }

  // =========================================================================
  // DEMO 1: GNB vs. Logistic Regression (Generative vs. Discriminative)
  // =========================================================================
  function initGnbLrDemo() {
    const canvas = document.getElementById('gnb-lr-canvas');
    if (!canvas) return;

    let rho = 0.75;
    let hasOutlier = false;

    // Seeded random points for reproducibility
    function generateBasePoints() {
      const c0 = [];
      const c1 = [];
      const n = 45;
      
      let seed = 123456;
      function rnd() {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      }
      function randn() {
        const u = rnd() || 0.0001;
        const v = rnd() || 0.0001;
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
      }

      for (let i = 0; i < n; i++) {
        c0.push({ z1: randn(), z2: randn(), label: 0 });
      }
      for (let i = 0; i < n; i++) {
        c1.push({ z1: randn(), z2: randn(), label: 1 });
      }
      return { c0, c1 };
    }

    const rawData = generateBasePoints();

    // DOM Controls
    const rhoSlider = document.getElementById('gnb-lr-rho');
    const rhoValSpan = document.getElementById('gnb-lr-rho-val');
    const gnbAccSpan = document.getElementById('gnb-acc-val');
    const lrAccSpan = document.getElementById('lr-acc-val');
    const statusSpan = document.getElementById('gnb-lr-status');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      // Strict 1:1 isometric aspect ratio: 1 unit on X == 1 unit on Y in pixels
      const spanY = 5.6;
      const scale = H / spanY; // px per data unit = 350 / 5.6 = 62.5
      const spanX = W / scale; // data units across width = 940 / 62.5 = 15.04
      const minX = -spanX / 2, maxX = spanX / 2;
      const minY = -spanY / 2, maxY = spanY / 2;

      function toX(x) { return W * 0.5 + x * scale; }
      function toY(y) { return H * 0.5 - y * scale; }

      // Equal isotropic base variances: s1 = s2 = 1.0 (circular when rho = 0)
      const s1 = 1.0, s2 = 1.0;
      const r = Math.max(-0.95, Math.min(0.95, rho));
      const l21 = r;
      const l22 = Math.sqrt(Math.max(0.001, 1 - r * r));

      const points = [];
      const mean0 = [-2.2, -0.6];
      const mean1 = [2.2, 0.6];

      rawData.c0.forEach(p => {
        const x = mean0[0] + s1 * p.z1;
        const y = mean0[1] + s2 * (l21 * p.z1 + l22 * p.z2);
        points.push({ x, y, label: 0 });
      });

      rawData.c1.forEach(p => {
        const x = mean1[0] + s1 * p.z1;
        const y = mean1[1] + s2 * (l21 * p.z1 + l22 * p.z2);
        points.push({ x, y, label: 1 });
      });

      if (hasOutlier) {
        points.push({ x: 5.2, y: 2.2, label: 1, isOutlier: true });
        points.push({ x: 5.6, y: 1.8, label: 1, isOutlier: true });
      }

      // Compute GNB parameters (assumes DIAGONAL covariance!)
      const c0Pts = points.filter(p => p.label === 0);
      const c1Pts = points.filter(p => p.label === 1);

      let mu0x = 0, mu0y = 0;
      c0Pts.forEach(p => { mu0x += p.x; mu0y += p.y; });
      mu0x /= c0Pts.length; mu0y /= c0Pts.length;

      let mu1x = 0, mu1y = 0;
      c1Pts.forEach(p => { mu1x += p.x; mu1y += p.y; });
      mu1x /= c1Pts.length; mu1y /= c1Pts.length;

      // Pooled diagonal variances
      let varX = 0, varY = 0;
      c0Pts.forEach(p => { varX += (p.x - mu0x) ** 2; varY += (p.y - mu0y) ** 2; });
      c1Pts.forEach(p => { varX += (p.x - mu1x) ** 2; varY += (p.y - mu1y) ** 2; });
      varX /= (points.length - 2);
      varY /= (points.length - 2);
      varX = Math.max(0.05, varX);
      varY = Math.max(0.05, varY);

      const wGnbX = (mu1x - mu0x) / varX;
      const wGnbY = (mu1y - mu0y) / varY;
      const bGnb = 0.5 * ((mu0x * mu0x - mu1x * mu1x) / varX + (mu0y * mu0y - mu1y * mu1y) / varY);

      // Logistic Regression via fast IRLS
      let wLrX = (mu1x - mu0x);
      let wLrY = (mu1y - mu0y);
      let bLr = -0.5 * (wLrX * (mu0x + mu1x) + wLrY * (mu0y + mu1y));

      for (let iter = 0; iter < 12; iter++) {
        let gx = 0, gy = 0, gb = 0;
        let hxx = 0, hxy = 0, hxb = 0, hyy = 0, hyb = 0, hbb = 0;
        for (let p of points) {
          const z = wLrX * p.x + wLrY * p.y + bLr;
          const s = 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));
          const err = s - p.label;
          gx += err * p.x;
          gy += err * p.y;
          gb += err;
          const wgt = Math.max(1e-5, s * (1.0 - s));
          hxx += wgt * p.x * p.x;
          hxy += wgt * p.x * p.y;
          hxb += wgt * p.x;
          hyy += wgt * p.y * p.y;
          hyb += wgt * p.y;
          hbb += wgt;
        }
        hxx += 0.05; hyy += 0.05; hbb += 0.05;

        const A = hxx, B = hxy, C = hxb;
        const D = hxy, E = hyy, F = hyb;
        const G = hxb, H_val = hyb, I = hbb;
        const det = A*(E*I - F*H_val) - B*(D*I - F*G) + C*(D*H_val - E*G);
        if (Math.abs(det) > 1e-7) {
          const inv00 =  (E*I - F*H_val) / det;
          const inv01 = -(B*I - C*H_val) / det;
          const inv02 =  (B*F - C*E) / det;
          const inv10 = -(D*I - F*G) / det;
          const inv11 =  (A*I - C*G) / det;
          const inv12 = -(A*F - C*D) / det;
          const inv20 =  (D*H_val - E*G) / det;
          const inv21 = -(A*H_val - B*G) / det;
          const inv22 =  (A*E - B*D) / det;

          wLrX -= (inv00 * gx + inv01 * gy + inv02 * gb);
          wLrY -= (inv10 * gx + inv11 * gy + inv12 * gb);
          bLr  -= (inv20 * gx + inv21 * gy + inv22 * gb);
        }
      }

      // Accuracies
      let gnbCorrect = 0, lrCorrect = 0;
      points.forEach(p => {
        const predGnb = (wGnbX * p.x + wGnbY * p.y + bGnb >= 0) ? 1 : 0;
        const predLr = (wLrX * p.x + wLrY * p.y + bLr >= 0) ? 1 : 0;
        if (predGnb === p.label) gnbCorrect++;
        if (predLr === p.label) lrCorrect++;
      });
      const gnbAcc = (gnbCorrect / points.length) * 100;
      const lrAcc = (lrCorrect / points.length) * 100;

      if (gnbAccSpan) gnbAccSpan.textContent = gnbAcc.toFixed(1) + '%';
      if (lrAccSpan) lrAccSpan.textContent = lrAcc.toFixed(1) + '%';
      if (rhoValSpan) rhoValSpan.textContent = rho.toFixed(2);

      if (statusSpan) {
        if (Math.abs(rho) < 0.15 && !hasOutlier) {
          statusSpan.innerHTML = '<span style="color:#859900;"><i class="fa fa-check-circle"></i> <b>Independent Features:</b> Isotropic circles; GNB assumption holds and boundaries align!</span>';
        } else if (hasOutlier) {
          statusSpan.innerHTML = '<span style="color:#dc322f;"><i class="fa fa-exclamation-circle"></i> <b>Outliers Present:</b> GNB mean/variance corrupted; LR saturates gracefully!</span>';
        } else {
          statusSpan.innerHTML = `<span style="color:#b58900;"><i class="fa fa-exclamation-triangle"></i> <b>Correlated Features (&rho;=${rho.toFixed(2)}):</b> True density tilts at 45°; GNB constrained to axis-aligned ellipses!</span>`;
        }
      }

      // 1. Draw Grid (Strict 1x1 Squares in Isometric Scale)
      ctx.strokeStyle = '#eee8d5';
      ctx.lineWidth = 1;
      for (let x = Math.ceil(minX); x <= Math.floor(maxX); x += 1) {
        ctx.beginPath(); ctx.moveTo(toX(x), 0); ctx.lineTo(toX(x), H); ctx.stroke();
      }
      for (let y = Math.ceil(minY); y <= Math.floor(maxY); y += 1) {
        ctx.beginPath(); ctx.moveTo(0, toY(y)); ctx.lineTo(W, toY(y)); ctx.stroke();
      }

      // 2A. Draw True Empirical Class Density Contours (Exact sample covariance and principal angle)
      function drawClassDensity(pts, strokeCol, fillCol) {
        if (!pts.length) return;
        let mx = 0, my = 0;
        pts.forEach(p => { mx += p.x; my += p.y; });
        mx /= pts.length; my /= pts.length;

        let sxx = 0, syy = 0, sxy = 0;
        pts.forEach(p => {
          const dx = p.x - mx;
          const dy = p.y - my;
          sxx += dx * dx;
          syy += dy * dy;
          sxy += dx * dy;
        });
        sxx /= (pts.length - 1);
        syy /= (pts.length - 1);
        sxy /= (pts.length - 1);

        const T = sxx + syy;
        const D = sxx * syy - sxy * sxy;
        const term = Math.sqrt(Math.max(0, (T * T) / 4.0 - D));
        const lam1 = T / 2.0 + term;
        const lam2 = Math.max(0.001, T / 2.0 - term);

        // Cartesian angle of major eigenvector (slanted up-right if sxy > 0)
        const angleCart = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        // Canvas screen Y points DOWN, so screen rotation is inverted (-angleCart)
        const angleScreen = -angleCart;

        ctx.save();
        ctx.strokeStyle = strokeCol;
        ctx.fillStyle = fillCol;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([]);

        [1.0, 2.0].forEach(k => {
          const rx = k * Math.sqrt(lam1) * scale;
          const ry = k * Math.sqrt(lam2) * scale;
          ctx.beginPath();
          ctx.ellipse(toX(mx), toY(my), rx, ry, angleScreen, 0, 2 * Math.PI);
          ctx.stroke();
          if (k === 2.0) ctx.fill();
        });
        ctx.restore();
      }

      const inliers0 = c0Pts.filter(p => !p.isOutlier);
      const inliers1 = c1Pts.filter(p => !p.isOutlier);
      drawClassDensity(inliers0, 'rgba(38, 139, 210, 0.65)', 'rgba(38, 139, 210, 0.05)');
      drawClassDensity(inliers1, 'rgba(203, 75, 22, 0.65)', 'rgba(203, 75, 22, 0.05)');

      // 2B. Draw GNB Estimated Model Ellipses (Constrained to Axis-Aligned / Diagonal Covariance)
      function drawGnbAssumedEllipses(mx, my, color) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        [1.0, 2.0].forEach(k => {
          const rx = k * Math.sqrt(varX) * scale;
          const ry = k * Math.sqrt(varY) * scale;
          ctx.beginPath();
          ctx.ellipse(toX(mx), toY(my), rx, ry, 0, 0, 2 * Math.PI);
          ctx.stroke();
        });
        ctx.restore();
      }
      drawGnbAssumedEllipses(mu0x, mu0y, 'rgba(38, 139, 210, 0.75)');
      drawGnbAssumedEllipses(mu1x, mu1y, 'rgba(203, 75, 22, 0.75)');

      // 3. Draw Decision Boundaries with Non-Overlapping Pill Labels
      function drawBoundaryLine(wx, wy, b, color, isDashed, label, labelT) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3.0;
        if (isDashed) ctx.setLineDash([6, 5]);

        const pts = [];
        const yAtMinX = -(wx * minX + b) / (wy || 1e-5);
        if (yAtMinX >= minY && yAtMinX <= maxY) pts.push([minX, yAtMinX]);
        const yAtMaxX = -(wx * maxX + b) / (wy || 1e-5);
        if (yAtMaxX >= minY && yAtMaxX <= maxY) pts.push([maxX, yAtMaxX]);
        const xAtMinY = -(wy * minY + b) / (wx || 1e-5);
        if (xAtMinY >= minX && xAtMinY <= maxX) pts.push([xAtMinY, minY]);
        const xAtMaxY = -(wy * maxY + b) / (wx || 1e-5);
        if (xAtMaxY >= minX && xAtMaxY <= maxX) pts.push([xAtMaxY, maxY]);

        const unique = [];
        for (let p of pts) {
          if (!unique.some(u => Math.hypot(u[0] - p[0], u[1] - p[1]) < 0.05)) {
            unique.push(p);
          }
        }

        if (unique.length >= 2) {
          // Sort consistently by Y so p1 is always bottom-most and p2 is top-most
          unique.sort((a, b) => a[1] - b[1]);
          const p1 = unique[0];
          const p2 = unique[unique.length - 1];

          ctx.beginPath();
          ctx.moveTo(toX(p1[0]), toY(p1[1]));
          ctx.lineTo(toX(p2[0]), toY(p2[1]));
          ctx.stroke();

          // Pill label at fraction labelT along the segment (0.15 = bottom, 0.85 = top)
          const lx = p1[0] + labelT * (p2[0] - p1[0]);
          const ly = p1[1] + labelT * (p2[1] - p1[1]);
          const sx = toX(lx);
          const sy = toY(ly);

          ctx.save();
          ctx.setLineDash([]);
          ctx.font = 'bold 11px sans-serif';
          const textW = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(253, 246, 227, 0.94)';
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.roundRect(sx - textW / 2 - 6, sy - 9, textW + 12, 18, 4);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = color;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(label, sx, sy);
          ctx.restore();
        }
        ctx.restore();
      }

      drawBoundaryLine(wGnbX, wGnbY, bGnb, SOL.orange, true, 'GNB', 0.82);
      drawBoundaryLine(wLrX, wLrY, bLr, SOL.green, false, 'Logistic Reg', 0.18);

      // 4. Draw Data Points
      points.forEach(p => {
        ctx.beginPath();
        const px = toX(p.x);
        const py = toY(p.y);
        ctx.arc(px, py, p.isOutlier ? 6 : 4.5, 0, 2 * Math.PI);
        if (p.label === 0) {
          ctx.fillStyle = SOL.blue;
          ctx.fill();
          ctx.strokeStyle = '#073642';
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          ctx.fillStyle = p.isOutlier ? SOL.red : SOL.orange;
          ctx.fill();
          ctx.strokeStyle = '#073642';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      });

      // Draw Means
      function drawMean(mx, my, color, label) {
        ctx.fillStyle = color;
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('✖', toX(mx) - 5, toY(my) + 5);
        ctx.font = '11px sans-serif';
        ctx.fillText(label, toX(mx) + 8, toY(my) + 4);
      }
      drawMean(mu0x, mu0y, SOL.blue, 'μ₀');
      drawMean(mu1x, mu1y, SOL.orange, 'μ₁');
    }

    if (rhoSlider) {
      rhoSlider.addEventListener('input', (e) => {
        rho = parseFloat(e.target.value);
        draw();
      });
    }

    const btnIndep = document.getElementById('gnb-btn-indep');
    const btnCorr = document.getElementById('gnb-btn-corr');
    const btnOutlier = document.getElementById('gnb-btn-outlier');

    if (btnIndep) btnIndep.addEventListener('click', () => {
      rho = 0.0;
      hasOutlier = false;
      if (rhoSlider) rhoSlider.value = 0.0;
      draw();
    });
    if (btnCorr) btnCorr.addEventListener('click', () => {
      rho = 0.8;
      hasOutlier = false;
      if (rhoSlider) rhoSlider.value = 0.8;
      draw();
    });
    if (btnOutlier) btnOutlier.addEventListener('click', () => {
      rho = 0.75;
      hasOutlier = !hasOutlier;
      if (rhoSlider) rhoSlider.value = 0.75;
      btnOutlier.style.background = hasOutlier ? '#eee8d5' : '#fdf6e3';
      draw();
    });

    window.addEventListener('resize', draw);
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) draw();
      });
    }
    draw();
  }

  // =========================================================================
  // DEMO 2: 1D Classification (Why Not Linear Regression? Outlier Vulnerability)
  // =========================================================================
  function init1dLinearVsLogisticDemo() {
    const canvas = document.getElementById('linear-vs-logistic-canvas');
    if (!canvas) return;

    let outlierX = 7.5;
    let includeOutlier = false;

    const basePts = [
      { x: -3.5, y: 0 }, { x: -3.0, y: 0 }, { x: -2.5, y: 0 },
      { x: -2.0, y: 0 }, { x: -1.5, y: 0 }, { x: -1.0, y: 0 },
      { x: 0.5, y: 1 },  { x: 1.0, y: 1 },  { x: 1.5, y: 1 },
      { x: 2.0, y: 1 },  { x: 2.5, y: 1 },  { x: 3.0, y: 1 }
    ];

    const slider = document.getElementById('outlier-x-slider');
    const sliderVal = document.getElementById('outlier-x-val');
    const linThreshSpan = document.getElementById('lin-thresh-val');
    const logThreshSpan = document.getElementById('log-thresh-val');
    const alertBox = document.getElementById('outlier-alert-box');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const minX = -4.5, maxX = 9.5;
      const minY = -0.3, maxY = 1.3;

      function toX(x) { return ((x - minX) / (maxX - minX)) * W; }
      function toY(y) { return H - ((y - minY) / (maxY - minY)) * H; }

      const pts = basePts.map(p => ({ ...p }));
      if (includeOutlier) {
        pts.push({ x: outlierX, y: 1, isOutlier: true });
      }

      // OLS Linear Regression
      let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
      pts.forEach(p => {
        sumX += p.x; sumY += p.y;
        sumXY += p.x * p.y; sumXX += p.x * p.x;
      });
      const N = pts.length;
      const wLin = (N * sumXY - sumX * sumY) / (N * sumXX - sumX * sumX);
      const bLin = (sumY - wLin * sumX) / N;
      const threshLin = (0.5 - bLin) / (wLin || 1e-5);

      // Logistic Regression
      let wLog = 1.5, bLog = 0.0;
      for (let iter = 0; iter < 15; iter++) {
        let gw = 0, gb = 0;
        let hww = 0, hwb = 0, hbb = 0;
        for (let p of pts) {
          const z = wLog * p.x + bLog;
          const s = 1.0 / (1.0 + Math.exp(-Math.max(-25, Math.min(25, z))));
          const diff = s - p.y;
          gw += diff * p.x;
          gb += diff;
          const h = Math.max(1e-5, s * (1.0 - s));
          hww += h * p.x * p.x;
          hwb += h * p.x;
          hbb += h;
        }
        hww += 0.01; hbb += 0.01;
        const det = hww * hbb - hwb * hwb;
        if (Math.abs(det) > 1e-6) {
          wLog -= (hbb * gw - hwb * gb) / det;
          bLog -= (-hwb * gw + hww * gb) / det;
        }
      }
      const threshLog = -bLog / (wLog || 1e-5);

      if (sliderVal) sliderVal.textContent = outlierX.toFixed(1);
      if (linThreshSpan) linThreshSpan.textContent = `x = ${threshLin.toFixed(2)}`;
      if (logThreshSpan) logThreshSpan.textContent = `x = ${threshLog.toFixed(2)}`;

      if (alertBox) {
        if (!includeOutlier) {
          alertBox.innerHTML = '<span style="color:#859900;"><i class="fa fa-check-circle"></i> Clean Data: Both models find a valid decision boundary near x = -0.25.</span>';
        } else if (threshLin > 0.6) {
          alertBox.innerHTML = `<span style="color:#dc322f;"><i class="fa fa-exclamation-triangle"></i> <b>THRESHOLD CORRUPTED:</b> Outlier rotated linear fit! Threshold drifted to x=${threshLin.toFixed(2)} (misclassifying positives). Logistic stays robust at x=${threshLog.toFixed(2)}!</span>`;
        } else {
          alertBox.innerHTML = `<span style="color:#b58900;"><i class="fa fa-info-circle"></i> Outlier active at x=${outlierX.toFixed(1)}. Notice the linear line tilting toward the outlier.</span>`;
        }
      }

      ctx.strokeStyle = '#eee8d5';
      ctx.lineWidth = 1;
      [0, 0.5, 1].forEach(yVal => {
        ctx.beginPath();
        ctx.moveTo(toX(minX), toY(yVal));
        ctx.lineTo(toX(maxX), toY(yVal));
        ctx.stroke();
      });

      // Decision line P = 0.5
      ctx.strokeStyle = '#93a1a1';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(toX(minX), toY(0.5));
      ctx.lineTo(toX(maxX), toY(0.5));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#93a1a1';
      ctx.font = '11px monospace';
      ctx.fillText('Decision Threshold P = 0.5', toX(minX) + 8, toY(0.5) - 6);

      // Plot Linear Fit line
      ctx.strokeStyle = SOL.blue;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(toX(minX), toY(wLin * minX + bLin));
      ctx.lineTo(toX(maxX), toY(wLin * maxX + bLin));
      ctx.stroke();

      // Plot Logistic Sigmoid Curve
      ctx.strokeStyle = SOL.cyan;
      ctx.lineWidth = 3.0;
      ctx.beginPath();
      for (let px = 0; px <= W; px += 2) {
        const x = minX + (px / W) * (maxX - minX);
        const z = wLog * x + bLog;
        const p = 1.0 / (1.0 + Math.exp(-Math.max(-25, Math.min(25, z))));
        if (px === 0) ctx.moveTo(px, toY(p));
        else ctx.lineTo(px, toY(p));
      }
      ctx.stroke();

      // Plot Threshold vertical markers
      function drawThresholdMarker(thX, color, label, isDashed) {
        if (thX < minX || thX > maxX) return;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        if (isDashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(toX(thX), toY(-0.2));
        ctx.lineTo(toX(thX), toY(1.2));
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = 'bold 11px monospace';
        ctx.fillText(label, toX(thX) - 20, toY(1.15));
        ctx.restore();
      }

      drawThresholdMarker(threshLin, SOL.blue, `Lin: ${threshLin.toFixed(2)}`, true);
      drawThresholdMarker(threshLog, SOL.cyan, `Log: ${threshLog.toFixed(2)}`, false);

      // Plot Data Points
      pts.forEach(p => {
        ctx.beginPath();
        const px = toX(p.x);
        const py = toY(p.y);
        ctx.arc(px, py, p.isOutlier ? 7 : 5, 0, 2 * Math.PI);
        ctx.fillStyle = p.isOutlier ? SOL.red : (p.y === 1 ? SOL.orange : SOL.violet);
        ctx.fill();
        ctx.strokeStyle = '#073642';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        if (p.isOutlier) {
          ctx.fillStyle = SOL.red;
          ctx.font = 'bold 11px sans-serif';
          ctx.fillText('OUTLIER (y=1)', px - 35, py - 12);
        }
      });
    }

    if (slider) {
      slider.addEventListener('input', (e) => {
        outlierX = parseFloat(e.target.value);
        includeOutlier = true;
        draw();
      });
    }

    const btnNoOutlier = document.getElementById('btn-1d-clean');
    const btnFarOutlier = document.getElementById('btn-1d-outlier');

    if (btnNoOutlier) btnNoOutlier.addEventListener('click', () => {
      includeOutlier = false;
      draw();
    });
    if (btnFarOutlier) btnFarOutlier.addEventListener('click', () => {
      includeOutlier = true;
      outlierX = 8.0;
      if (slider) slider.value = 8.0;
      draw();
    });

    window.addEventListener('resize', draw);
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) draw();
      });
    }
    draw();
  }

  // =========================================================================
  // DEMO 3: Geometry of Logistic Regression (Linear Hyperplane Boundary vs Nonlinear Sigmoid Surface)
  // =========================================================================
  function initLrGeometryDemo() {
    const canvas = document.getElementById('lr-geometry-canvas');
    if (!canvas) return;

    let slopeAngle = 0.7; // Angle theta of normal vector w in radians
    let normW = 1.6;      // Magnitude of w (steepness of ramp)
    let w0 = -0.2;        // Bias / Intercept
    let rotX = 55 * Math.PI / 180;
    let rotZ = 35 * Math.PI / 180;

    let isDragging3D = false;
    let lastMouseX = 0, lastMouseY = 0;

    const slopeSlider = document.getElementById('geom-slope-slider');
    const biasSlider = document.getElementById('geom-bias-slider');
    const formulaBadge = document.getElementById('geom-formula-badge');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const w1 = normW * Math.cos(slopeAngle);
      const w2 = normW * Math.sin(slopeAngle);

      const w2d = W * 0.46;
      const w3d = W * 0.54;

      // -------------------------------------------------------------------
      // LEFT HALF: 2D Feature Space & Linear Hyperplane Boundary
      // -------------------------------------------------------------------
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, w2d, H);
      ctx.clip();

      const minX = -3.0, maxX = 3.0;
      const minY = -3.0, maxY = 3.0;
      function to2dX(x) { return 40 + ((x - minX) / (maxX - minX)) * (w2d - 60); }
      function to2dY(y) { return (H - 40) - ((y - minY) / (maxY - minY)) * (H - 70); }

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText('2D Feature Space: Linear Decision Boundary', 40, 22);

      // Grid
      ctx.strokeStyle = '#eee8d5';
      ctx.lineWidth = 1;
      for (let x = -2; x <= 2; x += 1) {
        ctx.beginPath(); ctx.moveTo(to2dX(x), to2dY(minY)); ctx.lineTo(to2dX(x), to2dY(maxY)); ctx.stroke();
      }
      for (let y = -2; y <= 2; y += 1) {
        ctx.beginPath(); ctx.moveTo(to2dX(minX), to2dY(y)); ctx.lineTo(to2dX(maxX), to2dY(y)); ctx.stroke();
      }

      // Draw Contours of P = [0.1, 0.3, 0.5, 0.7, 0.9]
      // In LR, w1*x1 + w2*x2 + w0 = logit(p) -> x2 = (logit(p) - w0 - w1*x1) / w2
      const probLevels = [
        { p: 0.1, color: 'rgba(38, 139, 210, 0.45)', dash: [3, 3] },
        { p: 0.3, color: 'rgba(42, 161, 152, 0.55)', dash: [3, 3] },
        { p: 0.5, color: SOL.orange, dash: [], isBoundary: true },
        { p: 0.7, color: 'rgba(181, 137, 0, 0.55)', dash: [3, 3] },
        { p: 0.9, color: 'rgba(220, 50, 47, 0.45)', dash: [3, 3] }
      ];

      probLevels.forEach(lvl => {
        const logit = Math.log(lvl.p / (1 - lvl.p));
        ctx.save();
        ctx.strokeStyle = lvl.color;
        ctx.lineWidth = lvl.isBoundary ? 3.0 : 1.5;
        if (lvl.dash.length) ctx.setLineDash(lvl.dash);

        // Find boundary line endpoints intersecting bounding box
        const pts = [];
        if (Math.abs(w2) > 1e-4) {
          const yAtMinX = (logit - w0 - w1 * minX) / w2;
          if (yAtMinX >= minY && yAtMinX <= maxY) pts.push([minX, yAtMinX]);
          const yAtMaxX = (logit - w0 - w1 * maxX) / w2;
          if (yAtMaxX >= minY && yAtMaxX <= maxY) pts.push([maxX, yAtMaxX]);
        }
        if (Math.abs(w1) > 1e-4) {
          const xAtMinY = (logit - w0 - w2 * minY) / w1;
          if (xAtMinY >= minX && xAtMinY <= maxX) pts.push([xAtMinY, minY]);
          const xAtMaxY = (logit - w0 - w2 * maxY) / w1;
          if (xAtMaxY >= minX && xAtMaxY <= maxX) pts.push([xAtMaxY, maxY]);
        }

        if (pts.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(to2dX(pts[0][0]), to2dY(pts[0][1]));
          ctx.lineTo(to2dX(pts[1][0]), to2dY(pts[1][1]));
          ctx.stroke();

          if (lvl.isBoundary) {
            ctx.fillStyle = SOL.orange;
            ctx.font = 'bold 11px monospace';
            const midX = (pts[0][0] + pts[1][0]) / 2;
            const midY = (pts[0][1] + pts[1][1]) / 2;
            ctx.fillText('P=0.5 (Hyperplane)', to2dX(midX) + 6, to2dY(midY) - 8);
          }
        }
        ctx.restore();
      });

      // Sample Data Clusters
      const c0 = [
        { x: -1.8, y: -1.2 }, { x: -1.4, y: -0.8 }, { x: -2.0, y: -0.3 },
        { x: -1.1, y: -1.6 }, { x: -0.8, y: -1.4 }, { x: -1.5, y: -1.9 }
      ];
      const c1 = [
        { x: 1.5, y: 1.2 }, { x: 1.8, y: 0.8 }, { x: 1.2, y: 1.7 },
        { x: 0.8, y: 1.5 }, { x: 2.1, y: 1.4 }, { x: 1.6, y: 2.0 }
      ];

      c0.forEach(p => {
        ctx.beginPath();
        ctx.arc(to2dX(p.x), to2dY(p.y), 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = SOL.blue; ctx.fill();
        ctx.strokeStyle = '#073642'; ctx.lineWidth = 1; ctx.stroke();
      });
      c1.forEach(p => {
        ctx.beginPath();
        ctx.arc(to2dX(p.x), to2dY(p.y), 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = SOL.orange; ctx.fill();
        ctx.strokeStyle = '#073642'; ctx.lineWidth = 1; ctx.stroke();
      });

      ctx.fillStyle = SOL.base01;
      ctx.font = '10px sans-serif';
      ctx.fillText('Solid Orange: Hyperplane Boundary (wᵀx + w₀ = 0)', 40, H - 12);
      ctx.fillText('Dashed: Parallel Probability Contours', 40, H - 24);
      ctx.restore();

      // Divider line
      ctx.strokeStyle = '#93a1a1';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(w2d, 15); ctx.lineTo(w2d, H - 15); ctx.stroke();

      // -------------------------------------------------------------------
      // RIGHT HALF: 3D Perspective Probability Surface P(Y=1 | x₁, x₂)
      // -------------------------------------------------------------------
      ctx.save();
      ctx.beginPath();
      ctx.rect(w2d, 0, w3d, H);
      ctx.clip();

      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText('3D Probability Surface: P(Y=1|x) = σ(wᵀx + w₀)', w2d + 20, 22);
      ctx.fillStyle = SOL.base01;
      ctx.font = '10px sans-serif';
      ctx.fillText('(Drag mouse to rotate 3D perspective)', w2d + 20, 36);

      const cx3d = w2d + w3d * 0.5;
      const cy3d = H * 0.58;
      const scale3d = 42;

      function project3D(x, y, z) {
        const cosZ = Math.cos(rotZ), sinZ = Math.sin(rotZ);
        const cosX = Math.cos(rotX), sinX = Math.sin(rotX);

        const x1 = x * cosZ - y * sinZ;
        const y1 = x * sinZ + y * cosZ;
        const z1 = z;

        const x2 = x1;
        const y2 = y1 * cosX - z1 * sinX;
        const z2 = y1 * sinX + z1 * cosX;

        return {
          px: cx3d + x2 * scale3d,
          py: cy3d - z2 * scale3d,
          depth: y2
        };
      }

      // Base bounding box
      ctx.strokeStyle = '#eee8d5';
      ctx.lineWidth = 1;
      const boxCorners = [
        project3D(-2.5, -2.5, 0), project3D(2.5, -2.5, 0),
        project3D(2.5, 2.5, 0), project3D(-2.5, 2.5, 0),
        project3D(-2.5, -2.5, 1), project3D(2.5, -2.5, 1),
        project3D(2.5, 2.5, 1), project3D(-2.5, 2.5, 1)
      ];

      ctx.beginPath();
      ctx.moveTo(boxCorners[0].px, boxCorners[0].py);
      ctx.lineTo(boxCorners[1].px, boxCorners[1].py);
      ctx.lineTo(boxCorners[2].px, boxCorners[2].py);
      ctx.lineTo(boxCorners[3].px, boxCorners[3].py);
      ctx.closePath();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(boxCorners[4].px, boxCorners[4].py);
      ctx.lineTo(boxCorners[5].px, boxCorners[5].py);
      ctx.lineTo(boxCorners[6].px, boxCorners[6].py);
      ctx.lineTo(boxCorners[7].px, boxCorners[7].py);
      ctx.closePath();
      ctx.stroke();

      for (let i = 0; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(boxCorners[i].px, boxCorners[i].py);
        ctx.lineTo(boxCorners[i + 4].px, boxCorners[i + 4].py);
        ctx.stroke();
      }

      // Cut plane at P = 0.5
      const pHalfCorners = [
        project3D(-2.5, -2.5, 0.5), project3D(2.5, -2.5, 0.5),
        project3D(2.5, 2.5, 0.5), project3D(-2.5, 2.5, 0.5)
      ];
      ctx.fillStyle = 'rgba(203, 75, 22, 0.08)';
      ctx.beginPath();
      ctx.moveTo(pHalfCorners[0].px, pHalfCorners[0].py);
      ctx.lineTo(pHalfCorners[1].px, pHalfCorners[1].py);
      ctx.lineTo(pHalfCorners[2].px, pHalfCorners[2].py);
      ctx.lineTo(pHalfCorners[3].px, pHalfCorners[3].py);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = 'rgba(203, 75, 22, 0.4)';
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Generate 3D Sigmoidal Surface Mesh
      const gridN = 22;
      const quads = [];

      for (let i = 0; i < gridN; i++) {
        for (let j = 0; j < gridN; j++) {
          const xA = -2.5 + (i / gridN) * 5.0;
          const xB = -2.5 + ((i + 1) / gridN) * 5.0;
          const yA = -2.5 + (j / gridN) * 5.0;
          const yB = -2.5 + ((j + 1) / gridN) * 5.0;

          function sig(x, y) {
            const z = w1 * x + w2 * y + w0;
            return 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));
          }

          const z00 = sig(xA, yA);
          const z10 = sig(xB, yA);
          const z11 = sig(xB, yB);
          const z01 = sig(xA, yB);

          const p00 = project3D(xA, yA, z00);
          const p10 = project3D(xB, yA, z10);
          const p11 = project3D(xB, yB, z11);
          const p01 = project3D(xA, yB, z01);

          const avgDepth = (p00.depth + p10.depth + p11.depth + p01.depth) / 4;
          const avgZ = (z00 + z10 + z11 + z01) / 4;

          quads.push({ p00, p10, p11, p01, depth: avgDepth, z: avgZ });
        }
      }

      quads.sort((a, b) => a.depth - b.depth);

      quads.forEach(q => {
        ctx.beginPath();
        ctx.moveTo(q.p00.px, q.p00.py);
        ctx.lineTo(q.p10.px, q.p10.py);
        ctx.lineTo(q.p11.px, q.p11.py);
        ctx.lineTo(q.p01.px, q.p01.py);
        ctx.closePath();

        const alpha = 0.65;
        if (q.z < 0.45) {
          ctx.fillStyle = `rgba(38, 139, 210, ${alpha})`;
        } else if (q.z > 0.55) {
          ctx.fillStyle = `rgba(203, 75, 22, ${alpha})`;
        } else {
          ctx.fillStyle = `rgba(181, 137, 0, ${alpha})`;
        }
        ctx.fill();
        ctx.strokeStyle = 'rgba(7, 54, 66, 0.25)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      });

      const labelX = project3D(2.8, 0, 0);
      const labelY = project3D(0, 2.8, 0);
      const labelZ = project3D(0, 0, 1.15);
      ctx.fillStyle = SOL.base02;
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText('x₁', labelX.px, labelX.py);
      ctx.fillText('x₂', labelY.px, labelY.py);
      ctx.fillText('P(Y=1)', labelZ.px - 20, labelZ.py - 6);

      ctx.restore();

      if (formulaBadge) {
        formulaBadge.innerHTML = `Boundary: <b>${w1.toFixed(2)} x₁ ${w2 >= 0 ? '+' : '-'} ${Math.abs(w2).toFixed(2)} x₂ ${w0 >= 0 ? '+' : '-'} ${Math.abs(w0).toFixed(2)} = 0</b> &nbsp;|&nbsp; Surface: <b>P(Y=1|x) = &sigma;(wᵀx + w₀)</b>`;
      }
    }

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      if (x > canvas.width * 0.45 / (window.devicePixelRatio || 1)) {
        isDragging3D = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging3D) return;
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;

      rotZ += dx * 0.01;
      rotX += dy * 0.01;
      rotX = Math.max(0.1, Math.min(Math.PI * 0.48, rotX));
      draw();
    });

    window.addEventListener('mouseup', () => { isDragging3D = false; });

    if (slopeSlider) {
      slopeSlider.addEventListener('input', (e) => {
        slopeAngle = parseFloat(e.target.value);
        draw();
      });
    }
    if (biasSlider) {
      biasSlider.addEventListener('input', (e) => {
        w0 = parseFloat(e.target.value);
        draw();
      });
    }

    // Presets
    const btnDef = document.getElementById('geom-btn-default');
    const btnVert = document.getElementById('geom-btn-vertical');
    const btnHoriz = document.getElementById('geom-btn-horizontal');
    const btnSteep = document.getElementById('geom-btn-steep');

    function resetPresetButtons() {
      [btnDef, btnVert, btnHoriz, btnSteep].forEach(b => {
        if (b) { b.style.fontWeight = 'normal'; b.style.background = '#fdf6e3'; }
      });
    }

    if (btnDef) btnDef.addEventListener('click', () => {
      slopeAngle = 0.7; normW = 1.6; w0 = -0.2;
      if (slopeSlider) slopeSlider.value = 0.7;
      if (biasSlider) biasSlider.value = -0.2;
      resetPresetButtons();
      btnDef.style.fontWeight = 'bold'; btnDef.style.background = '#eee8d5';
      draw();
    });
    if (btnVert) btnVert.addEventListener('click', () => {
      slopeAngle = 0.0; normW = 1.8; w0 = 0.0;
      if (slopeSlider) slopeSlider.value = 0.0;
      if (biasSlider) biasSlider.value = 0.0;
      resetPresetButtons();
      btnVert.style.fontWeight = 'bold'; btnVert.style.background = '#eee8d5';
      draw();
    });
    if (btnHoriz) btnHoriz.addEventListener('click', () => {
      slopeAngle = Math.PI / 2; normW = 1.8; w0 = 0.0;
      if (slopeSlider) slopeSlider.value = 1.57;
      if (biasSlider) biasSlider.value = 0.0;
      resetPresetButtons();
      btnHoriz.style.fontWeight = 'bold'; btnHoriz.style.background = '#eee8d5';
      draw();
    });
    if (btnSteep) btnSteep.addEventListener('click', () => {
      slopeAngle = 0.7; normW = 3.8; w0 = -0.2;
      resetPresetButtons();
      btnSteep.style.fontWeight = 'bold'; btnSteep.style.background = '#eee8d5';
      draw();
    });

    window.addEventListener('resize', draw);
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) draw();
      });
    }
    draw();
  }

  // =========================================================================
  // DEMO 4: Interactive Softmax & Temperature Explorer (Multi-Sample Segmentation)
  // =========================================================================
  function initSoftmaxDemo() {
    const canvas = document.getElementById('softmax-canvas');
    if (!canvas) return;

    let temperature = 1.0;
    let mode = '1000'; // '1000' or '8'
    let currentSampleIdx = 0;

    const K1000 = 1000;

    // Define multiple samples replicating the 3 SVGs + rivalry + random
    const samples = [
      {
        id: 0,
        name: 'Sample 1 (#720 Target)',
        desc: 'Replicating laplace_softmax0: strong peak at #720, rival at #650',
        spikeIdx: 720, spikeVal: 1.0, rivalIdx: 650, rivalVal: 0.75, seed: 101
      },
      {
        id: 1,
        name: 'Sample 2 (#525 Target)',
        desc: 'Replicating laplace_softmax1: isolated sharp signal at #525',
        spikeIdx: 525, spikeVal: 1.0, rivalIdx: -1, rivalVal: 0, seed: 202
      },
      {
        id: 2,
        name: 'Sample 3 (#950 Target)',
        desc: 'Replicating laplace_softmax2: isolated signal shifted to #950',
        spikeIdx: 950, spikeVal: 0.95, rivalIdx: -1, rivalVal: 0, seed: 303
      },
      {
        id: 3,
        name: 'Sample 4 (Two Rivals)',
        desc: 'Multi-modal rivalry: identical peaks at #280 and #750',
        spikeIdx: 280, spikeVal: 0.95, rivalIdx: 750, rivalVal: 0.94, seed: 404
      }
    ];

    let currentLogits1000 = new Float32Array(K1000);

    function buildSample(sampleObj) {
      let seed = sampleObj.seed;
      function rnd() {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      }
      function randn() {
        const u = rnd() || 0.0001;
        const v = rnd() || 0.0001;
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
      }

      for (let i = 0; i < K1000; i++) {
        currentLogits1000[i] = randn() * 0.25;
      }
      if (sampleObj.spikeIdx >= 0 && sampleObj.spikeIdx < K1000) {
        currentLogits1000[sampleObj.spikeIdx] = sampleObj.spikeVal;
      }
      if (sampleObj.rivalIdx >= 0 && sampleObj.rivalIdx < K1000) {
        currentLogits1000[sampleObj.rivalIdx] = sampleObj.rivalVal;
      }
    }

    buildSample(samples[0]);

    // Mode 2: 8 Classes state
    let logits8 = [0.8, -0.4, 2.1, 0.3, -1.5, 1.4, -0.2, 0.9];
    const classLabels8 = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'];

    // DOM Controls
    const tempSlider = document.getElementById('softmax-temp-slider');
    const tempValSpan = document.getElementById('softmax-temp-val');
    const maxProbSpan = document.getElementById('softmax-max-p');
    const entropySpan = document.getElementById('softmax-entropy');
    const sampleLabelSpan = document.getElementById('softmax-sample-label');
    const winnerSpan = document.getElementById('softmax-winner-idx');

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const T = Math.max(0.02, temperature);

      if (mode === '1000') {
        const topH = H * 0.44;
        const botH = H * 0.44;
        const splitY = H * 0.48;

        let maxL = -Infinity;
        let maxIdx = 0;
        for (let i = 0; i < K1000; i++) {
          if (currentLogits1000[i] > maxL) { maxL = currentLogits1000[i]; maxIdx = i; }
        }

        let sumExp = 0;
        const probs = new Float32Array(K1000);
        for (let i = 0; i < K1000; i++) {
          probs[i] = Math.exp((currentLogits1000[i] - maxL) / T);
          sumExp += probs[i];
        }
        let entropy = 0;
        let maxP = 0;
        for (let i = 0; i < K1000; i++) {
          probs[i] /= sumExp;
          if (probs[i] > maxP) maxP = probs[i];
          if (probs[i] > 1e-9) {
            entropy -= probs[i] * Math.log2(probs[i]);
          }
        }

        if (tempValSpan) tempValSpan.textContent = T.toFixed(2);
        if (maxProbSpan) maxProbSpan.textContent = (maxP * 100).toFixed(1) + '%';
        if (entropySpan) entropySpan.textContent = `${entropy.toFixed(2)} bits (max ${Math.log2(1000).toFixed(1)})`;
        if (winnerSpan) winnerSpan.textContent = `#${maxIdx} (z=${maxL.toFixed(2)})`;
        if (sampleLabelSpan) {
          const sName = (currentSampleIdx < samples.length) ? samples[currentSampleIdx].name : 'Custom Random Draw';
          sampleLabelSpan.textContent = sName;
        }

        // TOP PANE: Input Logits z_i
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 12px sans-serif';
        const sampleDesc = (currentSampleIdx < samples.length) ? ` &mdash; ${samples[currentSampleIdx].desc}` : '';
        ctx.fillText(`Input Signal zᵢ (Sample ${currentSampleIdx + 1}${sampleDesc})`, 20, 18);

        const padX = 40, pw = W - 60;
        ctx.strokeStyle = '#eee8d5';
        ctx.lineWidth = 1;
        ctx.strokeRect(padX, 24, pw, topH - 28);

        const zeroY = 24 + (topH - 28) * 0.65;
        ctx.strokeStyle = '#93a1a1';
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(padX, zeroY); ctx.lineTo(padX + pw, zeroY); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#93a1a1';
        ctx.font = '10px monospace';
        ctx.fillText('z = 0', padX - 30, zeroY + 3);

        for (let i = 0; i < K1000; i++) {
          const x = padX + (i / K1000) * pw;
          const y = zeroY - (currentLogits1000[i] / 1.3) * (topH * 0.45);
          ctx.beginPath();
          const isTarget = (i === maxIdx);
          ctx.arc(x, y, isTarget ? 3.5 : 1.5, 0, 2 * Math.PI);
          ctx.fillStyle = isTarget ? SOL.orange : SOL.blue;
          ctx.fill();
        }

        // BOTTOM PANE: Softmax Probabilities p_i
        const botTop = splitY + 12;
        ctx.fillStyle = SOL.base02;
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText(`Softmax Segmentation Probabilities pᵢ = exp(zᵢ / T) / ∑ exp(zⱼ / T)    [T = ${T.toFixed(2)}]`, 20, botTop);

        ctx.strokeStyle = '#eee8d5';
        ctx.strokeRect(padX, botTop + 8, pw, botH - 12);

        const probBaseY = botTop + 8 + (botH - 12);

        for (let i = 0; i < K1000; i++) {
          const p = probs[i];
          if (p < 0.0005 && i !== maxIdx) continue;
          const x = padX + (i / K1000) * pw;
          const barH = p * (botH - 18);
          const y = probBaseY - barH;

          ctx.beginPath();
          ctx.moveTo(x, probBaseY);
          ctx.lineTo(x, y);
          ctx.strokeStyle = (i === maxIdx) ? SOL.orange : 'rgba(203, 75, 22, 0.4)';
          ctx.lineWidth = (i === maxIdx) ? 2.5 : 1.0;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(x, y, (i === maxIdx) ? 4.0 : 1.5, 0, 2 * Math.PI);
          ctx.fillStyle = (i === maxIdx) ? SOL.orange : SOL.yellow;
          ctx.fill();

          if (i === maxIdx) {
            ctx.fillStyle = SOL.orange;
            ctx.font = 'bold 12px monospace';
            ctx.fillText(`p_${i} = ${(p * 100).toFixed(1)}%`, x - 30, y - 8);
          }
        }

      } else {
        // Mode 2: 8 Classes
        const K8 = logits8.length;
        let maxL = -Infinity;
        let maxIdx = 0;
        logits8.forEach((z, i) => {
          if (z > maxL) { maxL = z; maxIdx = i; }
        });

        let sumExp = 0;
        const probs = logits8.map(z => {
          const e = Math.exp((z - maxL) / T);
          sumExp += e;
          return e;
        });
        const pNorm = probs.map(e => e / sumExp);

        let entropy = 0;
        pNorm.forEach(p => {
          if (p > 1e-9) entropy -= p * Math.log2(p);
        });

        if (tempValSpan) tempValSpan.textContent = T.toFixed(2);
        if (maxProbSpan) maxProbSpan.textContent = (pNorm[maxIdx] * 100).toFixed(1) + '%';
        if (entropySpan) entropySpan.textContent = `${entropy.toFixed(2)} bits (max ${Math.log2(8).toFixed(1)})`;
        if (winnerSpan) winnerSpan.textContent = `${classLabels8[maxIdx]} (z=${maxL.toFixed(2)})`;
        if (sampleLabelSpan) sampleLabelSpan.textContent = '8-Class Model';

        const colW = (W - 80) / K8;
        for (let i = 0; i < K8; i++) {
          const cx = 50 + i * colW + colW * 0.5;

          const zeroLogitY = 120;
          const z = logits8[i];
          const logitH = (z / 3.0) * 80;

          ctx.fillStyle = (i === maxIdx) ? SOL.blue : '#93a1a1';
          ctx.fillRect(cx - 16, (logitH >= 0) ? zeroLogitY - logitH : zeroLogitY, 32, Math.abs(logitH));
          ctx.strokeStyle = '#073642';
          ctx.lineWidth = 1;
          ctx.strokeRect(cx - 16, (logitH >= 0) ? zeroLogitY - logitH : zeroLogitY, 32, Math.abs(logitH));

          ctx.fillStyle = SOL.base02;
          ctx.font = 'bold 12px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`z=${z.toFixed(1)}`, cx, zeroLogitY - logitH - 6);
          ctx.fillText(classLabels8[i], cx, zeroLogitY + 22);

          const probBaseY = H - 35;
          const p = pNorm[i];
          const pH = p * 130;

          ctx.fillStyle = (i === maxIdx) ? SOL.orange : SOL.yellow;
          ctx.fillRect(cx - 16, probBaseY - pH, 32, pH);
          ctx.strokeStyle = '#073642';
          ctx.strokeRect(cx - 16, probBaseY - pH, 32, pH);

          ctx.fillStyle = SOL.base02;
          ctx.font = 'bold 11px monospace';
          ctx.fillText(`${(p * 100).toFixed(1)}%`, cx, probBaseY - pH - 6);
          ctx.textAlign = 'left';
        }

        ctx.fillStyle = SOL.base01;
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText('Raw Logits zᵢ:', 20, 25);
        ctx.fillText(`Softmax Probabilities P(Y=i | z)  [T = ${T.toFixed(2)}]:`, 20, 185);
      }
    }

    if (tempSlider) {
      tempSlider.addEventListener('input', (e) => {
        temperature = parseFloat(e.target.value);
        draw();
      });
    }

    // Quick Temperature buttons
    const btnCold = document.getElementById('temp-btn-cold');
    const btnStd = document.getElementById('temp-btn-std');
    const btnHot = document.getElementById('temp-btn-hot');

    if (btnCold) btnCold.addEventListener('click', () => {
      temperature = 0.15;
      if (tempSlider) tempSlider.value = 0.15;
      draw();
    });
    if (btnStd) btnStd.addEventListener('click', () => {
      temperature = 1.0;
      if (tempSlider) tempSlider.value = 1.0;
      draw();
    });
    if (btnHot) btnHot.addEventListener('click', () => {
      temperature = 3.5;
      if (tempSlider) tempSlider.value = 3.5;
      draw();
    });

    // Mode Toggle
    const btnMode1000 = document.getElementById('softmax-mode-1000');
    const btnMode8 = document.getElementById('softmax-mode-8');

    if (btnMode1000) btnMode1000.addEventListener('click', () => {
      mode = '1000';
      btnMode1000.style.fontWeight = 'bold'; btnMode1000.style.background = '#eee8d5';
      btnMode8.style.fontWeight = 'normal'; btnMode8.style.background = '#fdf6e3';
      draw();
    });
    if (btnMode8) btnMode8.addEventListener('click', () => {
      mode = '8';
      btnMode8.style.fontWeight = 'bold'; btnMode8.style.background = '#eee8d5';
      btnMode1000.style.fontWeight = 'normal'; btnMode1000.style.background = '#fdf6e3';
      draw();
    });

    // Sample Selector Buttons
    function selectSample(idx) {
      mode = '1000';
      if (btnMode1000) { btnMode1000.style.fontWeight = 'bold'; btnMode1000.style.background = '#eee8d5'; }
      if (btnMode8) { btnMode8.style.fontWeight = 'normal'; btnMode8.style.background = '#fdf6e3'; }

      document.querySelectorAll('.softmax-sample-btn').forEach(b => {
        b.style.fontWeight = 'normal'; b.style.background = '#fdf6e3';
      });

      if (idx >= 0 && idx < samples.length) {
        currentSampleIdx = idx;
        buildSample(samples[idx]);
        const activeBtn = document.getElementById(`softmax-sample-${idx + 1}`);
        if (activeBtn) { activeBtn.style.fontWeight = 'bold'; activeBtn.style.background = '#eee8d5'; }
      }
      draw();
    }

    const btnS1 = document.getElementById('softmax-sample-1');
    const btnS2 = document.getElementById('softmax-sample-2');
    const btnS3 = document.getElementById('softmax-sample-3');
    const btnS4 = document.getElementById('softmax-sample-4');
    const btnPrev = document.getElementById('softmax-prev-sample');
    const btnNext = document.getElementById('softmax-next-sample');
    const btnRand = document.getElementById('softmax-sample-rand');

    if (btnS1) btnS1.addEventListener('click', () => selectSample(0));
    if (btnS2) btnS2.addEventListener('click', () => selectSample(1));
    if (btnS3) btnS3.addEventListener('click', () => selectSample(2));
    if (btnS4) btnS4.addEventListener('click', () => selectSample(3));

    if (btnPrev) btnPrev.addEventListener('click', () => {
      const nextIdx = (currentSampleIdx - 1 + samples.length) % samples.length;
      selectSample(nextIdx);
    });
    if (btnNext) btnNext.addEventListener('click', () => {
      const nextIdx = (currentSampleIdx + 1) % samples.length;
      selectSample(nextIdx);
    });

    if (btnRand) btnRand.addEventListener('click', () => {
      mode = '1000';
      currentSampleIdx = samples.length;
      const randTarget = Math.floor(Math.random() * 900) + 50;
      buildSample({
        id: 99,
        name: `Random Sample (#${randTarget})`,
        desc: `Random target peak generated at index #${randTarget}`,
        spikeIdx: randTarget,
        spikeVal: 0.95 + Math.random() * 0.15,
        rivalIdx: Math.random() < 0.5 ? Math.floor(Math.random() * 900) + 50 : -1,
        rivalVal: 0.7 + Math.random() * 0.15,
        seed: Math.floor(Math.random() * 100000)
      });
      document.querySelectorAll('.softmax-sample-btn').forEach(b => {
        b.style.fontWeight = 'normal'; b.style.background = '#fdf6e3';
      });
      if (btnRand) { btnRand.style.fontWeight = 'bold'; btnRand.style.background = '#eee8d5'; }
      draw();
    });

    window.addEventListener('resize', draw);
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) draw();
      });
    }
    draw();
  }

  // =========================================================================
  // DEMO 5: Interactive Taylor Expansion & Newton-Raphson Explorer (Slide 475)
  // =========================================================================
  function initTaylorDemo() {
    const canvas = document.getElementById('taylor-canvas');
    if (!canvas) return;

    let funcType = 'logistic';
    let expansionPoint = 0.5;
    let order = 2;

    const sliderA = document.getElementById('taylor-a-slider');
    const valA = document.getElementById('taylor-a-val');
    const formulaBadge = document.getElementById('taylor-formula-badge');
    const newtonStepBadge = document.getElementById('taylor-newton-badge');

    const funcs = {
      logistic: {
        name: 'Logistic Loss: ℓ(w) = ln(1 + e⁻ʷ)',
        f: (w) => Math.log(1.0 + Math.exp(-w)),
        df: (w) => -1.0 / (1.0 + Math.exp(w)),
        d2f: (w) => {
          const s = 1.0 / (1.0 + Math.exp(-w));
          return s * (1.0 - s);
        },
        d3f: (w) => {
          const s = 1.0 / (1.0 + Math.exp(-w));
          return s * (1.0 - s) * (1.0 - 2.0 * s);
        },
        d4f: (w) => {
          const s = 1.0 / (1.0 + Math.exp(-w));
          return s * (1.0 - s) * (1.0 - 6.0 * s + 6.0 * s * s);
        },
        d5f: (w) => {
          const s = 1.0 / (1.0 + Math.exp(-w));
          return s * (1.0 - s) * (1.0 - 2.0 * s) * (1.0 - 12.0 * s + 12.0 * s * s);
        },
        minX: -4.0, maxX: 4.0, minY: -0.5, maxY: 4.5
      },
      sin: {
        name: 'Sine Function: f(x) = sin(x)',
        f: (x) => Math.sin(x),
        df: (x) => Math.cos(x),
        d2f: (x) => -Math.sin(x),
        d3f: (x) => -Math.cos(x),
        d4f: (x) => Math.sin(x),
        d5f: (x) => Math.cos(x),
        minX: -6.0, maxX: 6.0, minY: -2.0, maxY: 2.0
      },
      quartic: {
        name: 'Quartic Well: f(x) = ¼ x⁴ - ½ x² + 0.2 x + 1',
        f: (x) => 0.25 * x ** 4 - 0.5 * x ** 2 + 0.2 * x + 1.0,
        df: (x) => x ** 3 - x + 0.2,
        d2f: (x) => 3.0 * x ** 2 - 1.0,
        d3f: (x) => 6.0 * x,
        d4f: (x) => 6.0,
        d5f: (x) => 0.0,
        minX: -2.5, maxX: 2.5, minY: -0.5, maxY: 3.5
      }
    };

    function factorial(k) {
      if (k <= 1) return 1;
      let res = 1;
      for (let i = 2; i <= k; i++) res *= i;
      return res;
    }

    function draw() {
      const { ctx, width: W, height: H } = setupHiDPI(canvas);
      ctx.clearRect(0, 0, W, H);

      const fn = funcs[funcType];
      const a = expansionPoint;

      function toX(x) { return ((x - fn.minX) / (fn.maxX - fn.minX)) * W; }
      function toY(y) { return H - ((y - fn.minY) / (fn.maxY - fn.minY)) * H; }

      const f0 = fn.f(a);
      const f1 = fn.df(a);
      const f2 = fn.d2f(a);
      const f3 = fn.d3f(a);
      const f4 = fn.d4f(a);
      const f5 = fn.d5f(a);

      const derivs = [f0, f1, f2, f3, f4, f5];

      function taylorPoly(x) {
        let val = 0;
        const dx = x - a;
        for (let k = 0; k <= order; k++) {
          val += (derivs[k] / factorial(k)) * Math.pow(dx, k);
        }
        return val;
      }

      let newtonX = null;
      let newtonY = null;
      if (Math.abs(f2) > 1e-4) {
        newtonX = a - f1 / f2;
        newtonY = taylorPoly(newtonX);
      }

      if (valA) valA.textContent = a.toFixed(2);
      if (formulaBadge) {
        let formStr = `T_${order}(x) = ${f0.toFixed(2)}`;
        if (order >= 1) formStr += ` ${f1 >= 0 ? '+' : '-'} ${Math.abs(f1).toFixed(2)}(x &minus; ${a.toFixed(2)})`;
        if (order >= 2) formStr += ` ${f2 >= 0 ? '+' : '-'} ${(Math.abs(f2) / 2).toFixed(2)}(x &minus; ${a.toFixed(2)})²`;
        if (order >= 3) formStr += ` + ...`;
        formulaBadge.innerHTML = formStr;
      }

      if (newtonStepBadge) {
        if (order === 1) {
          newtonStepBadge.innerHTML = `<span style="color:#268bd2; font-weight:bold;"><i class="fa fa-arrow-right"></i> Order 1 (Tangent Line):</span> Gradient Descent direction &minus;f'(a) = ${( -f1 ).toFixed(2)}`;
        } else if (order === 2 && newtonX !== null) {
          const delta = -f1 / f2;
          newtonStepBadge.innerHTML = `<span style="color:#6c71c4; font-weight:bold;"><i class="fa fa-crosshairs"></i> Order 2 (Newton-Raphson Parabola):</span> Minimum vertex at x = a &minus; f'/f'' = <b>${newtonX.toFixed(2)}</b> (Step &Delta; = ${delta.toFixed(2)})`;
        } else {
          newtonStepBadge.innerHTML = `<span style="color:#859900;">Higher-order polynomial (n=${order}) expanding curvature radius.</span>`;
        }
      }

      ctx.strokeStyle = '#eee8d5';
      ctx.lineWidth = 1;
      for (let x = Math.ceil(fn.minX); x <= fn.maxX; x += 1) {
        ctx.beginPath(); ctx.moveTo(toX(x), 0); ctx.lineTo(toX(x), H); ctx.stroke();
      }
      for (let y = Math.ceil(fn.minY); y <= fn.maxY; y += 1) {
        ctx.beginPath(); ctx.moveTo(0, toY(y)); ctx.lineTo(W, toY(y)); ctx.stroke();
      }

      ctx.strokeStyle = '#93a1a1';
      ctx.lineWidth = 1.5;
      if (fn.minY <= 0 && fn.maxY >= 0) {
        ctx.beginPath(); ctx.moveTo(0, toY(0)); ctx.lineTo(W, toY(0)); ctx.stroke();
      }
      if (fn.minX <= 0 && fn.maxX >= 0) {
        ctx.beginPath(); ctx.moveTo(toX(0), 0); ctx.lineTo(toX(0), H); ctx.stroke();
      }

      ctx.strokeStyle = SOL.base02;
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      for (let px = 0; px <= W; px += 2) {
        const x = fn.minX + (px / W) * (fn.maxX - fn.minX);
        const y = fn.f(x);
        if (px === 0) ctx.moveTo(px, toY(y));
        else ctx.lineTo(px, toY(y));
      }
      ctx.stroke();

      ctx.strokeStyle = SOL.red;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      let started = false;
      for (let px = 0; px <= W; px += 2) {
        const x = fn.minX + (px / W) * (fn.maxX - fn.minX);
        const y = taylorPoly(x);
        if (y >= fn.minY - 2 && y <= fn.maxY + 2) {
          if (!started) { ctx.moveTo(px, toY(y)); started = true; }
          else { ctx.lineTo(px, toY(y)); }
        } else {
          started = false;
        }
      }
      ctx.stroke();

      if (order === 2 && newtonX !== null && newtonX >= fn.minX && newtonX <= fn.maxX) {
        const vx = toX(newtonX);
        const vy = toY(newtonY);

        ctx.strokeStyle = SOL.violet;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(vx, toY(fn.minY));
        ctx.lineTo(vx, vy);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = SOL.violet;
        ctx.beginPath();
        ctx.moveTo(vx, vy - 7);
        ctx.lineTo(vx + 7, vy);
        ctx.lineTo(vx, vy + 7);
        ctx.lineTo(vx - 7, vy);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = SOL.violet;
        ctx.font = 'bold 11px monospace';
        ctx.fillText(`Newton Step x = ${newtonX.toFixed(2)}`, vx + 8, vy + 4);
      }

      const ax = toX(a);
      const ay = toY(f0);

      ctx.strokeStyle = SOL.orange;
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ax, toY(fn.minY));
      ctx.lineTo(ax, ay);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(ax, ay, 6, 0, 2 * Math.PI);
      ctx.fillStyle = SOL.orange;
      ctx.fill();
      ctx.strokeStyle = '#073642';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = SOL.orange;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(`Center a = ${a.toFixed(2)}`, ax + 8, ay - 8);
    }

    if (sliderA) {
      sliderA.addEventListener('input', (e) => {
        expansionPoint = parseFloat(e.target.value);
        draw();
      });
    }

    const btnLogLoss = document.getElementById('taylor-fn-logloss');
    const btnSin = document.getElementById('taylor-fn-sin');
    const btnQuartic = document.getElementById('taylor-fn-quartic');

    function setActiveFnBtn(activeBtn) {
      [btnLogLoss, btnSin, btnQuartic].forEach(b => {
        if (b) { b.style.fontWeight = 'normal'; b.style.background = '#fdf6e3'; }
      });
      if (activeBtn) { activeBtn.style.fontWeight = 'bold'; activeBtn.style.background = '#eee8d5'; }
    }

    if (btnLogLoss) btnLogLoss.addEventListener('click', () => {
      funcType = 'logistic';
      expansionPoint = 0.5;
      if (sliderA) { sliderA.min = -3.0; sliderA.max = 3.0; sliderA.value = 0.5; }
      setActiveFnBtn(btnLogLoss);
      draw();
    });
    if (btnSin) btnSin.addEventListener('click', () => {
      funcType = 'sin';
      expansionPoint = 0.0;
      if (sliderA) { sliderA.min = -4.0; sliderA.max = 4.0; sliderA.value = 0.0; }
      setActiveFnBtn(btnSin);
      draw();
    });
    if (btnQuartic) btnQuartic.addEventListener('click', () => {
      funcType = 'quartic';
      expansionPoint = 0.8;
      if (sliderA) { sliderA.min = -2.0; sliderA.max = 2.0; sliderA.value = 0.8; }
      setActiveFnBtn(btnQuartic);
      draw();
    });

    document.querySelectorAll('.taylor-order-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        order = parseInt(e.target.getAttribute('data-order'), 10);
        document.querySelectorAll('.taylor-order-btn').forEach(b => {
          b.style.fontWeight = 'normal'; b.style.background = '#fdf6e3';
        });
        e.target.style.fontWeight = 'bold';
        e.target.style.background = '#eee8d5';
        draw();
      });
    });

    window.addEventListener('resize', draw);
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) draw();
      });
    }
    draw();
  }

  // =========================================================================
  // Initialize all demonstrations when DOM is ready
  // =========================================================================
  function initAll() {
    initGnbLrDemo();
    init1dLinearVsLogisticDemo();
    initLrGeometryDemo();
    initSoftmaxDemo();
    initTaylorDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

})();
