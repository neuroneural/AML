/**
 * Interactive Demos for CS8850 Lecture 07: Curse of Dimensionality
 * Includes:
 * 1. Hyper-Watermelon Explorer (Step-by-step & Continuous)
 * 2. Hamming's 4x4 Box Sphere Packing Paradox Explorer
 * 3. Lp Minkowski Metric Unit Ball & Distance Explorer
 */

(function() {
  'use strict';

  // Helper to handle Retina / high DPR displays
  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || canvas.width;
    const h = rect.height || canvas.height;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: w, height: h };
  }

  /* =========================================================================
   * 1. HYPER-WATERMELON EXPLORER
   * ========================================================================= */
  function initWatermelonDemo() {
    const canvas = document.getElementById('wm-canvas');
    if (!canvas) return;

    let step = 0; // 0: Realistic sliced watermelon, 1: 1D Core plug, 2: 1D Bar, 3: 1D 15% text, 4: 2D Disc, 5: 2D 28% text, 6: 3D Sphere 39%, 7: High-D Curve
    let eps = 0.15; // rind thickness fraction
    let userDim = 10; // dimension in high-D mode

    const epsSlider = document.getElementById('wm-eps-slider');
    const epsVal = document.getElementById('wm-eps-val');
    const dimSlider = document.getElementById('wm-dim-slider');
    const dimVal = document.getElementById('wm-dim-val');
    const prevBtn = document.getElementById('wm-prev-btn');
    const nextBtn = document.getElementById('wm-next-btn');
    const resetBtn = document.getElementById('wm-reset-btn');
    const stepLabel = document.getElementById('wm-step-label');
    const elFleshPct = document.getElementById('wm-flesh-pct');
    const elRindPct = document.getElementById('wm-rind-pct');
    const elDimDisplay = document.getElementById('wm-dim-display');
    const elEpsDisplay = document.getElementById('wm-eps-display');
    const dimControlGroup = document.getElementById('wm-dim-control-group');

    // Preset buttons
    const btnReal = document.getElementById('wm-btn-real');
    const btn1D = document.getElementById('wm-btn-1d');
    const btn2D = document.getElementById('wm-btn-2d');
    const btn3D = document.getElementById('wm-btn-3d');
    const btnHighD = document.getElementById('wm-btn-highd');

    const stepTitles = [
      "Step 1/8: Real Watermelon: Sliced cross-section",
      "Step 2/8: 1D Core Plug: Extracting a column",
      "Step 3/8: 1D Strip: Rind on ends (15% of radius)",
      "Step 4/8: 1D Fraction: V_rind ≈ 15% (one side)",
      "Step 5/8: 2D Disc: Concentric circles",
      "Step 6/8: 2D Fraction: Area of rind ≈ 28%",
      "Step 7/8: 3D Sphere: Volume of rind ≈ 39%",
      "Step 8/8: Hyper-D: Rind fraction = 1 - (1 - ε)ᴰ"
    ];

    function updateControls() {
      if (stepLabel) stepLabel.textContent = stepTitles[step];
      if (dimControlGroup) {
        dimControlGroup.style.display = (step === 7) ? 'flex' : 'none';
      }

      let curDim = 1;
      if (step <= 3) curDim = 1;
      else if (step <= 5) curDim = 2;
      else if (step === 6) curDim = 3;
      else curDim = userDim;

      const fleshFraction = Math.pow(Math.max(0, 1 - eps), curDim);
      const rindFraction = 1 - fleshFraction;

      if (elFleshPct) elFleshPct.textContent = (fleshFraction * 100).toFixed(1) + '%';
      if (elRindPct) elRindPct.textContent = (rindFraction * 100).toFixed(1) + '%';
      if (elDimDisplay) elDimDisplay.textContent = curDim;
      if (elEpsDisplay) elEpsDisplay.textContent = (eps * 100).toFixed(0) + '%';
    }

    function draw() {
      const { ctx, width, height } = setupCanvas(canvas);
      ctx.clearRect(0, 0, width, height);

      let curDim = (step <= 3) ? 1 : (step <= 5) ? 2 : (step === 6) ? 3 : userDim;

      // Draw background styling
      ctx.fillStyle = '#fdf6e3';
      ctx.fillRect(0, 0, width, height);

      if (step === 0) {
        drawRealMelon(ctx, width / 2, height / 2 - 5, 150, eps);
      } else if (step === 1) {
        drawRealMelon(ctx, width / 2, height / 2 - 5, 150, eps);
        drawCylinderCut(ctx, width / 2, height / 2 - 5, 150, eps);
      } else if (step === 2 || step === 3) {
        draw1DStrip(ctx, width / 2, height / 2 - 10, 560, 48, eps, step === 3);
      } else if (step === 4 || step === 5) {
        draw2DDisc(ctx, width / 2, height / 2 - 10, 145, eps, step === 5);
      } else if (step === 6) {
        draw3DSphere(ctx, width / 2, height / 2 - 10, 145, eps, true);
      } else if (step === 7) {
        drawHighDCurve(ctx, width, height, eps, userDim);
      }

      updateControls();
    }

    // Drawing helpers
    function drawRealMelon(ctx, cx, cy, r, epsFrac) {
      // Outer dark green striped rind
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = '#265828';
      ctx.fill();

      // Green rind stripes
      ctx.lineWidth = 14;
      ctx.strokeStyle = '#1b3f1c';
      for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 6) {
        ctx.beginPath();
        ctx.arc(cx, cy, r - 3, angle, angle + Math.PI / 14);
        ctx.stroke();
      }

      // Light green / white inner rind
      ctx.beginPath();
      ctx.arc(cx, cy, r * (1 - epsFrac * 0.4), 0, Math.PI * 2);
      ctx.fillStyle = '#b7e4b2';
      ctx.fill();

      // Red flesh
      ctx.beginPath();
      ctx.arc(cx, cy, r * (1 - epsFrac), 0, Math.PI * 2);
      ctx.fillStyle = '#e73838';
      ctx.fill();

      // Seeds
      const seedDist = r * (1 - epsFrac) * 0.65;
      const seedAngles = [0.3, 0.9, 1.5, 2.1, 2.7, 3.4, 4.0, 4.6, 5.3, 5.9];
      seedAngles.forEach(a => {
        const sx = cx + Math.cos(a) * seedDist;
        const sy = cy + Math.sin(a) * seedDist;
        drawSeed(ctx, sx, sy, a + Math.PI / 2);
      });

      // Dimension calipers
      ctx.strokeStyle = '#073642';
      ctx.lineWidth = 1.5;
      // Radius marker
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + r, cy);
      ctx.stroke();

      ctx.fillStyle = '#073642';
      ctx.font = 'bold 15px sans-serif';
      ctx.fillText('Radius R', cx + r * 0.4, cy - 8);

      // Rind marker
      ctx.strokeStyle = '#cb4b16';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx + r * (1 - epsFrac), cy + 18);
      ctx.lineTo(cx + r, cy + 18);
      ctx.stroke();
      ctx.fillStyle = '#cb4b16';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`Rind: ε = ${(epsFrac * 100).toFixed(0)}%`, cx + r * (1 - epsFrac) - 20, cy + 40);

      ctx.restore();
    }

    function drawSeed(ctx, x, y, rot) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rot);
      ctx.beginPath();
      ctx.ellipse(0, 0, 3.5, 7, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#22110c';
      ctx.fill();
      ctx.restore();
    }

    function drawCylinderCut(ctx, cx, cy, r, epsFrac) {
      ctx.save();
      ctx.strokeStyle = '#268bd2';
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 4]);
      // Draw rectangular plug cutting across
      const plugH = 34;
      ctx.strokeRect(cx - r, cy - plugH / 2, r * 2, plugH);

      ctx.fillStyle = 'rgba(38, 139, 210, 0.2)';
      ctx.fillRect(cx - r, cy - plugH / 2, r * 2, plugH);
      ctx.restore();
    }

    function draw1DStrip(ctx, cx, cy, barW, barH, epsFrac, showText) {
      ctx.save();
      const x0 = cx - barW / 2;
      const y0 = cy - barH / 2;
      const rindW = (barW / 2) * epsFrac;
      const fleshW = barW - 2 * rindW;

      // Outer border
      ctx.strokeStyle = '#073642';
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, barW, barH);

      // Left rind
      ctx.fillStyle = '#268b28';
      ctx.fillRect(x0, y0, rindW, barH);

      // Center flesh
      ctx.fillStyle = '#e73838';
      ctx.fillRect(x0 + rindW, y0, fleshW, barH);

      // Right rind
      ctx.fillStyle = '#268b28';
      ctx.fillRect(x0 + barW - rindW, y0, rindW, barH);

      // Ticks & Brackets
      ctx.fillStyle = '#073642';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';

      // Dimensions
      ctx.fillText('1D Cut / Column', cx, y0 - 35);
      ctx.font = 'bold 15px sans-serif';
      ctx.fillText(`Left Rind: ${(epsFrac * 100).toFixed(0)}%`, x0 + rindW / 2, y0 + barH + 28);
      ctx.fillText(`Flesh Core: ${((1 - 2 * epsFrac) * 100).toFixed(0)}%`, cx, y0 + barH + 28);
      ctx.fillText(`Right Rind: ${(epsFrac * 100).toFixed(0)}%`, x0 + barW - rindW / 2, y0 + barH + 28);

      if (showText) {
        ctx.fillStyle = '#cb4b16';
        ctx.font = 'bold 24px sans-serif';
        ctx.fillText(`Rind takes ${(epsFrac * 100).toFixed(0)}% of radius`, cx, y0 - 8);
        ctx.fillStyle = '#268bd2';
        ctx.font = '18px monospace';
        ctx.fillText(`V_rind / V_total = 1 - (1 - ε)¹ = ${(epsFrac * 100).toFixed(1)}% (one side)`, cx, y0 + barH + 65);
      }
      ctx.restore();
    }

    function draw2DDisc(ctx, cx, cy, r, epsFrac, showText) {
      ctx.save();
      // Outer Green Rind
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = '#268b28';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#073642';
      ctx.stroke();

      // Inner Red Flesh
      const rInner = r * (1 - epsFrac);
      ctx.beginPath();
      ctx.arc(cx, cy, rInner, 0, Math.PI * 2);
      ctx.fillStyle = '#e73838';
      ctx.fill();
      ctx.stroke();

      // Seeds
      for (let i = 0; i < 8; i++) {
        const a = (i * Math.PI * 2) / 8 + 0.2;
        drawSeed(ctx, cx + Math.cos(a) * (rInner * 0.6), cy + Math.sin(a) * (rInner * 0.6), a + Math.PI / 2);
      }

      ctx.fillStyle = '#073642';
      ctx.font = 'bold 18px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('2D Watermelon Disc', cx, cy - r - 25);

      if (showText) {
        const areaFrac = (1 - Math.pow(1 - epsFrac, 2)) * 100;
        ctx.fillStyle = '#cb4b16';
        ctx.font = 'bold 24px sans-serif';
        ctx.fillText(`Rind occupies ${areaFrac.toFixed(0)}% of area!`, cx, cy - r - 5);

        ctx.fillStyle = '#268bd2';
        ctx.font = '18px monospace';
        ctx.fillText(`V_rind / V_total = 1 - (1 - ${(epsFrac).toFixed(2)})² = ${areaFrac.toFixed(1)}%`, cx, cy + r + 35);
      }
      ctx.restore();
    }

    function draw3DSphere(ctx, cx, cy, r, epsFrac, showText) {
      ctx.save();
      const rInner = r * (1 - epsFrac);

      // Outer sphere 3D gradient
      const gradOuter = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r);
      gradOuter.addColorStop(0, '#4aa84e');
      gradOuter.addColorStop(0.7, '#268b28');
      gradOuter.addColorStop(1, '#0e4a11');

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = gradOuter;
      ctx.fill();
      ctx.strokeStyle = '#073642';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Cutout wedge to reveal red interior
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, rInner, -Math.PI / 4, Math.PI / 2, false);
      ctx.closePath();

      const gradInner = ctx.createRadialGradient(cx, cy, 10, cx, cy, rInner);
      gradInner.addColorStop(0, '#ff6b6b');
      gradInner.addColorStop(0.8, '#e73838');
      gradInner.addColorStop(1, '#b71c1c');

      ctx.fillStyle = gradInner;
      ctx.fill();
      ctx.strokeStyle = '#073642';
      ctx.stroke();

      // Seeds inside wedge
      drawSeed(ctx, cx + rInner * 0.4, cy + rInner * 0.2, 0.4);
      drawSeed(ctx, cx + rInner * 0.5, cy - rInner * 0.1, -0.2);

      ctx.fillStyle = '#073642';
      ctx.font = 'bold 18px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('3D Watermelon Sphere', cx, cy - r - 25);

      if (showText) {
        const volFrac = (1 - Math.pow(1 - epsFrac, 3)) * 100;
        ctx.fillStyle = '#cb4b16';
        ctx.font = 'bold 24px sans-serif';
        ctx.fillText(`Rind occupies ${volFrac.toFixed(0)}% of volume!`, cx, cy - r - 5);

        ctx.fillStyle = '#268bd2';
        ctx.font = '18px monospace';
        ctx.fillText(`V_rind / V_total = 1 - (1 - ${(epsFrac).toFixed(2)})³ = ${volFrac.toFixed(1)}%`, cx, cy + r + 35);
      }
      ctx.restore();
    }

    function drawHighDCurve(ctx, width, height, epsFrac, dVal) {
      ctx.save();
      // Margins configured so plot and x-axis labels are completely unobstructed
      const margin = { left: 90, right: 40, top: 48, bottom: 58 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;

      // Plot background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(margin.left, margin.top, plotW, plotH);
      ctx.strokeStyle = '#93a1a1';
      ctx.lineWidth = 1;
      ctx.strokeRect(margin.left, margin.top, plotW, plotH);

      // Y-axis grid lines and percentage labels
      ctx.strokeStyle = '#eee8d5';
      ctx.lineWidth = 1;
      for (let yPct = 0; yPct <= 100; yPct += 20) {
        const y = margin.top + plotH * (1 - yPct / 100);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotW, y);
        ctx.stroke();

        ctx.fillStyle = '#586e75';
        ctx.font = '13px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`${yPct}%`, margin.left - 10, y + 4);
      }

      // X-axis grid lines and dimension numbers
      for (let d = 0; d <= 50; d += 10) {
        const x = margin.left + (d / 50) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + plotH);
        ctx.stroke();

        ctx.fillStyle = '#073642';
        ctx.font = 'bold 14px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(d.toString(), x, margin.top + plotH + 18);
      }

      // Clearly visible X-axis Title
      ctx.fillStyle = '#073642';
      ctx.font = 'bold 16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Dimension D  (Number of Coordinates / Features)', margin.left + plotW / 2, margin.top + plotH + 42);

      // Clearly visible Y-axis Title
      ctx.save();
      ctx.translate(margin.left - 55, margin.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.font = 'bold 15px sans-serif';
      ctx.fillText('Rind Volume Fraction:  1 - (1 - ε)ᴰ', 0, 0);
      ctx.restore();

      // Draw Curve
      ctx.beginPath();
      for (let d = 1; d <= 50; d += 0.2) {
        const frac = 1 - Math.pow(Math.max(0, 1 - epsFrac), d);
        const x = margin.left + (d / 50) * plotW;
        const y = margin.top + plotH * (1 - frac);
        if (d === 1) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#268b28';
      ctx.lineWidth = 3.5;
      ctx.stroke();

      // Shaded area under curve (rind volume)
      ctx.lineTo(margin.left + plotW, margin.top + plotH);
      ctx.lineTo(margin.left + (1 / 50) * plotW, margin.top + plotH);
      ctx.closePath();
      ctx.fillStyle = 'rgba(38, 139, 40, 0.12)';
      ctx.fill();

      // Mark current dimension
      const curFrac = 1 - Math.pow(Math.max(0, 1 - epsFrac), dVal);
      const curX = margin.left + (dVal / 50) * plotW;
      const curY = margin.top + plotH * (1 - curFrac);

      ctx.strokeStyle = '#cb4b16';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(curX, margin.top + plotH);
      ctx.lineTo(curX, curY);
      ctx.lineTo(margin.left, curY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Marker point
      ctx.beginPath();
      ctx.arc(curX, curY, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#cb4b16';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Callout tag positioned so it never overlaps the top edge or badges
      ctx.fillStyle = '#cb4b16';
      ctx.font = 'bold 15px monospace';
      const textX = (dVal > 35) ? curX - 12 : curX + 12;
      const textY = (curY < margin.top + 32) ? curY + 22 : curY - 12;
      ctx.textAlign = (dVal > 35) ? 'right' : 'left';
      ctx.fillText(`D=${dVal}: ${(curFrac * 100).toFixed(1)}% RIND`, textX, textY);

      ctx.restore();
    }

    // Event listeners
    if (epsSlider) {
      epsSlider.addEventListener('input', (e) => {
        eps = parseFloat(e.target.value);
        if (epsVal) epsVal.textContent = (eps * 100).toFixed(0) + '%';
        draw();
      });
    }

    if (dimSlider) {
      dimSlider.addEventListener('input', (e) => {
        userDim = parseInt(e.target.value, 10);
        if (dimVal) dimVal.textContent = userDim;
        draw();
      });
    }

    if (prevBtn) {
      prevBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        step = Math.max(0, step - 1);
        draw();
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        step = Math.min(7, step + 1);
        draw();
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        step = 0;
        eps = 0.15;
        userDim = 10;
        if (epsSlider) epsSlider.value = 0.15;
        if (epsVal) epsVal.textContent = "15%";
        if (dimSlider) dimSlider.value = 10;
        if (dimVal) dimVal.textContent = "10";
        draw();
      });
    }

    if (btnReal) btnReal.addEventListener('click', (e) => { e.stopPropagation(); step = 0; draw(); });
    if (btn1D) btn1D.addEventListener('click', (e) => { e.stopPropagation(); step = 3; draw(); });
    if (btn2D) btn2D.addEventListener('click', (e) => { e.stopPropagation(); step = 5; draw(); });
    if (btn3D) btn3D.addEventListener('click', (e) => { e.stopPropagation(); step = 6; draw(); });
    if (btnHighD) btnHighD.addEventListener('click', (e) => { e.stopPropagation(); step = 7; draw(); });

    // Clicking on the canvas advances the step just like Reveal / SVG!
    canvas.addEventListener('click', () => {
      step = (step + 1) % 8;
      draw();
    });

    // Reveal.js slide transition hook
    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) {
          draw();
        }
      });
    }

    // Initial draw
    draw();
  }



  /* =========================================================================
   * 2. ANGLES BETWEEN HIGH-DIMENSIONAL VECTORS EXPLORER (3D ROTATABLE)
   * ========================================================================= */
  function initAngleDemo() {
    const canvas = document.getElementById('angle-canvas');
    if (!canvas) return;

    let d = 2;
    const numPairs = 600;
    let sampledAngles = [];
    let sampledVectors3D = []; // { x, y, z, angle, isOrth }

    // 3D rotation angles (pitch and yaw)
    let rotX = 0.38; // pitch (tilt down towards equator)
    let rotY = -0.52; // yaw (turn slightly to side)
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const compassCxRatio = 0.22;
    const compassCyRatio = 0.52;
    const compassR = 120;

    const slider = document.getElementById('angle-d-slider');
    const valDisplay = document.getElementById('angle-d-val');
    const resampleBtn = document.getElementById('angle-resample-btn');
    const resetBtn = document.getElementById('angle-reset-btn');

    const btnView3D = document.getElementById('angle-view-3d');
    const btnViewTop = document.getElementById('angle-view-top');
    const btnViewSide = document.getElementById('angle-view-side');

    const elDim = document.getElementById('angle-dim');
    const elStd = document.getElementById('angle-std');
    const elPct = document.getElementById('angle-pct');
    const elStatus = document.getElementById('angle-status');

    function randomGaussian() {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function sampleAngles(dim) {
      sampledAngles = new Float32Array(numPairs);
      sampledVectors3D = [];

      for (let i = 0; i < numPairs; i++) {
        let dot = 0, nx = 0, ny = 0;
        for (let k = 0; k < dim; k++) {
          const x = randomGaussian();
          const y = randomGaussian();
          dot += x * y;
          nx += x * x;
          ny += y * y;
        }
        const denom = Math.sqrt(nx) * Math.sqrt(ny);
        const cosTheta = denom > 1e-12 ? Math.max(-1, Math.min(1, dot / denom)) : 0;
        const thetaDeg = Math.acos(cosTheta) * (180 / Math.PI);
        sampledAngles[i] = thetaDeg;

        // 3D projection:
        // u = (0, 1, 0) is North Pole.
        // Polar angle is theta.
        // Azimuth phi is random in [0, 2*PI).
        // For D=2, vectors lie in a single plane (z = 0).
        let phi = 0;
        if (dim === 2) {
          phi = (i % 2 === 0) ? 0 : Math.PI;
        } else {
          phi = Math.random() * Math.PI * 2;
        }

        const sinT = Math.sin(thetaDeg * Math.PI / 180);
        const cosT = Math.cos(thetaDeg * Math.PI / 180);

        // y is along the reference vector u (North Pole)
        const vx = sinT * Math.cos(phi);
        const vz = sinT * Math.sin(phi);
        const vy = cosT;

        if (i < 160) {
          sampledVectors3D.push({
            x: vx,
            y: vy,
            z: vz,
            angle: thetaDeg,
            isOrth: Math.abs(thetaDeg - 90) <= 10
          });
        }
      }
    }

    function project3D(x, y, z, cx, cy, r) {
      // 1. Rotate around Y axis (yaw)
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const x1 = x * cosY + z * sinY;
      const y1 = y;
      const z1 = -x * sinY + z * cosY;

      // 2. Rotate around X axis (pitch)
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      const x2 = x1;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      return {
        sx: cx + x2 * r,
        sy: cy - y2 * r,
        depth: z2
      };
    }

    function updateBadge(dim) {
      if (elDim) elDim.textContent = dim;
      const stdRad = 1 / Math.sqrt(Math.max(1, dim - 1));
      const stdDeg = stdRad * (180 / Math.PI);
      if (elStd) elStd.textContent = stdDeg.toFixed(1) + '°';

      let countInside10 = 0;
      for (let i = 0; i < numPairs; i++) {
        if (Math.abs(sampledAngles[i] - 90) <= 10) countInside10++;
      }
      const pct = (countInside10 / numPairs) * 100;
      if (elPct) elPct.textContent = pct.toFixed(1) + '%';

      if (elStatus) {
        if (dim === 2) {
          elStatus.innerHTML = '<span style="color:#268bd2;">D = 2:</span> 2D circle: angles are uniformly distributed from 0° to 180°.';
        } else if (dim === 3) {
          elStatus.innerHTML = '<span style="color:#2aa198;">D = 3:</span> 3D Sphere: points are uniform on surface (grab & rotate to verify!).';
        } else if (dim <= 10) {
          elStatus.innerHTML = `<span style="color:#859900;">D = ${dim}:</span> Measure concentrates: ~${pct.toFixed(0)}% gather near the equator.`;
        } else if (dim <= 40) {
          elStatus.innerHTML = `<span style="color:#b58900;">D = ${dim}:</span> Strong orthogonality: ~${pct.toFixed(0)}% form an equatorial ribbon.`;
        } else {
          elStatus.innerHTML = `<span style="color:#dc322f; font-weight:bold;">D = ${dim} (HYPER-ORTHOGONALITY):</span> ${pct.toFixed(1)}% trapped on razor-thin equator!`;
        }
      }
    }

    function draw() {
      const { ctx, width, height } = setupCanvas(canvas);
      ctx.clearRect(0, 0, width, height);

      // Background
      ctx.fillStyle = '#fdf6e3';
      ctx.fillRect(0, 0, width, height);

      const compassCx = width * compassCxRatio;
      const compassCy = height * compassCyRatio;

      // -------------------------------------------------------------
      // LEFT PANEL: 3D Rotatable Hypersphere Projection
      // -------------------------------------------------------------
      // Panel Header
      ctx.fillStyle = '#073642';
      ctx.font = 'bold 14px "Helvetica Neue", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('3D Sphere Projection (Grab & Rotate)', compassCx, 26);

      // Sphere base shading with 3D gradient
      ctx.save();
      ctx.beginPath();
      ctx.arc(compassCx, compassCy, compassR, 0, Math.PI * 2);
      const grad = ctx.createRadialGradient(
        compassCx - compassR * 0.35, compassCy - compassR * 0.35, compassR * 0.1,
        compassCx, compassCy, compassR
      );
      grad.addColorStop(0, '#ffffff');
      grad.addColorStop(0.7, '#fbf3de');
      grad.addColorStop(1, '#ece2cb');
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.strokeStyle = '#93a1a1';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.clip(); // Clip everything inside the sphere

      // 1. Draw Orthogonal Equatorial Band (-10° to +10° latitude)
      const bandSegments = 40;
      ctx.fillStyle = 'rgba(133, 153, 0, 0.22)';
      ctx.beginPath();
      const latTop = 10 * (Math.PI / 180);
      const latBot = -10 * (Math.PI / 180);
      for (let s = 0; s <= bandSegments; s++) {
        const phi = (s / bandSegments) * Math.PI * 2;
        const pt = project3D(Math.cos(latTop) * Math.cos(phi), Math.sin(latTop), Math.cos(latTop) * Math.sin(phi), compassCx, compassCy, compassR);
        if (s === 0) ctx.moveTo(pt.sx, pt.sy);
        else ctx.lineTo(pt.sx, pt.sy);
      }
      for (let s = bandSegments; s >= 0; s--) {
        const phi = (s / bandSegments) * Math.PI * 2;
        const pb = project3D(Math.cos(latBot) * Math.cos(phi), Math.sin(latBot), Math.cos(latBot) * Math.sin(phi), compassCx, compassCy, compassR);
        ctx.lineTo(pb.sx, pb.sy);
      }
      ctx.closePath();
      ctx.fill();

      // 2. Latitude Lines (parallels)
      [-60, -30, 0, 30, 60].forEach(latDeg => {
        const lat = latDeg * (Math.PI / 180);
        const rLat = Math.cos(lat);
        const yLat = Math.sin(lat);
        const isEquator = (latDeg === 0);

        ctx.beginPath();
        for (let s = 0; s <= bandSegments; s++) {
          const phi = (s / bandSegments) * Math.PI * 2;
          const p = project3D(rLat * Math.cos(phi), yLat, rLat * Math.sin(phi), compassCx, compassCy, compassR);
          if (s === 0) ctx.moveTo(p.sx, p.sy);
          else ctx.lineTo(p.sx, p.sy);
        }
        if (isEquator) {
          ctx.strokeStyle = '#859900';
          ctx.lineWidth = 2.4;
          ctx.stroke();
        } else {
          ctx.strokeStyle = 'rgba(147, 161, 161, 0.35)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      });

      // 3. Longitude Meridians
      [0, 45, 90, 135].forEach(lonDeg => {
        const lon = lonDeg * (Math.PI / 180);
        ctx.beginPath();
        for (let s = 0; s <= bandSegments; s++) {
          const lat = -Math.PI / 2 + (s / bandSegments) * Math.PI;
          const p = project3D(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon), compassCx, compassCy, compassR);
          if (s === 0) ctx.moveTo(p.sx, p.sy);
          else ctx.lineTo(p.sx, p.sy);
        }
        ctx.strokeStyle = 'rgba(147, 161, 161, 0.28)';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // 4. Sampled Vectors Projected on 3D Sphere (sorted by depth)
      const projectedVecs = sampledVectors3D.map(v => {
        const proj = project3D(v.x, v.y, v.z, compassCx, compassCy, compassR);
        return { ...v, sx: proj.sx, sy: proj.sy, depth: proj.depth };
      });
      projectedVecs.sort((a, b) => a.depth - b.depth);

      projectedVecs.forEach(v => {
        const isFront = v.depth > 0;
        if (isFront) {
          ctx.strokeStyle = v.isOrth ? 'rgba(203, 75, 22, 0.45)' : 'rgba(88, 110, 117, 0.28)';
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(compassCx, compassCy);
          ctx.lineTo(v.sx, v.sy);
          ctx.stroke();

          ctx.fillStyle = v.isOrth ? '#cb4b16' : '#586e75';
          ctx.beginPath();
          ctx.arc(v.sx, v.sy, v.isOrth ? 3.5 : 2.5, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillStyle = v.isOrth ? 'rgba(203, 75, 22, 0.22)' : 'rgba(88, 110, 117, 0.15)';
          ctx.beginPath();
          ctx.arc(v.sx, v.sy, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // 5. Reference Vector u (North Pole arrow)
      const pCenter = project3D(0, 0, 0, compassCx, compassCy, compassR);
      const pPole = project3D(0, 1.0, 0, compassCx, compassCy, compassR);
      const pPoleTip = project3D(0, 1.38, 0, compassCx, compassCy, compassR);

      ctx.restore(); // End sphere clipping

      ctx.strokeStyle = '#dc322f';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(pCenter.sx, pCenter.sy);
      ctx.lineTo(pPoleTip.sx, pPoleTip.sy);
      ctx.stroke();

      // Arrowhead for u
      const angleU = Math.atan2(pPoleTip.sy - pPole.sy, pPoleTip.sx - pPole.sx);
      ctx.fillStyle = '#dc322f';
      ctx.beginPath();
      ctx.moveTo(pPoleTip.sx, pPoleTip.sy);
      ctx.lineTo(pPoleTip.sx - 10 * Math.cos(angleU - Math.PI / 6), pPoleTip.sy - 10 * Math.sin(angleU - Math.PI / 6));
      ctx.lineTo(pPoleTip.sx - 10 * Math.cos(angleU + Math.PI / 6), pPoleTip.sy - 10 * Math.sin(angleU + Math.PI / 6));
      ctx.closePath();
      ctx.fill();

      // Label for u
      ctx.fillStyle = '#dc322f';
      ctx.font = 'bold 14px serif';
      ctx.textAlign = 'left';
      ctx.fillText('u (ref pole)', pPoleTip.sx + 8, pPoleTip.sy + 4);

      // Equator callout label
      ctx.fillStyle = '#859900';
      ctx.font = 'bold 12px serif';
      ctx.textAlign = 'center';
      ctx.fillText('Equator: 90° Orthogonal Belt', compassCx, compassCy + compassR + 18);

      // Drag Hint
      ctx.fillStyle = '#93a1a1';
      ctx.font = 'italic 11px sans-serif';
      ctx.fillText('🖱️ Drag with mouse to rotate 3D sphere', compassCx, compassCy + compassR + 33);

      // -------------------------------------------------------------
      // RIGHT PANEL: Angle Distribution p(θ) & Histogram
      // -------------------------------------------------------------
      const plotLeft = width * 0.47;
      const plotRight = width * 0.95;
      const plotTop = 50;
      const plotBottom = height - 40;
      const plotW = plotRight - plotLeft;
      const plotH = plotBottom - plotTop;

      // Panel Header
      ctx.fillStyle = '#073642';
      ctx.font = 'bold 14px "Helvetica Neue", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Angle Density p(θ) & Monte Carlo Histogram', plotLeft + plotW / 2, 28);

      // Orthogonal Zone Shading on plot (80° to 100°)
      const x80 = plotLeft + (80 / 180) * plotW;
      const x100 = plotLeft + (100 / 180) * plotW;
      ctx.fillStyle = 'rgba(133, 153, 0, 0.14)';
      ctx.fillRect(x80, plotTop, x100 - x80, plotH);

      // Histogram of sampledAngles
      const numBins = 36; // 5° per bin
      const bins = new Array(numBins).fill(0);
      for (let i = 0; i < sampledAngles.length; i++) {
        const binIdx = Math.min(numBins - 1, Math.max(0, Math.floor(sampledAngles[i] / 5)));
        bins[binIdx]++;
      }
      const maxBinCount = Math.max(...bins, 1);
      ctx.fillStyle = 'rgba(38, 139, 210, 0.35)';
      ctx.strokeStyle = '#268bd2';
      ctx.lineWidth = 1;
      const binWidthPx = plotW / numBins;
      for (let b = 0; b < numBins; b++) {
        const bh = (bins[b] / maxBinCount) * (plotH * 0.82);
        const bx = plotLeft + b * binWidthPx;
        const by = plotBottom - bh;
        ctx.fillRect(bx, by, binWidthPx - 1, bh);
        ctx.strokeRect(bx, by, binWidthPx - 1, bh);
      }

      // Theoretical Curve: p(θ) ∝ (sin θ)^(d-2)
      ctx.strokeStyle = '#cb4b16';
      ctx.lineWidth = 3;
      ctx.beginPath();
      let started = false;
      for (let step = 0; step <= 180; step += 0.5) {
        const rad = step * (Math.PI / 180);
        const sinVal = Math.sin(rad);
        let val = 0;
        if (d === 2) {
          val = 1.0;
        } else if (sinVal > 0) {
          val = Math.pow(sinVal, d - 2);
        }
        const py = plotBottom - val * (plotH * 0.82);
        const px = plotLeft + (step / 180) * plotW;
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();

      // 90° Center Line
      const x90 = plotLeft + (90 / 180) * plotW;
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#859900';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(x90, plotTop);
      ctx.lineTo(x90, plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Axes
      ctx.strokeStyle = '#073642';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(plotLeft, plotBottom);
      ctx.lineTo(plotRight, plotBottom);
      ctx.moveTo(plotLeft, plotBottom);
      ctx.lineTo(plotLeft, plotTop);
      ctx.stroke();

      // X-Axis Ticks & Labels
      ctx.fillStyle = '#073642';
      ctx.font = '13px serif';
      ctx.textAlign = 'center';
      [
        { deg: 0, label: '0°' },
        { deg: 45, label: '45°' },
        { deg: 90, label: '90° (π/2)' },
        { deg: 135, label: '135°' },
        { deg: 180, label: '180°' }
      ].forEach(t => {
        const tx = plotLeft + (t.deg / 180) * plotW;
        ctx.beginPath();
        ctx.moveTo(tx, plotBottom);
        ctx.lineTo(tx, plotBottom + 5);
        ctx.stroke();
        ctx.fillText(t.label, tx, plotBottom + 20);
      });

      // Legend
      ctx.font = '12px "Helvetica Neue", sans-serif';
      ctx.textAlign = 'left';
      ctx.strokeStyle = '#cb4b16';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(plotLeft + 15, plotTop + 15);
      ctx.lineTo(plotLeft + 40, plotTop + 15);
      ctx.stroke();
      ctx.fillStyle = '#073642';
      ctx.fillText('Theory: p(θ) ∝ sin^(D-2)(θ)', plotLeft + 46, plotTop + 19);

      ctx.fillStyle = 'rgba(38, 139, 210, 0.4)';
      ctx.fillRect(plotLeft + 15, plotTop + 28, 25, 12);
      ctx.strokeRect(plotLeft + 15, plotTop + 28, 25, 12);
      ctx.fillStyle = '#073642';
      ctx.fillText(`Monte Carlo (N=${numPairs})`, plotLeft + 46, plotTop + 38);
    }

    // -------------------------------------------------------------
    // Mouse / Touch Rotation Event Handlers
    // -------------------------------------------------------------
    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const compassCx = canvas.clientWidth * compassCxRatio;
      if (x < compassCx + compassR + 40) {
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        canvas.style.cursor = 'grabbing';
        e.stopPropagation();
      }
    });

    window.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const compassCx = canvas.clientWidth * compassCxRatio;

      if (!isDragging) {
        if (x >= 0 && x < compassCx + compassR + 40 && y >= 0 && y <= canvas.clientHeight) {
          canvas.style.cursor = 'grab';
        } else {
          canvas.style.cursor = 'default';
        }
        return;
      }

      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;

      rotY += dx * 0.009;
      rotX += dy * 0.009;
      rotX = Math.max(-1.48, Math.min(1.48, rotX));
      draw();
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        canvas.style.cursor = 'grab';
      }
    });

    // Touch Support
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        const rect = canvas.getBoundingClientRect();
        const x = e.touches[0].clientX - rect.left;
        const compassCx = canvas.clientWidth * compassCxRatio;
        if (x < compassCx + compassR + 40) {
          isDragging = true;
          lastMouseX = e.touches[0].clientX;
          lastMouseY = e.touches[0].clientY;
          e.preventDefault();
        }
      }
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
      if (isDragging && e.touches.length === 1) {
        const dx = e.touches[0].clientX - lastMouseX;
        const dy = e.touches[0].clientY - lastMouseY;
        lastMouseX = e.touches[0].clientX;
        lastMouseY = e.touches[0].clientY;
        rotY += dx * 0.009;
        rotX += dy * 0.009;
        rotX = Math.max(-1.48, Math.min(1.48, rotX));
        draw();
        e.preventDefault();
      }
    }, { passive: false });

    canvas.addEventListener('touchend', () => {
      isDragging = false;
    });

    // -------------------------------------------------------------
    // Control Bar Handlers
    // -------------------------------------------------------------
    if (slider) {
      slider.addEventListener('input', (e) => {
        d = parseInt(e.target.value, 10);
        if (valDisplay) valDisplay.textContent = d;
        sampleAngles(d);
        updateBadge(d);
        draw();
      });
    }

    if (resampleBtn) {
      resampleBtn.addEventListener('click', () => {
        sampleAngles(d);
        updateBadge(d);
        draw();
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        d = 2;
        rotX = 0.38;
        rotY = -0.52;
        if (slider) slider.value = 2;
        if (valDisplay) valDisplay.textContent = '2';
        sampleAngles(2);
        updateBadge(2);
        draw();
      });
    }

    // View Presets
    if (btnView3D) {
      btnView3D.addEventListener('click', () => {
        rotX = 0.38; rotY = -0.52; draw();
      });
    }
    if (btnViewTop) {
      btnViewTop.addEventListener('click', () => {
        rotX = 1.45; rotY = 0; draw();
      });
    }
    if (btnViewSide) {
      btnViewSide.addEventListener('click', () => {
        rotX = 0.02; rotY = 0; draw();
      });
    }

    [2, 3, 10, 50, 100].forEach(dimVal => {
      const btn = document.getElementById(`angle-btn-${dimVal}`);
      if (btn) {
        btn.addEventListener('click', () => {
          d = dimVal;
          if (slider) slider.value = dimVal;
          if (valDisplay) valDisplay.textContent = dimVal.toString();
          sampleAngles(d);
          updateBadge(d);
          draw();
        });
      }
    });

    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) {
          draw();
        }
      });
    }

    // Initialize with D = 2
    sampleAngles(2);
    updateBadge(2);
    draw();
  }

  /* =========================================================================
   * 3. Lp MINKOWSKI METRIC UNIT BALL & DISTANCE EXPLORER
   * ========================================================================= */
  function initLpDemo() {
    const canvas = document.getElementById('lp-canvas');
    if (!canvas) return;

    let p = 2.0;
    let isInf = false;
    let probeX = 0.8;
    let probeY = 0.6;
    let isDragging = false;

    const pSlider = document.getElementById('lp-p-slider');
    const pValDisplay = document.getElementById('lp-p-val');
    const resetBtn = document.getElementById('lp-reset-btn');

    const elNormName = document.getElementById('lp-norm-name');
    const elFormula = document.getElementById('lp-formula');
    const elProbePos = document.getElementById('lp-probe-pos');
    const elL1Val = document.getElementById('lp-l1-val');
    const elL2Val = document.getElementById('lp-l2-val');
    const elLpVal = document.getElementById('lp-lp-val');
    const elLinfVal = document.getElementById('lp-linf-val');

    function updateBadge() {
      if (elNormName) {
        if (isInf) elNormName.textContent = 'L∞ (Chebyshev / Maximum Norm)';
        else if (Math.abs(p - 1.0) < 0.01) elNormName.textContent = 'L₁ (Manhattan / Taxicab Norm)';
        else if (Math.abs(p - 2.0) < 0.01) elNormName.textContent = 'L₂ (Euclidean Norm)';
        else if (p < 1.0) elNormName.textContent = `L_${p.toFixed(1)} (Non-convex Fractional Quasi-Norm)`;
        else elNormName.textContent = `L_${p.toFixed(1)} Minkowski Norm`;
      }

      if (elFormula) {
        if (isInf) elFormula.innerHTML = '‖<b>x</b>‖<sub>∞</sub> = max(|<i>x</i><sub>1</sub>|, |<i>x</i><sub>2</sub>|) = 1';
        else if (Math.abs(p - 1.0) < 0.01) elFormula.innerHTML = '‖<b>x</b>‖<sub>1</sub> = |<i>x</i><sub>1</sub>| + |<i>x</i><sub>2</sub>| = 1';
        else if (Math.abs(p - 2.0) < 0.01) elFormula.innerHTML = '‖<b>x</b>‖<sub>2</sub> = &radic;(<i>x</i><sub>1</sub><sup>2</sup> + <i>x</i><sub>2</sub><sup>2</sup>) = 1';
        else elFormula.innerHTML = `‖<b>x</b>‖<sub>${p.toFixed(1)}</sub> = (|<i>x</i><sub>1</sub>|<sup>${p.toFixed(1)}</sup> + |<i>x</i><sub>2</sub>|<sup>${p.toFixed(1)}</sup>)<sup>1/${p.toFixed(1)}</sup> = 1`;
      }

      if (elProbePos) elProbePos.textContent = `(${probeX.toFixed(2)}, ${probeY.toFixed(2)})`;

      const d1 = Math.abs(probeX) + Math.abs(probeY);
      const d2 = Math.sqrt(probeX * probeX + probeY * probeY);
      const dinf = Math.max(Math.abs(probeX), Math.abs(probeY));
      const dp = isInf ? dinf : Math.pow(Math.pow(Math.abs(probeX), p) + Math.pow(Math.abs(probeY), p), 1 / p);

      if (elL1Val) elL1Val.textContent = d1.toFixed(2);
      if (elL2Val) elL2Val.textContent = d2.toFixed(2);
      if (elLpVal) elLpVal.textContent = isInf ? dinf.toFixed(2) : dp.toFixed(2);
      if (elLinfVal) elLinfVal.textContent = dinf.toFixed(2);
    }

    function draw() {
      const { ctx, width, height } = setupCanvas(canvas);
      ctx.clearRect(0, 0, width, height);

      // Background
      ctx.fillStyle = '#fdf6e3';
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const scale = 120; // 120 px per unit (unit 1 is 120px)

      // 1. Draw Grid Lines
      ctx.strokeStyle = '#eee8d5';
      ctx.lineWidth = 1;
      for (let g = -1.5; g <= 1.5; g += 0.5) {
        ctx.beginPath();
        ctx.moveTo(cx + g * scale, 0);
        ctx.lineTo(cx + g * scale, height);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, cy - g * scale);
        ctx.lineTo(width, cy - g * scale);
        ctx.stroke();
      }

      // 2. Draw Axes
      ctx.strokeStyle = '#073642';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(30, cy);
      ctx.lineTo(width - 30, cy);
      ctx.moveTo(cx, height - 20);
      ctx.lineTo(cx, 20);
      ctx.stroke();

      // Axis labels & ticks
      ctx.fillStyle = '#073642';
      ctx.font = 'bold 15px serif';
      ctx.textAlign = 'center';
      ctx.fillText('x', width - 20, cy + 18);
      ctx.fillText('y', cx + 18, 25);

      [-1, 1].forEach(t => {
        // x ticks
        ctx.beginPath();
        ctx.moveTo(cx + t * scale, cy - 4);
        ctx.lineTo(cx + t * scale, cy + 4);
        ctx.stroke();
        ctx.fillText(t.toString(), cx + t * scale, cy + 20);

        // y ticks
        ctx.beginPath();
        ctx.moveTo(cx - 4, cy - t * scale);
        ctx.lineTo(cx + 4, cy - t * scale);
        ctx.stroke();
        ctx.fillText(t.toString(), cx - 18, cy - t * scale + 5);
      });

      // 3. Draw Unit Ball Contour
      ctx.beginPath();
      if (isInf) {
        ctx.rect(cx - scale, cy - scale, scale * 2, scale * 2);
      } else {
        const numPts = 360;
        for (let i = 0; i <= numPts; i++) {
          const theta = (i * Math.PI * 2) / numPts;
          const cosT = Math.cos(theta);
          const sinT = Math.sin(theta);
          const rTheta = Math.pow(Math.pow(Math.abs(cosT), p) + Math.pow(Math.abs(sinT), p), -1 / p);
          const px = cx + rTheta * cosT * scale;
          const py = cy - rTheta * sinT * scale;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
      }

      ctx.fillStyle = 'rgba(38, 139, 210, 0.18)';
      ctx.fill();
      ctx.strokeStyle = '#268bd2';
      ctx.lineWidth = 3.5;
      ctx.stroke();

      // 4. Draw Draggable Probe Point x
      const prx = cx + probeX * scale;
      const pry = cy - probeY * scale;

      // Projection lines
      ctx.strokeStyle = '#93a1a1';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(prx, cy);
      ctx.lineTo(prx, pry);
      ctx.lineTo(cx, pry);
      ctx.stroke();

      // Vector line from origin
      ctx.strokeStyle = '#cb4b16';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(prx, pry);
      ctx.stroke();

      // Probe point dot
      ctx.beginPath();
      ctx.arc(prx, pry, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#cb4b16';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Probe label
      ctx.fillStyle = '#cb4b16';
      ctx.font = 'bold 15px serif';
      ctx.textAlign = 'left';
      ctx.fillText('x', prx + 10, pry - 10);

      updateBadge();
    }

    function setNorm(newP, infMode) {
      isInf = infMode;
      p = newP;
      if (pSlider) {
        pSlider.value = isInf ? 8 : p;
      }
      if (pValDisplay) {
        pValDisplay.textContent = isInf ? '∞' : p.toFixed(1);
      }
      draw();
    }

    if (pSlider) {
      pSlider.addEventListener('input', (e) => {
        isInf = false;
        p = parseFloat(e.target.value);
        if (pValDisplay) pValDisplay.textContent = p.toFixed(1);
        draw();
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        probeX = 0.8;
        probeY = 0.6;
        setNorm(2.0, false);
      });
    }

    // Preset buttons
    const btn05 = document.getElementById('lp-btn-05');
    const btn1 = document.getElementById('lp-btn-1');
    const btn2 = document.getElementById('lp-btn-2');
    const btn4 = document.getElementById('lp-btn-4');
    const btnInf = document.getElementById('lp-btn-inf');

    if (btn05) btn05.addEventListener('click', () => setNorm(0.5, false));
    if (btn1) btn1.addEventListener('click', () => setNorm(1.0, false));
    if (btn2) btn2.addEventListener('click', () => setNorm(2.0, false));
    if (btn4) btn4.addEventListener('click', () => setNorm(4.0, false));
    if (btnInf) btnInf.addEventListener('click', () => setNorm(8.0, true));

    // Mouse drag interaction for probe point
    function handlePointer(e) {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const scale = 120;
      probeX = (mouseX - rect.width / 2) / scale;
      probeY = -(mouseY - rect.height / 2) / scale;
      // Clamp within visible bounds
      probeX = Math.max(-1.5, Math.min(1.5, probeX));
      probeY = Math.max(-1.5, Math.min(1.5, probeY));
      draw();
    }

    canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      handlePointer(e);
      canvas.style.cursor = 'crosshair';
    });

    window.addEventListener('mousemove', (e) => {
      if (isDragging) handlePointer(e);
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        canvas.style.cursor = 'default';
      }
    });

    if (window.Reveal) {
      Reveal.addEventListener('slidechanged', (e) => {
        if (e.currentSlide && e.currentSlide.contains(canvas)) draw();
      });
    }

    draw();
  }

  // Initialize all demos on DOMContentLoaded or immediately if ready
  function initAll() {
    initWatermelonDemo();
    initAngleDemo();
    initLpDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

})();
