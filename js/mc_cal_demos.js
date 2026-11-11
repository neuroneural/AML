/**
 * Interactive Visualizations for CS8850 Lecture 20:
 * Model Comparison & Calibration (AML/cs8850_20_model_comparison_1.html & cs8850_22_calibration.html)
 *
 * Includes:
 *   1. Decision Threshold, Confusion Matrix & ROC / Profit Explorer (#mc-roc-canvas)
 *   2. Reliability Diagram & Temperature Scaling Calibrator (#mc-cal-canvas)
 *
 * Strictly enforces 1:1 geometric aspect ratio on the [0,1]x[0,1] ROC and Reliability squares.
 */
(function () {
  'use strict';

  var C = {
    bg: '#fdf6e3',
    panel: '#f7f0dd',
    ink: '#073642',
    muted: '#586e75',
    grid: '#dcd5c0',
    border: '#93a1a1',
    blue: '#268bd2',
    orange: '#cb4b16',
    green: '#859900',
    red: '#dc322f',
    violet: '#6c71c4',
    yellow: '#b58900',
    cyan: '#2aa198'
  };

  function erf(x) {
    var sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    var a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    var t = 1.0 / (1.0 + p * x);
    var y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  function normCDF(x) {
    return 0.5 * (1 + erf(x / Math.SQRT2));
  }

  function normPDF(x) {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  }

  function logit(p) {
    var cl = Math.max(1e-5, Math.min(1 - 1e-5, p));
    return Math.log(cl / (1 - cl));
  }

  // =========================================================================
  // DEMO 1: DECISION THRESHOLD, CONFUSION MATRIX, ROC & PROFIT CURVE EXPLORER
  // =========================================================================
  function initRocDemo() {
    var canvas = document.getElementById('mc-roc-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');

    var selScenario = document.getElementById('mc-roc-scenario');
    var sliderTau = document.getElementById('mc-roc-tau');
    var valTau = document.getElementById('mc-roc-tau-val');
    var sliderDprime = document.getElementById('mc-roc-sep') || document.getElementById('mc-roc-dprime');
    var valDprime = document.getElementById('mc-roc-sep-val') || document.getElementById('mc-roc-dprime-val');
    var sliderPrior = document.getElementById('mc-roc-prior');
    var valPrior = document.getElementById('mc-roc-prior-val');
    var sliderBtp = document.getElementById('mc-roc-btp');
    var valBtp = document.getElementById('mc-roc-btp-val');
    var sliderCfp = document.getElementById('mc-roc-cfp');
    var valCfp = document.getElementById('mc-roc-cfp-val');
    var btnOpt = document.getElementById('mc-roc-opt-btn') || document.getElementById('mc-roc-btn-opt');
    var btnBal = document.getElementById('mc-roc-btn-bal');
    var statusBanner = document.getElementById('mc-roc-status');
    var readTpr = document.getElementById('mc-roc-tpr');
    var readFpr = document.getElementById('mc-roc-fpr');
    var readAuc = document.getElementById('mc-roc-auc');
    var readEv = document.getElementById('mc-roc-ev');

    var state = {
      tau: sliderTau ? parseFloat(sliderTau.value) : 0.35,
      dprime: sliderDprime ? parseFloat(sliderDprime.value) : 1.50,
      piPos: sliderPrior ? parseFloat(sliderPrior.value) : 0.55,
      bTP: sliderBtp ? parseFloat(sliderBtp.value) : 99,
      cFP: sliderCfp ? parseFloat(sliderCfp.value) : 1,
      dragging: null
    };

    var scenarios = {
      balanced: { piPos: 0.50, bTP: 20, cFP: 20, label: 'Balanced Priors & Costs (π₊ = 0.50, b_TP = $20, c_FP = $20)' },
      marketing: { piPos: 0.15, bTP: 99, cFP: 5, label: 'Targeted Marketing (π₊ = 0.15, b_TP = $99 profit, c_FP = $5 mailer)' },
      fraud: { piPos: 0.08, bTP: 120, cFP: 8, label: 'High-Recall Fraud / Medical Triage (π₊ = 0.08, b_TP = $120, c_FP = $8)' },
      spam: { piPos: 0.40, bTP: 5, cFP: 45, label: 'Low-FPR Spam Filter (π₊ = 0.40, b_TP = $5, c_FP = $45 false-block cost)' }
    };

    // Map score s in (0,1) to latent normal z = logit(s)*0.85
    function scoreToZ(s) {
      return logit(s) * 0.85;
    }

    // Class 0 has latent mean -d'/2, Class 1 has latent mean +d'/2, unit variance
    function ratesAtTau(tau, dp) {
      var zCut = scoreToZ(tau);
      var fpr = 1 - normCDF(zCut - (-0.5 * dp));
      var tpr = 1 - normCDF(zCut - (0.5 * dp));
      return { tpr: tpr, fpr: fpr, fnr: 1 - tpr, tnr: 1 - fpr };
    }

    function evAtTau(tau, dp) {
      var r = ratesAtTau(tau, dp);
      return state.piPos * r.tpr * state.bTP - (1 - state.piPos) * r.fpr * state.cFP;
    }

    function computeAUC(dp) {
      // Exact Gaussian equal-variance AUC = Phi(d' / sqrt(2))
      return normCDF(dp / Math.SQRT2);
    }

    function findOptimalTau(dp) {
      var bestTau = 0.50;
      var bestEV = -1e9;
      for (var i = 1; i <= 199; i++) {
        var t = i / 200;
        var ev = evAtTau(t, dp);
        if (ev > bestEV) {
          bestEV = ev;
          bestTau = t;
        }
      }
      return { tau: bestTau, ev: bestEV };
    }

    function draw() {
      var W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.bg;
      ctx.fillRect(0, 0, W, H);

      var dp = state.dprime;
      var tau = state.tau;
      var cur = ratesAtTau(tau, dp);
      var curEV = evAtTau(tau, dp);
      var opt = findOptimalTau(dp);
      var auc = computeAUC(dp);

      // Update DOM readouts
      if (valTau) valTau.textContent = tau.toFixed(2);
      if (valDprime) valDprime.textContent = dp.toFixed(1);
      if (valPrior) valPrior.textContent = state.piPos.toFixed(2);
      if (valBtp) valBtp.textContent = '$' + Math.round(state.bTP);
      if (valCfp) valCfp.textContent = '$' + Math.round(state.cFP);
      if (readTpr) readTpr.textContent = (cur.tpr * 100).toFixed(1) + '%';
      if (readFpr) readFpr.textContent = (cur.fpr * 100).toFixed(1) + '%';
      if (readAuc) readAuc.textContent = auc.toFixed(3);
      if (readEv) readEv.textContent = (curEV >= 0 ? '+$' : '-$') + Math.abs(curEV).toFixed(2);

      if (statusBanner) {
        var diffOpt = Math.abs(tau - opt.tau);
        statusBanner.innerHTML =
          '<b>Threshold τ = ' + tau.toFixed(2) + '</b> → TPR = <b>' + (cur.tpr * 100).toFixed(1) + '%</b>, FPR = <b>' + (cur.fpr * 100).toFixed(1) + '%</b> | ' +
          'Max Expected Value at <b>τ* = ' + opt.tau.toFixed(2) + '</b> (+$' + opt.ev.toFixed(2) + '/sample)' +
          (diffOpt < 0.03 ? ' <span style="color:#859900;font-weight:bold;">[At Optimal Operating Point!]</span>' : ' — drag the threshold line or click "Jump to Optimal τ*"');
      }

      // ---------------------------------------------------------------------
      // PANEL 1 (LEFT): Score Distributions p(s|Y=0), p(s|Y=1) & Confusion Matrix
      // ---------------------------------------------------------------------
      var p1 = { x: 46, y: 34, w: 320, h: 205 };
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(p1.x, p1.y, p1.w, p1.h);
      ctx.strokeRect(p1.x, p1.y, p1.w, p1.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('1. Classifier Score Densities & Threshold τ', p1.x + p1.w / 2, p1.y - 10);

      // Compute density curves in score space s in [0.02, 0.98]
      var maxDens = 0.01;
      var nSteps = 120;
      var sArr = [], d0Arr = [], d1Arr = [];
      for (var i = 0; i <= nSteps; i++) {
        var s = 0.02 + (0.96 * i) / nSteps;
        var z = scoreToZ(s);
        // Include smooth Jacobian taper so endpoints go cleanly to 0
        var jac = Math.pow(s * (1 - s), 0.35);
        var d0 = normPDF(z + 0.5 * dp) * jac;
        var d1 = normPDF(z - 0.5 * dp) * jac;
        sArr.push(s);
        d0Arr.push(d0);
        d1Arr.push(d1);
        if (d0 > maxDens) maxDens = d0;
        if (d1 > maxDens) maxDens = d1;
      }

      function sToX(s) { return p1.x + ((s - 0.0) / 1.0) * p1.w; }
      function dToY(d) { return p1.y + p1.h - 22 - (d / (maxDens * 1.12)) * (p1.h - 34); }
      var baseY = p1.y + p1.h - 22;

      // Shade FP (s >= tau under d0) and TP (s >= tau under d1)
      // TP shading (greenish)
      ctx.beginPath();
      var startedTP = false;
      for (var i = 0; i <= nSteps; i++) {
        if (sArr[i] >= tau) {
          var x = sToX(sArr[i]), y = dToY(d1Arr[i]);
          if (!startedTP) { ctx.moveTo(sToX(tau), baseY); ctx.lineTo(x, y); startedTP = true; }
          else ctx.lineTo(x, y);
        }
      }
      if (startedTP) {
        ctx.lineTo(sToX(sArr[nSteps]), baseY);
        ctx.closePath();
        ctx.fillStyle = 'rgba(133, 153, 0, 0.25)';
        ctx.fill();
      }

      // FP shading (reddish)
      ctx.beginPath();
      var startedFP = false;
      for (var i = 0; i <= nSteps; i++) {
        if (sArr[i] >= tau) {
          var x = sToX(sArr[i]), y = dToY(d0Arr[i]);
          if (!startedFP) { ctx.moveTo(sToX(tau), baseY); ctx.lineTo(x, y); startedFP = true; }
          else ctx.lineTo(x, y);
        }
      }
      if (startedFP) {
        ctx.lineTo(sToX(sArr[nSteps]), baseY);
        ctx.closePath();
        ctx.fillStyle = 'rgba(220, 50, 47, 0.30)';
        ctx.fill();
      }

      // Draw Class 0 curve (Negative class, Blue)
      ctx.beginPath();
      for (var i = 0; i <= nSteps; i++) {
        var x = sToX(sArr[i]), y = dToY(d0Arr[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = C.blue;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Draw Class 1 curve (Positive class, Green)
      ctx.beginPath();
      for (var i = 0; i <= nSteps; i++) {
        var x = sToX(sArr[i]), y = dToY(d1Arr[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Baseline
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p1.x, baseY);
      ctx.lineTo(p1.x + p1.w, baseY);
      ctx.stroke();

      // X-axis ticks
      ctx.fillStyle = C.muted;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      [0, 0.25, 0.5, 0.75, 1.0].forEach(function (v) {
        ctx.fillText(v.toFixed(2), sToX(v), baseY + 15);
      });

      // Legend inside density plot
      ctx.font = 'bold 11.5px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillStyle = C.blue;
      ctx.fillText('— Y = 0 (Neg)', p1.x + 8, p1.y + 16);
      ctx.fillStyle = C.green;
      ctx.fillText('— Y = 1 (Pos)', p1.x + 102, p1.y + 16);

      // Threshold vertical line tau
      var tauX = sToX(tau);
      ctx.save();
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(tauX, p1.y + 4);
      ctx.lineTo(tauX, baseY);
      ctx.stroke();
      ctx.restore();

      // Threshold pill handle
      ctx.fillStyle = C.orange;
      ctx.beginPath();
      ctx.roundRect(tauX - 30, p1.y + 22, 60, 18, 4);
      ctx.fill();
      ctx.fillStyle = '#fdf6e3';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('τ=' + tau.toFixed(2), tauX, p1.y + 35);

      // 2x2 Confusion Matrix Card below Panel 1
      var cm = { x: 46, y: 256, w: 320, h: 178 };
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(cm.x, cm.y, cm.w, cm.h);
      ctx.strokeRect(cm.x, cm.y, cm.w, cm.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Confusion Matrix (N = 1,000, π₊ = ' + (state.piPos * 100).toFixed(0) + '%)', cm.x + cm.w / 2, cm.y + 18);

      var nPos = Math.round(1000 * state.piPos);
      var nNeg = 1000 - nPos;
      var TP = Math.round(nPos * cur.tpr);
      var FN = nPos - TP;
      var FP = Math.round(nNeg * cur.fpr);
      var TN = nNeg - FP;
      var acc = (TP + TN) / 1000;
      var prec = (TP + FP) > 0 ? TP / (TP + FP) : 1.0;

      // Grid cells for 2x2 matrix
      var gx = cm.x + 82, gy = cm.y + 44, cw = 112, ch = 50;
      ctx.font = 'bold 11.5px sans-serif';
      ctx.fillStyle = C.muted;
      ctx.fillText('True p (Pos)', gx + cw * 0.5, gy - 6);
      ctx.fillText('True n (Neg)', gx + cw * 1.5, gy - 6);

      ctx.textAlign = 'right';
      ctx.fillText('Pred P', gx - 8, gy + ch * 0.5 + 4);
      ctx.fillText('Pred N', gx - 8, gy + ch * 1.5 + 4);

      var cells = [
        { r: 0, c: 0, bg: 'rgba(133,153,0,0.18)', border: C.green, title: 'TP = ' + TP, sub: 'TPR: ' + (cur.tpr * 100).toFixed(1) + '%', col: C.green },
        { r: 0, c: 1, bg: 'rgba(220,50,47,0.16)', border: C.red, title: 'FP = ' + FP, sub: 'FPR: ' + (cur.fpr * 100).toFixed(1) + '%', col: C.red },
        { r: 1, c: 0, bg: 'rgba(203,75,22,0.13)', border: C.orange, title: 'FN = ' + FN, sub: 'FNR: ' + (cur.fnr * 100).toFixed(1) + '%', col: C.orange },
        { r: 1, c: 1, bg: 'rgba(38,139,210,0.14)', border: C.blue, title: 'TN = ' + TN, sub: 'TNR: ' + (cur.tnr * 100).toFixed(1) + '%', col: C.blue }
      ];

      ctx.textAlign = 'center';
      cells.forEach(function (cell) {
        var cx = gx + cell.c * cw, cy = gy + cell.r * ch;
        ctx.fillStyle = cell.bg;
        ctx.fillRect(cx + 2, cy + 2, cw - 4, ch - 4);
        ctx.strokeStyle = cell.border;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(cx + 2, cy + 2, cw - 4, ch - 4);
        ctx.fillStyle = cell.col;
        ctx.font = 'bold 13px monospace';
        ctx.fillText(cell.title, cx + cw / 2, cy + 21);
        ctx.fillStyle = C.ink;
        ctx.font = '11px sans-serif';
        ctx.fillText(cell.sub, cx + cw / 2, cy + 38);
      });

      ctx.fillStyle = C.ink;
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(
        'Accuracy: ' + (acc * 100).toFixed(1) + '%   |   Precision: ' + (prec * 100).toFixed(1) + '%',
        cm.x + cm.w / 2,
        cm.y + cm.h - 10
      );

      // ---------------------------------------------------------------------
      // PANEL 2 (CENTER): Exact 1:1 Square ROC Curve (FPR vs TPR)
      // ---------------------------------------------------------------------
      var rocSize = 350;
      var p2 = { x: 424, y: 44, w: rocSize, h: rocSize }; // Exact 350x350 1:1 square!
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(p2.x, p2.y, p2.w, p2.h);
      ctx.strokeRect(p2.x, p2.y, p2.w, p2.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('2. ROC Space (1:1 Square, AUC = ' + auc.toFixed(3) + ')', p2.x + p2.w / 2, p2.y - 12);

      function fprToX(fpr) { return p2.x + fpr * p2.w; }
      function tprToY(tpr) { return p2.y + (1 - tpr) * p2.h; }

      // Grid lines inside 1:1 ROC square
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      [0.2, 0.4, 0.6, 0.8].forEach(function (g) {
        ctx.beginPath();
        ctx.moveTo(fprToX(g), p2.y);
        ctx.lineTo(fprToX(g), p2.y + p2.h);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(p2.x, tprToY(g));
        ctx.lineTo(p2.x + p2.w, tprToY(g));
        ctx.stroke();
      });

      // Shaded AUC polygon
      ctx.beginPath();
      ctx.moveTo(fprToX(0), tprToY(0));
      for (var i = 200; i >= 0; i--) {
        var z = -4.5 + (9.0 * i) / 200;
        var fpr = 1 - normCDF(z + 0.5 * dp);
        var tpr = 1 - normCDF(z - 0.5 * dp);
        ctx.lineTo(fprToX(fpr), tprToY(tpr));
      }
      ctx.lineTo(fprToX(1), tprToY(0));
      ctx.closePath();
      ctx.fillStyle = 'rgba(38, 139, 210, 0.14)';
      ctx.fill();

      // Diagonal random guess line
      ctx.save();
      ctx.strokeStyle = C.border;
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(fprToX(0), tprToY(0));
      ctx.lineTo(fprToX(1), tprToY(1));
      ctx.stroke();
      ctx.restore();

      // ROC Curve
      ctx.beginPath();
      for (var i = 200; i >= 0; i--) {
        var z = -4.5 + (9.0 * i) / 200;
        var fpr = 1 - normCDF(z + 0.5 * dp);
        var tpr = 1 - normCDF(z - 0.5 * dp);
        var x = fprToX(fpr), y = tprToY(tpr);
        if (i === 200) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = C.blue;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Optimal iso-profit tangent line at opt.tau
      var optRates = ratesAtTau(opt.tau, dp);
      var slope = ((1 - state.piPos) * state.cFP) / Math.max(1e-6, state.piPos * state.bTP);
      ctx.save();
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      var dx = 0.28;
      var x1 = Math.max(0, optRates.fpr - dx), y1 = optRates.tpr - slope * (optRates.fpr - x1);
      var x2 = Math.min(1, optRates.fpr + dx), y2 = optRates.tpr + slope * (x2 - optRates.fpr);
      ctx.moveTo(fprToX(x1), tprToY(Math.max(0, Math.min(1, y1))));
      ctx.lineTo(fprToX(x2), tprToY(Math.max(0, Math.min(1, y2))));
      ctx.stroke();
      ctx.restore();

      // Optimal operating point marker (Green star/ring)
      ctx.fillStyle = C.green;
      ctx.beginPath();
      ctx.arc(fprToX(optRates.fpr), tprToY(optRates.tpr), 6, 0, 2 * Math.PI);
      ctx.fill();

      // Current threshold operating point (Orange circle)
      var curX = fprToX(cur.fpr), curY = tprToY(cur.tpr);
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(curX, tprToY(0));
      ctx.lineTo(curX, curY);
      ctx.lineTo(fprToX(0), curY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = C.orange;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(curX, curY, 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // ROC axis labels
      ctx.fillStyle = C.muted;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) {
        ctx.fillText(v.toFixed(1), fprToX(v), p2.y + p2.h + 15);
      });
      ctx.textAlign = 'right';
      [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) {
        ctx.fillText(v.toFixed(1), p2.x - 6, tprToY(v) + 4);
      });
      ctx.fillStyle = C.ink;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('False Positive Rate (FPR = 1 − Specificity)', p2.x + p2.w / 2, p2.y + p2.h + 32);

      ctx.save();
      ctx.translate(p2.x - 32, p2.y + p2.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('True Positive Rate (TPR = Recall)', 0, 0);
      ctx.restore();

      // ---------------------------------------------------------------------
      // PANEL 3 (RIGHT): Expected Value / Profit Curve vs. Threshold tau
      // ---------------------------------------------------------------------
      var p3 = { x: 816, y: 44, w: 344, h: 350 };
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(p3.x, p3.y, p3.w, p3.h);
      ctx.strokeRect(p3.x, p3.y, p3.w, p3.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('3. Expected Profit Curve EV(τ) per Sample', p3.x + p3.w / 2, p3.y - 12);

      var evMin = 0, evMax = 0.5;
      var tSamples = [], evSamples = [];
      for (var i = 1; i <= 199; i++) {
        var t = i / 200;
        var ev = evAtTau(t, dp);
        tSamples.push(t);
        evSamples.push(ev);
        if (ev < evMin) evMin = ev;
        if (ev > evMax) evMax = ev;
      }
      var padEV = Math.max(1.0, 0.15 * (evMax - evMin));
      var yLo = evMin - padEV, yHi = evMax + padEV;

      function tToX3(t) { return p3.x + 36 + t * (p3.w - 50); }
      function evToY3(ev) { return p3.y + p3.h - 28 - ((ev - yLo) / (yHi - yLo)) * (p3.h - 52); }

      // Zero profit line
      var zeroY = evToY3(0);
      if (zeroY >= p3.y + 15 && zeroY <= p3.y + p3.h - 20) {
        ctx.save();
        ctx.strokeStyle = C.red;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(p3.x + 36, zeroY);
        ctx.lineTo(p3.x + p3.w - 14, zeroY);
        ctx.stroke();
        ctx.fillStyle = C.red;
        ctx.font = '10.5px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Break-even ($0)', p3.x + 42, zeroY - 5);
        ctx.restore();
      }

      // Draw EV(tau) curve
      ctx.beginPath();
      for (var i = 0; i < tSamples.length; i++) {
        var x = tToX3(tSamples[i]), y = evToY3(evSamples[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = C.violet;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Optimal tau* vertical line & star
      var optX3 = tToX3(opt.tau), optY3 = evToY3(opt.ev);
      ctx.save();
      ctx.strokeStyle = C.green;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(optX3, optY3);
      ctx.lineTo(optX3, p3.y + p3.h - 28);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = C.green;
      ctx.beginPath();
      ctx.arc(optX3, optY3, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = 'bold 11.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Max EV at τ*=' + opt.tau.toFixed(2), Math.min(p3.x + p3.w - 62, Math.max(p3.x + 82, optX3)), optY3 - 10);

      // Current threshold marker on EV curve
      var curX3 = tToX3(tau), curY3 = evToY3(curEV);
      ctx.fillStyle = C.orange;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(curX3, curY3, 7.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // X-axis ticks on EV plot
      ctx.fillStyle = C.muted;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      [0, 0.25, 0.5, 0.75, 1.0].forEach(function (v) {
        ctx.fillText(v.toFixed(2), tToX3(v), p3.y + p3.h - 10);
      });
      ctx.fillStyle = C.ink;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Decision Threshold τ (Cut-off Score)', p3.x + p3.w / 2, p3.y + p3.h + 22);

      // Formula box inside EV plot
      ctx.fillStyle = C.panel;
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(p3.x + 44, p3.y + p3.h - 86, p3.w - 58, 48);
      ctx.strokeRect(p3.x + 44, p3.y + p3.h - 86, p3.w - 58, 48);
      ctx.fillStyle = C.ink;
      ctx.font = 'bold 11.5px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('EV(τ) = π₊·TPR(τ)·b_TP − (1−π₊)·FPR(τ)·c_FP', p3.x + 44 + (p3.w - 58) / 2, p3.y + p3.h - 66);
      ctx.fillStyle = C.muted;
      ctx.font = '11px sans-serif';
      ctx.fillText(
        'π₊=' + state.piPos.toFixed(2) + ', b_TP=$' + state.bTP + ', c_FP=$' + state.cFP + ' → τ*=' + opt.tau.toFixed(2),
        p3.x + 44 + (p3.w - 58) / 2,
        p3.y + p3.h - 47
      );
    }

    // Pointer interaction on canvas (drag threshold in Panel 1, 2, or 3)
    function handlePointer(e) {
      var rect = canvas.getBoundingClientRect();
      var mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      var my = (e.clientY - rect.top) * (canvas.height / rect.height);

      if (state.dragging === 'p1') {
        var s = (mx - 46) / 320;
        state.tau = Math.max(0.02, Math.min(0.98, s));
      } else if (state.dragging === 'p2') {
        var fprTarget = Math.max(0.005, Math.min(0.995, (mx - 424) / 350));
        // Find tau whose fpr is closest to fprTarget
        var bestT = 0.5, bestDiff = 999;
        for (var i = 1; i <= 199; i++) {
          var t = i / 200;
          var r = ratesAtTau(t, state.dprime);
          var d = Math.abs(r.fpr - fprTarget);
          if (d < bestDiff) { bestDiff = d; bestT = t; }
        }
        state.tau = bestT;
      } else if (state.dragging === 'p3') {
        var s3 = (mx - (816 + 36)) / (344 - 50);
        state.tau = Math.max(0.02, Math.min(0.98, s3));
      }
      if (sliderTau) sliderTau.value = state.tau;
      draw();
    }

    canvas.addEventListener('mousedown', function (e) {
      var rect = canvas.getBoundingClientRect();
      var mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      if (mx <= 380) state.dragging = 'p1';
      else if (mx <= 790) state.dragging = 'p2';
      else state.dragging = 'p3';
      handlePointer(e);
    });
    window.addEventListener('mousemove', function (e) {
      if (state.dragging) handlePointer(e);
    });
    window.addEventListener('mouseup', function () {
      state.dragging = null;
    });

    if (sliderTau) {
      sliderTau.addEventListener('input', function () {
        state.tau = parseFloat(sliderTau.value);
        draw();
      });
    }
    if (sliderDprime) {
      sliderDprime.addEventListener('input', function () {
        state.dprime = parseFloat(sliderDprime.value);
        draw();
      });
    }
    if (sliderPrior) {
      sliderPrior.addEventListener('input', function () {
        state.piPos = parseFloat(sliderPrior.value);
        draw();
      });
    }
    if (sliderBtp) {
      sliderBtp.addEventListener('input', function () {
        state.bTP = parseFloat(sliderBtp.value);
        draw();
      });
    }
    if (sliderCfp) {
      sliderCfp.addEventListener('input', function () {
        state.cFP = parseFloat(sliderCfp.value);
        draw();
      });
    }
    if (selScenario) {
      selScenario.addEventListener('change', function () {
        var sc = scenarios[selScenario.value] || scenarios.marketing;
        state.piPos = sc.piPos;
        state.bTP = sc.bTP;
        state.cFP = sc.cFP;
        draw();
      });
    }
    if (btnOpt) {
      btnOpt.addEventListener('click', function () {
        var opt = findOptimalTau(state.dprime);
        state.tau = opt.tau;
        if (sliderTau) sliderTau.value = state.tau;
        draw();
      });
    }
    if (btnBal) {
      btnBal.addEventListener('click', function () {
        state.tau = 0.50;
        if (sliderTau) sliderTau.value = 0.50;
        draw();
      });
    }

    draw();
  }

  // =========================================================================
  // DEMO 2: RELIABILITY DIAGRAM & TEMPERATURE SCALING CALIBRATOR
  // =========================================================================
  function initCalDemo() {
    var canvas = document.getElementById('mc-cal-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');

    var selModel = document.getElementById('mc-cal-preset') || document.getElementById('mc-cal-model');
    var sliderTemp = document.getElementById('mc-cal-temp');
    var valTemp = document.getElementById('mc-cal-temp-val');
    var sliderBins = document.getElementById('mc-cal-bins');
    var valBins = document.getElementById('mc-cal-bins-val');
    var btnUncal = document.getElementById('mc-cal-reset-btn') || document.getElementById('mc-cal-btn-uncal');
    var btnOpt = document.getElementById('mc-cal-opt-btn') || document.getElementById('mc-cal-btn-opt');
    var btnSharp = document.getElementById('mc-cal-btn-sharp');
    var statusBanner = document.getElementById('mc-cal-status');
    var readAcc = document.getElementById('mc-cal-acc');
    var readEce = document.getElementById('mc-cal-ece');
    var readMce = document.getElementById('mc-cal-mce');
    var readNll = document.getElementById('mc-cal-nll');

    // Generate deterministic synthetic K=10 class validation set (N=600 samples)
    // where ground-truth probabilities follow temperature T_true, and model logits
    // are scaled by an uncalibrated factor.
    var N = 600, K = 10;
    var baseSamples = [];
    var seed = 1337;
    function rand() {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    }
    function randn() {
      var u = Math.max(1e-7, rand()), v = Math.max(1e-7, rand());
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    for (var i = 0; i < N; i++) {
      // Latent logits z_k at true calibration scale
      var zTrue = [];
      for (var k = 0; k < K; k++) zTrue.push(randn() * 1.15);
      // Boost one dominant class to give ~72% top-1 accuracy
      var dom = i % K;
      zTrue[dom] += 2.35 + 0.5 * randn();

      // Compute true softmax probabilities pTrue
      var maxZ = Math.max.apply(null, zTrue);
      var exps = zTrue.map(function (z) { return Math.exp(z - maxZ); });
      var sumExp = exps.reduce(function (a, b) { return a + b; }, 0);
      var pTrue = exps.map(function (e) { return e / sumExp; });

      // Sample ground-truth label y from pTrue
      var r = rand(), cum = 0, y = K - 1;
      for (var k = 0; k < K; k++) {
        cum += pTrue[k];
        if (r <= cum) { y = k; break; }
      }
      baseSamples.push({ zTrue: zTrue, y: y });
    }

    var modelConfigs = {
      resnet: { scale: 2.35, optT: 2.35, name: 'ResNet-110 on CIFAR-100 (Overconfident Deep Net)' },
      resnet110: { scale: 2.35, optT: 2.35, name: 'ResNet-110 on CIFAR-100 (Overconfident Deep Net)' },
      lenet: { scale: 1.08, optT: 1.08, name: 'LeNet-5 (Nearly Calibrated Shallow Net)' },
      lenet5: { scale: 1.08, optT: 1.08, name: 'LeNet-5 (Nearly Calibrated Shallow Net)' },
      under: { scale: 0.55, optT: 0.55, name: 'Margin-Pushed Classifier (Underconfident)' },
      underconf: { scale: 0.55, optT: 0.55, name: 'Margin-Pushed Classifier (Underconfident)' }
    };

    var state = {
      modelKey: selModel ? selModel.value : 'resnet',
      T: sliderTemp ? parseFloat(sliderTemp.value) : 1.00,
      numBins: sliderBins ? parseInt(sliderBins.value, 10) : 10
    };

    function evaluateAtTemperature(T, modelScale, M) {
      var nBins = M || state.numBins || 10;
      var bins = [];
      for (var m = 0; m < nBins; m++) {
        bins.push({ count: 0, correctSum: 0, confSum: 0 });
      }
      var totalCorrect = 0;
      var totalNLL = 0;
      var totalConf = 0;

      var effScale = modelScale / Math.max(0.15, T);
      for (var i = 0; i < N; i++) {
        var s = baseSamples[i];
        var logits = s.zTrue.map(function (z) { return z * effScale; });
        var maxL = Math.max.apply(null, logits);
        var exps = logits.map(function (l) { return Math.exp(l - maxL); });
        var sumE = exps.reduce(function (a, b) { return a + b; }, 0);
        var probs = exps.map(function (e) { return e / sumE; });

        var predK = 0, maxP = probs[0];
        for (var k = 1; k < K; k++) {
          if (probs[k] > maxP) { maxP = probs[k]; predK = k; }
        }
        var isCorrect = (predK === s.y) ? 1 : 0;
        totalCorrect += isCorrect;
        totalConf += maxP;
        totalNLL += -Math.log(Math.max(1e-12, probs[s.y]));

        var bIdx = Math.min(nBins - 1, Math.floor(maxP * nBins));
        bins[bIdx].count += 1;
        bins[bIdx].correctSum += isCorrect;
        bins[bIdx].confSum += maxP;
      }

      var ece = 0, mce = 0;
      bins.forEach(function (b, m) {
        if (b.count > 0) {
          b.acc = b.correctSum / b.count;
          b.conf = b.confSum / b.count;
          var gap = Math.abs(b.acc - b.conf);
          b.gap = gap;
          ece += (b.count / N) * gap;
          if (gap > mce) mce = gap;
        } else {
          b.acc = (m + 0.5) / nBins;
          b.conf = (m + 0.5) / nBins;
          b.gap = 0;
        }
      });

      return {
        bins: bins,
        acc: totalCorrect / N,
        avgConf: totalConf / N,
        ece: ece,
        mce: mce,
        nll: totalNLL / N
      };
    }

    function draw() {
      var W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.bg;
      ctx.fillRect(0, 0, W, H);

      var mCfg = modelConfigs[state.modelKey] || modelConfigs.resnet110;
      var T = state.T;
      var nBins = state.numBins || 10;
      var res = evaluateAtTemperature(T, mCfg.scale, nBins);

      if (valTemp) valTemp.textContent = T.toFixed(2);
      if (valBins) valBins.textContent = String(nBins);
      if (readAcc) readAcc.textContent = (res.acc * 100).toFixed(1) + '%';
      if (readEce) readEce.textContent = (res.ece * 100).toFixed(2) + '%';
      if (readMce) readMce.textContent = (res.mce * 100).toFixed(1) + '%';
      if (readNll) readNll.textContent = res.nll.toFixed(3);

      var diffConf = res.avgConf - res.acc;
      var regime = Math.abs(diffConf) < 0.025
        ? '<span style="color:#859900;font-weight:bold;">Well-Calibrated (Avg Conf ≈ Accuracy)</span>'
        : (diffConf > 0
          ? '<span style="color:#dc322f;font-weight:bold;">Overconfident (Avg Conf ' + (res.avgConf * 100).toFixed(1) + '% > Acc ' + (res.acc * 100).toFixed(1) + '%)</span>'
          : '<span style="color:#268bd2;font-weight:bold;">Underconfident (Avg Conf ' + (res.avgConf * 100).toFixed(1) + '% < Acc ' + (res.acc * 100).toFixed(1) + '%)</span>');

      if (statusBanner) {
        statusBanner.innerHTML =
          '<b>' + mCfg.name + '</b> at <b>T = ' + T.toFixed(2) + '</b>: ' + regime +
          ' | Notice Top-1 Accuracy stays locked at <b>' + (res.acc * 100).toFixed(1) + '%</b> for all T > 0!';
      }

      // ---------------------------------------------------------------------
      // LEFT COLUMN TOP: Confidence Histogram across 10 Bins
      // ---------------------------------------------------------------------
      var pHist = { x: 48, y: 34, w: 390, h: 175 };
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(pHist.x, pHist.y, pHist.w, pHist.h);
      ctx.strokeRect(pHist.x, pHist.y, pHist.w, pHist.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('1. Confidence Histogram P(p̂ ∈ B_m)', pHist.x + pHist.w / 2, pHist.y - 10);

      var maxFrac = 0.35;
      res.bins.forEach(function (b) {
        var f = b.count / N;
        if (f > maxFrac) maxFrac = f;
      });

      var hBaseY = pHist.y + pHist.h - 22;
      var usableH = pHist.h - 38;
      for (var m = 0; m < nBins; m++) {
        var frac = res.bins[m].count / N;
        var bw = pHist.w / nBins;
        var bx = pHist.x + m * bw;
        var bh = (frac / (maxFrac * 1.1)) * usableH;
        ctx.fillStyle = 'rgba(38, 139, 210, 0.65)';
        ctx.fillRect(bx + 2, hBaseY - bh, bw - 4, bh);
        ctx.strokeStyle = C.blue;
        ctx.strokeRect(bx + 2, hBaseY - bh, bw - 4, bh);
      }

      // Vertical lines for Avg Accuracy vs Avg Confidence
      var accX = pHist.x + res.acc * pHist.w;
      var confX = pHist.x + res.avgConf * pHist.w;

      ctx.save();
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(accX, pHist.y + 8);
      ctx.lineTo(accX, hBaseY);
      ctx.stroke();

      ctx.strokeStyle = C.red;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(confX, pHist.y + 8);
      ctx.lineTo(confX, hBaseY);
      ctx.stroke();
      ctx.restore();

      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillStyle = C.green;
      ctx.fillText('Avg Acc: ' + (res.acc * 100).toFixed(1) + '%', pHist.x + 8, pHist.y + 16);
      ctx.fillStyle = C.red;
      ctx.fillText('Avg Conf: ' + (res.avgConf * 100).toFixed(1) + '%', pHist.x + 135, pHist.y + 16);

      ctx.fillStyle = C.muted;
      ctx.font = '10.5px sans-serif';
      ctx.textAlign = 'center';
      [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) {
        ctx.fillText(v.toFixed(1), pHist.x + v * pHist.w, hBaseY + 14);
      });

      // ---------------------------------------------------------------------
      // LEFT COLUMN BOTTOM: Validation NLL(T) & Optimal Temperature T*
      // ---------------------------------------------------------------------
      var pNll = { x: 48, y: 242, w: 390, h: 182 };
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(pNll.x, pNll.y, pNll.w, pNll.h);
      ctx.strokeRect(pNll.x, pNll.y, pNll.w, pNll.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 13.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('2. Validation NLL(T) & ECE(T) vs. Temperature T', pNll.x + pNll.w / 2, pNll.y - 10);

      var tGrid = [], nllGrid = [], eceGrid = [];
      var minNll = 999, maxNll = 0;
      for (var i = 0; i <= 64; i++) {
        var tVal = 0.35 + (3.15 * i) / 64;
        var evRes = evaluateAtTemperature(tVal, mCfg.scale);
        tGrid.push(tVal);
        nllGrid.push(evRes.nll);
        eceGrid.push(evRes.ece);
        if (evRes.nll < minNll) minNll = evRes.nll;
        if (evRes.nll > maxNll) maxNll = evRes.nll;
      }

      function tToXN(t) { return pNll.x + 32 + ((t - 0.35) / 3.15) * (pNll.w - 48); }
      function nllToYN(v) {
        var cl = Math.min(maxNll, Math.max(minNll, v));
        return pNll.y + pNll.h - 24 - ((cl - minNll) / Math.max(0.2, maxNll - minNll)) * (pNll.h - 46);
      }

      // Draw NLL curve
      ctx.beginPath();
      for (var i = 0; i < tGrid.length; i++) {
        var x = tToXN(tGrid[i]), y = nllToYN(nllGrid[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = C.violet;
      ctx.lineWidth = 2.8;
      ctx.stroke();

      // Optimal T* vertical marker
      var optTX = tToXN(mCfg.optT);
      var optRes = evaluateAtTemperature(mCfg.optT, mCfg.scale);
      var optTY = nllToYN(optRes.nll);
      ctx.save();
      ctx.strokeStyle = C.green;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(optTX, pNll.y + 12);
      ctx.lineTo(optTX, pNll.y + pNll.h - 24);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = C.green;
      ctx.beginPath();
      ctx.arc(optTX, optTY, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('T* = ' + mCfg.optT.toFixed(2) + ' (Min NLL)', optTX, optTY - 10);

      // Current T marker on NLL curve
      var curTX = tToXN(Math.max(0.35, Math.min(3.5, T)));
      var curTY = nllToYN(res.nll);
      ctx.fillStyle = C.orange;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(curTX, curTY, 7.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = C.muted;
      ctx.font = '10.5px sans-serif';
      ctx.textAlign = 'center';
      [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5].forEach(function (tv) {
        ctx.fillText('T=' + tv.toFixed(1), tToXN(tv), pNll.y + pNll.h - 8);
      });

      // ---------------------------------------------------------------------
      // CENTER-RIGHT: Exact 1:1 Square Reliability Diagram (350 x 350)
      // ---------------------------------------------------------------------
      var relSize = 350;
      var pRel = { x: 492, y: 44, w: relSize, h: relSize }; // Exact 1:1 square!
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(pRel.x, pRel.y, pRel.w, pRel.h);
      ctx.strokeRect(pRel.x, pRel.y, pRel.w, pRel.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('3. Reliability Diagram (1:1 Square: Accuracy vs. Confidence)', pRel.x + pRel.w / 2, pRel.y - 12);

      function cToX(c) { return pRel.x + c * pRel.w; }
      function aToY(a) { return pRel.y + (1 - a) * pRel.h; }

      // Grid lines
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      [0.2, 0.4, 0.6, 0.8].forEach(function (g) {
        ctx.beginPath();
        ctx.moveTo(cToX(g), pRel.y);
        ctx.lineTo(cToX(g), pRel.y + pRel.h);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pRel.x, aToY(g));
        ctx.lineTo(pRel.x + pRel.w, aToY(g));
        ctx.stroke();
      });

      // Draw bins: blue accuracy bar + red calibration gap bar (Guo et al. 2017 style)
      var binW = pRel.w / nBins;
      for (var m = 0; m < nBins; m++) {
        var b = res.bins[m];
        var bx = pRel.x + m * binW;
        if (b.count === 0) continue;

        var accY = aToY(b.acc);
        var confY = aToY(b.conf);
        var baseRelY = aToY(0);

        // Accuracy bar (Blue)
        ctx.fillStyle = 'rgba(38, 139, 210, 0.72)';
        ctx.fillRect(bx + 2, accY, binW - 4, baseRelY - accY);
        ctx.strokeStyle = C.blue;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(bx + 2, accY, binW - 4, baseRelY - accY);

        // Gap bar between b.acc and b.conf (Red translucent with border)
        var topGapY = Math.min(accY, confY);
        var botGapY = Math.max(accY, confY);
        ctx.fillStyle = 'rgba(220, 50, 47, 0.34)';
        ctx.fillRect(bx + 2, topGapY, binW - 4, botGapY - topGapY);
        ctx.strokeStyle = C.red;
        ctx.lineWidth = 1.4;
        ctx.strokeRect(bx + 2, topGapY, binW - 4, botGapY - topGapY);
      }

      // Perfect calibration diagonal y = x
      ctx.save();
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(cToX(0), aToY(0));
      ctx.lineTo(cToX(1), aToY(1));
      ctx.stroke();
      ctx.restore();

      // Legend inside Reliability Diagram
      ctx.fillStyle = 'rgba(253, 246, 227, 0.92)';
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(pRel.x + 12, pRel.y + 12, 165, 62);
      ctx.strokeRect(pRel.x + 12, pRel.y + 12, 165, 62);

      ctx.fillStyle = 'rgba(38, 139, 210, 0.75)';
      ctx.fillRect(pRel.x + 20, pRel.y + 22, 14, 12);
      ctx.fillStyle = C.ink;
      ctx.font = 'bold 11.5px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Outputs: acc(B_m)', pRel.x + 40, pRel.y + 32);

      ctx.fillStyle = 'rgba(220, 50, 47, 0.40)';
      ctx.fillRect(pRel.x + 20, pRel.y + 42, 14, 12);
      ctx.strokeStyle = C.red;
      ctx.strokeRect(pRel.x + 20, pRel.y + 42, 14, 12);
      ctx.fillStyle = C.red;
      ctx.fillText('Gap: |acc − conf|', pRel.x + 40, pRel.y + 52);

      ctx.fillStyle = C.muted;
      ctx.font = '10.5px sans-serif';
      ctx.fillText('- - Perfect: P(Ŷ=Y|P̂=p)=p', pRel.x + 20, pRel.y + 68);

      // Axis ticks & titles for Reliability Diagram
      ctx.fillStyle = C.muted;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) {
        ctx.fillText(v.toFixed(1), cToX(v), pRel.y + pRel.h + 15);
      });
      ctx.textAlign = 'right';
      [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) {
        ctx.fillText(v.toFixed(1), pRel.x - 6, aToY(v) + 4);
      });

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Confidence conf(B_m)', pRel.x + pRel.w / 2, pRel.y + pRel.h + 32);

      ctx.save();
      ctx.translate(pRel.x - 32, pRel.y + pRel.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Empirical Accuracy acc(B_m)', 0, 0);
      ctx.restore();

      // ---------------------------------------------------------------------
      // RIGHT SCORECARD PANEL: Temperature Scaling Math & Metrics
      // ---------------------------------------------------------------------
      var pCard = { x: 862, y: 44, w: 298, h: 350 };
      ctx.fillStyle = C.panel;
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.fillRect(pCard.x, pCard.y, pCard.w, pCard.h);
      ctx.strokeRect(pCard.x, pCard.y, pCard.w, pCard.h);

      ctx.fillStyle = C.ink;
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Temperature Scaling Summary', pCard.x + pCard.w / 2, pCard.y + 24);

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(pCard.x + 12, pCard.y + 36, pCard.w - 24, 54);
      ctx.strokeRect(pCard.x + 12, pCard.y + 36, pCard.w - 24, 54);
      ctx.fillStyle = C.ink;
      ctx.font = 'bold 12.5px monospace';
      ctx.fillText('q̂_i = max_k softmax(z_i / T)^(k)', pCard.x + pCard.w / 2, pCard.y + 58);
      ctx.fillStyle = C.muted;
      ctx.font = '11px sans-serif';
      ctx.fillText('Single scalar T > 0 fit on Validation NLL', pCard.x + pCard.w / 2, pCard.y + 78);

      var rows = [
        { label: 'Temperature T:', val: T.toFixed(2) + (Math.abs(T - mCfg.optT) < 0.06 ? ' (Optimal T*)' : ''), col: C.orange },
        { label: 'Top-1 Accuracy:', val: (res.acc * 100).toFixed(1) + '% (Invariant!)', col: C.green },
        { label: 'Average Confidence:', val: (res.avgConf * 100).toFixed(1) + '%', col: C.ink },
        { label: 'Expected Cal. Error (ECE):', val: (res.ece * 100).toFixed(2) + '%', col: res.ece < 0.03 ? C.green : C.red },
        { label: 'Maximum Cal. Error (MCE):', val: (res.mce * 100).toFixed(1) + '%', col: C.violet },
        { label: 'Validation NLL (Cross-Ent):', val: res.nll.toFixed(3), col: C.blue }
      ];

      var ry = pCard.y + 114;
      rows.forEach(function (r) {
        ctx.textAlign = 'left';
        ctx.fillStyle = C.muted;
        ctx.font = '12px sans-serif';
        ctx.fillText(r.label, pCard.x + 16, ry);
        ctx.textAlign = 'right';
        ctx.fillStyle = r.col;
        ctx.font = 'bold 12.5px monospace';
        ctx.fillText(r.val, pCard.x + pCard.w - 16, ry);
        ry += 30;
      });

      // Key insight box at bottom of scorecard
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 1.4;
      ctx.fillRect(pCard.x + 12, pCard.y + 292, pCard.w - 24, 46);
      ctx.strokeRect(pCard.x + 12, pCard.y + 292, pCard.w - 24, 46);
      ctx.fillStyle = C.ink;
      ctx.font = 'bold 11.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Why Accuracy Never Changes:', pCard.x + pCard.w / 2, pCard.y + 310);
      ctx.font = '11px monospace';
      ctx.fillStyle = C.green;
      ctx.fillText('argmax_k (z_ik / T) = argmax_k (z_ik)', pCard.x + pCard.w / 2, pCard.y + 328);
    }

    // Allow clicking/dragging on Panel 2 (NLL vs T plot) to change T
    var draggingT = false;
    function handleTPointer(e) {
      var rect = canvas.getBoundingClientRect();
      var mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      var t = 0.35 + ((mx - (48 + 32)) / (390 - 48)) * 3.15;
      state.T = Math.max(0.35, Math.min(3.50, t));
      if (sliderTemp) sliderTemp.value = state.T;
      draw();
    }

    canvas.addEventListener('mousedown', function (e) {
      var rect = canvas.getBoundingClientRect();
      var mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      var my = (e.clientY - rect.top) * (canvas.height / rect.height);
      if (mx >= 48 && mx <= 440 && my >= 230) {
        draggingT = true;
        handleTPointer(e);
      }
    });
    window.addEventListener('mousemove', function (e) {
      if (draggingT) handleTPointer(e);
    });
    window.addEventListener('mouseup', function () {
      draggingT = false;
    });

    if (sliderTemp) {
      sliderTemp.addEventListener('input', function () {
        state.T = parseFloat(sliderTemp.value);
        draw();
      });
    }
    if (sliderBins) {
      sliderBins.addEventListener('input', function () {
        state.numBins = parseInt(sliderBins.value, 10) || 10;
        draw();
      });
    }
    if (selModel) {
      selModel.addEventListener('change', function () {
        state.modelKey = selModel.value;
        state.T = 1.00;
        if (sliderTemp) sliderTemp.value = 1.00;
        draw();
      });
    }
    if (btnUncal) {
      btnUncal.addEventListener('click', function () {
        state.T = 1.00;
        if (sliderTemp) sliderTemp.value = 1.00;
        draw();
      });
    }
    if (btnOpt) {
      btnOpt.addEventListener('click', function () {
        var mCfg = modelConfigs[state.modelKey] || modelConfigs.resnet110;
        state.T = mCfg.optT;
        if (sliderTemp) sliderTemp.value = state.T;
        draw();
      });
    }
    if (btnSharp) {
      btnSharp.addEventListener('click', function () {
        state.T = 0.55;
        if (sliderTemp) sliderTemp.value = 0.55;
        draw();
      });
    }

    draw();
  }

  function initAll() {
    initRocDemo();
    initCalDemo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }
})();
