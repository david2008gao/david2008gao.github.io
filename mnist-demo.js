/*
 * MNIST digit-recognizer demo.
 * - Lets the visitor draw a digit on a canvas.
 * - Downsamples the drawing to a 28x28 grayscale image (same preprocessing
 *   as MNIST: centered strokes, values in [0, 1]).
 * - Runs a hand-written forward pass of the MLP trained in train_mnist.py
 *   using the weights exported to weights.json. No ML library needed —
 *   it's just matrix multiplies.
 *
 * Requires this HTML structure (see mnist-demo.html):
 *   <canvas id="mnist-canvas">
 *   <button id="mnist-clear">
 *   <button id="mnist-predict">
 *   <div id="mnist-result">
 */

(function () {
  const CANVAS_SIZE = 280;      // on-screen canvas size (px)
  const MODEL_INPUT_SIZE = 28;  // MNIST native resolution
  const WEIGHTS_URL = "weights.json"; // adjust path if you host it elsewhere

  let weights = null;
  let canvas, ctx, resultEl, predictBtn;
  let drawing = false;
  let lastX = 0, lastY = 0;

  // ---------------------------------------------------------------------
  // Load trained weights once on page load
  // ---------------------------------------------------------------------
  async function loadWeights() {
    const res = await fetch(WEIGHTS_URL);
    if (!res.ok) throw new Error(`Could not load ${WEIGHTS_URL} (${res.status})`);
    weights = await res.json();
  }

  // ---------------------------------------------------------------------
  // Canvas drawing
  // ---------------------------------------------------------------------
  function setupCanvas() {
    canvas = document.getElementById("mnist-canvas");
    ctx = canvas.getContext("2d");
    resetCanvas();

    const getPos = (e) => {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: (clientX - rect.left) * (canvas.width / rect.width),
        y: (clientY - rect.top) * (canvas.height / rect.height),
      };
    };

    const start = (e) => {
      e.preventDefault();
      drawing = true;
      const p = getPos(e);
      lastX = p.x;
      lastY = p.y;
    };
    const move = (e) => {
      if (!drawing) return;
      e.preventDefault();
      const p = getPos(e);
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 18;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
      lastX = p.x;
      lastY = p.y;
    };
    const end = () => { drawing = false; };

    canvas.addEventListener("mousedown", start);
    canvas.addEventListener("mousemove", move);
    window.addEventListener("mouseup", end);

    canvas.addEventListener("touchstart", start, { passive: false });
    canvas.addEventListener("touchmove", move, { passive: false });
    canvas.addEventListener("touchend", end);
  }

  function resetCanvas() {
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    ctx.fillStyle = "#000000"; // MNIST digits are white-on-black
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // ---------------------------------------------------------------------
  // Preprocess: canvas -> 28x28 grayscale array in [0, 1].
  //
  // This mirrors how the MNIST images themselves were built, which matters a
  // lot: the network only ever saw digits prepared this exact way, so a raw
  // 280 -> 28 squash (whatever size/position the user happened to draw at) is
  // off-distribution and predicts badly. The original recipe is:
  //   1. crop to the bounding box of the ink,
  //   2. scale that box, preserving aspect ratio, so its larger side is 20px,
  //   3. paste into a 28x28 field positioned so the digit's centre of mass
  //      lands on the centre of the field (not the bounding box centre).
  // Step 2 makes the model size-invariant, step 3 makes it position-invariant.
  // ---------------------------------------------------------------------
  const DIGIT_BOX = 20;     // MNIST fits the digit into a 20x20 box...
  const INK_THRESHOLD = 8;  // 0-255; ignore stray anti-aliasing when finding the bbox

  // Downscale a source region in repeated halving steps. A single large
  // drawImage() downscale skips source pixels in some browsers, which eats
  // thin strokes; halving keeps the box-filter averaging intact.
  function downscaleRegion(src, sx, sy, sw, sh, dw, dh) {
    let cur = document.createElement("canvas");
    cur.width = sw;
    cur.height = sh;
    let cctx = cur.getContext("2d");
    cctx.imageSmoothingEnabled = true;
    cctx.imageSmoothingQuality = "high";
    cctx.drawImage(src, sx, sy, sw, sh, 0, 0, sw, sh);

    let cw = sw, ch = sh;
    while (cw > dw * 2 && ch > dh * 2) {
      const nw = Math.max(dw, Math.round(cw / 2));
      const nh = Math.max(dh, Math.round(ch / 2));
      const next = document.createElement("canvas");
      next.width = nw;
      next.height = nh;
      const nctx = next.getContext("2d");
      nctx.imageSmoothingEnabled = true;
      nctx.imageSmoothingQuality = "high";
      nctx.drawImage(cur, 0, 0, cw, ch, 0, 0, nw, nh);
      cur = next;
      cctx = nctx;
      cw = nw;
      ch = nh;
    }

    const out = document.createElement("canvas");
    out.width = dw;
    out.height = dh;
    const octx = out.getContext("2d");
    octx.imageSmoothingEnabled = true;
    octx.imageSmoothingQuality = "high";
    octx.drawImage(cur, 0, 0, cw, ch, 0, 0, dw, dh);
    return octx.getImageData(0, 0, dw, dh).data;
  }

  // Returns a 784-length array, or null if the canvas is blank.
  function canvasToModelInput() {
    const src = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    // --- 1. bounding box of the ink -------------------------------------
    let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        if (src[(y * canvas.width + x) * 4] > INK_THRESHOLD) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null; // nothing drawn

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;

    // --- 2. scale the longer side to 20px, keeping aspect ratio ---------
    const scale = DIGIT_BOX / Math.max(boxW, boxH);
    const dw = Math.max(1, Math.round(boxW * scale));
    const dh = Math.max(1, Math.round(boxH * scale));
    const small = downscaleRegion(canvas, minX, minY, boxW, boxH, dw, dh);

    // --- 3. centre of mass of the scaled digit --------------------------
    let mass = 0, mx = 0, my = 0;
    const px = new Float32Array(dw * dh);
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        const v = small[(y * dw + x) * 4] / 255;
        px[y * dw + x] = v;
        mass += v;
        mx += v * x;
        my += v * y;
      }
    }
    // Fall back to the geometric centre if the digit is somehow empty.
    const comX = mass > 0 ? mx / mass : (dw - 1) / 2;
    const comY = mass > 0 ? my / mass : (dh - 1) / 2;

    // Paste so the centre of mass sits at the centre of the 28x28 field,
    // clamped so the digit can never be pushed off the edge.
    const centre = MODEL_INPUT_SIZE / 2;
    const offX = clamp(Math.round(centre - comX), 0, MODEL_INPUT_SIZE - dw);
    const offY = clamp(Math.round(centre - comY), 0, MODEL_INPUT_SIZE - dh);

    const input = new Array(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE).fill(0);
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        input[(y + offY) * MODEL_INPUT_SIZE + (x + offX)] = px[y * dw + x];
      }
    }

    drawPreview(input);
    return input;
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  // Optional: if the page has a <canvas id="mnist-preview">, show the exact
  // 28x28 the model sees. Handy for eyeballing that preprocessing is sane.
  function drawPreview(input) {
    const el = document.getElementById("mnist-preview");
    if (!el) return;
    el.width = MODEL_INPUT_SIZE;
    el.height = MODEL_INPUT_SIZE;
    const pctx = el.getContext("2d");
    const img = pctx.createImageData(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    for (let i = 0; i < input.length; i++) {
      const v = Math.round(input[i] * 255);
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    pctx.putImageData(img, 0, 0);
  }

  // ---------------------------------------------------------------------
  // Forward pass: Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax
  // Mirrors the PyTorch MLP in train_mnist.py exactly.
  // ---------------------------------------------------------------------
  function linear(x, layer) {
    const { W, b } = layer; // W: [out][in], b: [out]
    const out = new Array(W.length);
    for (let o = 0; o < W.length; o++) {
      let sum = b[o];
      const row = W[o];
      for (let i = 0; i < row.length; i++) {
        sum += row[i] * x[i];
      }
      out[o] = sum;
    }
    return out;
  }

  function relu(x) {
    return x.map((v) => (v > 0 ? v : 0));
  }

  function softmax(x) {
    const max = Math.max(...x);
    const exps = x.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  function predict(input) {
    let h = relu(linear(input, weights.fc1));
    h = relu(linear(h, weights.fc2));
    const logits = linear(h, weights.fc3);
    return softmax(logits);
  }

  // ---------------------------------------------------------------------
  // UI wiring
  // ---------------------------------------------------------------------
  function renderResult(probs) {
    const top = probs.reduce(
      (best, p, i) => (p > best.p ? { digit: i, p } : best),
      { digit: -1, p: -1 }
    );
    const confidence = (top.p * 100).toFixed(1);

    const bars = probs
      .map((p, digit) => {
        const pct = (p * 100).toFixed(1);
        return `
          <div class="mnist-bar-row">
            <span class="mnist-bar-label">${digit}</span>
            <div class="mnist-bar-track">
              <div class="mnist-bar-fill" style="width:${pct}%"></div>
            </div>
            <span class="mnist-bar-pct">${pct}%</span>
          </div>`;
      })
      .join("");

    resultEl.innerHTML = `
      <div class="mnist-top-prediction">Prediction: <strong>${top.digit}</strong> (${confidence}% confidence)</div>
      <div class="mnist-bars">${bars}</div>
    `;
  }

  async function handlePredict() {
    if (!weights) {
      resultEl.textContent = "Model is still loading, try again in a second…";
      return;
    }
    const input = canvasToModelInput();
    if (!input) {
      resultEl.textContent = "Draw a digit first!";
      return;
    }
    predictBtn.disabled = true;
    const probs = predict(input);
    renderResult(probs);
    predictBtn.disabled = false;
  }

  function handleClear() {
    resetCanvas();
    resultEl.innerHTML = "";
  }

  async function init() {
    const canvasEl = document.getElementById("mnist-canvas");
    const clearBtn = document.getElementById("mnist-clear");
    resultEl = document.getElementById("mnist-result");
    predictBtn = document.getElementById("mnist-predict");

    if (!canvasEl || !clearBtn || !predictBtn || !resultEl) {
      console.error(
        "mnist-demo.js: couldn't find #mnist-canvas / #mnist-clear / #mnist-predict / #mnist-result in the page. " +
        "Make sure mnist-demo.html's markup is on the page before this script runs."
      );
      return;
    }

    setupCanvas();
    clearBtn.addEventListener("click", handleClear);
    predictBtn.addEventListener("click", handlePredict);

    resultEl.textContent = "Loading model…";
    try {
      await loadWeights();
      resultEl.textContent = "Draw a digit and click Predict.";
    } catch (err) {
      resultEl.textContent = "Failed to load model: " + err.message;
      console.error(err);
    }
  }

  // Run now if the DOM is already parsed (e.g. this script was injected
  // after page load by a CMS/site builder), otherwise wait for it.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();