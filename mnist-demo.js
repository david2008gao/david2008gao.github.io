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
  // Preprocess: canvas -> 28x28 grayscale array, normalized to [0, 1],
  // roughly centered the way MNIST digits are.
  // ---------------------------------------------------------------------
  function canvasToModelInput() {
    // Downscale directly to 28x28 using an offscreen canvas.
    const off = document.createElement("canvas");
    off.width = MODEL_INPUT_SIZE;
    off.height = MODEL_INPUT_SIZE;
    const octx = off.getContext("2d");
    octx.fillStyle = "#000000";
    octx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    octx.drawImage(canvas, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

    const imgData = octx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
    const input = new Array(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    for (let i = 0; i < MODEL_INPUT_SIZE * MODEL_INPUT_SIZE; i++) {
      // Use the red channel (grayscale strokes) as the intensity, scaled to [0,1]
      input[i] = imgData[i * 4] / 255;
    }
    return input;
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
    predictBtn.disabled = true;
    const input = canvasToModelInput();
    const probs = predict(input);
    renderResult(probs);
    predictBtn.disabled = false;
  }

  function handleClear() {
    resetCanvas();
    resultEl.innerHTML = "";
  }

  document.addEventListener("DOMContentLoaded", async () => {
    setupCanvas();
    resultEl = document.getElementById("mnist-result");
    predictBtn = document.getElementById("mnist-predict");
    document.getElementById("mnist-clear").addEventListener("click", handleClear);
    predictBtn.addEventListener("click", handlePredict);

    resultEl.textContent = "Loading model…";
    try {
      await loadWeights();
      resultEl.textContent = "Draw a digit and click Predict.";
    } catch (err) {
      resultEl.textContent = "Failed to load model: " + err.message;
      console.error(err);
    }
  });
})();
