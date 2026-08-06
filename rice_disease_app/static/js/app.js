(() => {
  "use strict";

  const qs = (sel, el = document) => el.querySelector(sel);

  // ---------------------------------------------------------------------
  // Elements
  // ---------------------------------------------------------------------
  const dropzone = qs("#dropzone");
  const fileInput = qs("#fileInput");
  const dzEmpty = qs("#dzEmpty");
  const dzPreview = qs("#dzPreview");
  const dzCamera = qs("#dzCamera");
  const previewImg = qs("#previewImg");
  const scanSweep = qs("#scanSweep");
  const browseBtn = qs("#browseBtn");
  const cameraBtn = qs("#cameraBtn");
  const cancelCameraBtn = qs("#cancelCameraBtn");
  const captureBtn = qs("#captureBtn");
  const cameraVideo = qs("#cameraVideo");
  const analyzeBtn = qs("#analyzeBtn");
  const resetBtn = qs("#resetBtn");
  const benchStatus = qs("#benchStatus");
  const ticket = qs("#ticket");
  const statusDot = qs("#statusDot");
  const statusText = qs("#statusText");
  const statAccuracy = qs("#statAccuracy");

  let currentFile = null;
  let cameraStream = null;
  let modelInfo = null;

  // ---------------------------------------------------------------------
  // Status pill — is the model actually loaded on the backend?
  // ---------------------------------------------------------------------
  async function checkStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.model_loaded) {
        statusDot.className = "status-dot is-ready";
        statusText.textContent = `Model loaded (${data.model_file})`;
      } else {
        statusDot.className = "status-dot is-error";
        statusText.textContent = "No model found — see README";
      }
    } catch {
      statusDot.className = "status-dot is-error";
      statusText.textContent = "Backend unreachable";
    }
  }

  // ---------------------------------------------------------------------
  // Model info — pathology cards + calibration data
  // ---------------------------------------------------------------------
  async function loadModelInfo() {
    try {
      const res = await fetch("/api/model-info");
      modelInfo = await res.json();
      if (modelInfo.headline_accuracy) {
        statAccuracy.textContent = (modelInfo.headline_accuracy * 100).toFixed(1) + "%";
      }
      renderPathologyCards();
      initChart();
    } catch (err) {
      console.error("Failed to load model info", err);
    }
  }

  function renderPathologyCards() {
    const grid = qs("#pathGrid");
    const classes = modelInfo.classes;
    grid.innerHTML = classes
      .map((cls) => {
        const info = modelInfo.disease_info[cls];
        const solutionItems = (info.solutions || [])
          .map((s, i) => `<li class="solution-item"><span class="solution-num">${i + 1}</span><span>${s}</span></li>`)
          .join("");
        return `
          <article class="path-card" id="card-${cls}">
            <div class="path-card-top ${info.color}"></div>
            <p class="path-card-tag">${info.type} pathogen</p>
            <h3 class="path-card-name">${info.display_name}</h3>
            <p class="path-card-pathogen">${info.pathogen}</p>
            <p class="path-card-text"><b>Symptoms.</b> ${info.symptoms}</p>
            <p class="path-card-text"><b>Conditions.</b> ${info.conditions}</p>
            <p class="path-card-note">${info.notes}</p>
            ${solutionItems ? `
            <div class="solutions-block">
              <p class="solutions-title">&#9679; Treatment &amp; Solutions</p>
              <ol class="solutions-list">${solutionItems}</ol>
            </div>` : ""}
          </article>
        `;
      })
      .join("");
  }

  // ---------------------------------------------------------------------
  // Dropzone: click / drag-drop / keyboard
  // ---------------------------------------------------------------------
  browseBtn.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("click", (e) => {
    if (dzEmpty.hidden) return; // only open browser when in empty state
    if (e.target === browseBtn || e.target === cameraBtn) return;
    fileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !dzEmpty.hidden) {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-drag");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-drag");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      setBenchStatus("That doesn't look like an image file.", true);
      return;
    }
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    dzEmpty.hidden = true;
    dzCamera.hidden = true;
    dzPreview.hidden = false;
    ticket.hidden = true;
    setBenchStatus("");
    stopCamera();
  }

  resetBtn.addEventListener("click", () => {
    currentFile = null;
    fileInput.value = "";
    dzPreview.hidden = true;
    dzEmpty.hidden = false;
    ticket.hidden = true;
    setBenchStatus("");
  });

  // ---------------------------------------------------------------------
  // Camera capture
  // ---------------------------------------------------------------------
  cameraBtn.addEventListener("click", async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      cameraVideo.srcObject = cameraStream;
      dzEmpty.hidden = true;
      dzCamera.hidden = false;
    } catch {
      setBenchStatus("Couldn't access the camera — check permissions.", true);
    }
  });

  cancelCameraBtn.addEventListener("click", () => {
    stopCamera();
    dzCamera.hidden = true;
    dzEmpty.hidden = false;
  });

  captureBtn.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;
    canvas.getContext("2d").drawImage(cameraVideo, 0, 0);
    canvas.toBlob((blob) => {
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      handleFile(file);
    }, "image/jpeg", 0.92);
  });

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
  }

  // ---------------------------------------------------------------------
  // Run diagnosis
  // ---------------------------------------------------------------------
  analyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    scanSweep.hidden = false;
    setBenchStatus("Reading specimen…");
    ticket.hidden = true;

    const form = new FormData();
    form.append("image", currentFile);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        setBenchStatus(data.error || "Something went wrong.", true);
      } else {
        setBenchStatus("Diagnosis complete.");
        renderTicket(data);
      }
    } catch {
      setBenchStatus("Couldn't reach the backend. Is app.py running?", true);
    } finally {
      analyzeBtn.disabled = false;
      scanSweep.hidden = true;
    }
  });

  function setBenchStatus(msg, isError = false) {
    benchStatus.textContent = msg;
    benchStatus.classList.toggle("is-error", isError);
  }

  function confidenceTier(p) {
    if (p >= 0.9) return "High confidence";
    if (p >= 0.7) return "Medium confidence";
    return "Low confidence";
  }

  function renderTicket(data) {
    qs("#ticketCase").textContent = data.case_id;
    qs("#ticketThumb").src = previewImg.src;
    qs("#ticketDisease").textContent = data.display_name;
    qs("#ticketLink").href = `#card-${data.predicted_class}`;
    qs("#ticketLink").onclick = () => highlightPathCard(data.predicted_class);

    const pct = data.confidence * 100;
    qs("#dialNum").textContent = pct.toFixed(1) + "%";
    qs("#dialTier").textContent = confidenceTier(data.confidence);

    const circumference = 2 * Math.PI * 50;
    const dialFill = qs("#dialFill");
    dialFill.style.strokeDasharray = `${circumference}`;
    dialFill.style.strokeDashoffset = `${circumference}`;
    requestAnimationFrame(() => {
      dialFill.style.strokeDashoffset = `${circumference * (1 - data.confidence)}`;
    });

    const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    const ledger = qs("#ledger");
    ledger.innerHTML = sorted
      .map(([cls, p], i) => {
        const name = modelInfo?.disease_info?.[cls]?.display_name || cls;
        return `
          <div class="ledger-row ${i === 0 ? "is-top" : ""}">
            <span class="ledger-name">${name}</span>
            <span class="ledger-bar"><span class="ledger-bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
            <span class="ledger-pct">${(p * 100).toFixed(1)}%</span>
          </div>
        `;
      })
      .join("");

    ticket.hidden = false;
  }

  function highlightPathCard(cls) {
    document.querySelectorAll(".path-card").forEach((c) => c.classList.remove("is-highlight"));
    const card = qs(`#card-${cls}`);
    if (card) {
      card.classList.add("is-highlight");
      setTimeout(() => card.classList.remove("is-highlight"), 2200);
    }
  }

  // ---------------------------------------------------------------------
  // Calibration chart — hand-rolled SVG line chart, no dependencies
  // ---------------------------------------------------------------------
  let activeMetric = "accuracy";

  function initChart() {
    document.querySelectorAll(".chart-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".chart-tab").forEach((t) => {
          t.classList.remove("is-active");
          t.setAttribute("aria-selected", "false");
        });
        tab.classList.add("is-active");
        tab.setAttribute("aria-selected", "true");
        activeMetric = tab.dataset.metric;
        drawChart();
      });
    });
    drawChart();
  }

  function drawChart() {
    const mount = qs("#chartMount");
    if (!modelInfo) return;

    const baseline = modelInfo.baseline;
    const augmented = modelInfo.augmented;

    const key = activeMetric === "accuracy" ? "val_accuracy" : "val_loss";
    const baseSeries = baseline[key];
    const augSeries = augmented[key];

    const W = 900, H = 300;
    const padL = 44, padR = 16, padT = 16, padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const maxEpoch = augmented.epochs; // 26, the longer run
    const allVals = baseSeries.concat(augSeries);
    let yMax = Math.max(...allVals);
    let yMin = 0;
    if (activeMetric === "loss") {
      yMax = Math.ceil(yMax * 10) / 10 + 0.1;
    } else {
      yMax = 1;
    }

    const xAt = (i) => padL + (plotW * i) / (maxEpoch - 1);
    const yAt = (v) => padT + plotH - (plotH * (v - yMin)) / (yMax - yMin);

    const toPath = (arr) =>
      arr.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");

    const gridCount = 4;
    let gridLines = "";
    let yLabels = "";
    for (let g = 0; g <= gridCount; g++) {
      const v = yMin + ((yMax - yMin) * g) / gridCount;
      const y = yAt(v).toFixed(1);
      gridLines += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--ink-line)" stroke-width="1"/>`;
      const label = activeMetric === "accuracy" ? `${Math.round(v * 100)}%` : v.toFixed(2);
      yLabels += `<text x="${padL - 8}" y="${Number(y) + 3}" font-size="10" text-anchor="end" fill="var(--text-ink-dim)" font-family="var(--font-mono)">${label}</text>`;
    }

    let xLabels = "";
    for (let e = 0; e < maxEpoch; e += 5) {
      xLabels += `<text x="${xAt(e).toFixed(1)}" y="${H - 8}" font-size="10" text-anchor="middle" fill="var(--text-ink-dim)" font-family="var(--font-mono)">${e + 1}</text>`;
    }
    xLabels += `<text x="${xAt(maxEpoch - 1).toFixed(1)}" y="${H - 8}" font-size="10" text-anchor="middle" fill="var(--text-ink-dim)" font-family="var(--font-mono)">${maxEpoch}</text>`;

    const dots = (arr, color) =>
      arr
        .map(
          (v, i) =>
            `<circle cx="${xAt(i).toFixed(1)}" cy="${yAt(v).toFixed(1)}" r="2.5" fill="${color}" class="chart-dot" data-i="${i}" data-v="${v}"/>`
        )
        .join("");

    mount.innerHTML = `
      <div style="position:relative">
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
          ${gridLines}
          <path d="${toPath(baseSeries)}" fill="none" stroke="var(--text-ink-dim)" stroke-width="2"/>
          <path d="${toPath(augSeries)}" fill="none" stroke="var(--gold)" stroke-width="2.5"/>
          ${dots(baseSeries, "var(--text-ink-dim)")}
          ${dots(augSeries, "var(--gold)")}
          ${yLabels}
          ${xLabels}
          <text x="${W / 2}" y="${H - 8}" font-size="10" text-anchor="middle" fill="var(--text-ink-dim)" font-family="var(--font-mono)" dy="14" opacity="0">epoch</text>
        </svg>
        <div class="chart-tooltip" id="chartTooltip"></div>
      </div>
    `;

    const svg = mount.querySelector("svg");
    const tooltip = qs("#chartTooltip");
    svg.addEventListener("mousemove", (e) => {
      const rect = svg.getBoundingClientRect();
      const scaleX = W / rect.width;
      const mx = (e.clientX - rect.left) * scaleX;
      let i = Math.round(((mx - padL) / plotW) * (maxEpoch - 1));
      i = Math.max(0, Math.min(maxEpoch - 1, i));
      const bVal = i < baseSeries.length ? baseSeries[i] : null;
      const aVal = i < augSeries.length ? augSeries[i] : null;
      const fmt = (v) => (activeMetric === "accuracy" ? (v * 100).toFixed(1) + "%" : v.toFixed(3));
      tooltip.innerHTML = `Epoch ${i + 1}<br>Baseline: ${bVal !== null ? fmt(bVal) : "—"}<br>Calibrated: ${aVal !== null ? fmt(aVal) : "—"}`;
      const px = ((xAt(i) / W) * rect.width);
      const py = ((yAt(aVal ?? bVal ?? 0) / H) * rect.height);
      tooltip.style.left = `${px}px`;
      tooltip.style.top = `${py}px`;
      tooltip.style.opacity = "1";
    });
    svg.addEventListener("mouseleave", () => (tooltip.style.opacity = "0"));
  }

  // ---------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------
  checkStatus();
  loadModelInfo();
})();
