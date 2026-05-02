/* Dashboard — gauge + SSE updates + recent alerts strip */

// ── Gauge renderer ─────────────────────────────────────────────────────────
const canvas  = document.getElementById("gauge-canvas");
const ctx     = canvas.getContext("2d");
let   _score  = 0;
let   _thr    = 0.6;

function drawGauge(score, threshold) {
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 16;
  const r  = 90;
  const startA = Math.PI, endA = 2 * Math.PI;

  ctx.clearRect(0, 0, W, H);

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startA, endA);
  ctx.lineWidth  = 16;
  ctx.strokeStyle = "#1c2030";
  ctx.stroke();

  // Coloured fill arc
  const fillEnd = startA + (endA - startA) * Math.min(score / 1.0, 1);
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  if (score < threshold * 0.7) {
    grad.addColorStop(0, "#29d984");
    grad.addColorStop(1, "#29d984");
  } else if (score < threshold) {
    grad.addColorStop(0, "#29d984");
    grad.addColorStop(1, "#f0c040");
  } else {
    grad.addColorStop(0, "#f0c040");
    grad.addColorStop(1, "#f04060");
  }
  ctx.beginPath();
  ctx.arc(cx, cy, r, startA, fillEnd);
  ctx.lineWidth   = 16;
  ctx.strokeStyle = grad;
  ctx.lineCap     = "round";
  ctx.stroke();

  // Threshold tick mark
  const thrAngle = startA + (endA - startA) * threshold;
  const ix = cx + (r - 8) * Math.cos(thrAngle);
  const iy = cy + (r - 8) * Math.sin(thrAngle);
  const ox = cx + (r + 8) * Math.cos(thrAngle);
  const oy = cy + (r + 8) * Math.sin(thrAngle);
  ctx.beginPath();
  ctx.moveTo(ix, iy);
  ctx.lineTo(ox, oy);
  ctx.lineWidth   = 2.5;
  ctx.strokeStyle = "#f0c040";
  ctx.lineCap     = "butt";
  ctx.stroke();
}

function animateGauge(target, thr) {
  const delta = (target - _score) * 0.18;
  _score += Math.abs(delta) < 0.001 ? (target - _score) : delta;
  _thr = thr;
  drawGauge(_score, _thr);
  document.getElementById("g-score").textContent = (_score * 100).toFixed(0);
  document.getElementById("g-threshold").textContent =
    `Threshold ${(thr * 100).toFixed(0)}%`;
  if (Math.abs(target - _score) > 0.001) {
    requestAnimationFrame(() => animateGauge(target, thr));
  }
}

// ── SSE callback (called by base.html) ─────────────────────────────────────
function onStatusUpdate(s) {
  // Stat cards
  document.getElementById("s-fps").textContent    = s.fps || "—";
  document.getElementById("s-tracks").textContent = s.tracks;
  document.getElementById("s-loiters").textContent = s.loiters;
  document.getElementById("s-dets").textContent   = s.total_detections;

  // Gauge
  animateGauge(s.risk.score, s.risk.threshold);

  // Reasons
  const rl = document.getElementById("reason-list");
  if (s.risk.reasons.length === 0) {
    rl.innerHTML = '<p class="reason-empty">No signals</p>';
  } else {
    rl.innerHTML = s.risk.reasons
      .map(r => `<span class="reason-tag">${r}</span>`)
      .join("");
  }

  // Component bars
  const cg = document.getElementById("component-grid");
  cg.innerHTML = Object.entries(s.risk.components)
    .map(([k, v]) => {
      const pct = Math.min(v / 0.5 * 100, 100);
      return `<div class="comp-item">${k}
        <div class="comp-bar" style="width:${pct}%;max-width:100%;"></div>
      </div>`;
    })
    .join("");
}

// ── Recent alerts ──────────────────────────────────────────────────────────
async function loadRecent() {
  const strip = document.getElementById("recent-strip");
  const res   = await fetch("/api/events?limit=20");
  const evs   = await res.json();
  if (!evs.length) {
    strip.innerHTML = '<p class="empty-msg">No alerts yet.</p>';
    return;
  }
  strip.innerHTML = evs
    .slice()
    .reverse()
    .map(e => {
      const name = e.image_path ? e.image_path.split(/[\\/]/).pop() : "";
      const img  = name
        ? `<img src="/events/image/${name}" alt="event"
               onerror="this.style.display='none'" />`
        : "";
      return `<a class="thumb-card" href="/events" title="${e.reasons.join(', ')}">
        ${img}
        <div class="thumb-meta">
          <div>${e.iso_time.slice(11)}</div>
          <div class="thumb-score">${(e.score * 100).toFixed(0)}%</div>
        </div>
      </a>`;
    })
    .join("");
}

// Initial load
loadRecent();
// Refresh strip every 15 s
setInterval(loadRecent, 15_000);

// Initial gauge draw
drawGauge(0, 0.6);
