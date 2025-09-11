// Vanilla JS UI + ROI overlay interactions + REST wiring

(() => {
  // ----- DOM refs -----
  const qs = (s) => document.querySelector(s);
  const stream = qs('#stream');
  const canvas = qs('#roiOverlay');
  const ctx = canvas.getContext('2d');

  const cameraToggle = qs('#cameraToggle');
  const saveBtn = qs('#saveBtn');

  const expSlider = qs('#expSlider');
  const expLabel = qs('#expLabel');
  const cmSelect = qs('#colormapSelect');

  const statsExposure = qs('#statsExposure');
  const statsFps = qs('#statsFps');

  const barPanel = qs('#barPanel');
  const roiGridPanel = qs('#roiGridPanel');
  const perRoiPanels = qs('#perRoiPanels');
  const perRoiGrid = qs('#perRoiGrid');

  const roiList = qs('#roiList');

  // ----- State -----
  const state = {
    rois: [],                 // [{id,x,y,w,h}, ...] in stream pixel coords
    metrics: null,            // snapshot JSON
    selectedId: null,
    naturalW: 0,
    naturalH: 0,
    // Interaction
    mouse: { x:0, y:0, down:false },
    drag: null,               // {mode:'move'|'resize'|'create', roiId, start:{x,y}, orig:{x,y,w,h}, handle:null}
    preview: null,            // temp preview rect for create
    // Debounce timers
    putTimer: null
  };

  // ----- REST helpers -----
  async function logToServer(level, message, data) {
    try {
      // fire and forget
      fetch('/log/js', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ level, message, data })
      });
    } catch (_) { /* ignore logging errors */ }
  }

  async function getJSON(url) {
    const cacheBustUrl = new URL(url, window.location.origin);
    cacheBustUrl.searchParams.set('t', Date.now());
    const res = await fetch(cacheBustUrl.href, { cache: 'no-store' });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
  async function postJSON(url, body) {
    const res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
  async function putJSON(url, body) {
    const res = await fetch(url, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
  async function del(url) {
    const res = await fetch(url, { method:'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
  async function resetMax(roiId) {
    try {
      await postJSON(`/roi/${encodeURIComponent(roiId)}/reset_max`, {});
      logToServer('info', 'Reset max for ROI', { roiId });
    } catch(e) {
      logToServer('error', 'Failed to reset max for ROI', { roiId, error: e.toString() });
    }
  }


  // ----- Canvas sizing -----
  function resizeCanvasToImage() {
    const rect = stream.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    drawOverlay();
  }

  function scaleFactors() {
    const dw = Math.max(1, stream.clientWidth);
    const dh = Math.max(1, stream.clientHeight);
    const sx = state.naturalW > 0 ? state.naturalW / dw : 1;
    const sy = state.naturalH > 0 ? state.naturalH / dh : 1;
    return { sx, sy, dw, dh };
  }

  function toStreamCoords(px, py) {
    const { sx, sy } = scaleFactors();
    return { x: Math.round(px * sx), y: Math.round(py * sy) };
  }
  function toDisplayRect(r) {
    const { sx, sy } = scaleFactors();
    return {
      x: r.x / sx,
      y: r.y / sy,
      w: r.w / sx,
      h: r.h / sy,
    };
  }

  function clampRectStream(x, y, w, h) {
    // Bounds using natural size
    const W = Math.max(1, state.naturalW);
    const H = Math.max(1, state.naturalH);
    x = Math.max(0, x|0);
    y = Math.max(0, y|0);
    w = Math.max(1, w|0);
    h = Math.max(1, h|0);
    if (x >= W) { x = W-1; w = 1; }
    if (y >= H) { y = H-1; h = 1; }
    if (x + w > W) w = Math.max(1, W - x);
    if (y + h > H) h = Math.max(1, H - y);
    return { x, y, w, h };
  }

  // ----- ROI CRUD -----
  async function refreshRois() {
    try {
      const list = await getJSON('/rois');
      logToServer('info', 'Refreshed ROI list from server', { count: list.length, ids: list.map(r => r.id) });
      state.rois = Array.isArray(list) ? list : [];
      if (state.selectedId && !state.rois.find(r => r.id === state.selectedId)) {
        state.selectedId = null;
      }
      renderRoiList();
      renderPerRoiPanels();
      drawOverlay();
    } catch (e) {
      logToServer('error', 'Failed to refresh ROIs', { error: e.toString() });
    }
  }

  function findRoi(id) {
    return state.rois.find(r => r.id === id) || null;
  }

  async function createRoi(rect) {
    logToServer('info', 'Attempting to create ROI', { rect });
    const r = clampRectStream(rect.x, rect.y, rect.w, rect.h);
    const created = await postJSON('/rois', r);
    await refreshRois();
    state.selectedId = created?.id || null;
  }

  function debouncePut(roiId, rect) {
    if (state.putTimer) {
      clearTimeout(state.putTimer);
      state.putTimer = null;
    }
    state.putTimer = setTimeout(async () => {
      try {
        const r = clampRectStream(rect.x, rect.y, rect.w, rect.h);
        await putJSON(`/rois/${encodeURIComponent(roiId)}`, r);
        await refreshRois();
      } catch (_) {}
    }, 60);
  }

  async function deleteRoi(roiId) {
    logToServer('info', 'Attempting to delete ROI', { roiId });
    try {
      await del(`/rois/${encodeURIComponent(roiId)}`);
      await refreshRois();
    } catch (e) {
      logToServer('error', 'Failed to delete ROI', { roiId, error: e.toString() });
    }
  }

  // ----- Metrics polling -----
  async function pollMetrics() {
    try {
      const snap = await getJSON('/metrics');
      state.metrics = snap;
      const exp = snap?.exposure_us ?? 0;
      const fps = snap?.fps ?? 0;
      statsExposure.textContent = `exp: ${exp}`;
      statsFps.textContent = `fps: ${fps.toFixed(1)}`;

      const hasRois = (snap?.rois?.length || state.rois.length) > 0;
      barPanel.hidden = true;
      roiGridPanel.hidden = true;
      perRoiPanels.hidden = !hasRois;

      renderRoiList();
    } catch (_) {
      // ignore transient errors
    } finally {
      setTimeout(pollMetrics, 300);
    }
  }

  // ----- Sidebar rendering -----
  function renderRoiList() {
    const metricsMap = new Map();
    if (state.metrics?.rois) {
      for (const m of state.metrics.rois) metricsMap.set(m.id, m);
    }
    roiList.innerHTML = '';
    for (const r of state.rois) {
      const m = metricsMap.get(r.id);
      const card = document.createElement('div');
      card.className = 'roi-card';
      if (state.selectedId === r.id) card.style.outline = '2px solid var(--accent)';

      const head = document.createElement('div');
      head.className = 'head';
      head.innerHTML = `<span>${r.id}</span>`;

      const btnGroup = document.createElement('div');
      btnGroup.style.display = 'flex';
      btnGroup.style.gap = '6px';

      const resetBtn = document.createElement('button');
      resetBtn.textContent = 'Reset Max';
      resetBtn.onclick = () => resetMax(r.id);

      const delBtn = document.createElement('button');
      delBtn.className = 'danger';
      delBtn.textContent = 'Delete';
      delBtn.onclick = () => deleteRoi(r.id);

      btnGroup.appendChild(resetBtn);
      btnGroup.appendChild(delBtn);

      const coords = document.createElement('div');
      coords.className = 'coords';
      coords.textContent = `x:${r.x} y:${r.y} w:${r.w} h:${r.h}`;

      const metrics = document.createElement('div');
      metrics.className = 'metrics';
      const sum = m ? m.sum_gray : '-';
      const vms = m ? m.value_per_ms.toFixed(1) : '-';
      metrics.innerHTML = `<span>Sum: ${sum}</span><span>Value/ms: ${vms}</span>`;

      card.appendChild(head);
      card.appendChild(btnGroup);
      card.appendChild(coords);
      card.appendChild(metrics);
      card.onclick = (e) => {
        if (e.target === delBtn || e.target === resetBtn) return;
        state.selectedId = r.id;
        drawOverlay();
      };
      roiList.appendChild(card);
    }
  }

  function renderPerRoiPanels() {
    const ids = state.rois.map(r => r.id);
    for (const child of Array.from(perRoiGrid.children)) {
      const id = child.getAttribute('data-roi');
      if (!ids.includes(id)) child.remove();
    }
    for (const r of state.rois) {
      let card = perRoiGrid.querySelector(`.per-roi-card[data-roi="${r.id}"]`);
      if (!card) {
        card = document.createElement('div');
        card.className = 'per-roi-card';
        card.setAttribute('data-roi', r.id);

        const header = document.createElement('div');
        header.className = 'header';
        header.innerHTML = `<span class="title">${r.id}</span><span class="dims">${r.w}x${r.h}</span>`;

        const plots = document.createElement('div');
        plots.className = 'plots';

        const profileImg = document.createElement('img');
        profileImg.className = 'roi-profile';
        profileImg.alt = `Profile ${r.id}`;
        profileImg.src = `/roi_profile_feed/${encodeURIComponent(r.id)}`;

        const integrationImg = document.createElement('img');
        integrationImg.className = 'roi-integration';
        integrationImg.alt = `Integration ${r.id}`;
        integrationImg.src = `/roi_integration_feed/${encodeURIComponent(r.id)}`;

        plots.appendChild(profileImg);
        plots.appendChild(integrationImg);

        const toolbar = document.createElement('div');
        toolbar.className = 'toolbar';
        const resetBtn = document.createElement('button');
        resetBtn.textContent = 'Reset Max';
        resetBtn.onclick = () => resetMax(r.id);
        toolbar.appendChild(resetBtn);

        card.appendChild(header);
        card.appendChild(plots);
        card.appendChild(toolbar);

        perRoiGrid.appendChild(card);
      }
    }
  }


  // ----- Overlay drawing -----
  function drawOverlay() {
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    for (const r of state.rois) {
      drawRoi(r, r.id === state.selectedId ? '#50e3c2' : '#4a90e2');
    }
    if (state.preview) {
      ctx.save();
      ctx.strokeStyle = '#ffd166';
      ctx.lineWidth = 2;
      const disp = toDisplayRect(state.preview);
      strokeRectRounded(disp.x, disp.y, disp.w, disp.h, 3);
      ctx.restore();
    }
  }

  function strokeRectRounded(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.stroke();
  }

  function drawRoi(roi, color) {
    const disp = toDisplayRect(roi);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    strokeRectRounded(disp.x, disp.y, disp.w, disp.h, 4);

    ctx.fillStyle = 'rgba(74,144,226,0.12)';
    ctx.fillRect(disp.x, disp.y, disp.w, disp.h);

    drawHandles(disp, color);
    ctx.restore();
  }

  function drawHandles(d, color) {
    const hs = 7;
    const pts = handlePoints(d);
    ctx.fillStyle = color;
    for (const p of pts) {
      ctx.fillRect(p.x - hs, p.y - hs, hs * 2, hs * 2);
    }
  }

  function handlePoints(d) {
    const x0 = d.x, y0 = d.y, x1 = d.x + d.w, y1 = d.y + d.h, xm = x0 + d.w/2, ym = y0 + d.h/2;
    return [
      {name:'tl', x:x0, y:y0}, {name:'tr', x:x1, y:y0}, {name:'bl', x:x0, y:y1}, {name:'br', x:x1, y:y1},
      {name:'tm', x:xm, y:y0}, {name:'bm', x:xm, y:y1}, {name:'ml', x:x0, y:ym}, {name:'mr', x:x1, y:ym},
    ];
  }

  function hitTest(px, py) {
    const handleRadius = 9;
    for (const r of state.rois) {
      const disp = toDisplayRect(r);
      const pts = handlePoints(disp);
      for (const p of pts) {
        if (Math.abs(px - p.x) <= handleRadius && Math.abs(py - p.y) <= handleRadius) {
          return { roi: r, kind: 'handle', handle: p.name };
        }
      }
    }
    for (const r of state.rois) {
      const d = toDisplayRect(r);
      if (px >= d.x && px <= d.x + d.w && py >= d.y && py <= d.y + d.h) {
        return { roi: r, kind: 'inside' };
      }
    }
    return { roi: null, kind: 'none' };
  }

  // ----- Mouse interactions -----
  function canvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
  }

  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    const p = canvasPoint(e);
    state.mouse = { x:p.x, y:p.y, down:true };
    const hit = hitTest(p.x, p.y);
    if (hit.kind === 'handle' && hit.roi) {
      state.selectedId = hit.roi.id;
      state.drag = { mode: 'resize', roiId: hit.roi.id, start: { x:p.x, y:p.y }, handle: hit.handle, orig: { ...hit.roi } };
    } else if (hit.kind === 'inside' && hit.roi) {
      state.selectedId = hit.roi.id;
      state.drag = { mode: 'move', roiId: hit.roi.id, start: { x:p.x, y:p.y }, orig: { ...hit.roi } };
    } else {
      state.selectedId = null;
      state.drag = { mode: 'create', roiId: null, start: { x:p.x, y:p.y } };
      state.preview = null;
    }
    drawOverlay();
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.mouse.down) return;
    const p = canvasPoint(e);
    state.mouse = { x:p.x, y:p.y, down:true };
    if (!state.drag) return;
    if (state.drag.mode === 'move' && state.drag.roiId) {
      const sel = findRoi(state.drag.roiId);
      if (!sel) return;
      const { sx, sy } = scaleFactors();
      const nx = state.drag.orig.x + Math.round((p.x - state.drag.start.x) * sx);
      const ny = state.drag.orig.y + Math.round((p.y - state.drag.start.y) * sy);
      const rect = clampRectStream(nx, ny, sel.w, sel.h);
      sel.x = rect.x; sel.y = rect.y;
      debouncePut(sel.id, rect);
      drawOverlay();
    } else if (state.drag.mode === 'resize' && state.drag.roiId) {
      const sel = findRoi(state.drag.roiId);
      if (!sel) return;
      const o = state.drag.orig;
      const { sx, sy } = scaleFactors();
      const ddx = Math.round((p.x - state.drag.start.x) * sx);
      const ddy = Math.round((p.y - state.drag.start.y) * sy);
      let x = o.x, y = o.y, w = o.w, h = o.h;
      switch (state.drag.handle) {
        case 'tl': x = o.x + ddx; y = o.y + ddy; w = o.w - ddx; h = o.h - ddy; break;
        case 'tr': y = o.y + ddy; w = o.w + ddx; h = o.h - ddy; break;
        case 'bl': x = o.x + ddx; w = o.w - ddx; h = o.h + ddy; break;
        case 'br': w = o.w + ddx; h = o.h + ddy; break;
        case 'tm': y = o.y + ddy; h = o.h - ddy; break;
        case 'bm': h = o.h + ddy; break;
        case 'ml': x = o.x + ddx; w = o.w - ddx; break;
        case 'mr': w = o.w + ddx; break;
      }
      if (w < 1) { x = x + w - 1; w = 1; }
      if (h < 1) { y = y + h - 1; h = 1; }
      const rect = clampRectStream(x, y, w, h);
      sel.x = rect.x; sel.y = rect.y; sel.w = rect.w; sel.h = rect.h;
      debouncePut(sel.id, rect);
      drawOverlay();
    } else if (state.drag.mode === 'create') {
      const { sx, sy } = scaleFactors();
      const s = state.drag.start;
      const p0 = toStreamCoords(s.x, s.y);
      const p1 = toStreamCoords(p.x, p.y);
      const x = Math.min(p0.x, p1.x);
      const y = Math.min(p0.y, p1.y);
      const w = Math.max(1, Math.abs(p1.x - p0.x));
      const h = Math.max(1, Math.abs(p1.y - p0.y));
      state.preview = clampRectStream(x, y, w, h);
      drawOverlay();
    }
  });

  window.addEventListener('mouseup', async (e) => {
    if (!state.mouse.down) return;
    state.mouse.down = false;
    const drag = state.drag;
    state.drag = null;
    if (drag?.mode === 'create' && state.preview) {
      const r = state.preview;
      state.preview = null;
      if (r.w >= 3 && r.h >= 3) {
        try { await createRoi(r); } catch (_) {}
      }
      drawOverlay();
    }
  });

  // ----- Controls wiring -----
  async function syncControls() {
    try {
      const [exp, cm] = await Promise.all([ getJSON('/exposure'), getJSON('/colormap') ]);
      if (typeof exp?.value === 'number') {
        expSlider.value = String(exp.value);
        expLabel.textContent = String(exp.value);
      }
      if (cm?.value) {
        cmSelect.value = cm.value;
      }
    } catch (_) {}
  }

  expSlider.addEventListener('input', () => { expLabel.textContent = String(expSlider.value); });
  expSlider.addEventListener('change', async () => {
    const v = parseInt(expSlider.value, 10);
    try {
      const res = await postJSON('/exposure', { value: v });
      if (typeof res.value === 'number') {
        expSlider.value = String(res.value);
        expLabel.textContent = String(res.value);
      }
    } catch (_) {}
  });

  cmSelect.addEventListener('change', async () => {
    const value = cmSelect.value;
    try {
      const res = await postJSON('/colormap', { value });
      if (res?.value) cmSelect.value = res.value;
    } catch (_) {}
  });

  cameraToggle.addEventListener('change', async () => {
    const enabled = cameraToggle.checked;
    try {
      const res = await postJSON('/camera', { enabled });
      cameraToggle.checked = !!res.enabled;
      stream.src = enabled ? '/video_feed?ts=' + Date.now() : '';
    } catch (_) {
      cameraToggle.checked = !enabled;
    }
  });

  saveBtn.addEventListener('click', async () => {
    const iso = new Date().toISOString().replace(/[:]/g, '-').replace(/\..+/, '');
    const def = `snapshot_${iso}`;
    const base = window.prompt('Save as…', def);
    if (!base) return;
    try {
      const res = await fetch(`/save_bundle?base=${encodeURIComponent(base)}`);
      if (!res.ok) throw new Error('Save failed');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${base}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.warn(e);
    }
  });

  // ----- Image events -----
  stream.addEventListener('load', () => {
    state.naturalW = stream.naturalWidth || 0;
    state.naturalH = stream.naturalHeight || 0;
    resizeCanvasToImage();
    drawOverlay();
  });

  window.addEventListener('resize', resizeCanvasToImage);

  // ----- Bootstrap -----
  (async function init() {
    await syncControls();
    await refreshRois();
    pollMetrics();
    if (stream.complete) {
      state.naturalW = stream.naturalWidth || state.naturalW;
      state.naturalH = stream.naturalHeight || state.naturalH;
      resizeCanvasToImage();
    }
  })();
})();
