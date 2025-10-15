// Vanilla JS UI + ROI overlay interactions + REST wiring

(() => {
  // ----- DOM refs -----
  const qs = (s) => document.querySelector(s);
  const stream = qs('#stream');
  const canvas = qs('#roiOverlay');
  const ctx = canvas.getContext('2d');

  const cameraToggle = qs('#cameraToggle');
  const cameraSelect = qs('#cameraSelect');
  const saveAllBtn = qs('#saveAllBtn');
  const backgroundSubtractionToggle = qs('#backgroundSubtractionToggle');
  const backgroundSubtractionModal = qs('#backgroundSubtractionModal');
  const closeModalButton = backgroundSubtractionModal.querySelector('.close-button');
  const startBgSubtractionBtn = qs('#startBgSubtraction');
  const numFramesInput = qs('#numFrames');

  const expSlider = qs('#expSlider');
  const expLabel = qs('#expLabel');
  const cmSelect = qs('#colormapSelect');

  // Beam analysis controls
  const baPixelSize = qs('#baPixelSize');
  const baCompute = qs('#baCompute');
  const baClipNegatives = qs('#baClipNegatives');
  const baAngleClip = qs('#baAngleClip');
  const baBackground = qs('#baBackground');
  const baRotation = qs('#baRotation');
  const baFixedAngle = qs('#baFixedAngle');

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

  const roiImageTimers = new Map(); // Map<roiId, { profile:number, cuts:number }>

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
    // Do not attempt to parse a JSON body, as there may not be one.
    return;
  }
  async function resetMax(roiId) {
    try {
      await postJSON(`/roi/${encodeURIComponent(roiId)}/reset_max_values`, {});
      logToServer('info', 'Reset max for ROI', { roiId });
    } catch(e) {
      logToServer('error', 'Failed to reset max for ROI', { roiId, error: e.toString() });
    }
  }



  async function downloadRoiZip(rid, kind) {
    const iso = new Date().toISOString().replace(/[:]/g, '-').replace(/\..+/, '');
    const def = `${rid}_${kind}_${iso}`;
    const promptLabel = kind === 'profile' ? 'Save profile as...' : 'Save cuts as...';
    const input = window.prompt(promptLabel, def);
    if (!input) return;
    const baseName = input.trim();
    if (!baseName) return;

    const endpoint = kind === 'profile'
      ? `/roi_profile_save/${encodeURIComponent(rid)}`
      : `/roi_cuts_save/${encodeURIComponent(rid)}`;

    try {
      const res = await fetch(`${endpoint}?base=${encodeURIComponent(baseName)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${baseName}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      logToServer('error', `Failed to save ROI ${kind}`, { roiId: rid, error: err.toString() });
      alert(`Failed to save ROI ${kind}. See logs for details.`);
    }
  }


  // ----- Beam options sync -----
  async function syncBeamOptions() {
    try {
      const opts = await getJSON('/beam_options');
      if (baPixelSize) baPixelSize.value = opts.pixel_size != null ? String(opts.pixel_size) : '';
      if (baCompute) baCompute.value = opts.compute || 'both';
      if (baClipNegatives) {
        const clip = opts.clip_negatives;
        let clipVal;
        if (typeof clip === 'string') {
          clipVal = clip.toLowerCase();
        } else if (clip === true) {
          clipVal = 'zero';
        } else {
          clipVal = 'none';
        }
        baClipNegatives.value = clipVal;
      }
      if (baAngleClip) baAngleClip.value = opts.angle_clip_mode || 'otsu';
      if (baBackground) baBackground.checked = !!opts.background_subtraction;
      if (baRotation) baRotation.value = opts.rotation || 'auto';
      updateFixedAngleAvailability();
      if (baFixedAngle) baFixedAngle.value = opts.fixed_angle != null ? String(opts.fixed_angle) : '';
    } catch (_) {}
  }

  function postBeamOptions(patch) {
    try {
      postJSON('/beam_options', patch);
    } catch (_) {}
  }

  function updateFixedAngleAvailability() {
    if (!baFixedAngle) return;
    const mode = baRotation ? baRotation.value : 'auto';
    const enable = mode === 'fixed';
    baFixedAngle.disabled = !enable;
  }

  updateFixedAngleAvailability();

  if (baPixelSize) baPixelSize.addEventListener('change', () => {
    const v = baPixelSize.value.trim();
    postBeamOptions({ pixel_size: v === '' ? null : parseFloat(v) });
  });
  if (baCompute) baCompute.addEventListener('change', () => {
    postBeamOptions({ compute: baCompute.value });
  });
  if (baClipNegatives) baClipNegatives.addEventListener('change', () => {
    postBeamOptions({ clip_negatives: baClipNegatives.value });
  });
  if (baAngleClip) baAngleClip.addEventListener('change', () => {
    postBeamOptions({ angle_clip_mode: baAngleClip.value });
  });
  if (baBackground) baBackground.addEventListener('change', () => {
    postBeamOptions({ background_subtraction: !!baBackground.checked });
  });
  if (baRotation) baRotation.addEventListener('change', () => {
    updateFixedAngleAvailability();
    postBeamOptions({ rotation: baRotation.value });
  });
  if (baFixedAngle) baFixedAngle.addEventListener('change', () => {
    const v = baFixedAngle.value.trim();
    postBeamOptions({ fixed_angle: v === '' ? null : parseFloat(v) });
  });


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
      updateIntegrationPlots();
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
      roiGridPanel.hidden = true;
      perRoiPanels.hidden = !hasRois;

      updateIntegrationPlots();
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

  function clearRoiImageTimers() {
    for (const timers of roiImageTimers.values()) {
      if (timers.profile) clearInterval(timers.profile);
      if (timers.cuts) clearInterval(timers.cuts);
    }
    roiImageTimers.clear();
  }

  function renderPerRoiPanels() {
    clearRoiImageTimers();
    perRoiGrid.innerHTML = ''; // Clear the container
    for (const r of state.rois) {
      const card = document.createElement('div');
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

      const cutsImg = document.createElement('img');
      cutsImg.className = 'roi-cuts';
      cutsImg.alt = `Cuts ${r.id}`;

      const integrationPlot = document.createElement('div');
      integrationPlot.className = 'roi-integration-plot';
      const barContainer = document.createElement('div');
      barContainer.className = 'bar-container';
      const bar = document.createElement('div');
      bar.className = 'bar';
      const label = document.createElement('span');
      label.className = 'label';
      barContainer.appendChild(bar);
      barContainer.appendChild(label);
      integrationPlot.appendChild(barContainer);

      plots.appendChild(profileImg);
      plots.appendChild(cutsImg);
      plots.appendChild(integrationPlot);

      const updateProfile = () => {
        profileImg.src = `/roi_profile_image/${encodeURIComponent(r.id)}?ts=${Date.now()}`;
      };
      const updateCuts = () => {
        cutsImg.src = `/roi_cuts_image/${encodeURIComponent(r.id)}?ts=${Date.now()}`;
      };
      updateProfile();
      updateCuts();
      const timers = {
        profile: setInterval(updateProfile, 400),
        cuts: setInterval(updateCuts, 400),
      };
      roiImageTimers.set(r.id, timers);

      const toolbar = document.createElement('div');
      toolbar.className = 'toolbar';

      const saveProfileBtn = document.createElement('button');
      saveProfileBtn.textContent = 'Save Profile';
      saveProfileBtn.onclick = () => downloadRoiZip(r.id, 'profile');
      toolbar.appendChild(saveProfileBtn);

      const saveCutsBtn = document.createElement('button');
      saveCutsBtn.textContent = 'Save Cuts';
      saveCutsBtn.onclick = () => downloadRoiZip(r.id, 'cuts');
      toolbar.appendChild(saveCutsBtn);

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

  function updateIntegrationPlots() {
    if (!state.metrics?.rois) return;

    const metricsMap = new Map();
    for (const m of state.metrics.rois) metricsMap.set(m.id, m);

    const yMaxMap = state.metrics.y_max_integral || {};

    for (const r of state.rois) {
      const plot = perRoiGrid.querySelector(`.per-roi-card[data-roi="${r.id}"] .roi-integration-plot`);
      if (!plot) continue;

      const m = metricsMap.get(r.id);
      const bar = plot.querySelector('.bar');
      const label = plot.querySelector('.label');

      if (m) {
        const yMax = yMaxMap[r.id] || 1.0;
        const pct = (m.value_per_ms / Math.max(1.0, yMax)) * 100;
        bar.style.height = `${Math.min(100, Math.max(0, pct))}%`;
        label.textContent = `${m.value_per_ms.toFixed(1)}/ms`;
      } else {
        bar.style.height = '0%';
        label.textContent = '-';
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
  async function syncCameraList() {
    try {
      const cameras = await getJSON('/cameras');
      cameraSelect.innerHTML = '';
      if (!cameras || cameras.length === 0) {
        cameraSelect.disabled = true;
        cameraToggle.disabled = true;
        const opt = document.createElement('option');
        opt.textContent = 'No cameras found';
        cameraSelect.appendChild(opt);
        return;
      }
      cameraSelect.disabled = false;
      cameraToggle.disabled = false;
      for (const cam of cameras) {
        const opt = document.createElement('option');
        opt.value = cam.id;
        opt.textContent = cam.name;
        cameraSelect.appendChild(opt);
      }
    } catch (e) {
      logToServer('error', 'Failed to get camera list', { error: e.toString() });
      cameraSelect.disabled = true;
      cameraToggle.disabled = true;
    }
  }

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
    const cameraId = cameraSelect.value;
    cameraSelect.disabled = enabled; // Disable selector when camera is on
    try {
      const res = await postJSON('/camera', { enabled, camera_id: cameraId });
      const success = !!res.enabled;
      cameraToggle.checked = success;
      cameraSelect.disabled = success;
      stream.src = success ? '/video_feed?ts=' + Date.now() : '';
      if (!success && res.error) {
        logToServer('error', 'Failed to toggle camera', { error: res.error });
        alert(`Error: ${res.error}`);
      }
    } catch (e) {
      logToServer('error', 'Failed to toggle camera', { error: e.toString() });
      cameraToggle.checked = !enabled;
      cameraSelect.disabled = !enabled;
    }
  });

  if (saveAllBtn) {
    saveAllBtn.addEventListener('click', async () => {
      const iso = new Date().toISOString().replace(/[:]/g, '-').replace(/\..+/, '');
      const def = `snapshot_${iso}`;
      const base = window.prompt('Save as...', def);
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
  }

  // ----- Image events -----
  stream.addEventListener('load', () => {
    state.naturalW = stream.naturalWidth || 0;
    state.naturalH = stream.naturalHeight || 0;
    resizeCanvasToImage();
    drawOverlay();
  });

  window.addEventListener('resize', resizeCanvasToImage);

  // ----- Background Subtraction Logic -----
  backgroundSubtractionToggle.addEventListener('change', async () => {
    if (backgroundSubtractionToggle.checked) {
      backgroundSubtractionModal.hidden = false;
    } else {
      try {
        await postJSON('/background_subtraction', { enabled: false });
        logToServer('info', 'Background subtraction disabled');
      } catch (e) {
        logToServer('error', 'Failed to disable background subtraction', { error: e.toString() });
      }
    }
  });

  closeModalButton.addEventListener('click', () => {
    backgroundSubtractionModal.hidden = true;
    backgroundSubtractionToggle.checked = false;
  });

  startBgSubtractionBtn.addEventListener('click', async () => {
    const numFrames = parseInt(numFramesInput.value, 10);
    if (isNaN(numFrames) || numFrames <= 0) {
      alert('Please enter a valid number of frames.');
      return;
    }
    backgroundSubtractionModal.hidden = true;
    try {
      await postJSON('/background_subtraction', { enabled: true, num_frames: numFrames });
      logToServer('info', 'Background subtraction enabled', { num_frames: numFrames });
    } catch (e) {
      logToServer('error', 'Failed to enable background subtraction', { error: e.toString() });
      let message = e && e.message ? e.message : 'Unknown error';
      try {
        const parsed = JSON.parse(message);
        if (parsed && typeof parsed.error === 'string') {
          message = parsed.error;
        }
      } catch (_) {}
      alert(`Background subtraction failed: ${message}`);
      backgroundSubtractionModal.hidden = false;
      backgroundSubtractionToggle.checked = false;
    }
  });

  // ----- Bootstrap -----
  (async function init() {
    await Promise.all([
      syncControls(),
      syncCameraList(),
      refreshRois(),
      syncBeamOptions()
    ]);
    pollMetrics();
    if (stream.complete) {
      state.naturalW = stream.naturalWidth || state.naturalW;
      state.naturalH = stream.naturalHeight || state.naturalH;
      resizeCanvasToImage();
    }
  })();
})();
