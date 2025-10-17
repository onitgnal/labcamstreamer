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
  const autoExposureBtn = qs('#autoExposureBtn');

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
  const statsMax = qs('#statsMaxPixel');

  const barPanel = qs('#barPanel');
  const roiGridPanel = qs('#roiGridPanel');
  const perRoiPanels = qs('#perRoiPanels');
  const perRoiGrid = qs('#perRoiGrid');

  const roiList = qs('#roiList');

  // Caustic controls
  const causticPanel = qs('#causticPanel');
  const causticCollapseBtn = qs('#causticCollapseBtn');
  const causticWavelength = qs('#causticWavelength');
  const causticUnit = qs('#causticUnit');
  const causticSource = qs('#causticSource');
  const causticFitBtn = qs('#causticFitBtn');
  const causticSaveBtn = qs('#causticSaveBtn');
  const causticPointList = qs('#causticPointList');
  const causticImagesSection = qs('#causticImagesSection');
  const causticImagesGrid = qs('#causticImagesGrid');
  const causticPlotSection = qs('#causticPlotSection');
  const causticPlotCanvas = qs('#causticPlotCanvas');
  const causticPlotEmpty = qs('#causticPlotEmpty');
  const causticFitSummary = qs('#causticFitSummary');
  const causticModal = qs('#causticAddModal');
  const causticModalClose = qs('#causticModalClose');
  const causticModalCancel = qs('#causticModalCancel');
  const causticModalAdd = qs('#causticModalAdd');
  const causticModalZ = qs('#causticModalZ');
  const causticModalRoi = qs('#causticModalRoi');
  const causticModalUnit = qs('#causticModalUnit');
  const causticLoadBtn = qs('#causticLoadBtn');
  const causticImportModal = qs('#causticImportModal');
  const causticImportClose = qs('#causticImportClose');
  const causticImportFolder = qs('#causticImportFolder');
  const causticImportRecursive = qs('#causticImportRecursive');
  const causticImportCancel = qs('#causticImportCancel');
  const causticImportConfirm = qs('#causticImportConfirm');
  const causticImportProgress = qs('#causticImportProgress');
  const causticImportProgressText = qs('#causticImportProgressText');
  const causticImportProgressFill = causticImportProgress ? causticImportProgress.querySelector('.progress-fill') : null;
  const toastContainer = qs('#toastContainer');

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
    putTimer: null,
    caustic: {
      config: {
        wavelength_nm: 1030,
        position_unit: 'mm',
        radii_source: 'gauss_1e2',
      },
      points: [],
      series: { z: [], wx: [], wy: [] },
      fits: {},
      selectedPointId: null,
      collapsed: false,
      pendingRoiId: null,
    },
  };

  const roiImageTimers = new Map(); // Map<roiId, { profile:number, cuts:number }>
  let causticConfigTimer = null;
  let causticImportTask = null;
  let causticImportPollTimer = null;
  let causticImportActive = false;

  const SATURATION_THRESHOLD = 250;
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

  function showToast(message, options = {}) {
    const opts = options || {};
    const variant = opts.variant || 'info';
    const duration = typeof opts.duration === 'number' ? opts.duration : 5000;
    if (!toastContainer) {
      window.alert(String(message));
      return;
    }
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.dataset.variant = variant;

    const msg = document.createElement('span');
    msg.className = 'toast-message';
    msg.textContent = String(message);

    const closeBtn = document.createElement('button');
    closeBtn.className = 'toast-close';
    closeBtn.type = 'button';
    closeBtn.innerHTML = '&times;';

    toast.appendChild(msg);
    toast.appendChild(closeBtn);
    toastContainer.appendChild(toast);

    let hideTimer = null;
    let removing = false;
    const removeToast = () => {
      if (removing) return;
      removing = true;
      toast.classList.remove('show');
      const cleanup = () => {
        if (toast.parentNode === toastContainer) {
          toastContainer.removeChild(toast);
        }
      };
      toast.addEventListener('transitionend', cleanup, { once: true });
      setTimeout(cleanup, 300);
    };

    closeBtn.addEventListener('click', () => {
      if (hideTimer) clearTimeout(hideTimer);
      removeToast();
    });

    requestAnimationFrame(() => {
      toast.classList.add('show');
    });

    if (duration > 0) {
      hideTimer = setTimeout(removeToast, duration);
    }
  }

  function setCausticImportBusy(busy) {
    if (causticLoadBtn) causticLoadBtn.disabled = !!busy;
    if (causticImportFolder) causticImportFolder.disabled = !!busy;
    if (causticImportRecursive) causticImportRecursive.disabled = !!busy;
    if (causticImportConfirm) causticImportConfirm.disabled = !!busy;
  }

  function resetCausticImportModal() {
    if (causticImportFolder) causticImportFolder.value = '';
    if (causticImportRecursive) causticImportRecursive.checked = false;
    if (causticImportProgress) causticImportProgress.hidden = true;
    if (causticImportProgressFill) causticImportProgressFill.style.width = '0%';
    if (causticImportProgressText) causticImportProgressText.textContent = 'Waiting...';
  }

  function openCausticImportModal() {
    if (!causticImportModal) return;
    causticImportModal.hidden = false;
    if (!causticImportActive) {
      resetCausticImportModal();
      setCausticImportBusy(false);
      if (causticImportFolder) {
        requestAnimationFrame(() => causticImportFolder.focus());
      }
    } else if (causticImportProgress) {
      causticImportProgress.hidden = false;
    }
  }

  function closeCausticImportModal() {
    if (!causticImportModal) return;
    causticImportModal.hidden = true;
  }

  function stopCausticImportPolling() {
    if (causticImportPollTimer) {
      clearTimeout(causticImportPollTimer);
      causticImportPollTimer = null;
    }
  }

  function updateCausticImportProgress(snapshot) {
    if (!causticImportProgress) return;
    causticImportProgress.hidden = false;
    const status = (snapshot?.status || '').toLowerCase();
    const total = Number(snapshot?.total_files ?? snapshot?.total ?? 0) || 0;
    const processed = Number(snapshot?.processed_files ?? snapshot?.processed ?? 0) || 0;
    const counts = snapshot?.counts || {};
    const imported = Number(counts.imported || 0);
    const duplicates = Number(counts.duplicates || 0);
    const malformed = Number(counts.malformed || 0);
    const ioErrors = Number(counts.io_errors || 0);
    const skipped = duplicates + malformed + ioErrors;

    let percent = 0;
    if (status === 'completed') {
      const total = Number(snapshot?.total_files ?? snapshot?.total ?? 0) || 0;
      if (snapshot?.caustic_state) {
        applyCausticState(snapshot.caustic_state);
      } else {
        refreshCausticState().catch(() => {});
      }
      if (total === 0) {
        showToast('No BMP files found in the selected folder.', { variant: 'info', duration: 5000 });
      } else {
        const variant = summary.imported > 0 ? 'success' : 'info';
        showToast(summary.message, { variant, duration: 6000 });
      }
    } else if (status === 'failed') {
      const reason = snapshot?.error || 'Import failed';
      showToast(reason, { variant: 'error', duration: 7000 });
    }

    causticImportActive = false;
    setCausticImportBusy(false);

    if (snapshot?.skipped?.length) {
      console.info('Caustic import skipped files:', snapshot.skipped);
    }

    setTimeout(() => {
      resetCausticImportModal();
      closeCausticImportModal();
    }, 200);
  }

  function finalizeCausticImport(snapshot) {
    stopCausticImportPolling();
    causticImportTask = snapshot;
    updateCausticImportProgress(snapshot);
    const status = (snapshot?.status || '').toLowerCase();
    const counts = snapshot?.counts || {};
    const summary = summarizeImportCounts(counts);
    if (status === 'completed') {
      const total = Number(snapshot?.total_files ?? snapshot?.total ?? 0) || 0;
      if (snapshot?.caustic_state) {
        applyCausticState(snapshot.caustic_state);
      } else {
        refreshCausticState().catch(() => {});
      }
      if (total === 0) {
        showToast('No BMP files found in the selected folder.', { variant: 'info', duration: 5000 });
      } else {
        const variant = summary.imported > 0 ? 'success' : 'info';
        showToast(summary.message, { variant, duration: 6000 });
      }
    } else if (status === 'failed') {
      const reason = snapshot?.error || 'Import failed';
      showToast(reason, { variant: 'error', duration: 7000 });
    }
    causticImportActive = false;
    setCausticImportBusy(false);
    if (snapshot?.skipped?.length) {
      console.info('Caustic import skipped files:', snapshot.skipped);
    }
    setTimeout(() => {
      resetCausticImportModal();
      closeCausticImportModal();
    }, 200);
    causticImportTask = null;
  }

  async function startCausticImport(folder, recursive) {
    stopCausticImportPolling();
    causticImportActive = true;
    setCausticImportBusy(true);
    if (causticImportProgress) {
      causticImportProgress.hidden = false;
      if (causticImportProgressFill) causticImportProgressFill.style.width = '0%';
      if (causticImportProgressText) causticImportProgressText.textContent = 'Starting...';
    }

    try {
      const res = await fetch('/api/caustic/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder, recursive: !!recursive }),
      });
      if (!res.ok) {
        let message = res.statusText || 'Failed to start import';
        try {
          const errData = await res.json();
          if (errData?.error) message = errData.error;
        } catch (_) {
          try {
            message = await res.text();
          } catch (__){ /* ignore */ }
        }
        throw new Error(message);
      }
      const snapshot = await res.json();
      causticImportTask = snapshot;
      updateCausticImportProgress(snapshot);
      if (snapshot?.task_id) {
        causticImportPollTimer = setTimeout(() => pollCausticImport(snapshot.task_id), 600);
      } else {
        finalizeCausticImport(snapshot);
      }
    } catch (error) {
      causticImportActive = false;
      setCausticImportBusy(false);
      if (causticImportProgress) causticImportProgress.hidden = true;
      showToast(`Import failed to start: ${parseServerError(error)}`, { variant: 'error' });
    }
  }
  async function runAutoExposure() {
    if (!autoExposureBtn) return;
    const originalText = autoExposureBtn.textContent;
    autoExposureBtn.disabled = true;
    autoExposureBtn.textContent = 'Auto...';
    try {
      const res = await postJSON('/exposure/auto', {});
      if (typeof res?.value === 'number') {
        expSlider.value = String(res.value);
        expLabel.textContent = String(res.value);
      }
      if (res?.message) {
        autoExposureBtn.setAttribute('title', res.message);
        logToServer('info', 'Auto exposure result', res);
      }
    } catch (err) {
      const message = err?.message || err?.toString?.() || 'Unknown error';
      logToServer('error', 'Auto exposure failed', { error: message });
      alert(`Auto exposure failed: ${message}`);
    } finally {
      autoExposureBtn.textContent = originalText;
      autoExposureBtn.disabled = false;
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


  // ----- Caustic helpers -----
  const CAUSTIC_SOURCE_LABELS = {
    gauss_1e2: 'Gaussian 1/e² radii (x, y)',
    moment_2sigma: '2σ second-moment radii (x, y)',
  };

  const LENGTH_UNIT_FACTORS = {
    m: 1,
    meter: 1,
    meters: 1,
    mm: 1e-3,
    millimeter: 1e-3,
    millimeters: 1e-3,
    cm: 1e-2,
    centimeter: 1e-2,
    centimeters: 1e-2,
    um: 1e-6,
    micrometer: 1e-6,
    micrometers: 1e-6,
    in: 0.0254,
    inch: 0.0254,
    inches: 0.0254,
    ft: 0.3048,
    foot: 0.3048,
    feet: 0.3048,
  };

  function formatNumber(value, digits = 3) {
    if (!Number.isFinite(value)) return '—';
    if (Math.abs(value) >= 1000 || Math.abs(value) < 1e-2) {
      return value.toExponential(digits - 1);
    }
    return value.toFixed(digits);
  }

  function getCausticSourceLabel(key) {
    return CAUSTIC_SOURCE_LABELS[key] || key;
  }

  function convertMetersToUnit(value, unit) {
    if (!Number.isFinite(value)) return null;
    const factor = LENGTH_UNIT_FACTORS[(unit || '').toLowerCase()];
    if (!factor) return null;
    return value / factor;
  }

  function getUniformPixelSize(points) {
    const values = (points || [])
      .map((p) => Number(p.pixel_size_m))
      .filter((v) => Number.isFinite(v) && v > 0);
    if (!values.length) return null;
    const sum = values.reduce((acc, v) => acc + v, 0);
    return sum / values.length;
  }

  function niceNumber(value, round) {
    if (!Number.isFinite(value) || value === 0) return 0;
    const exponent = Math.floor(Math.log10(Math.abs(value)));
    const fraction = Math.abs(value) / Math.pow(10, exponent);
    let niceFraction;
    if (round) {
      if (fraction < 1.5) niceFraction = 1;
      else if (fraction < 3) niceFraction = 2;
      else if (fraction < 7) niceFraction = 5;
      else niceFraction = 10;
    } else {
      if (fraction <= 1) niceFraction = 1;
      else if (fraction <= 2) niceFraction = 2;
      else if (fraction <= 5) niceFraction = 5;
      else niceFraction = 10;
    }
    return Math.sign(value) * niceFraction * Math.pow(10, exponent);
  }

  function computeTicks(min, max, maxTicks = 6) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
    if (min === max) {
      if (min === 0) return [0];
      const step = niceNumber(Math.abs(min), true) || 1;
      return [min - step, min, min + step];
    }
    const span = Math.abs(max - min);
    if (!Number.isFinite(span) || span === 0) return [min];
    const niceSpan = niceNumber(span, false) || span;
    const step = Math.max(niceNumber(niceSpan / Math.max(1, maxTicks - 1), true), 1e-12);
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + step * 0.5; v += step) {
      ticks.push(Number(v.toPrecision(12)));
      if (ticks.length > maxTicks * 4) break;
    }
    return ticks;
  }

  function parseServerError(error) {
    if (!error) return 'Unknown error';
    if (typeof error === 'string') {
      try {
        const parsed = JSON.parse(error);
        if (parsed && typeof parsed.error === 'string') {
          return parsed.error;
        }
      } catch (_) {}
      return error;
    }
    if (error instanceof Error) {
      const message = error.message || '';
      if (message) {
        try {
          const parsed = JSON.parse(message);
          if (parsed && typeof parsed.error === 'string') {
            return parsed.error;
          }
        } catch (_) {}
        return message;
      }
    }
    if (typeof error === 'object' && error.error) {
      return String(error.error);
    }
    return String(error);
  }

  function applyCausticState(payload) {
    if (!payload || typeof payload !== 'object') return;
    const prevSelected = state.caustic.selectedPointId;

    if (payload.config) {
      state.caustic.config = {
        ...state.caustic.config,
        ...payload.config,
      };
    }
    if (Array.isArray(payload.points)) {
      state.caustic.points = payload.points;
    }
    if (payload.series) {
      state.caustic.series = {
        z: Array.isArray(payload.series.z) ? payload.series.z : [],
        wx: Array.isArray(payload.series.wx) ? payload.series.wx : [],
        wy: Array.isArray(payload.series.wy) ? payload.series.wy : [],
      };
    }
    if (payload.fits) {
      state.caustic.fits = payload.fits;
    }

    let selected = prevSelected;
    if (payload.last_added_point_id) {
      selected = payload.last_added_point_id;
    } else if (selected && !state.caustic.points.some((pt) => pt.id === selected)) {
      selected = null;
    }
    if (!selected && state.caustic.points.length) {
      selected = state.caustic.points[state.caustic.points.length - 1].id;
    }
    state.caustic.selectedPointId = selected || null;

    updateCausticUi();
  }

  function updateCausticUi() {
    renderCausticControls();
    renderCausticPointList();
    renderCausticImages();
    renderCausticPlot();
    renderCausticFitSummary();
  }

  function renderCausticControls() {
    if (!causticPanel) return;
    const config = state.caustic.config || {};

    if (causticWavelength) {
      const next = config.wavelength_nm != null ? String(config.wavelength_nm) : '';
      if (causticWavelength.value !== next) causticWavelength.value = next;
    }
    if (causticUnit) {
      const next = config.position_unit || 'mm';
      if (causticUnit.value !== next) causticUnit.value = next;
      if (causticModalUnit) causticModalUnit.textContent = next;
    }
    if (causticSource) {
      const next = config.radii_source || 'gauss_1e2';
      if (causticSource.value !== next) causticSource.value = next;
    }

    causticPanel.classList.toggle('collapsed', !!state.caustic.collapsed);
    if (causticCollapseBtn) {
      causticCollapseBtn.textContent = state.caustic.collapsed ? '+' : '−';
    }

    const points = state.caustic.points || [];
    const hasEnoughPoints = points.length >= 3;
    const pixelSize = getUniformPixelSize(points);

    if (causticFitBtn) {
      causticFitBtn.disabled = !hasEnoughPoints;
      if (!hasEnoughPoints) {
        causticFitBtn.title = 'Collect at least three points to run the M² fit.';
      } else if (!pixelSize) {
        causticFitBtn.title = 'Set the pixel size in Beam Analysis before running the M² fit.';
      } else {
        causticFitBtn.title = '';
      }
    }

    if (causticSaveBtn) {
      const hasPoints = points.length > 0;
      causticSaveBtn.disabled = !hasPoints;
      causticSaveBtn.title = hasPoints ? '' : 'Add caustic points before saving.';
    }
  }

  function renderCausticPointList() {
    if (!causticPointList) return;
    causticPointList.innerHTML = '';
    const points = state.caustic.points || [];
    const sourceKey = state.caustic.config?.radii_source || 'gauss_1e2';
    const unitLabel = state.caustic.config?.position_unit || 'mm';

    if (!points.length) {
      const empty = document.createElement('div');
      empty.className = 'caustic-point-empty';
      empty.textContent = 'No caustic points collected yet.';
      causticPointList.appendChild(empty);
      return;
    }

    for (const point of points) {
      const row = document.createElement('div');
      row.className = 'caustic-point-row';
      row.dataset.pointId = point.id;
      if (point.id === state.caustic.selectedPointId) row.classList.add('selected');

      const radii = point.radii && point.radii[sourceKey] ? point.radii[sourceKey] : {};

      const roi = document.createElement('span');
      roi.className = 'label roi';
      roi.textContent = point.roi_id || 'ROI';

      const z = document.createElement('span');
      z.className = 'z';
      const zDisplay = Number(point.z_display);
      const numericZ = Number.isFinite(zDisplay) ? zDisplay : Number(point.z_m);
      z.textContent = `${formatNumber(numericZ, 3)} ${unitLabel}`;

      const wxSpan = document.createElement('span');
      wxSpan.className = 'wx';
      wxSpan.textContent = formatNumber(radii.wx);

      const wySpan = document.createElement('span');
      wySpan.className = 'wy';
      wySpan.textContent = formatNumber(radii.wy);

      const source = document.createElement('span');
      source.className = 'src';
      source.textContent = getCausticSourceLabel(sourceKey);

      const removeCell = document.createElement('span');
      removeCell.className = 'remove';
      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.textContent = '×';
      removeBtn.title = 'Remove point';
      removeBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        removeCausticPoint(point.id);
      });
      removeCell.appendChild(removeBtn);

      row.appendChild(roi);
      row.appendChild(z);
      row.appendChild(wxSpan);
      row.appendChild(wySpan);
      row.appendChild(source);
      row.appendChild(removeCell);

      row.addEventListener('click', () => selectCausticPoint(point.id));
      causticPointList.appendChild(row);
    }
  }

  function renderCausticImages() {
    if (!causticImagesSection || !causticImagesGrid) return;
    const points = state.caustic.points || [];
    const selectedId = state.caustic.selectedPointId;
    const unitLabel = state.caustic.config?.position_unit || 'mm';

    causticImagesSection.hidden = !points.length;
    causticImagesGrid.innerHTML = '';
    if (!points.length) return;

    const cacheBust = Date.now();
    for (const point of points) {
      const card = document.createElement('div');
      card.className = 'caustic-card';
      card.dataset.pointId = point.id;
      if (point.id === selectedId) card.classList.add('selected');

      const header = document.createElement('div');
      header.className = 'caustic-card-header';

      const meta = document.createElement('div');
      meta.className = 'meta';
      const roiLabel = document.createElement('strong');
      roiLabel.textContent = point.roi_id || 'ROI';
      const zLine = document.createElement('span');
      const zDisplay = Number(point.z_display);
      const numericZ = Number.isFinite(zDisplay) ? zDisplay : Number(point.z_m);
      zLine.textContent = `z: ${formatNumber(numericZ, 3)} ${unitLabel}`;
      const sourceLine = document.createElement('span');
      sourceLine.textContent = getCausticSourceLabel(state.caustic.config?.radii_source || 'gauss_1e2');
      meta.appendChild(roiLabel);
      meta.appendChild(zLine);
      meta.appendChild(sourceLine);

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.textContent = 'Remove';
      removeBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        removeCausticPoint(point.id);
      });

      header.appendChild(meta);
      header.appendChild(removeBtn);
      card.appendChild(header);

      const images = document.createElement('div');
      images.className = 'caustic-card-images';

      if (point.images?.profile) {
        const img = document.createElement('img');
        img.alt = 'Caustic profile';
        img.src = `${point.images.profile}?t=${cacheBust}`;
        images.appendChild(img);
      }

      if (point.images?.cut_x || point.images?.cut_y) {
        const cutsWrap = document.createElement('div');
        cutsWrap.className = 'caustic-cut-images';
        if (point.images.cut_x) {
          const cutX = document.createElement('img');
          cutX.alt = 'Cut X';
          cutX.src = `${point.images.cut_x}?t=${cacheBust}`;
          cutsWrap.appendChild(cutX);
        }
        if (point.images.cut_y) {
          const cutY = document.createElement('img');
          cutY.alt = 'Cut Y';
          cutY.src = `${point.images.cut_y}?t=${cacheBust}`;
          cutsWrap.appendChild(cutY);
        }
        images.appendChild(cutsWrap);
      }

      card.appendChild(images);
      card.addEventListener('click', () => selectCausticPoint(point.id));
      causticImagesGrid.appendChild(card);
    }
  }

  function renderCausticPlot() {
    if (!causticPlotSection || !causticPlotCanvas || !causticPlotEmpty) return;
    const points = state.caustic.points || [];

    if (!points.length) {
      causticPlotSection.hidden = true;
      return;
    }
    causticPlotSection.hidden = false;

    const ctx2d = causticPlotCanvas.getContext('2d');
    if (!ctx2d) return;

    const series = state.caustic.series || { z: [], wx: [], wy: [] };
    const pointsForPlot = points.slice().sort((a, b) => Number(a.z_m) - Number(b.z_m));
    const unit = state.caustic.config?.position_unit || 'mm';
    const pixelSize = getUniformPixelSize(points);

    let zValues = (series.z || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
    let wxValues = (series.wx || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
    let wyValues = (series.wy || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
    let count = Math.min(zValues.length, wxValues.length, wyValues.length);

    const sourceKey = state.caustic.config?.radii_source || 'gauss_1e2';

    if (!count && points.length) {
      zValues = [];
      wxValues = [];
      wyValues = [];
      for (const point of pointsForPlot) {
        const radii = point.radii && point.radii[sourceKey] ? point.radii[sourceKey] : {};
        const zCandidate = Number(point.z_display);
        const zValue = Number.isFinite(zCandidate)
          ? zCandidate
          : convertMetersToUnit(Number(point.z_m), unit);
        const wxCandidate = Number(radii.wx);
        const wyCandidate = Number(radii.wy);
        if (!Number.isFinite(zValue) || !Number.isFinite(wxCandidate) || !Number.isFinite(wyCandidate)) continue;
        zValues.push(zValue);
        wxValues.push(wxCandidate);
        wyValues.push(wyCandidate);
      }
      count = Math.min(zValues.length, wxValues.length, wyValues.length);
    }

    const perPointScales = pointsForPlot.map((pt) => {
      const raw = Number(pt.pixel_size_m);
      return Number.isFinite(raw) && raw > 0 ? raw : null;
    });

    let useMeterUnits = true;
    if (count > 0) {
      const wxMeters = [];
      const wyMeters = [];
      const pixelSizeFallback = Number.isFinite(pixelSize) && pixelSize > 0 ? pixelSize : null;
      for (let i = 0; i < count; i++) {
        const scaleCandidate = perPointScales[i];
        const scale = Number.isFinite(scaleCandidate) && scaleCandidate > 0 ? scaleCandidate : pixelSizeFallback;
        const wxVal = wxValues[i];
        const wyVal = wyValues[i];
        if (!Number.isFinite(wxVal) || !Number.isFinite(wyVal) || !Number.isFinite(scale) || scale <= 0) {
          useMeterUnits = false;
          break;
        }
        wxMeters.push(wxVal * scale);
        wyMeters.push(wyVal * scale);
      }
      if (useMeterUnits && wxMeters.length === count && wyMeters.length === count) {
        wxValues = wxMeters;
        wyValues = wyMeters;
      } else {
        useMeterUnits = false;
      }
    }

    if (!count) {
      causticPlotEmpty.hidden = false;
      const wrapper = causticPlotCanvas.parentElement;
      const width = (wrapper?.clientWidth || 640);
      const height = (wrapper?.clientHeight || 320);
      const dpr = window.devicePixelRatio || 1;
      causticPlotCanvas.width = width * dpr;
      causticPlotCanvas.height = height * dpr;
      causticPlotCanvas.style.width = `${width}px`;
      causticPlotCanvas.style.height = `${height}px`;
      ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx2d.clearRect(0, 0, width, height);
      return;
    }

    causticPlotEmpty.hidden = true;

    const wrapper = causticPlotCanvas.parentElement;
    const width = (wrapper?.clientWidth || 640);
    const height = (wrapper?.clientHeight || 320);
    const dpr = window.devicePixelRatio || 1;
    causticPlotCanvas.width = width * dpr;
    causticPlotCanvas.height = height * dpr;
    causticPlotCanvas.style.width = `${width}px`;
    causticPlotCanvas.style.height = `${height}px`;
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2d.clearRect(0, 0, width, height);

    const margins = { left: 70, right: 20, top: 20, bottom: 60 };
    const plotW = Math.max(10, width - margins.left - margins.right);
    const plotH = Math.max(10, height - margins.top - margins.bottom);

    const zMin = Math.min(...zValues);
    const zMax = Math.max(...zValues);
    const zPadding = (zMax - zMin) * 0.1 || 1;
    const zLower = zMin - zPadding;
    const zUpper = zMax + zPadding;
    const zRange = Math.max(1e-9, zUpper - zLower);

    const allW = wxValues.concat(wyValues);
    const wMinRaw = Math.min(...allW);
    const wMaxRaw = Math.max(...allW);
    const wPadding = (wMaxRaw - wMinRaw) * 0.2 || 1;
    const wLower = Math.max(0, wMinRaw - wPadding);
    const wUpper = wMaxRaw + wPadding;
    const wRange = Math.max(1e-9, wUpper - wLower);

    const toX = (z) => margins.left + ((z - zLower) / zRange) * plotW;
    const toY = (w) => (height - margins.bottom) - ((w - wLower) / wRange) * plotH;

    ctx2d.fillStyle = '#141924';
    ctx2d.fillRect(margins.left, margins.top, plotW, plotH);

    ctx2d.strokeStyle = '#525860';
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    ctx2d.moveTo(margins.left, height - margins.bottom);
    ctx2d.lineTo(width - margins.right, height - margins.bottom);
    ctx2d.moveTo(margins.left, height - margins.bottom);
    ctx2d.lineTo(margins.left, margins.top);
    ctx2d.stroke();

    const xTicks = computeTicks(zLower, zUpper, 6);
    const yTicks = computeTicks(wLower, wUpper, 6);

    ctx2d.strokeStyle = '#424854';
    ctx2d.fillStyle = '#8b93a7';
    ctx2d.font = '12px sans-serif';

    ctx2d.textAlign = 'center';
    ctx2d.textBaseline = 'top';
    for (const tick of xTicks) {
      if (!Number.isFinite(tick)) continue;
      if (tick < zLower - zRange * 0.01 || tick > zUpper + zRange * 0.01) continue;
      const x = toX(tick);
      if (x < margins.left - 1 || x > width - margins.right + 1) continue;
      ctx2d.beginPath();
      ctx2d.moveTo(x, height - margins.bottom);
      ctx2d.lineTo(x, height - margins.bottom + 6);
      ctx2d.stroke();
      ctx2d.fillText(formatNumber(tick, 4), x, height - margins.bottom + 8);
    }

    ctx2d.textAlign = 'right';
    ctx2d.textBaseline = 'middle';
    for (const tick of yTicks) {
      if (!Number.isFinite(tick)) continue;
      if (tick < wLower - wRange * 0.01 || tick > wUpper + wRange * 0.01) continue;
      const y = toY(tick);
      if (y < margins.top - 1 || y > height - margins.bottom + 1) continue;
      ctx2d.beginPath();
      ctx2d.moveTo(margins.left - 6, y);
      ctx2d.lineTo(margins.left, y);
      ctx2d.stroke();
      ctx2d.fillText(formatNumber(tick, 4), margins.left - 8, y);
    }

    ctx2d.textAlign = 'center';
    ctx2d.textBaseline = 'bottom';
    ctx2d.fillText(`z (${unit})`, margins.left + plotW / 2, height - 8);
    ctx2d.save();
    ctx2d.translate(20, margins.top + plotH / 2);
    ctx2d.rotate(-Math.PI / 2);
    ctx2d.textAlign = 'center';
    ctx2d.textBaseline = 'middle';
    ctx2d.fillText(useMeterUnits ? 'Radius (m)' : 'Radius (pixels)', 0, 0);
    ctx2d.restore();

    const drawSeries = (values, color) => {
      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = 2;
      ctx2d.beginPath();
      for (let i = 0; i < count; i++) {
        const x = toX(zValues[i]);
        const y = toY(values[i]);
        if (i === 0) ctx2d.moveTo(x, y);
        else ctx2d.lineTo(x, y);
      }
      ctx2d.stroke();
    };

    drawSeries(wxValues, '#e2904a');
    drawSeries(wyValues, '#c2e350');

    const selectedId = state.caustic.selectedPointId;
    const selectedIndex = pointsForPlot.findIndex((pt) => pt.id === selectedId);

    const drawPoints = (values, color) => {
      for (let i = 0; i < count; i++) {
        const x = toX(zValues[i]);
        const y = toY(values[i]);
        ctx2d.fillStyle = color;
        ctx2d.beginPath();
        ctx2d.arc(x, y, 4, 0, Math.PI * 2);
        ctx2d.fill();
        if (i === selectedIndex) {
          ctx2d.strokeStyle = '#50e3c2';
          ctx2d.lineWidth = 2;
          ctx2d.beginPath();
          ctx2d.arc(x, y, 6, 0, Math.PI * 2);
          ctx2d.stroke();
        }
      }
    };

    drawPoints(wxValues, '#e2904a');
    drawPoints(wyValues, '#c2e350');

    const fits = state.caustic.fits || {};

    const plotFit = (axisKey, color) => {
      const fit = fits[axisKey];
      if (!fit) return;
      if (!Number.isFinite(fit.w0_m) || !Number.isFinite(fit.z0_m) || !Number.isFinite(fit.zR_prime_m)) return;
      if (!useMeterUnits && (!Number.isFinite(pixelSize) || pixelSize <= 0)) return;
      const zMeters = pointsForPlot.map((pt) => Number(pt.z_m)).filter((v) => Number.isFinite(v));
      if (!zMeters.length) return;
      const minM = Math.min(...zMeters);
      const maxM = Math.max(...zMeters);
      const span = Math.max(1e-9, maxM - minM);
      const start = minM - span * 0.1;
      const end = maxM + span * 0.1;
      const samples = [];
      const steps = 80;
      for (let i = 0; i < steps; i++) {
        const z = start + (end - start) * (i / (steps - 1));
        const term = (z - fit.z0_m) / fit.zR_prime_m;
        const w = Math.sqrt(Math.max(0, fit.w0_m * fit.w0_m * (1 + term * term)));
        const zDisplay = convertMetersToUnit(z, unit);
        const wDisplay = useMeterUnits ? w : w / pixelSize;
        if (!Number.isFinite(zDisplay) || !Number.isFinite(wDisplay)) continue;
        samples.push({ z: zDisplay, w: wDisplay });
      }
      if (!samples.length) return;

      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = 1.5;
      ctx2d.setLineDash([6, 4]);
      ctx2d.beginPath();
      samples.forEach((sample, idx) => {
        const x = toX(sample.z);
        const y = toY(sample.w);
        if (idx === 0) ctx2d.moveTo(x, y);
        else ctx2d.lineTo(x, y);
      });
      ctx2d.stroke();
      ctx2d.setLineDash([]);
    };

    plotFit('x', '#e2904a');
    plotFit('y', '#c2e350');
  }

  function renderCausticFitSummary() {
    if (!causticFitSummary) return;
    causticFitSummary.innerHTML = '';
    const fits = state.caustic.fits || {};
    const unit = state.caustic.config?.position_unit || 'mm';

    const entries = [];
    for (const axis of ['x', 'y']) {
      const fit = fits[axis];
      if (!fit || !Number.isFinite(fit.w0_m) || !Number.isFinite(fit.M2)) continue;
      const sigma = fit.sigma || {};
      const w0Um = convertMetersToUnit(fit.w0_m, 'um');
      const w0SigmaUm = convertMetersToUnit(sigma.w0_m, 'um');
      const z0Val = convertMetersToUnit(fit.z0_m, unit);
      const z0Sigma = convertMetersToUnit(sigma.z0_m, unit);
      const zRVal = convertMetersToUnit(fit.zR_prime_m, unit);
      const zRSigma = convertMetersToUnit(sigma.zR_prime_m, unit);

      const axisDiv = document.createElement('div');
      axisDiv.className = 'axis';
      const heading = document.createElement('strong');
      heading.textContent = axis === 'x' ? 'Axis X' : 'Axis Y';
      axisDiv.appendChild(heading);

      const row = (label, value, sigmaValue, suffix) => {
        const line = document.createElement('span');
        const valueStr = formatNumber(value);
        const sigmaStr = sigmaValue != null ? formatNumber(sigmaValue) : null;
        line.textContent = sigmaStr
          ? `${label}: ${valueStr}${suffix} ± ${sigmaStr}${suffix}`
          : `${label}: ${valueStr}${suffix}`;
        axisDiv.appendChild(line);
      };

      row('w0', w0Um, w0SigmaUm, ' µm');
      row('z0', z0Val, z0Sigma, ` ${unit}`);
      row(`zR'`, zRVal, zRSigma, ` ${unit}`);
      const mLine = document.createElement('span');
      const mSigma = sigma.M2 != null ? formatNumber(sigma.M2) : null;
      mLine.textContent = mSigma ? `M²: ${formatNumber(fit.M2)} ± ${mSigma}` : `M²: ${formatNumber(fit.M2)}`;
      axisDiv.appendChild(mLine);
      entries.push(axisDiv);
    }

    if (!entries.length) {
      const note = document.createElement('span');
      note.className = 'caustic-point-empty';
      note.textContent = 'Run M² fit to populate results.';
      causticFitSummary.appendChild(note);
    } else {
      for (const entry of entries) {
        causticFitSummary.appendChild(entry);
      }
    }
  }

  function selectCausticPoint(pointId) {
    state.caustic.selectedPointId = pointId;
    renderCausticPointList();
    renderCausticImages();
    renderCausticPlot();
  }

  function openCausticModal(roiId) {
    state.caustic.pendingRoiId = roiId;
    if (causticModalRoi) causticModalRoi.textContent = roiId;
    if (causticModalZ) {
      causticModalZ.value = '';
      setTimeout(() => causticModalZ.focus(), 20);
    }
    if (causticModal) causticModal.hidden = false;
  }

  function closeCausticModal() {
    state.caustic.pendingRoiId = null;
    if (causticModal) causticModal.hidden = true;
  }

  function scheduleCausticConfigUpdate() {
    if (causticConfigTimer) clearTimeout(causticConfigTimer);
    causticConfigTimer = setTimeout(() => {
      const payload = {};
      if (causticWavelength) {
        const val = Number(causticWavelength.value);
        if (Number.isFinite(val) && val > 0) payload.wavelength_nm = val;
      }
      if (causticUnit) {
        const unitVal = causticUnit.value.trim();
        if (unitVal) payload.position_unit = unitVal;
      }
      if (causticSource) {
        payload.radii_source = causticSource.value;
      }
      updateCausticConfig(payload);
    }, 200);
  }

  async function updateCausticConfig(changes) {
    try {
      const res = await postJSON('/caustic/config', changes);
      applyCausticState(res);
    } catch (error) {
      alert(`Failed to update caustic settings: ${parseServerError(error)}`);
    }
  }

  async function refreshCausticState() {
    try {
      const data = await getJSON('/caustic/state');
      applyCausticState(data);
    } catch (error) {
      console.warn('Failed to load caustic state', error);
    }
  }

  async function submitCausticPoint() {
    const roiId = state.caustic.pendingRoiId;
    if (!roiId) {
      closeCausticModal();
      return;
    }
    const zValue = Number(causticModalZ?.value);
    if (!Number.isFinite(zValue)) {
      alert('Enter a valid z-position.');
      return;
    }
    try {
      const res = await postJSON('/caustic/add', { roi_id: roiId, z: zValue });
      applyCausticState(res);
      closeCausticModal();
      logToServer('info', 'Caustic point added', { roi: roiId, z: zValue });
    } catch (error) {
      alert(`Failed to add caustic point: ${parseServerError(error)}`);
    }
  }

  async function removeCausticPoint(pointId) {
    try {
      const res = await fetch(`/caustic/${encodeURIComponent(pointId)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      applyCausticState(data);
    } catch (error) {
      alert(`Failed to remove point: ${parseServerError(error)}`);
    }
  }

  async function requestCausticFit() {
    const points = state.caustic.points || [];
    if (points.length < 3) {
      alert('Collect at least three caustic points before running the M² fit.');
      return;
    }
    if (!getUniformPixelSize(points)) {
      alert('Set the pixel size in the Beam Analysis panel before running the M² fit.');
      return;
    }
    try {
      const res = await postJSON('/caustic/fit', {});
      applyCausticState(res);
    } catch (error) {
      alert(`M² fit failed: ${parseServerError(error)}`);
    }
  }

  async function requestCausticSave() {
    const timestamp = new Date().toISOString().replace(/[:]/g, '').replace(/\..*/, '');
    const defaultName = `caustic_${timestamp}`;
    const input = window.prompt('Save caustic as...', defaultName);
    if (input === null) return;
    const base = input.trim();
    const query = base ? `?base=${encodeURIComponent(base)}` : '';
    try {
      const res = await fetch(`/caustic/save${query}`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const downloadName = res.headers.get('X-Download-Filename') || `${base || defaultName}.zip`;
      const a = document.createElement('a');
      a.href = url;
      a.download = downloadName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      alert(`Failed to save caustic dataset: ${parseServerError(error)}`);
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
      if (statsMax) {
        const peakRaw = Number.isFinite(snap?.frame_max_pixel_raw) ? snap.frame_max_pixel_raw : null;
        const peakProcessed = Number.isFinite(snap?.frame_max_pixel) ? snap.frame_max_pixel : null;
        if (peakRaw !== null) {
          const peakRawRounded = Math.round(peakRaw);
          const peakProcessedRounded = peakProcessed !== null ? Math.round(peakProcessed) : null;
          statsMax.textContent = peakProcessedRounded !== null && Math.abs(peakProcessedRounded - peakRawRounded) > 1
            ? `peak: ${peakRawRounded} raw / ${peakProcessedRounded} proc`
            : `peak: ${peakRawRounded}`;
          statsMax.classList.toggle('warn', peakRawRounded >= SATURATION_THRESHOLD);
          if (peakProcessedRounded !== null) {
            statsMax.title = `Raw peak ${peakRawRounded}, processed ${peakProcessedRounded}`;
          } else {
            statsMax.title = `Raw peak ${peakRawRounded}`;
          }
        } else {
          statsMax.textContent = 'peak: -';
          statsMax.classList.remove('warn');
          statsMax.removeAttribute('title');
        }
      }

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
      const peakRaw = Number.isFinite(m?.max_pixel_raw) ? Math.round(m.max_pixel_raw) : null;
      const isSaturated = peakRaw !== null && peakRaw >= SATURATION_THRESHOLD;
      card.classList.toggle('saturated', isSaturated);

      const head = document.createElement('div');
      head.className = 'head';
      head.innerHTML = `<span>${r.id}</span>`;

      const btnGroup = document.createElement('div');
      btnGroup.style.display = 'flex';
      btnGroup.style.gap = '6px';

      const addBtn = document.createElement('button');
      addBtn.textContent = 'Add to caustic';
      addBtn.onclick = (ev) => { ev.stopPropagation(); openCausticModal(r.id); };

      const resetBtn = document.createElement('button');
      resetBtn.textContent = 'Reset Max';
      resetBtn.onclick = () => resetMax(r.id);

      const delBtn = document.createElement('button');
      delBtn.className = 'danger';
      delBtn.textContent = 'Delete';
      delBtn.onclick = () => deleteRoi(r.id);

      btnGroup.appendChild(addBtn);
      btnGroup.appendChild(resetBtn);
      btnGroup.appendChild(delBtn);

      const coords = document.createElement('div');
      coords.className = 'coords';
      coords.textContent = `x:${r.x} y:${r.y} w:${r.w} h:${r.h}`;

      const metrics = document.createElement('div');
      metrics.className = 'metrics';
      const sumSpan = document.createElement('span');
      sumSpan.textContent = `Sum: ${m ? m.sum_gray : '-'}`;
      const vmsSpan = document.createElement('span');
      vmsSpan.textContent = `Value/ms: ${m ? m.value_per_ms.toFixed(1) : '-'}`;
      const peakSpan = document.createElement('span');
      if (peakRaw !== null) {
        peakSpan.textContent = `Peak: ${peakRaw}`;
        peakSpan.classList.toggle('warn', isSaturated);
        if (m && Number.isFinite(m.max_pixel_per_ms)) {
          peakSpan.title = `Exposure-normalized peak: ${m.max_pixel_per_ms.toFixed(1)}/ms`;
        } else {
          peakSpan.removeAttribute('title');
        }
      } else {
        peakSpan.textContent = 'Peak: -';
        peakSpan.classList.remove('warn');
        peakSpan.removeAttribute('title');
      }
      metrics.appendChild(sumSpan);
      metrics.appendChild(vmsSpan);
      metrics.appendChild(peakSpan);

      card.appendChild(head);
      card.appendChild(btnGroup);
      card.appendChild(coords);
      card.appendChild(metrics);
      card.onclick = (e) => {
        if (btnGroup.contains(e.target)) return;
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
      const titleSpan = document.createElement('span');
      titleSpan.className = 'title';
      titleSpan.textContent = r.id;
      const dimsSpan = document.createElement('span');
      dimsSpan.className = 'dims';
      dimsSpan.textContent = `${r.w}x${r.h}`;
      const headerGroup = document.createElement('div');
      headerGroup.className = 'id-group';
      headerGroup.appendChild(titleSpan);
      headerGroup.appendChild(dimsSpan);
      const peakSpan = document.createElement('span');
      peakSpan.className = 'peak';
      peakSpan.dataset.role = 'peak';
      peakSpan.textContent = 'peak: -';
      header.appendChild(headerGroup);
      header.appendChild(peakSpan);

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

      const addCausticBtn = document.createElement('button');
      addCausticBtn.textContent = 'Add to caustic';
      addCausticBtn.onclick = () => openCausticModal(r.id);
      toolbar.appendChild(addCausticBtn);

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
      const card = perRoiGrid.querySelector(`.per-roi-card[data-roi="${r.id}"]`);
      if (!card) continue;
      const plot = card.querySelector('.roi-integration-plot');
      if (!plot) continue;

      const m = metricsMap.get(r.id);
      const bar = plot.querySelector('.bar');
      const label = plot.querySelector('.label');
      const peakEl = card.querySelector('.header .peak');

      if (m) {
        const yMax = yMaxMap[r.id] || 1.0;
        const pct = (m.value_per_ms / Math.max(1.0, yMax)) * 100;
        bar.style.height = `${Math.min(100, Math.max(0, pct))}%`;
        label.textContent = `${m.value_per_ms.toFixed(1)}/ms`;
        if (peakEl) {
          if (Number.isFinite(m.max_pixel_raw)) {
            const peakRaw = Math.round(m.max_pixel_raw);
            peakEl.textContent = `peak: ${peakRaw}`;
            peakEl.classList.toggle('warn', peakRaw >= SATURATION_THRESHOLD);
            card.classList.toggle('saturated', peakRaw >= SATURATION_THRESHOLD);
            if (Number.isFinite(m.max_pixel_per_ms)) {
              peakEl.title = `Exposure-normalized peak: ${m.max_pixel_per_ms.toFixed(1)}/ms`;
            } else {
              peakEl.removeAttribute('title');
            }
          } else {
            peakEl.textContent = 'peak: -';
            peakEl.classList.remove('warn');
            peakEl.removeAttribute('title');
            card.classList.remove('saturated');
          }
        }
      } else {
        bar.style.height = '0%';
        label.textContent = '-';
        if (peakEl) {
          peakEl.textContent = 'peak: -';
          peakEl.classList.remove('warn');
          peakEl.removeAttribute('title');
        }
        card.classList.remove('saturated');
      }
    }
  }


  // ----- Overlay drawing -----
  function drawOverlay() {
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    const saturated = new Set();
    if (state.metrics?.rois) {
      for (const m of state.metrics.rois) {
        if (Number.isFinite(m.max_pixel_raw) && m.max_pixel_raw >= SATURATION_THRESHOLD) {
          saturated.add(m.id);
        }
      }
    }
    for (const r of state.rois) {
      const isSelected = r.id === state.selectedId;
      const color = saturated.has(r.id) ? '#ff5c6c' : (isSelected ? '#50e3c2' : '#4a90e2');
      drawRoi(r, color);
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
    if (autoExposureBtn) {
      autoExposureBtn.disabled = !cameraToggle.checked;
    }
  }

  if (autoExposureBtn) {
    autoExposureBtn.addEventListener('click', (ev) => {
      ev.preventDefault();
      if (!autoExposureBtn.disabled) {
        runAutoExposure();
      }
    });
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
    if (autoExposureBtn) {
      autoExposureBtn.disabled = !enabled;
    }
    try {
      const res = await postJSON('/camera', { enabled, camera_id: cameraId });
      const success = !!res.enabled;
      cameraToggle.checked = success;
      cameraSelect.disabled = success;
      if (autoExposureBtn) {
        autoExposureBtn.disabled = !success;
      }
      stream.src = success ? '/video_feed?ts=' + Date.now() : '';
      if (!success && res.error) {
        logToServer('error', 'Failed to toggle camera', { error: res.error });
        alert(`Error: ${res.error}`);
      }
    } catch (e) {
      logToServer('error', 'Failed to toggle camera', { error: e.toString() });
      cameraToggle.checked = !enabled;
      cameraSelect.disabled = !enabled;
      if (autoExposureBtn) {
        autoExposureBtn.disabled = !cameraToggle.checked;
      }
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

  if (causticCollapseBtn) {
    causticCollapseBtn.addEventListener('click', () => {
      state.caustic.collapsed = !state.caustic.collapsed;
      updateCausticUi();
    });
  }
  if (causticWavelength) {
    causticWavelength.addEventListener('change', scheduleCausticConfigUpdate);
    causticWavelength.addEventListener('blur', scheduleCausticConfigUpdate);
  }
  if (causticUnit) {
    causticUnit.addEventListener('change', scheduleCausticConfigUpdate);
    causticUnit.addEventListener('blur', scheduleCausticConfigUpdate);
  }
  if (causticSource) {
    causticSource.addEventListener('change', scheduleCausticConfigUpdate);
  }
  if (causticLoadBtn) {
    causticLoadBtn.addEventListener('click', openCausticImportModal);
  }
  if (causticImportClose) {
    causticImportClose.addEventListener('click', closeCausticImportModal);
  }
  if (causticImportCancel) {
    causticImportCancel.addEventListener('click', () => {
      if (causticImportActive) {
        closeCausticImportModal();
        showToast('Import continues in background.', { variant: 'info', duration: 4000 });
      } else {
        closeCausticImportModal();
      }
    });
  }
  if (causticImportConfirm) {
    causticImportConfirm.addEventListener('click', (ev) => {
      ev.preventDefault();
      if (causticImportActive) return;
      const folderValue = (causticImportFolder?.value || '').trim();
      if (!folderValue) {
        alert('Enter a folder path to import.');
        return;
      }
      const recursive = !!(causticImportRecursive && causticImportRecursive.checked);
      startCausticImport(folderValue, recursive);
    });
  }
  if (causticImportFolder) {
    causticImportFolder.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') {
        ev.preventDefault();
        if (!causticImportActive && causticImportConfirm) {
          causticImportConfirm.click();
        }
      }
    });
  }

  setCausticImportBusy(false);
  resetCausticImportModal();

  if (causticFitBtn) {
    causticFitBtn.addEventListener('click', requestCausticFit);
  }
  if (causticSaveBtn) {
    causticSaveBtn.addEventListener('click', requestCausticSave);
  }
  if (causticModalClose) {
    causticModalClose.addEventListener('click', closeCausticModal);
  }
  if (causticModalCancel) {
    causticModalCancel.addEventListener('click', closeCausticModal);
  }
  if (causticModalAdd) {
    causticModalAdd.addEventListener('click', submitCausticPoint);
  }
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && causticModal && !causticModal.hidden) {
      closeCausticModal();
    }
  });

  window.addEventListener('resize', () => {
    renderCausticPlot();
  });

  updateCausticUi();

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
      syncBeamOptions(),
      refreshCausticState()
    ]);
    pollMetrics();
    if (stream.complete) {
      state.naturalW = stream.naturalWidth || state.naturalW;
      state.naturalH = stream.naturalHeight || state.naturalH;
      resizeCanvasToImage();
    }
  })();
})();




