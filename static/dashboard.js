async function fetchEncodeStatus() {
  try {
    const r = await fetch('/encode_status');
    if (!r.ok) return;
    const s = await r.json();
    const el = document.getElementById('encodeStatus');
    if (el) el.textContent = `${s.message} (${s.progress || 0}%)`;
  } catch (e) { /* ignore */ }
}

async function runRecognize(tolerance=0.5) {
  try {
    const r = await fetch('/recognize_ajax', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ tolerance })
    });
    if (!r.ok) {
      const j = await r.json().catch(()=>({}));
      alert('Recognition failed: ' + (j.error || r.status));
      return;
    }
    const data = await r.json();
    document.getElementById('recognitionPanel').style.display = 'block';
    document.getElementById('annotatedPreview').src = data.annotated_b64;
    const detList = document.getElementById('detectionsList');
    detList.innerHTML = '';
    data.detections.forEach(d => {
      const div = document.createElement('div');
      div.className = 'detection';
      div.innerHTML = `<strong>${d.name}</strong> — ${(d.confidence*100).toFixed(1)}%`;
      detList.appendChild(div);
    });
  } catch (e) {
    alert('Recognition error: ' + e.message);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const runBtn = document.getElementById('runRecognize');
  const reRun = document.getElementById('reRun');
  const tolSlider = document.getElementById('tolSlider');
  const tolVal = document.getElementById('tolVal');
  if (runBtn) runBtn.addEventListener('click', () => runRecognize(parseFloat(tolSlider.value)));
  if (reRun) reRun.addEventListener('click', () => runRecognize(parseFloat(tolSlider.value)));
  if (tolSlider && tolVal) {
    tolVal.textContent = parseFloat(tolSlider.value).toFixed(2);
    tolSlider.addEventListener('input', () => {
      tolVal.textContent = parseFloat(tolSlider.value).toFixed(2);
    });
  }
  fetchEncodeStatus();
  setInterval(fetchEncodeStatus, 3000);
});
