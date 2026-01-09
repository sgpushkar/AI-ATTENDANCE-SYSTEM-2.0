// static/ui.js — toasts, preview modal, student modal, theme toggle
(function(){
  const toastsEl = document.getElementById('toasts');

  function makeToastElement(msg, cls, icon) {
    const el = document.createElement('div');
    el.className = `toast ${cls}`;
    el.innerHTML = `<div style="font-size:18px">${icon}</div><div style="min-width:140px">${msg}</div>`;
    return el;
  }

  function toast(msg, type='info', ttl=3000){
    if(!toastsEl) return;
    const icon = type === 'success' ? '✔️' : (type === 'error' ? '❌' : 'ℹ️');
    const cls = type === 'success' ? 'success' : (type === 'error' ? 'error' : 'info');
    const el = makeToastElement(msg, cls, icon);
    el.style.opacity = '0';
    el.style.transform = 'translateY(-6px)';
    toastsEl.appendChild(el);
    requestAnimationFrame(()=> { el.style.opacity='1'; el.style.transform='translateY(0)'; });
    setTimeout(()=> {
      el.style.opacity='0'; el.style.transform='translateY(-6px)';
      setTimeout(()=> el.remove(), 260);
    }, ttl);
  }
  window.toast = toast;

  // preview modal
  const previewModal = document.getElementById('previewModal');
  const previewImage = document.getElementById('previewImage');
  const previewUse = document.getElementById('previewUse');
  const previewRetake = document.getElementById('previewRetake');
  const previewDownload = document.getElementById('previewDownload');
  let currentPreviewDataURL = null;

  function openPreview(dataURL){
    currentPreviewDataURL = dataURL;
    previewImage.src = dataURL;
    previewModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }
  function closePreview(){
    previewModal.classList.add('hidden');
    document.body.style.overflow = '';
    currentPreviewDataURL = null;
  }
  previewRetake.addEventListener('click', () => { closePreview(); toast('Retake — starting camera', 'info'); });
  previewDownload.addEventListener('click', () => {
    if(!currentPreviewDataURL) return;
    const a = document.createElement('a'); a.href = currentPreviewDataURL; a.download = 'class_photo_preview.jpg'; a.click();
  });

  previewUse.addEventListener('click', async () => {
    if(!currentPreviewDataURL) return;
    toast('Uploading photo...', 'info');
    try {
      const form = new URLSearchParams();
      form.append('imageData', currentPreviewDataURL);
      const resp = await fetch('/upload_photo', { method:'POST', headers:{ 'Content-Type':'application/x-www-form-urlencoded', 'X-Requested-With':'XMLHttpRequest' }, body: form.toString() });
      const j = await resp.json().catch(()=>null);
      if(!resp.ok || !j || !j.success) { toast('Upload failed', 'error'); closePreview(); return; }
      toast('Uploaded — running recognition...', 'success');
      const r = await fetch('/recognize_ajax', { method:'POST', headers:{ 'Content-Type':'application/json', 'X-Requested-With':'XMLHttpRequest' }, body: JSON.stringify({ threshold:0.55 }) });
      const res = await r.json().catch(()=>null);
      if(r.ok && res && res.success){
        const win = window.open(); win.document.write(`<img src="${res.annotated_b64}" style="max-width:100%"/>`);
        toast(`Recognition: ${res.marked} marked / ${res.faces_found} faces`, 'success');
        setTimeout(()=> { window.location.href = '/attendance'; }, 900);
      } else {
        toast('Recognition failed', 'error'); closePreview();
      }
    } catch(e){
      toast('Upload/recognize error', 'error'); closePreview();
    }
  });

  window.openPreview = openPreview;
  window.closePreview = closePreview;

  // student modal
  const studentModal = document.getElementById('studentModal');
  const studentModalTitle = document.getElementById('studentModalTitle');
  const studentModalBody = document.getElementById('studentModalBody');
  const studentModalClose = document.getElementById('studentModalClose');

  studentModalClose.addEventListener('click', () => { studentModal.classList.add('hidden'); document.body.style.overflow = ''; });

  async function openStudentModal(name){
    studentModalTitle.textContent = name;
    studentModalBody.innerHTML = '<div class="text-xs text-slate-400">Loading…</div>';
    studentModal.classList.remove('hidden'); document.body.style.overflow = 'hidden';
    try {
      const rows = Array.from(document.querySelectorAll('table tbody tr')).map(tr => {
        const tds = tr.querySelectorAll('td');
        return { name: (tds[0]?.innerText||'').trim(), time: (tds[1]?.innerText||'').trim(), conf: (tds[2]?.innerText||'').trim() };
      }).filter(r => r.name === name);
      if(rows.length === 0) studentModalBody.innerHTML = '<div class="text-xs text-slate-400">No records found.</div>';
      else {
        studentModalBody.innerHTML = '<ul class="list-disc pl-5 space-y-1 text-sm"></ul>';
        const ul = studentModalBody.querySelector('ul');
        rows.forEach(r => {
          const li = document.createElement('li');
          li.innerHTML = `<strong>${r.time}</strong> — conf: ${r.conf}`;
          ul.appendChild(li);
        });
      }
    } catch(e) {
      studentModalBody.innerHTML = '<div class="text-xs text-red-500">Failed to load</div>';
    }
  }
  window.openStudentModal = openStudentModal;

  // student-row click
  document.addEventListener('click', (ev) => {
    const el = ev.target.closest('.student-row');
    if(el){
      const name = el.dataset.name || el.innerText.trim();
      if(name) openStudentModal(name);
    }
  });

  // theme toggle
  const themeToggle = document.getElementById('themeToggle');
  function setTheme(mode){
    if(mode === 'light') {
      document.documentElement.style.setProperty('--bg','white');
      document.body.classList.remove('bg-slate-950'); document.body.style.background = '#f8fafc'; document.body.style.color = '#0f172a';
      themeToggle.textContent = '☀️';
      localStorage.setItem('ui_theme','light');
    } else {
      document.body.style.background = ''; document.body.style.color = ''; document.body.classList.add('bg-slate-950');
      themeToggle.textContent = '🌙';
      localStorage.setItem('ui_theme','dark');
    }
  }
  themeToggle.addEventListener('click', () => {
    const cur = localStorage.getItem('ui_theme') || 'dark';
    setTheme(cur === 'dark' ? 'light' : 'dark');
  });
  // initialize
  const saved = localStorage.getItem('ui_theme') || 'dark';
  setTheme(saved);

})();
