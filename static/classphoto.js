// static/classphoto.js (patched)
// - same safeFetchForm helper and X-Requested-With header

document.addEventListener('DOMContentLoaded', function(){
  const startBtn = document.getElementById('startDashboardCam');
  const captureBtn = document.getElementById('captureClassMain');
  const video = document.getElementById('dashboardVideo');
  const canvas = document.getElementById('dashboardCanvas');
  const status = document.getElementById('classCaptureStatus');
  let stream = null;

  async function safeFetchForm(url, form) {
    const headers = { 'X-Requested-With': 'XMLHttpRequest' };
    try {
      const resp = await fetch(url, { method: 'POST', body: form, headers });
      const ct = resp.headers.get('content-type') || '';
      if (ct.includes('application/json')) return await resp.json();
      const text = await resp.text();
      return { success: resp.ok, message: text };
    } catch (err) {
      console.error(err);
      return { success: false, error: 'network' };
    }
  }

  startBtn && startBtn.addEventListener('click', async ()=>{
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; video.srcObject=null; startBtn.textContent='Start Camera'; status.textContent=''; return; }
    try{
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream; startBtn.textContent='Stop Camera';
      status.textContent = 'Camera started';
    }catch(e){ console.error(e); status.textContent='Camera error'; toast('Camera error','error',3000); }
  });

  captureBtn && captureBtn.addEventListener('click', async ()=>{
    if(!video.videoWidth){ status.textContent='Camera not started'; toast('Camera not started','error',2000); return; }
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

    const subject = document.getElementById('subjectSelect') ? document.getElementById('subjectSelect').value : 'General';
    const teacher = document.getElementById('teacherSelect') ? document.getElementById('teacherSelect').value : '';

    status.textContent = 'Uploading...';
    const form = new FormData();
    form.append('image', dataUrl);
    form.append('subject', subject);
    form.append('teacher', teacher);

    const j = await safeFetchForm('/upload_photo', form);
    if (j && j.success) {
      status.textContent = 'Uploaded. Running recognition...';
      const form2 = new FormData(); form2.append('subject', subject); form2.append('teacher', teacher);
      const r = await safeFetchForm('/recognize', form2);
      if (r && (r.success || r.annotated_b64 || !r.error)) {
        toast(`Attendance marked for subject ${subject}`, 'success', 3000);
        setTimeout(()=> location.href='/attendance', 800);
      } else {
        toast(r && (r.message || r.error) || 'Recognition failed', 'error', 4000);
        status.textContent = 'Recognition failed';
      }
    } else {
      status.textContent = 'Upload failed';
      toast(j && (j.message || j.error) || 'Upload failed', 'error', 4000);
    }
  });
});
