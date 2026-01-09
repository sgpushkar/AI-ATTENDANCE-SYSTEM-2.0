// static/webcam_capture.js — enhanced: FPS meter, clarity check, adaptive constraints, safe upload
document.addEventListener('DOMContentLoaded', function(){
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');

  // camera stats overlay
  const statsEl = document.createElement('div');
  statsEl.id = 'cameraStats';
  statsEl.style.position='fixed';
  statsEl.style.left='10px';
  statsEl.style.top='70px';
  statsEl.style.padding='6px 8px';
  statsEl.style.background='rgba(0,0,0,0.5)';
  statsEl.style.color='#fff';
  statsEl.style.fontSize='12px';
  statsEl.style.borderRadius='8px';
  statsEl.style.zIndex = 9999;
  document.body.appendChild(statsEl);

  let stream = null;
  let lastTime = performance.now(), frames = 0, fps = 0;

  async function startCamera(constraints){
    try{
      if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; video.srcObject=null; }
      const c = constraints || { video: { width: 640, height: 480, frameRate: 15 }, audio: false };
      stream = await navigator.mediaDevices.getUserMedia(c);
      video.srcObject = stream;
      video.play();
      requestAnimationFrame(measureFPS);
      requestAnimationFrame(clarityLoop);
    }catch(e){
      console.error(e);
      toast('Camera denied or not available','error',3000);
    }
  }

  if (video) startCamera();

  function measureFPS(now){
    frames++;
    const dt = now - lastTime;
    if(dt >= 1000){
      fps = Math.round((frames / dt) * 1000);
      frames = 0;
      lastTime = now;
      // adapt if fps is terrible
      if(fps < 8){
        startCamera({ video: { width: 320, height: 240, frameRate: 15 }, audio: false });
      }
    }
    statsEl.innerText = `FPS: ${fps}`;
    requestAnimationFrame(measureFPS);
  }

  function clarityCheckFrame(){
    if(!video || !video.videoWidth) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video,0,0);
    const data = ctx.getImageData(0,0,canvas.width,canvas.height);
    let sum=0;
    for(let i=0;i<data.data.length;i+=4){
      sum += 0.299*data.data[i] + 0.587*data.data[i+1] + 0.114*data.data[i+2];
    }
    const avgLum = sum / (canvas.width * canvas.height);
    let difsum = 0, difcount=0;
    for(let y=1;y<canvas.height-1;y+=6){
      for(let x=1;x<canvas.width-1;x+=6){
        const idx = ((y*canvas.width)+x)*4;
        const v = 0.299*data.data[idx] + 0.587*data.data[idx+1] + 0.114*data.data[idx+2];
        const idxR = ((y*canvas.width)+(x+1))*4;
        const vr = 0.299*data.data[idxR] + 0.587*data.data[idxR+1] + 0.114*data.data[idxR+2];
        difsum += Math.abs(v - vr);
        difcount++;
      }
    }
    const lapVar = difcount ? (difsum / difcount) : 0;
    return { avgLum, lapVar };
  }

  function clarityLoop(){
    const c = clarityCheckFrame();
    if(c){
      const lumText = c.avgLum < 50 ? 'dark' : (c.avgLum > 200 ? 'bright' : 'ok');
      let clarityText = 'ok';
      if(c.lapVar < 6) clarityText = 'blurry';
      else if(c.lapVar < 12) clarityText = 'soft';
      else clarityText = 'clear';
      statsEl.innerText = `FPS: ${fps} • ${clarityText} • ${lumText}`;
    }
    setTimeout(clarityLoop, 1000);
  }

  async function safeFetchForm(url, form) {
    const headers = { 'X-Requested-With': 'XMLHttpRequest' };
    try {
      const resp = await fetch(url, { method: 'POST', body: form, headers });
      const contentType = resp.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        return await resp.json();
      } else {
        const text = await resp.text();
        return { success: resp.ok, message: text };
      }
    } catch (err) {
      console.error('safeFetchForm error', err);
      return { success: false, error: 'network' };
    }
  }

  const captureBtn = document.getElementById('captureBtn');
  const captureClassBtn = document.getElementById('captureClassBtn');

  captureBtn && captureBtn.addEventListener('click', async ()=>{
    if(!video.videoWidth){ toast('Camera not ready','error',2000); return; }
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

    const name = document.getElementById('studentName').value || 'unknown';
    const subject = document.getElementById('subjectSelectCapture') ? document.getElementById('subjectSelectCapture').value : 'General';
    const teacher = document.getElementById('teacherSelectCapture') ? document.getElementById('teacherSelectCapture').value : '';

    const form = new FormData();
    form.append('image', dataUrl);
    form.append('student_name', name);
    form.append('subject', subject);
    form.append('teacher', teacher);

    toast('Uploading student photo...', 'info', 2000);
    const j = await safeFetchForm('/upload_photo', form);
    if (j && j.success) {
      toast('Student photo saved', 'success', 2000);
    } else {
      console.warn('upload response', j);
      toast(j && (j.message || j.error || j.err || j.detail || 'Save failed') || 'Save failed', 'error', 4000);
    }
  });

  captureClassBtn && captureClassBtn.addEventListener('click', async ()=>{
    if(!video.videoWidth){ toast('Camera not ready','error',2000); return; }
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    const subject = document.getElementById('subjectSelectCapture') ? document.getElementById('subjectSelectCapture').value : 'General';
    const teacher = document.getElementById('teacherSelectCapture') ? document.getElementById('teacherSelectCapture').value : '';

    const form = new FormData();
    form.append('image', dataUrl);
    form.append('subject', subject);
    form.append('teacher', teacher);

    toast('Uploading class photo...', 'info', 2000);
    const j = await safeFetchForm('/upload_photo', form);
    if (j && j.success) {
      toast('Class photo uploaded, running recognition...', 'info', 2000);
      const form2 = new FormData(); form2.append('subject', subject); form2.append('teacher', teacher);
      const r = await safeFetchForm('/recognize', form2);
      if (r && (r.success || r.annotated_b64 || !r.error)) {
        toast('Attendance marked', 'success', 2000);
        setTimeout(()=> location.href='/attendance', 700);
      } else {
        toast(r && (r.message || r.error) || 'Recognition failed', 'error', 4000);
      }
    } else {
      toast(j && (j.message || j.error) || 'Upload failed', 'error', 4000);
    }
  });

});
