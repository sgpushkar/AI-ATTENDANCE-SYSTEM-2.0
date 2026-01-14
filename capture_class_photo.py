const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureClassBtn = document.getElementById("captureClassBtn");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => alert("Camera access denied"));

// ✅ Capture class photo
captureClassBtn.addEventListener("click", async (e) => {
  e.preventDefault(); // 🔴 VERY IMPORTANT

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const imageData = canvas.toDataURL("image/jpeg");

  const res = await fetch("/capture_class", {
    method: "POST",
    body: new URLSearchParams({
      imageData: imageData
    })
  });

  const data = await res.json();

  if (data.success) {
    alert("✅ Class photo saved");
  } else {
    alert("❌ Failed to save class photo");
  }
});