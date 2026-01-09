document.addEventListener("DOMContentLoaded", () => {
  const uploadModal = document.getElementById("uploadModal");
  const openUpload = document.getElementById("openUpload");
  const closeUpload = document.getElementById("closeUpload");
  const cancelUpload = document.getElementById("cancelUpload");
  const uploadForm = document.getElementById("uploadForm");

  function show() { uploadModal.setAttribute("aria-hidden", "false"); }
  function hide() { uploadModal.setAttribute("aria-hidden", "true"); }

  if (openUpload) openUpload.addEventListener("click", show);
  if (closeUpload) closeUpload.addEventListener("click", hide);
  if (cancelUpload) cancelUpload.addEventListener("click", hide);

  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      const btn = uploadForm.querySelector("button[type='submit']");
      if (btn) { btn.disabled = true; btn.textContent = "Uploading…"; }
    });
  }

  document.querySelectorAll("form[action^='/students/delete']").forEach(f => {
    f.addEventListener('submit', (ev) => {
      if (!confirm('Delete student and all images? This cannot be undone.')) ev.preventDefault();
    });
  });
});
