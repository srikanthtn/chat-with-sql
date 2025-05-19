document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();  // prevent normal form submission
    
    const form = e.target;
    const fileInput = form.file;
    const tableNameInput = form.tablename;

    if (!fileInput.files.length) {
        showPopup('Please select a file to upload.', true);
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    if (tableNameInput.value.trim()) {
        formData.append('tablename', tableNameInput.value.trim());
    }

    try {
        const res = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const text = await res.text();

        if (res.ok) {
            showPopup(text, false);
            form.reset();  // Clear form inputs after success
        } else {
            showPopup('Upload failed: ' + text, true);
        }
    } catch (err) {
        showPopup('Upload error: ' + err.message, true);
    }
});

function showPopup(message, isError = false) {
    const popup = document.getElementById('popup');
    popup.textContent = message;
    popup.classList.remove('hidden');
    popup.classList.toggle('error', isError);

    setTimeout(() => {
        popup.classList.add('hidden');
    }, 4000);
}
