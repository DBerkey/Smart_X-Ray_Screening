function handleImageUpload(input) {
    if (input.files && input.files[0]) {
      document.getElementById('filenameField').value = input.files[0].name;
    }
  }

  (function () {
    const input   = document.getElementById('myFileInput');
    const field   = document.getElementById('filenameField');
    const preview = document.getElementById('imagePreview');

    let objectUrl = null;

    function clearPreview() {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
      }
      preview.src = '';
      preview.classList.add('d-none');
    }

    input.addEventListener('change', function () {
      const file = this.files && this.files[0];

      field.value = file ? file.name : '';

      clearPreview();

      if (!file) return;                        
      if (!file.type.startsWith('image/')) {    
        alert('Please select an image file.');
        this.value = '';
        return;
      }

      objectUrl = URL.createObjectURL(file);
      preview.src = objectUrl;
      preview.classList.remove('d-none');
    });

    const form = input.closest('form');
    if (form) {
      form.addEventListener('submit', () => {
        clearPreview();
      });
    }
  })();