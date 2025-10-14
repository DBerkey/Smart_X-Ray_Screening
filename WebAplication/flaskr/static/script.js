// Update the filename field when a file is selected
function handleImageUpload(input) {
  if (input.files && input.files[0]) {
    document.getElementById('filenameField').value = input.files[0].name;
  }
}

// Image preview functionality
(function () {
  const input = document.getElementById('myFileInput');
  const field = document.getElementById('filenameField');
  const preview = document.getElementById('imagePreview');

  let objectUrl = null;

  // Clear previous preview and revoke object URL
  function clearPreview() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    preview.src = '';
    preview.classList.add('d-none');
  }

  // Make image visible when uploaded
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

  // Clear preview on form submission
  const form = input.closest('form');
  if (form) {
    form.addEventListener('submit', () => {
      clearPreview();
    });
  }
})();

document.addEventListener('DOMContentLoaded', function () {

  const fileInput = document.getElementById('myFileInput');
  const filenameField = document.getElementById('filenameField');

  // Clear file input and filename field on page load
  if (fileInput) fileInput.value = ""; 
  if (filenameField) filenameField.value = ""; 

  const toggle = document.getElementById('togglePatient');
  const fields = document.getElementById('patientFields');
  const ageSelect = document.getElementById('patientAge');


  // Reset toggle and fields on page load
  if (toggle) {
    toggle.checked = false;
  }
  if (fields) {
    fields.classList.add('d-none');
  }


  // Populate age dropdown if empty
  if (ageSelect && ageSelect.options.length === 0) {
    const placeholder = document.createElement('option');
    placeholder.value = "";
    placeholder.textContent = "Choose...";
    placeholder.disabled = true;
    placeholder.selected = true;
    ageSelect.appendChild(placeholder);

    const infantOpt = document.createElement('option');
    infantOpt.value = "0";
    infantOpt.textContent = "Below 1 (infant)";
    ageSelect.appendChild(infantOpt);

    for (let i = 1; i <= 100; i++) {
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = String(i);
      ageSelect.appendChild(opt);
    }

    const seniorOpt = document.createElement('option');
    seniorOpt.value = "100+";
    seniorOpt.textContent = "100+";
    ageSelect.appendChild(seniorOpt);
  }

  // Toggle patient fields visibility
  if (toggle && fields) {
    toggle.addEventListener('change', function () {
      if (toggle.checked) {
        fields.classList.remove('d-none');
      } else {
        fields.classList.add('d-none');
        const sex = document.getElementById('patientSex');
        const age = document.getElementById('patientAge');
        if (sex) sex.selectedIndex = 0;
        if (age) age.selectedIndex = 0;
      }
    });
  }
});