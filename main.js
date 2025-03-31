const body = document.body;
    const imageInput = document.getElementById("imageInput");
    // imageInput.addEventListener("change", uploadImage);

    // Handle drag events
    body.addEventListener("dragover", (e) => {
      e.preventDefault();
      body.classList.add("dragover");
    });

    body.addEventListener("dragleave", () => {
      body.classList.remove("dragover");
    });

    body.addEventListener("drop", (e) => {
      e.preventDefault();
      body.classList.remove("dragover");

      const file = imageInput.files[0];
      if (file) uploadImage(file);
    });

    // Click anywhere to open file picker
    body.addEventListener("click", () => {
      imageInput.click();
    });

    // Handle manual file selection
    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      if (file) uploadImage(file);
    });

    async function uploadImage() {
      const fileInput = document.getElementById("imageInput");
      const file = fileInput.files[0];
      if (!file) return alert("Please select an image.");

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("https://ml-fossil-identifier.onrender.com/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      document.getElementById("result").innerHTML = `
        <p><strong>Prediction:</strong> ${data.prediction}</p>
        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
      `;
    }