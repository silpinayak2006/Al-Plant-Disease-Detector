const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const detectBtn = document.getElementById("detectBtn");
const result = document.getElementById("result");

// Display uploaded image
imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";

        result.innerText = "Image uploaded successfully!";
    }

});

// Connect to Flask AI prediction API
detectBtn.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        result.innerText = "Please upload a plant leaf image first.";
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    result.innerText = "AI is analyzing your plant leaf...";
    detectBtn.disabled = true;

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        // Display complete detection result
        result.innerHTML = `
            <h3>🌱 Detection Result</h3>

            <p><strong>Disease:</strong> ${data.disease}</p>

            <p><strong>Confidence:</strong> ${data.confidence}%</p>

            <hr>

            <h3>📖 Disease Information</h3>
            <p>${data.description}</p>

            <h3>🔍 Symptoms</h3>
            <p>${data.symptoms}</p>

            <h3>💊 Treatment</h3>
            <p>${data.treatment}</p>

            <h3>🛡️ Prevention</h3>
            <p>${data.prevention}</p>
        `;

    } catch (error) {

        result.innerText = "Error: " + error.message;

    } finally {

        detectBtn.disabled = false;

    }

});