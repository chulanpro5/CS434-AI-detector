import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;

const status = document.getElementById('status');
const fileUpload = document.getElementById('file-upload');
const imageContainer = document.getElementById('image-container');
const resultContainer = document.getElementById('result-container');
const loader = document.getElementById('loader'); // Reference to loader

status.textContent = 'Loading model...';
const classifier = await pipeline('image-classification', 'Organika/sdxl-detector', {
    device: 'webgpu',
    revision: 'main',
    dtype: 'fp32'
});
status.textContent = 'Ready';

fileUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e2) {
        imageContainer.innerHTML = '';
        const image = document.createElement('img');
        image.src = e2.target.result;
        imageContainer.appendChild(image);
        classify(image);
    };
    reader.readAsDataURL(file);
});

async function classify(img) {
    loader.style.display = 'block'; // Show loader
    status.textContent = 'Analyzing...';
    const output = await classifier(img.src);
    status.textContent = '';
    loader.style.display = 'none'; // Hide loader
    displayResults(output);
}

function displayResults(results) {
    resultContainer.innerHTML = '<h2>Classification Results:</h2>';
    const ul = document.createElement('ul');
    results.forEach(({ label, score }) => {
        const li = document.createElement('li');
        li.textContent = `${label}: ${(score * 100).toFixed(2)}%`;
        li.className = score >= 0.75 ? 'high-certainty' : 'low-certainty'; // Add class based on certainty
        ul.appendChild(li);
    });
    resultContainer.appendChild(ul);
}
