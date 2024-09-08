import { pipeline, env } from '@huggingface/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Reference the elements that we will need
const status = document.getElementById('status');
const fileUpload = document.getElementById('file-upload');
const imageContainer = document.getElementById('image-container');
const resultContainer = document.getElementById('result-container');

// Create a new image classification pipeline
status.textContent = 'Loading model...';
const classifier = await pipeline(
    'image-classification',
    // 'Xenova/vit-base-patch16-224',
    'Organika/sdxl-detector',
    {
        device: 'webgpu', // or 'wasm'
        revision: 'refs/pr/3',
        dtype: 'fp32'
    },
);
status.textContent = 'Ready';

fileUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();

    // Set up a callback when the file is loaded
    reader.onload = function (e2) {
        imageContainer.innerHTML = '';
        const image = document.createElement('img');
        image.src = e2.target.result;
        imageContainer.appendChild(image);
        classify(image);
    };
    reader.readAsDataURL(file);
});

// Classify the image
async function classify(img) {
    status.textContent = 'Analysing...';
    const output = await classifier(img.src);
    status.textContent = '';
    displayResults(output);
}

// Display the classification results
function displayResults(results) {
    resultContainer.innerHTML = '<h2>Classification Results:</h2>';
    const ul = document.createElement('ul');
    results.forEach(({ label, score }) => {
        const li = document.createElement('li');
        li.textContent = `${label}: ${(score * 100).toFixed(2)}%`;
        ul.appendChild(li);
    });
    resultContainer.appendChild(ul);
}