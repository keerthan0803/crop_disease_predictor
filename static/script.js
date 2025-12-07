// Handle file selection
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const resultsContainer = document.getElementById('resultsContainer');
const resultsContent = document.getElementById('resultsContent');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorContainer = document.getElementById('errorContainer');

let selectedFile = null;

// Upload area click handler
uploadArea.addEventListener('click', () => fileInput.click());

// File selection handler
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isValidFile(file)) {
            handleFileSelect(file);
        } else {
            showError('Invalid file type. Please upload PNG, JPG, or JPEG images.');
        }
    }
});

function isValidFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    return validTypes.includes(file.type);
}

function handleFileSelect(file) {
    if (!isValidFile(file)) {
        showError('Invalid file type. Please upload PNG, JPG, or JPEG images.');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadArea.style.display = 'none';
        predictBtn.disabled = false;
        errorContainer.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    predictBtn.disabled = true;
    resultsContent.style.display = 'none';
    resultsContainer.style.display = 'block';
    errorContainer.style.display = 'none';
}

function showError(message) {
    errorContainer.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    errorContainer.style.display = 'block';
}

function hideError() {
    errorContainer.style.display = 'none';
}

async function predictDisease() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Show loading spinner
    loadingSpinner.style.display = 'block';
    resultsContent.style.display = 'none';
    resultsContainer.style.display = 'none';
    hideError();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        showError(`Prediction error: ${error.message}`);
        loadingSpinner.style.display = 'none';
    }
}

function displayResults(result) {
    loadingSpinner.style.display = 'none';
    resultsContainer.style.display = 'none';
    resultsContent.style.display = 'block';

    // Update results
    document.getElementById('plantResult').textContent = result.plant || 'Unknown';
    
    const diseaseElement = document.getElementById('diseaseResult');
    diseaseElement.textContent = result.disease || 'Unknown';
    
    // Color code the disease result
    if (result.disease === 'Healthy') {
        diseaseElement.style.color = '#28a745';
        diseaseElement.style.background = '#d4edda';
    } else {
        diseaseElement.style.color = '#dc3545';
        diseaseElement.style.background = '#f8d7da';
    }

    // Update confidence bar
    const confidence = result.confidence.toFixed(2);
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = confidence + '%';
    confidenceBar.textContent = confidence + '%';
    document.getElementById('confidenceText').textContent = confidence + '%';

    // Display top 5 predictions
    const allPredictions = result.all_predictions;
    const topPredictions = Object.entries(allPredictions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    const topPredictionsContainer = document.getElementById('topPredictions');
    topPredictionsContainer.innerHTML = '';

    topPredictions.forEach((pred, index) => {
        const [className, score] = pred;
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="prediction-item-name">${index + 1}. ${className}</span>
            <span class="prediction-item-score">${score.toFixed(2)}%</span>
        `;
        topPredictionsContainer.appendChild(item);
    });

    hideError();
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Check model status on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/model-status');
        const status = await response.json();
        
        if (!status.loaded) {
            showError('Model is not loaded. Please train the model first using train_model.py');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
});
