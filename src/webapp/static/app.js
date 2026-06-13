document.addEventListener('DOMContentLoaded', () => {
    const btnSample = document.getElementById('btn-sample');
    const btnPredict = document.getElementById('btn-predict');
    const btnExport = document.getElementById('btn-export');
    const fileUpload = document.getElementById('file-upload');
    const badgeSampleId = document.getElementById('sample-id-badge');
    const badgeTrueDiag = document.getElementById('true-diag-badge');
    const canvas = document.getElementById('ecg-canvas');
    const ctx = canvas.getContext('2d');
    const modelsGrid = document.getElementById('models-grid');
    const loadingOverlay = document.getElementById('loading-overlay');

    let currentSampleIdx = null;
    let loadedSignal = null; // Stores the active downsampled signal for export

    // Resize canvas to match display size
    function resizeCanvas() {
        const rect = canvas.parentNode.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = 1200; // Fixed height for 12 leads (100px per lead)
        if (loadedSignal) {
            plotEcg(loadedSignal);
        } else {
            drawGrid();
        }
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function drawGrid() {
        // Clinical ECG pink/red paper background styling
        ctx.fillStyle = '#fff9f9'; 
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Major lines (20px intervals representing 0.2s at standard paper speed)
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.22)'; 
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x <= canvas.width; x += 20) {
            ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
        }
        for (let y = 0; y <= canvas.height; y += 20) {
            ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
        }
        ctx.stroke();

        // Minor lines (4px intervals representing 0.04s)
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.07)'; 
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        for (let x = 0; x <= canvas.width; x += 4) {
            ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
        }
        for (let y = 0; y <= canvas.height; y += 4) {
            ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
        }
        ctx.stroke();
    }

    function plotEcg(signalData) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawGrid();

        const N = signalData.length;
        if (N === 0) return;

        // Plot all 12 standard leads
        const leads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        const leadNames = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
        const sectionHeight = canvas.height / leads.length;

        leads.forEach((leadIndex, i) => {
            const baseY = i * sectionHeight + (sectionHeight / 2);

            ctx.beginPath();
            ctx.strokeStyle = '#0f172a'; // Clinical dark slate trace
            ctx.lineWidth = 1.5;

            const y_scale = 30; // standard amplitude scale
            const x_scale = canvas.width / N;

            for (let t = 0; t < N; t++) {
                const val = signalData[t][leadIndex];
                const x = t * x_scale;
                const y = baseY - (val * y_scale);

                if (t === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Lead Identifier label
            ctx.fillStyle = '#1e293b';
            ctx.font = 'bold 12px Outfit';
            ctx.fillText(leadNames[i], 10, i * sectionHeight + 20);
        });
    }

    async function fetchSample() {
        loadingOverlay.classList.remove('hidden');
        btnPredict.disabled = true;
        btnExport.disabled = true;

        try {
            const res = await fetch('/api/sample');
            if (!res.ok) throw new Error(await res.text());

            const data = await res.json();
            currentSampleIdx = data.index;
            loadedSignal = data.signal;

            badgeSampleId.textContent = `Sample ID: ${data.index}`;
            badgeTrueDiag.textContent = `True Labels: ${data.true_diagnoses.length > 0 ? data.true_diagnoses.join(', ') : 'None'}`;
            badgeTrueDiag.style.color = 'var(--text-main)';
            badgeTrueDiag.style.borderColor = 'var(--accent)';

            plotEcg(data.signal);
            btnPredict.disabled = false;
            btnExport.disabled = false;

            // reset results view
            modelsGrid.innerHTML = '<div class="empty-state"><p>Sample loaded. Click "Run Diagnostics" to execute inference.</p></div>';

        } catch (e) {
            console.error(e);
            alert('Failed to fetch sample. Is FastAPI backend running?');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    async function runPredictions() {
        if (currentSampleIdx === null) return;

        loadingOverlay.classList.remove('hidden');

        try {
            const res = await fetch(`/api/predict/${currentSampleIdx}`, { method: 'POST' });
            if (!res.ok) throw new Error(await res.text());

            const results = await res.json();
            displayPredictions(results);

        } catch (e) {
            console.error(e);
            alert('Prediction failed.');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    function displayPredictions(results) {
        modelsGrid.innerHTML = '';
        const classes_order = ['NORM', 'MI', 'STTC', 'CD', 'HYP'];

        results.forEach((resItem, i) => {
            const card = document.createElement('div');
            card.className = 'model-card slide-up';
            card.style.animationDelay = `${i * 0.1}s`;

            let html = `<div class="model-name">
                <span>${resItem.model.toUpperCase()}</span>
                <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
            </div>
            <ul class="pred-list">`;

            classes_order.forEach(cls => {
                const prob = resItem.probabilities[cls];
                const probPct = (prob * 100).toFixed(1);
                const isHigh = resItem.predicted_classes.includes(cls);

                html += `
                    <li class="pred-item ${isHigh ? 'high-prob' : ''}">
                        <div style="width:100%">
                            <div style="display:flex; justify-content:space-between">
                                <span class="pred-class">${cls}</span>
                                <span class="pred-prob">${probPct}%</span>
                            </div>
                            <div class="pred-bar-container">
                                <div class="pred-bar" style="width: 0%" data-target="${probPct}"></div>
                            </div>
                        </div>
                    </li>
                `;
            });

            html += `</ul>`;
            card.innerHTML = html;
            modelsGrid.appendChild(card);
        });

        // Trigger animations
        setTimeout(() => {
            document.querySelectorAll('.pred-bar').forEach(bar => {
                const target = bar.getAttribute('data-target');
                bar.style.width = target + '%';
            });
        }, 100);
    }

    // Export current loaded downsampled signal as a JSON file
    btnExport.addEventListener('click', () => {
        if (!loadedSignal) return;
        
        // Reconstruct full 5000-point size if user wants to upload it back, 
        // but wait! The API expects shape (5000, 12).
        // Since we downsampled signal by a factor of 5 for frontend performance, 
        // to make it perfectly uploadable back, we should upscale it back by repeating points 5 times 
        // or just interpolate, or we can write a simple repeat to get 5000 points.
        // Let's repeat each downsampled point 5 times to form a valid (5000, 12) signal!
        const fullSignal = [];
        for (let t = 0; t < loadedSignal.length; t++) {
            for (let r = 0; r < 5; r++) {
                fullSignal.push(loadedSignal[t]);
            }
        }

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(fullSignal));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", `ecg_signal_sample_${currentSampleIdx}.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
    });

    // File Upload Handler (JSON or CSV)
    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        loadingOverlay.classList.remove('hidden');
        btnPredict.disabled = true;
        btnExport.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/predict_upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error(await res.text());

            const data = await res.json();
            loadedSignal = data.signal;

            badgeSampleId.textContent = `Uploaded File: ${file.name}`;
            badgeTrueDiag.textContent = `True Labels: Unknown (User Uploaded)`;
            badgeTrueDiag.style.color = 'var(--text-muted)';
            badgeTrueDiag.style.borderColor = 'var(--panel-border)';

            plotEcg(loadedSignal);
            displayPredictions(data.predictions);

        } catch (err) {
            console.error(err);
            alert('Failed to process and predict uploaded ECG file: ' + err.message);
        } finally {
            loadingOverlay.classList.add('hidden');
            fileUpload.value = ''; // Reset input element
        }
    });

    btnSample.addEventListener('click', fetchSample);
    btnPredict.addEventListener('click', runPredictions);
});
