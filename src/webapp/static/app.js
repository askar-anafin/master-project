document.addEventListener('DOMContentLoaded', () => {
    const btnSample = document.getElementById('btn-sample');
    const btnPredict = document.getElementById('btn-predict');
    const badgeSampleId = document.getElementById('sample-id-badge');
    const badgeTrueDiag = document.getElementById('true-diag-badge');
    const canvas = document.getElementById('ecg-canvas');
    const ctx = canvas.getContext('2d');
    const modelsGrid = document.getElementById('models-grid');
    const loadingOverlay = document.getElementById('loading-overlay');

    let currentSampleIdx = null;

    // Resize canvas to match display size
    function resizeCanvas() {
        const rect = canvas.parentNode.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function drawGrid() {
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.15)'; // Classic ECG green grid
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x <= canvas.width; x += 20) {
            ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
        }
        for (let y = 0; y <= canvas.height; y += 20) {
            ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
        }
        ctx.stroke();

        ctx.strokeStyle = 'rgba(0, 255, 0, 0.05)'; // minor lines
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

        // signalData is array of length N (e.g. 1000) where each element is an array of 12 (leads)
        const N = signalData.length;
        if (N === 0) return;

        // Plot 4 leads: I, II, III, aVR (indices 0, 1, 2, 3)
        const leads = [0, 1, 2, 3];
        const leadNames = ['Lead I', 'Lead II', 'Lead III', 'aVR'];
        const sectionHeight = canvas.height / leads.length;

        leads.forEach((leadIndex, i) => {
            const baseY = i * sectionHeight + (sectionHeight / 2);

            ctx.beginPath();
            ctx.strokeStyle = '#22d3ee'; // Cyan color for signal
            ctx.lineWidth = 1.5;

            // Simple scaling
            const y_scale = 30;
            const x_scale = canvas.width / N;

            for (let t = 0; t < N; t++) {
                const val = signalData[t][leadIndex];
                const x = t * x_scale;
                const y = baseY - (val * y_scale); // invert Y since canvas 0 is top

                if (t === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Label
            ctx.fillStyle = '#f8fafc';
            ctx.font = '12px Outfit';
            ctx.fillText(leadNames[i], 10, i * sectionHeight + 20);
        });
    }

    async function fetchSample() {
        loadingOverlay.classList.remove('hidden');
        btnPredict.disabled = true;

        try {
            const res = await fetch('/api/sample');
            if (!res.ok) throw new Error(await res.text());

            const data = await res.json();
            currentSampleIdx = data.index;

            badgeSampleId.textContent = `Sample ID: ${data.index}`;
            badgeTrueDiag.textContent = `True Diagnosis: ${data.true_diagnoses.length > 0 ? data.true_diagnoses.join(', ') : 'None'}`;
            badgeTrueDiag.style.color = 'var(--text-main)';
            badgeTrueDiag.style.borderColor = 'var(--accent)';

            plotEcg(data.signal);
            btnPredict.disabled = false;

            // clear results
            modelsGrid.innerHTML = '<div class="empty-state"><p>Ready for analysis.</p></div>';

        } catch (e) {
            console.error(e);
            alert('Failed to fetch sample. Is backend running?');
        } finally {
            loadingOverlay.classList.remove('hidden');
            setTimeout(() => loadingOverlay.classList.add('hidden'), 300); // slight delay for smooth transition
        }
    }

    async function runPredictions() {
        if (currentSampleIdx === null) return;

        loadingOverlay.classList.remove('hidden');

        try {
            const res = await fetch(`/api/predict/${currentSampleIdx}`, { method: 'POST' });
            if (!res.ok) throw new Error(await res.text());

            const results = await res.json();

            modelsGrid.innerHTML = ''; // clear grid
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
                    const isHigh = prob > 0.5;

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

            // Animate bars
            setTimeout(() => {
                document.querySelectorAll('.pred-bar').forEach(bar => {
                    const target = bar.getAttribute('data-target');
                    bar.style.width = target + '%';
                });
            }, 100);

        } catch (e) {
            console.error(e);
            alert('Prediction failed.');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    btnSample.addEventListener('click', fetchSample);
    btnPredict.addEventListener('click', runPredictions);
});
