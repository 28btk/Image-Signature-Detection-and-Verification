const modeButtons = document.querySelectorAll(".mode-button");
const modePanels = document.querySelectorAll(".mode-panel");

const detectionForm = document.getElementById("detection-form");
const verificationForm = document.getElementById("verification-form");

const documentInput = document.getElementById("document-input");
const signatureAInput = document.getElementById("signature-a-input");
const signatureBInput = document.getElementById("signature-b-input");

const documentPreview = document.getElementById("document-preview");
const signatureAPreview = document.getElementById("signature-a-preview");
const signatureBPreview = document.getElementById("signature-b-preview");

const detectionStatus = document.getElementById("detection-status");
const verificationStatus = document.getElementById("verification-status");
const detectionResult = document.getElementById("detection-result");
const verificationResult = document.getElementById("verification-result");

function switchMode(mode) {
    modeButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === mode);
    });

    modePanels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === `${mode}-panel`);
    });
}

modeButtons.forEach((button) => {
    button.addEventListener("click", () => switchMode(button.dataset.mode));
});

function renderPreview(input, imgElement, emptyLabelId) {
    const emptyLabel = document.getElementById(emptyLabelId);
    const [file] = input.files;

    if (!file) {
        imgElement.style.display = "none";
        imgElement.removeAttribute("src");
        emptyLabel.style.display = "block";
        return;
    }

    const objectUrl = URL.createObjectURL(file);
    imgElement.src = objectUrl;
    imgElement.style.display = "block";
    emptyLabel.style.display = "none";
}

documentInput.addEventListener("change", () => renderPreview(documentInput, documentPreview, "document-preview-empty"));
signatureAInput.addEventListener("change", () => renderPreview(signatureAInput, signatureAPreview, "signature-a-empty"));
signatureBInput.addEventListener("change", () => renderPreview(signatureBInput, signatureBPreview, "signature-b-empty"));

function setStatus(element, message, state = "") {
    element.textContent = message;
    element.className = `status-text ${state}`.trim();
}

function metricCard(label, value) {
    return `
        <article class="metric-card">
            <div class="metric-label">${label}</div>
            <div class="metric-value">${value}</div>
        </article>
    `;
}

function imageCard(label, imageUrl) {
    return `
        <article class="result-card">
            <h3>${label}</h3>
            <img src="${imageUrl}" alt="${label}">
        </article>
    `;
}

function prettyJson(data) {
    return JSON.stringify(data, null, 2);
}

function renderDetectionResult(data) {
    const bbox = data.detection.bbox;
    detectionResult.innerHTML = `
        <div class="result-shell">
            <section class="result-hero">
                <span class="result-badge">Detection Complete</span>
                <h3>Bounding box: x=${bbox.x}, y=${bbox.y}, w=${bbox.width}, h=${bbox.height}</h3>
                <p>Pipeline: ${data.pipeline.join(" -> ")}</p>
            </section>

            <section class="metrics-grid">
                ${metricCard("Candidate score", data.detection.candidate_score)}
                ${metricCard("Candidate count", data.detection.candidate_count)}
                ${metricCard("Fill ratio", data.detection.fill_ratio)}
                ${metricCard("Contour area", data.detection.contour_area)}
            </section>

            <section class="image-grid">
                ${imageCard("Detection Mask", data.outputs.mask_image)}
                ${imageCard("Annotated Document", data.outputs.annotated_image)}
                ${imageCard("Cropped Signature", data.outputs.cropped_signature)}
                ${imageCard("Normalized Signature", data.outputs.normalized_signature)}
            </section>

            <section class="data-block">
                <pre>${prettyJson(data.detection)}</pre>
            </section>
        </div>
    `;
}

function renderVerificationResult(data) {
    const verification = data.verification;
    const evaluation = verification.evaluation;
    const baseline = verification.baseline;
    const improvement = verification.improvement;
    const alignment = verification.metrics.alignment_offset;

    verificationResult.innerHTML = `
        <div class="result-shell">
            <section class="result-hero">
                <span class="result-badge">${verification.prediction.toUpperCase()}</span>
                <h3>Advanced score: ${verification.score} | Baseline: ${baseline.score} | Threshold: ${verification.threshold}</h3>
                <p>${evaluation.message}</p>
            </section>

            <section class="metrics-grid">
                ${metricCard("Confidence", verification.confidence)}
                ${metricCard("Score delta", improvement.score_delta)}
                ${metricCard("Feature similarity", verification.metrics.feature_similarity)}
                ${metricCard("Overlap similarity", verification.metrics.overlap_similarity)}
                ${metricCard("Exact overlap", verification.metrics.exact_overlap_similarity)}
                ${metricCard("Contour similarity", verification.metrics.contour_similarity)}
                ${metricCard("Density similarity", verification.metrics.density_similarity)}
                ${metricCard("Chamfer similarity", verification.metrics.chamfer_similarity)}
                ${metricCard("Keypoint similarity", verification.metrics.keypoint_similarity)}
                ${metricCard("Keypoint method", verification.metrics.keypoint_method)}
                ${metricCard("Keypoint matches", verification.metrics.keypoint_good_matches)}
                ${metricCard("RANSAC inlier ratio", verification.metrics.keypoint_inlier_ratio)}
                ${metricCard("Alignment offset", `dx=${alignment.dx}, dy=${alignment.dy}`)}
                ${metricCard("Baseline prediction", baseline.prediction)}
                ${metricCard("Margin", evaluation.margin)}
            </section>

            <section class="image-grid">
                ${imageCard("Normalized Signature A", data.outputs.normalized_signature_a)}
                ${imageCard("Normalized Signature B", data.outputs.normalized_signature_b)}
                ${imageCard("Comparison Preview", data.outputs.comparison_preview)}
            </section>

            <section class="data-block">
                <pre>${prettyJson({
                    features: data.features,
                    baseline: baseline,
                    improvement: improvement,
                })}</pre>
            </section>
        </div>
    `;
}

async function submitForm(url, formData) {
    const response = await fetch(url, {
        method: "POST",
        body: formData,
    });

    const data = await response.json();
    if (!response.ok || !data.success) {
        throw new Error(data.error || "Request failed.");
    }
    return data;
}

detectionForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    detectionResult.innerHTML = "";
    setStatus(detectionStatus, "Running detection...", "");

    try {
        const formData = new FormData(detectionForm);
        const data = await submitForm("/api/detect", formData);
        renderDetectionResult(data);
        setStatus(detectionStatus, "Detection completed successfully.", "success");
    } catch (error) {
        setStatus(detectionStatus, error.message, "error");
    }
});

verificationForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    verificationResult.innerHTML = "";
    setStatus(verificationStatus, "Running verification...", "");

    try {
        const formData = new FormData(verificationForm);
        const thresholdInput = document.getElementById("threshold-input");
        if (!thresholdInput.value) {
            formData.delete("threshold");
        }
        const data = await submitForm("/api/verify", formData);
        renderVerificationResult(data);
        setStatus(verificationStatus, "Verification completed successfully.", "success");
    } catch (error) {
        setStatus(verificationStatus, error.message, "error");
    }
});
