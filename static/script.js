document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileNameSpan = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    
    const textInput = document.getElementById('text-input');
    const wordCountSpan = document.getElementById('word-count');
    const urlInput = document.getElementById('url-input');
    
    const modelChoice = document.getElementById('model-choice');
    const summaryLength = document.getElementById('summary-length');
    const summaryStyle = document.getElementById('summary-style');
    const summarizeBtn = document.getElementById('summarize-btn');
    
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const loadingStatus = document.getElementById('loading-status');
    const summaryWrapper = document.getElementById('summary-wrapper');
    const summaryOutput = document.getElementById('summary-output');
    const summaryActions = document.getElementById('summary-actions');
    const topicsList = document.getElementById('topics-list');
    
    const metricRougeL = document.getElementById('metric-rouge-l');
    const metricBert = document.getElementById('metric-bert');
    const metricCompression = document.getElementById('metric-compression');
    const metricTime = document.getElementById('metric-time');
    
    const listenBtn = document.getElementById('listen-btn');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');

    let currentTab = 'upload';
    let selectedFile = null;
    let synth = window.speechSynthesis;
    let speaking = false;

    // --- Tab Switching Logic ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            currentTab = btn.dataset.tab;
            document.getElementById(`tab-${currentTab}`).classList.add('active');
        });
    });

    // --- File Upload & Drag-and-Drop ---
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameSpan.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileInfo.style.display = 'inline-flex';
    }

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
    });

    // --- Word Count Tracker ---
    textInput.addEventListener('input', () => {
        const text = textInput.value.trim();
        const words = text ? text.split(/\s+/).length : 0;
        wordCountSpan.textContent = words;
    });

    // --- Summarize Execution ---
    summarizeBtn.addEventListener('click', async () => {
        const lengthVal = summaryLength.value;
        const styleVal = summaryStyle.value;
        const modelVal = modelChoice.value;

        // Reset UI States
        emptyState.style.display = 'none';
        summaryWrapper.style.display = 'none';
        summaryActions.style.display = 'none';
        loadingState.style.display = 'block';

        try {
            let data = null;

            if (currentTab === 'upload') {
                if (!selectedFile) {
                    alert('Please select or upload a document file (PDF, DOCX, TXT, or Image).');
                    resetToEmpty();
                    return;
                }

                loadingStatus.textContent = 'Extracting document & running PaddleOCR pipeline...';
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('length', lengthVal);
                formData.append('style', styleVal);
                formData.append('model_choice', modelVal);

                const response = await fetch('/api/summarize-file', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to summarize file.');
                }
                data = await response.json();

            } else if (currentTab === 'text') {
                const textVal = textInput.value.trim();
                if (!textVal || textVal.split(/\s+/).length < 15) {
                    alert('Please paste at least 15-20 words of news article text.');
                    resetToEmpty();
                    return;
                }

                loadingStatus.textContent = 'Analyzing syntax & running Pegasus transformer model...';
                const response = await fetch('/api/summarize-text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: textVal,
                        length: lengthVal,
                        style: styleVal,
                        model_choice: modelVal
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to summarize text.');
                }
                data = await response.json();

            } else if (currentTab === 'url') {
                const urlVal = urlInput.value.trim();
                if (!urlVal) {
                    alert('Please enter a valid news article URL.');
                    resetToEmpty();
                    return;
                }

                loadingStatus.textContent = 'Scraping article text & running NLP pipeline...';
                const response = await fetch('/api/summarize-url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: urlVal,
                        length: lengthVal,
                        style: styleVal,
                        model_choice: modelVal
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to summarize URL content.');
                }
                data = await response.json();
            }

            // Populate Output Results
            loadingState.style.display = 'none';
            summaryWrapper.style.display = 'block';
            summaryActions.style.display = 'flex';

            summaryOutput.textContent = data.summary;

            // Render Key Topics
            topicsList.innerHTML = '';
            if (data.keywords && data.keywords.length > 0) {
                data.keywords.forEach(kw => {
                    const tag = document.createElement('span');
                    tag.className = 'tag';
                    tag.textContent = kw;
                    topicsList.appendChild(tag);
                });
            } else {
                topicsList.innerHTML = '<span class="tag">News</span><span class="tag">Article</span>';
            }

            // Render Metrics
            metricRougeL.textContent = data.metrics.rouge_l.toFixed(2);
            metricBert.textContent = data.metrics.bert_score.toFixed(2);
            metricCompression.textContent = `${data.metrics.compression_ratio}%`;
            metricTime.textContent = `${data.metrics.reading_time_saved_min}m`;

        } catch (error) {
            alert(`Error: ${error.message}`);
            resetToEmpty();
        }
    });

    function resetToEmpty() {
        loadingState.style.display = 'none';
        summaryWrapper.style.display = 'none';
        summaryActions.style.display = 'none';
        emptyState.style.display = 'block';
    }

    // --- Action Handlers ---
    // 1. Text to Speech
    listenBtn.addEventListener('click', () => {
        if (!synth) {
            alert('Text-to-speech is not supported in your browser.');
            return;
        }

        if (speaking) {
            synth.cancel();
            speaking = false;
            listenBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            return;
        }

        const text = summaryOutput.textContent;
        if (!text) return;

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => {
            speaking = false;
            listenBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
        };

        synth.speak(utterance);
        speaking = true;
        listenBtn.innerHTML = '<i class="fa-solid fa-square"></i>';
    });

    // 2. Copy Summary
    copyBtn.addEventListener('click', () => {
        const text = summaryOutput.textContent;
        if (!text) return;

        navigator.clipboard.writeText(text).then(() => {
            copyBtn.innerHTML = '<i class="fa-solid fa-check" style="color: #10b981;"></i>';
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
            }, 2000);
        });
    });

    // 3. Download Summary TXT
    downloadBtn.addEventListener('click', () => {
        const text = summaryOutput.textContent;
        if (!text) return;

        const blob = new Blob([`AI NEWS ARTICLE SUMMARY\nCreated by Prabu Arvind M\n\n${text}`], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Article_Summary_PrabuArvindM.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
});
