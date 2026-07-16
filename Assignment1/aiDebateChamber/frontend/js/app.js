// --- PRODUCTION LOGIC (CONNECTS TO BACKEND) ---
let currentTopic = "";
let lastSpeaker = null;
let lastMessage = "";
let currentRound = 1;

const topicInput = document.getElementById('topicInput');
const startBtn = document.getElementById('startBtn');
const nextBtn = document.getElementById('nextTurnBtn');
const feedA = document.getElementById('feedA');
const feedB = document.getElementById('feedB');

// New UI Selectors
const displayTopic = document.getElementById('displayTopic');
const networkStatus = document.getElementById('networkStatus');
const statusText = document.getElementById('statusText');
const dotA = document.getElementById('dotA');
const dotB = document.getElementById('dotB');
const globalDot = document.getElementById('globalDot');

const API_URL = "http://127.0.0.1:5000/api/debate";

function toggleTyping(agent, show) {
    const feed = agent === 'A' ? feedA : feedB;
    const existing = document.getElementById('typingIndicator');
    
    if (show) {
        if (!existing) {
            const typingHTML = `<div class="typing-wrapper" id="typingIndicator"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>`;
            feed.insertAdjacentHTML('beforeend', typingHTML);
            feed.scrollTop = feed.scrollHeight;
        }
    } else {
        if (existing) existing.remove();
    }
}

function setActive(agent) {
    if (agent === 'A') {
        dotA.classList.add('active');
        dotB.classList.remove('active');
        statusText.textContent = `Awaiting API Response for Agent A... (Round ${currentRound})`;
    } else {
        dotB.classList.add('active');
        dotA.classList.remove('active');
        statusText.textContent = `Awaiting API Response for Agent B... (Round ${currentRound})`;
    }
}

function appendMessage(agent, text, round) {
    toggleTyping(agent, false); // Clear typing before posting
    
    const div = document.createElement('div');
    div.classList.add('msg', agent === 'A' ? 'msg-adv' : 'msg-chal');
    div.innerHTML = `
        <div class="round-tag">Round ${round} · ${agent === 'A' ? 'Advocate' : 'Challenger'}</div>
        <div class="msg-text">${text}</div>
    `;

    const feed = agent === 'A' ? feedA : feedB;
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
}

function setProcessingState(isLoading) {
    startBtn.disabled = isLoading;
    nextBtn.disabled = isLoading;
    if (isLoading) {
        topicInput.disabled = true;
        networkStatus.textContent = "COMPUTING";
        networkStatus.style.color = "#eab308";
        networkStatus.style.borderColor = "rgba(234,179,8,0.3)";
    } else {
        networkStatus.textContent = "IDLE";
        networkStatus.style.color = "#8b5cf6";
        networkStatus.style.borderColor = "rgba(139,92,246,0.3)";
    }
}

startBtn.addEventListener('click', async () => {
    const topic = topicInput.value.trim();
    if (!topic) return alert("Please enter a custom topic first.");
    
    currentTopic = topic;
    displayTopic.textContent = `"${topic}"`;
    feedA.innerHTML = '';
    feedB.innerHTML = '';
    currentRound = 1;
    
    setProcessingState(true);
    globalDot.classList.add('live');
    
    setActive('A');
    toggleTyping('A', true);

    try {
        const response = await fetch(`${API_URL}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic: currentTopic })
        });
        
        const data = await response.json();
        
        // Mocking the Backend logic returning 'Agent A' since the Python code is currently empty!
        lastSpeaker = "A"; 
        lastMessage = data.message || "I defend the topic! (Python Backend generated this)";
        
        appendMessage(lastSpeaker, lastMessage, currentRound);
        
        setProcessingState(false);
        dotA.classList.remove('active');
        statusText.textContent = "API Idle. Waiting for User Execution...";
        
    } catch (err) {
        console.error("Backend connection failed.", err);
        alert("Failed to connect to the AI Backend Python Server on port 5000!");
        setProcessingState(false);
        globalDot.classList.remove('live');
        toggleTyping('A', false);
    }
});

nextBtn.addEventListener('click', async () => {
    setProcessingState(true);
    
    // Switch to whichever agent DID NOT speak last
    const nextAgent = lastSpeaker === 'A' ? 'B' : 'A';
    setActive(nextAgent);
    toggleTyping(nextAgent, true);

    try {
        const response = await fetch(`${API_URL}/next-turn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                topic: currentTopic,
                last_speaker: lastSpeaker,
                last_message: lastMessage
            })
        });
        
        const data = await response.json();
        
        // Since backend is empty, fallback to nextAgent
        lastSpeaker = data.agent || nextAgent;
        lastMessage = data.message || "I attack the topic! (Python Backend generated this)";
        
        appendMessage(lastSpeaker, lastMessage, currentRound);
        
        if (lastSpeaker === 'B') currentRound++; // Increment round after B goes
        
        setProcessingState(false);
        dotA.classList.remove('active');
        dotB.classList.remove('active');
        statusText.textContent = "API Idle. Waiting for User Execution...";
        
    } catch (err) {
        console.error(err);
        alert("Agent failed to respond.");
        setProcessingState(false);
        toggleTyping(nextAgent, false);
    }
});
