// --- PRODUCTION LOGIC (CONNECTS TO BACKEND) ---
let currentTopic = "";
let lastSpeaker = null;
let lastMessage = "";
let currentRound = 1;
let debateTranscript = [];
let debateConcluded = false;

const MAX_DEBATE_ROUNDS = 5;

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
const judgeOverlay = document.getElementById('judgeOverlay');
const judgeTitle = document.getElementById('judgeTitle');
const judgeSubtitle = document.getElementById('judgeSubtitle');
const verdictLabel = document.getElementById('verdictLabel');
const winnerLabel = document.getElementById('winnerLabel');
const confidenceLabel = document.getElementById('confidenceLabel');
const reasoningLabel = document.getElementById('reasoningLabel');
const advocateScoreLabel = document.getElementById('advocateScoreLabel');
const challengerScoreLabel = document.getElementById('challengerScoreLabel');
const advocateFill = document.getElementById('advocateFill');
const challengerFill = document.getElementById('challengerFill');
const judgeMetricTraining = document.getElementById('judgeMetricTraining');
const judgeMetricMargin = document.getElementById('judgeMetricMargin');
const judgeMetricAdvantage = document.getElementById('judgeMetricAdvantage');
const judgeMetricComplexity = document.getElementById('judgeMetricComplexity');
const resetBtn = document.getElementById('resetBtn');

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

function recordTurn(agent, response) {
    debateTranscript.push({ agent, response });
}

function resetJudgeOverlay() {
    judgeOverlay.style.display = 'none';
    judgeTitle.textContent = 'Machine Learning Verdict';
    judgeSubtitle.textContent = 'Waiting for the regression judge to score the final debate';
    verdictLabel.textContent = 'Pending';
    winnerLabel.textContent = 'Waiting';
    confidenceLabel.textContent = '0%';
    reasoningLabel.textContent = 'No final verdict has been calculated yet.';
    advocateScoreLabel.textContent = '--';
    challengerScoreLabel.textContent = '--';
    advocateFill.style.width = '0%';
    challengerFill.style.width = '0%';
    judgeMetricTraining.textContent = '--';
    judgeMetricMargin.textContent = '--';
    judgeMetricAdvantage.textContent = '--';
    judgeMetricComplexity.textContent = '--';
}

function buildReasoning(result) {
    const winner = result.winner;
    const margin = Number(result.margin || 0).toFixed(2);
    const advocateScore = Number(result.advocate_score || 0).toFixed(2);
    const challengerScore = Number(result.challenger_score || 0).toFixed(2);
    const featureSummary = `word count, complexity, sentence structure, and persuasiveness`;

    if (winner === 'Tie') {
        return `The regression judge scored both sides evenly. Agent A earned ${advocateScore} and Agent B earned ${challengerScore}. The tie came from nearly identical feature profiles across ${featureSummary}.`;
    }

    return `${winner} won by ${margin} points. Agent A scored ${advocateScore} and Agent B scored ${challengerScore}. The model favored the stronger combination of ${featureSummary} and the final training metrics confirm the judge was fit on historical debate data.`;
}

function showJudgeResult(result) {
    const advocateScore = Number(result.advocate_score || 0);
    const challengerScore = Number(result.challenger_score || 0);
    const maxScore = Math.max(advocateScore, challengerScore, 10);
    const confidence = Math.max(50, Math.min(99, Math.round(50 + (Number(result.margin || 0) * 5))));
    const advocatePercent = Math.round((advocateScore / maxScore) * 100);
    const challengerPercent = Math.round((challengerScore / maxScore) * 100);

    judgeOverlay.style.display = 'flex';
    judgeTitle.textContent = 'Machine Learning Verdict';
    judgeSubtitle.textContent = 'Scikit-Learn regression judge evaluated the full debate transcript';
    verdictLabel.textContent = result.winner === 'Tie' ? 'Tie' : 'Winner';
    winnerLabel.textContent = result.winner;
    confidenceLabel.textContent = `${confidence}%`;
    reasoningLabel.textContent = buildReasoning(result);
    advocateScoreLabel.textContent = advocateScore.toFixed(2);
    challengerScoreLabel.textContent = challengerScore.toFixed(2);
    advocateFill.style.width = `${advocatePercent}%`;
    challengerFill.style.width = `${challengerPercent}%`;

    if (result.training_metrics) {
        judgeMetricTraining.textContent = `${result.training_metrics.rows ?? '--'} rows trained`;
    } else {
        judgeMetricTraining.textContent = 'Training metrics unavailable';
    }

    judgeMetricMargin.textContent = `${Number(result.margin || 0).toFixed(2)} points`;
    judgeMetricAdvantage.textContent = result.winner;
    judgeMetricComplexity.textContent = `A:${Number(result.advocate_features?.complexity_score || 0).toFixed(2)} / B:${Number(result.challenger_features?.complexity_score || 0).toFixed(2)}`;
}

async function finalizeDebate() {
    if (debateConcluded || debateTranscript.length < 2) {
        return;
    }

    debateConcluded = true;
    setProcessingState(true);
    statusText.textContent = 'Running ML Judge evaluation...';

    try {
        const advocateText = debateTranscript.filter(turn => turn.agent === 'A').map(turn => turn.response).join(' ');
        const challengerText = debateTranscript.filter(turn => turn.agent === 'B').map(turn => turn.response).join(' ');

        const response = await fetch('http://127.0.0.1:5000/api/machine-learning/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                advocate_text: advocateText,
                challenger_text: challengerText
            })
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.error || `Evaluation failed with HTTP ${response.status}`);
        }

        const result = await response.json();
        showJudgeResult(result);
        statusText.textContent = 'ML Judge verdict displayed.';
    } catch (err) {
        console.error('Failed to evaluate debate.', err);
        judgeOverlay.style.display = 'flex';
        judgeTitle.textContent = 'Machine Learning Verdict Unavailable';
        judgeSubtitle.textContent = 'The backend did not return a final judge result';
        verdictLabel.textContent = 'Error';
        winnerLabel.textContent = 'Unavailable';
        confidenceLabel.textContent = '0%';
        reasoningLabel.textContent = err.message;
    } finally {
        setProcessingState(false);
    }
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
    lastSpeaker = null;
    lastMessage = "";
    debateTranscript = [];
    debateConcluded = false;
    displayTopic.textContent = `"${topic}"`;
    feedA.innerHTML = '';
    feedB.innerHTML = '';
    currentRound = 1;
    resetJudgeOverlay();
    
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

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.error || `HTTP ${response.status}`);
        }

        const data = await response.json();

        lastSpeaker = data.agent || "A";
        lastMessage = data.message;

        if (!lastMessage) {
            throw new Error("Backend returned an empty opening response.");
        }

        recordTurn(lastSpeaker, lastMessage);
        
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
    if (debateConcluded) {
        return;
    }

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

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.error || `HTTP ${response.status}`);
        }

        const data = await response.json();

        lastSpeaker = data.agent || nextAgent;
        lastMessage = data.message;

        if (!lastMessage) {
            throw new Error("Backend returned an empty response.");
        }

        recordTurn(lastSpeaker, lastMessage);
        
        appendMessage(lastSpeaker, lastMessage, currentRound);
        
        if (lastSpeaker === 'B') currentRound++; // Increment round after B goes

        if (lastSpeaker === 'B' && currentRound > MAX_DEBATE_ROUNDS) {
            await finalizeDebate();
            return;
        }
        
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

if (resetBtn) {
    resetBtn.addEventListener('click', () => {
        window.location.reload();
    });
}

resetJudgeOverlay();
