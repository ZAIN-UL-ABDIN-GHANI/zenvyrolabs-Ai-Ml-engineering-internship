from flask import Flask, request, jsonify
from flask_cors import CORS
from services.aiService import DebateConductor
from services.mlJudge import DebateRegressionJudge

app = Flask(__name__)
CORS(app) # Allow Cross-Origin Requests from the UI

# System Modules
conductor = DebateConductor()
ml_judge = DebateRegressionJudge()

@app.route('/api/debate/start', methods=['POST'])
def start_debate():
    """Initializes the debate. Interns will link this to the UI later."""
    data = request.json
    topic = data.get('topic')
    return jsonify({"status": "active", "topic": topic})

@app.route('/api/debate/next-turn', methods=['POST'])
def next_turn():
    """Triggers the next LLM agent to generate via Local Ollama."""
    data = request.json
    # INTERN TASK: Extract variables, pass to aiService, and track history.
    return jsonify({"message": "Turn execution pending intern implementation."})

@app.route('/api/machine-learning/train', methods=['POST'])
def trigger_training():
    """Triggers the SciKit-Learn Regression Model Training Loop."""
    try:
        # INTERN TASK: Wire this to the mlJudge.train_model() function
        # Ensure it returns the MSE (Mean Squared Error) and Accuracy metrics!
        accuracy_metrics = ml_judge.train_model("historical_debates.csv")
        return jsonify({"status": "Training Completed", "metrics": accuracy_metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/machine-learning/evaluate', methods=['POST'])
def evaluate_debate():
    """Uses the Trained ML model to score the AI's arguments."""
    data = request.json
    advocate_text = data.get('advocate_text')
    challenger_text = data.get('challenger_text')
    
    # INTERN TASK: Use ml_judge.predict_score() to find out who won!
    return jsonify({"winner": "Pending", "advocate_score": 0.0, "challenger_score": 0.0})

if __name__ == '__main__':
    print("🚀 AI Server running on http://127.0.0.1:5000")
    print("Ensure Ollama is running locally on port 11434!")
    app.run(debug=True, port=5000)
