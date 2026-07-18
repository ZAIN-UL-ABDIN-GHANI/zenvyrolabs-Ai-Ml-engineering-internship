from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - keeps the app runnable in minimal envs.
    CORS = None

from services.aiService import DebateConductor
from services.mlJudge import DebateRegressionJudge

app = Flask(__name__)
if CORS is not None:
    CORS(app)


ALLOWED_ORIGINS = {
    "http://localhost:8080",
    "http://127.0.0.1:8080",
}


@app.before_request
def handle_api_preflight():
    """Return a lightweight response for API preflight requests."""
    if request.method == "OPTIONS" and request.path.startswith("/api/"):
        return ("", 204)


@app.after_request
def add_cors_headers(response):
    """Attach CORS headers to API responses so browser fetches can succeed."""
    origin = request.headers.get("Origin")

    if request.path.startswith("/api/"):
        if origin in ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        else:
            response.headers["Access-Control-Allow-Origin"] = "*"

        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Max-Age"] = "86400"

    return response

# Initialize Services
conductor = DebateConductor()
ml_judge = DebateRegressionJudge()


@app.route('/api/debate/start', methods=['POST'])
def start_debate():
    """Initialize a new debate."""

    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON body is required"}), 400

    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        opening = conductor.generate_agent_a_response(topic)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "status": "active",
        "topic": topic,
        "agent": opening["agent"],
        "message": opening["response"],
        "round": 1
    }), 200


@app.route('/api/debate/next-turn', methods=['POST'])
def next_turn():
    """Generate the next debate turn."""

    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON body is required"}), 400

    topic = data.get("topic", "").strip()
    last_speaker = data.get("last_speaker", "A")
    last_message = data.get("last_message", "").strip()

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        if last_speaker == "A":
            turn = conductor.generate_agent_b_response(topic, opponent_last_message=last_message)
        else:
            turn = conductor.generate_agent_a_response(topic, opponent_last_message=last_message)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "status": "active",
        "topic": topic,
        "agent": turn["agent"],
        "message": turn["response"],
        "round": conductor.get_round_number(topic),
    }), 200


@app.route('/api/machine-learning/train', methods=['POST'])
def trigger_training():
    """Train the ML debate judge."""

    try:
        metrics = ml_judge.train_model("data/historical_debates.csv")

        return jsonify({
            "status": "Training Completed",
            "metrics": metrics
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/machine-learning/evaluate', methods=['POST'])
def evaluate_debate():
    """Evaluate debate arguments using the trained ML model."""

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "JSON body is required"
        }), 400

    advocate_text = data.get("advocate_text", "").strip()
    challenger_text = data.get("challenger_text", "").strip()

    if not advocate_text or not challenger_text:
        return jsonify({
            "error": "Both advocate_text and challenger_text are required."
        }), 400

    try:
        result = ml_judge.evaluate_debate(advocate_text, challenger_text)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print(" AI Debate Chamber Backend")
    print("=" * 60)
    print("Server : http://127.0.0.1:5000")
    print("ML Judge : Ready")
    print("LLM : Requires Ollama on port 11434")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )