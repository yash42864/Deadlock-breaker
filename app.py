from flask import Flask, request, jsonify
import numpy as np
import logging

# Try importing flask_cors, handle missing package
try:
    from flask_cors import CORS
except ImportError:
    print("ERROR: flask-cors is not installed. Install it with: pip install flask-cors")
    exit(1)

try:
    from ai_model import predict_safety
except ImportError:
    print("ERROR: ai_model.py is not found or has issues. Ensure it exists in the same directory.")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Allow all origins for debugging (temporary)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request for /predict")
        return jsonify({}), 200

    logger.info("Received POST request to /predict")
    try:
        data = request.get_json()
        if not data or 'allocation' not in data or 'maxDemand' not in data or 'available' not in data:
            logger.error("Missing required fields")
            return jsonify({"error": "Missing required fields: allocation, maxDemand, available"}), 400
        alloc = np.array(data['allocation'], dtype=int)
        max_demand = np.array(data['maxDemand'], dtype=int)
        available = np.array(data['available'], dtype=int)
        logger.debug(f"Input data: alloc={alloc}, max_demand={max_demand}, available={available}")
        result = predict_safety(alloc, max_demand, available)
        logger.info("Prediction successful")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request for /health")
        return jsonify({}), 200

    logger.info("Received GET request to /health")
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)