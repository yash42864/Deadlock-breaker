import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic training data
def generate_training_data(num_samples, num_processes, num_resources):
    X = []
    y = []
    for _ in range(num_samples):
        alloc = np.random.randint(0, 5, (num_processes, num_resources))
        max_demand = alloc + np.random.randint(0, 5, (num_processes, num_resources))
        available = np.random.randint(0, 10, num_resources)
        need = max_demand - alloc
        work = available.copy()
        finish = np.zeros(num_processes, dtype=bool)
        safe = True
        for _ in range(num_processes):
            found = False
            for i in range(num_processes):
                if not finish[i] and all(need[i] <= work):
                    work += alloc[i]
                    finish[i] = True
                    found = True
                    break
            if not found:
                safe = False
                break
        X.append(np.concatenate([alloc.flatten(), max_demand.flatten(), available]))
        y.append(1 if safe else 0)
    return np.array(X), np.array(y)

# Train model
def train_model(num_processes=5, num_resources=3):
    X, y = generate_training_data(1000, num_processes, num_resources)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Predict safety
def predict_safety(alloc, max_demand, available):
    try:
        model = train_model(alloc.shape[0], alloc.shape[1])
        need = max_demand - alloc
        input_data = np.concatenate([alloc.flatten(), max_demand.flatten(), available])
        prob_safe = model.predict_proba([input_data])[0][1]
        prediction = model.predict([input_data])[0]
        risky = []
        for i, n in enumerate(need):
            if any(n > available * 1.5):  # Arbitrary threshold for risk
                risky.append(i)
        return {
            "is_safe": bool(prediction),
            "probability_safe": float(prob_safe),
            "risky_processes": risky
        }
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")