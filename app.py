from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import os

app = Flask(__name__)

# Adjust paths for Render
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "thermochauffage_model.h5")
scaler_path = os.path.join(base_path, "scaler.pkl")

# Load model and scaler
try:
    model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

@app.route('/calculate', methods=['POST'])
def calculate():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    data = request.json.get('molds', [])
    total_surface = 4096  # 40x20 cm
    puissance = 1026  # 380V * 2.7A
    grid_temperatures = np.zeros((5, 5))
    results = ""
    mold_positions = []

    for idx, mold in enumerate(data):
        try:
            # Extract and validate mold properties
            masse_moule = float(mold["masse_moule"])
            volume_moule = float(mold["volume_moule"])
            surface_moule = float(mold["surface_moule"])
            materiau_moule = mold["materiau_moule"]
            masse_contre_moule = float(mold["masse_contre_moule"])
            volume_contre_moule = float(mold["volume_contre_moule"])
            surface_contre_moule = float(mold["surface_contre_moule"])
            materiau_contre_moule = mold["materiau_contre_moule"]
            materiau_moulage = mold["materiau_moulage"]

            # Validation
            if not (0.1 <= masse_moule <= 2.0):
                raise ValueError(f"Moule {idx+1} : La masse du moule doit être entre 0.1 et 2.0 kg.")
            if not (20 <= volume_moule <= 200):
                raise ValueError(f"Moule {idx+1} : Le volume du moule doit être entre 20 et 200 cm³.")
            if not (10 <= surface_moule <= 100):
                raise ValueError(f"Moule {idx+1} : La surface du moule doit être entre 10 et 100 cm².")
            if not (0.1 <= masse_contre_moule <= 2.0):
                raise ValueError(f"Moule {idx+1} : La masse du contre-moule doit être entre 0.1 et 2.0 kg.")
            if not (20 <= volume_contre_moule <= 200):
                raise ValueError(f"Moule {idx+1} : Le volume du contre-moule doit être entre 20 et 200 cm³.")
            if not (10 <= surface_contre_moule <= 100):
                raise ValueError(f"Moule {idx+1} : La surface du contre-moule doit être entre 10 et 100 cm².")

            # Calculate number of pieces
            nb_pieces = int(total_surface / surface_moule)

            # Assign position on the grid
            x_pos = (idx % 2) * 20
            y_pos = (idx // 2) * 10
            mold_positions.append((x_pos, y_pos))

            # Predict temperature
            input_data = np.array([[puissance, masse_moule, surface_moule, volume_moule]])
            input_data_scaled = scaler.transform(input_data)
            temp_predite = model.predict(input_data_scaled, verbose=0)[0][0]

            # Adjust temperature based on material
            thermal_factor = 1.0
            if materiau_moulage.lower() in ["plastique", "plastic"]:
                thermal_factor = 0.9
            elif materiau_moulage.lower() in ["métal", "metal"]:
                thermal_factor = 1.1
            temp_predite *= thermal_factor

            # Update grid
            x_idx = x_pos // 8
            y_idx = y_pos // 4
            grid_temperatures[x_idx, y_idx] = temp_predite

            # Build results string
            results += (
                f"Moule {idx+1} :\n"
                f"  Position : X={x_pos} cm, Y={y_pos} cm\n"
                f"  Nombre de pièces possibles : {nb_pieces}\n"
                f"  Matériau du moule : {materiau_moule}\    f"  Matériau du contre-moule : {materiau_contre_moule}\n"
                f"  Matériau de moulage : {materiau_moulage}\n"
                f"  Température prédite : {temp_predite:.2f} °C\n\n"
            )

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Moule {idx+1} : Une erreur s'est produite : {e}"}), 500

    # Finalize grid temperatures
    for i in range(5):
        for j in range(5):
            if grid_temperatures[i, j] == 0:
                grid_temperatures[i, j] = 50
            else:
                grid_temperatures[i, j] += np.random.normal(0, 5)
    grid_temperatures = np.clip(grid_temperatures, 50, 300)

    return jsonify({
        "results": results,
        "gridTemperatures": grid_temperatures.tolist()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
