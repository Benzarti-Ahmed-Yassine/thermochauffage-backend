import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

# Verify TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Génération du dataset (200,000 échantillons)
def generate_dataset(n_samples=200000):
    puissance = 1026  # Fixed power (380V * 2.7A)
    masse = np.random.uniform(0.1, 2.0, n_samples)  # Mass between 0.1 and 2.0 kg
    surface = np.random.uniform(10, 100, n_samples)  # Surface between 10 and 100 cm²
    volume = np.random.uniform(20, 200, n_samples)  # Volume between 20 and 200 cm³
    temperature = (puissance / (masse * 500)) * (surface / volume) * 100 + np.random.normal(0, 10, n_samples)
    X = np.column_stack((np.full(n_samples, puissance), masse, surface, volume))
    y = temperature.clip(50, 300)  # Clip temperatures between 50°C and 300°C
    return X, y

print("Génération du dataset...")
X, y = generate_dataset(200000)

# 2. Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Séparation des données en entraînement et validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Conversion en dataset TensorFlow pour meilleure performance
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64).prefetch(tf.data.AUTOTUNE)

# 5. Construction du modèle amélioré
model = keras.Sequential([
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(4,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(1)
])

# 6. Compilation avec un optimiseur performant
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',  # Explicitly using default MSE
              metrics=['mae'])

# 7. Callbacks
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 8. Entraînement du modèle
print("Entraînement du modèle...")
history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=50, 
                    callbacks=[lr_scheduler, early_stopping], 
                    verbose=1)

# 9. Affichage des courbes d'entraînement
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Courbe de la perte (MSE)
axes[0].plot(history.history['loss'], label='Perte entraînement')
axes[0].plot(history.history['val_loss'], label='Perte validation')
axes[0].set_title("Perte")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()

# Courbe de l'erreur absolue moyenne (MAE)
axes[1].plot(history.history['mae'], label='MAE entraînement')
axes[1].plot(history.history['val_mae'], label='MAE validation')
axes[1].set_title("Erreur absolue moyenne")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE (°C)")
axes[1].legend()

plt.tight_layout()
plt.show()

# 10. Évaluation finale du modèle
test_loss, test_mae = model.evaluate(val_dataset, verbose=0)
print(f"\n📊 Performance du modèle sur les données de validation :\nMSE = {test_loss:.4f}, MAE = {test_mae:.4f} °C")

# 11. Sauvegarde du modèle with explicit HDF5 format and fallback to SavedModel
output_dir = os.path.dirname(__file__) or "."
try:
    model_path = os.path.join(output_dir, "thermochauffage_model.h5")
    model.save(model_path, save_format='h5', include_optimizer=True)
    print(f"Modèle sauvegardé sous '{model_path}' avec succès.")
except Exception as e:
    print(f"Erreur lors de la sauvegarde en HDF5 : {e}")
    model_path = os.path.join(output_dir, "thermochauffage_model")
    model.save(model_path)  # SavedModel format as fallback
    print(f"Modèle sauvegardé sous '{model_path}' (SavedModel) avec succès.")

# 12. Sauvegarde du scaler pour une utilisation ultérieure
scaler_path = os.path.join(output_dir, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler sauvegardé sous '{scaler_path}' avec succès.")

# 13. Test sur un échantillon
test_data = np.array([[1026, 0.8, 40, 80]])
test_data_scaled = scaler.transform(test_data)  # Normalisation des données de test
pred = model.predict(test_data_scaled, verbose=0)
print(f"Température prédite pour l'échantillon de test : {pred[0][0]:.2f} °C")