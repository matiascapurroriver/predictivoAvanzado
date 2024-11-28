from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from datetime import datetime, timedelta
import os  # Importar os para manejar las variables de entorno

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo Prophet guardado
with open('modelo_prophet_semanal.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Ruta principal para renderizar la interfaz
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para generar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capturar la fecha ingresada por el usuario
        date_input = request.form['week']  # Fecha en formato 'YYYY-MM-DD'
        
        # Crear un DataFrame con la fecha ingresada
        future_date = pd.DataFrame({'ds': [date_input]})
        
        # Realizar la predicción
        forecast = model.predict(future_date)
        predicted_value = forecast[['yhat', 'yhat_lower', 'yhat_upper']].iloc[0]
        
        # Devolver la predicción
        return jsonify({
            'date': date_input,
            'prediction': round(predicted_value['yhat'], 2),
            'lower_bound': round(predicted_value['yhat_lower'], 2),
            'upper_bound': round(predicted_value['yhat_upper'], 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# Ejecutar la aplicación Flask
if __name__ == '__main__':
    # Usar el puerto dinámico asignado por Render
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)