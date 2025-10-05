from flask import Flask, render_template, request, jsonify, send_file
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store trained models and data
models = {}
scalers = {}
available_cities = []
city_coordinates = {}
training_data = None

class WeatherPredictor:
    def __init__(self):
        self.models = {
            'T2M': None,
            'RH2M': None,
            'WS2M': None,
            'PRECTOTCORR': None
        }
        self.scalers = {
            'T2M': None,
            'RH2M': None,
            'WS2M': None,
            'PRECTOTCORR': None
        }
        
    def prepare_features(self, dates, data_values):
        """Prepare time-based features from dates"""
        features = []
        for date_str in dates:
            date = datetime.strptime(date_str, '%Y%m%d')
            day_of_year = date.timetuple().tm_yday
            month = date.month
            year = date.year
            
            # Cyclical encoding for seasonal patterns
            day_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_cos = np.cos(2 * np.pi * day_of_year / 365)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            features.append([
                year,
                month,
                day_of_year,
                day_sin,
                day_cos,
                month_sin,
                month_cos
            ])
        
        return np.array(features)
    
    def train_model(self, parameter_data, parameter_name):
        """Train a model for a specific weather parameter"""
        dates = list(parameter_data.keys())
        values = [parameter_data[date] for date in dates]
        
        # Remove missing values
        valid_indices = [i for i, v in enumerate(values) if v != -999.0]
        dates = [dates[i] for i in valid_indices]
        values = [values[i] for i in valid_indices]
        
        if len(dates) < 100:  # Need sufficient data
            return False
        
        X = self.prepare_features(dates, values)
        y = np.array(values)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.models[parameter_name] = model
        self.scalers[parameter_name] = scaler
        
        # Calculate accuracy
        score = model.score(X_test_scaled, y_test)
        print(f"{parameter_name} Model R² Score: {score:.3f}")
        
        return True
    
    def predict(self, target_date, parameter_name):
        """Predict weather parameter for a specific date"""
        if self.models[parameter_name] is None:
            return None
        
        # Prepare features for target date
        date = datetime.strptime(target_date, '%Y%m%d')
        day_of_year = date.timetuple().tm_yday
        month = date.month
        year = date.year
        
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        X = np.array([[year, month, day_of_year, day_sin, day_cos, month_sin, month_cos]])
        X_scaled = self.scalers[parameter_name].transform(X)
        
        prediction = self.models[parameter_name].predict(X_scaled)[0]
        return prediction
    
    def calculate_statistics(self, parameter_data, target_day_of_year):
        """Calculate historical statistics for a specific day of year"""
        dates = list(parameter_data.keys())
        values = [parameter_data[date] for date in dates]
        
        # Filter by day of year
        day_values = []
        for date_str, value in zip(dates, values):
            if value != -999.0:
                date = datetime.strptime(date_str, '%Y%m%d')
                if date.timetuple().tm_yday == target_day_of_year:
                    day_values.append(value)
        
        if not day_values:
            return None
        
        return {
            'mean': np.mean(day_values),
            'median': np.median(day_values),
            'std': np.std(day_values),
            'min': np.min(day_values),
            'max': np.max(day_values),
            'count': len(day_values)
        }

def load_and_train_models(json_file_path):
    """Load JSON data and train models for all cities and parameters"""
    global models, scalers, available_cities, city_coordinates, training_data
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        training_data = data
        available_cities = []
        city_coordinates = {}
        models = {}
        
        print("Training models for all cities...")
        
        for feature in data['features']:
            place = feature['place']
            coords = feature['geometry']['coordinates']
            parameters = feature['properties']['parameter']
            
            # Store city info
            available_cities.append(place)
            city_coordinates[place] = {
                'lon': coords[0],
                'lat': coords[1],
                'elevation': coords[2]
            }
            
            # Train models for this city
            predictor = WeatherPredictor()
            city_models = {}
            
            for param_name, param_data in parameters.items():
                if len(param_data) >= 100:
                    print(f"Training {param_name} model for {place}...")
                    success = predictor.train_model(param_data, param_name)
                    if success:
                        city_models[param_name] = {
                            'model': predictor.models[param_name],
                            'scaler': predictor.scalers[param_name],
                            'data': param_data
                        }
            
            models[place] = city_models
        
        # Save models
        save_models()
        
        print(f"Training complete! {len(available_cities)} cities available.")
        return True
        
    except Exception as e:
        print(f"Error loading and training: {str(e)}")
        return False

def save_models():
    """Save trained models to disk"""
    model_data = {
        'models': models,
        'available_cities': available_cities,
        'city_coordinates': city_coordinates
    }
    
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Models saved successfully!")

def load_models():
    """Load pre-trained models from disk"""
    global models, available_cities, city_coordinates
    
    try:
        if os.path.exists('trained_models.pkl'):
            with open('trained_models.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            models = model_data['models']
            available_cities = model_data['available_cities']
            city_coordinates = model_data['city_coordinates']
            
            print(f"Models loaded successfully! {len(available_cities)} cities available.")
            return True
        else:
            print("No saved models found. Please train models first.")
            return False
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities"""
    return jsonify({
        'cities': available_cities,
        'city_data': city_coordinates
    })

@app.route('/api/predict', methods=['POST'])
def predict_weather():
    """Predict weather for a specific location and date"""
    try:
        data = request.json
        city = data.get('city')
        date_str = data.get('date')  # Format: YYYYMMDD
        
        if city not in models:
            return jsonify({'error': 'City not found'}), 404
        
        # Parse date
        target_date = datetime.strptime(date_str, '%Y%m%d')
        day_of_year = target_date.timetuple().tm_yday
        
        predictions = {}
        statistics = {}
        
        city_models = models[city]
        
        for param_name, param_info in city_models.items():
            # Make prediction
            predictor = WeatherPredictor()
            predictor.models[param_name] = param_info['model']
            predictor.scalers[param_name] = param_info['scaler']
            
            pred_value = predictor.predict(date_str, param_name)
            predictions[param_name] = round(float(pred_value), 2)
            
            # Calculate historical statistics
            stats = predictor.calculate_statistics(param_info['data'], day_of_year)
            if stats:
                statistics[param_name] = {
                    'mean': round(stats['mean'], 2),
                    'median': round(stats['median'], 2),
                    'std': round(stats['std'], 2),
                    'min': round(stats['min'], 2),
                    'max': round(stats['max'], 2),
                    'historical_count': stats['count']
                }
        
        # Calculate probabilities for thresholds
        probabilities = calculate_threshold_probabilities(statistics, predictions)
        
        return jsonify({
            'city': city,
            'coordinates': city_coordinates[city],
            'date': date_str,
            'predictions': predictions,
            'statistics': statistics,
            'probabilities': probabilities,
            'metadata': {
                'units': {
                    'T2M': '°C',
                    'RH2M': '%',
                    'WS2M': 'm/s',
                    'PRECTOTCORR': 'mm/day'
                },
                'longnames': {
                    'T2M': 'Temperature at 2 Meters',
                    'RH2M': 'Relative Humidity at 2 Meters',
                    'WS2M': 'Wind Speed at 2 Meters',
                    'PRECTOTCORR': 'Precipitation Corrected'
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_threshold_probabilities(statistics, predictions):
    """Calculate probability of exceeding certain thresholds"""
    probabilities = {}
    
    for param, stats in statistics.items():
        if param == 'T2M':
            # Temperature thresholds (Celsius)
            thresholds = [30, 35, 40]  # Hot conditions
            threshold_probs = {}
            for threshold in thresholds:
                if stats['std'] > 0:
                    z_score = (threshold - stats['mean']) / stats['std']
                    # Using normal distribution approximation
                    prob = 1 - 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
                    threshold_probs[f'above_{threshold}C'] = round(prob * 100, 1)
            probabilities['T2M'] = threshold_probs
            
        elif param == 'PRECTOTCORR':
            # Precipitation thresholds
            thresholds = [5, 10, 20]  # mm/day
            threshold_probs = {}
            for threshold in thresholds:
                if stats['std'] > 0:
                    z_score = (threshold - stats['mean']) / stats['std']
                    prob = 1 - 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
                    threshold_probs[f'above_{threshold}mm'] = round(prob * 100, 1)
            probabilities['PRECTOTCORR'] = threshold_probs
    
    return probabilities

@app.route('/api/historical', methods=['POST'])
def get_historical_data():
    """Get historical data for visualization"""
    try:
        data = request.json
        city = data.get('city')
        parameter = data.get('parameter')
        
        if city not in models or parameter not in models[city]:
            return jsonify({'error': 'Data not found'}), 404
        
        param_data = models[city][parameter]['data']
        
        # Convert to list format
        historical = []
        for date_str, value in param_data.items():
            if value != -999.0:
                historical.append({
                    'date': date_str,
                    'value': round(float(value), 2)
                })
        
        return jsonify({
            'city': city,
            'parameter': parameter,
            'data': historical[-365:]  # Last year of data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download', methods=['POST'])
def download_data():
    """Download query results as CSV or JSON"""
    try:
        data = request.json
        city = data.get('city')
        date_str = data.get('date')
        format_type = data.get('format', 'json')  # 'json' or 'csv'
        
        # Get predictions
        predictions = {}
        city_models = models[city]
        
        for param_name, param_info in city_models.items():
            predictor = WeatherPredictor()
            predictor.models[param_name] = param_info['model']
            predictor.scalers[param_name] = param_info['scaler']
            
            pred_value = predictor.predict(date_str, param_name)
            predictions[param_name] = round(float(pred_value), 2)
        
        output_data = {
            'city': city,
            'coordinates': city_coordinates[city],
            'date': date_str,
            'predictions': predictions,
            'metadata': {
                'source': 'NASA POWER Data',
                'api_version': 'v2.8.0',
                'model': 'Random Forest Regressor',
                'units': {
                    'T2M': '°C',
                    'RH2M': '%',
                    'WS2M': 'm/s',
                    'PRECTOTCORR': 'mm/day'
                }
            }
        }
        
        if format_type == 'csv':
            # Convert to CSV
            df = pd.DataFrame([{
                'City': city,
                'Date': date_str,
                'Longitude': city_coordinates[city]['lon'],
                'Latitude': city_coordinates[city]['lat'],
                **predictions
            }])
            
            csv_file = f'weather_prediction_{city}_{date_str}.csv'
            df.to_csv(csv_file, index=False)
            return send_file(csv_file, as_attachment=True)
        else:
            # Return JSON
            json_file = f'weather_prediction_{city}_{date_str}.json'
            with open(json_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            return send_file(json_file, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_models_endpoint():
    """Endpoint to trigger model training"""
    try:
        data = request.json
        json_file_path = data.get('json_file_path', 'nasa_weather_data.json')
        
        success = load_and_train_models(json_file_path)
        
        if success:
            return jsonify({
                'message': 'Models trained successfully',
                'cities': available_cities
            })
        else:
            return jsonify({'error': 'Training failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    print("\n" + "="*60)
    print("NASA WEATHER PREDICTION DASHBOARD - STARTING")
    print("="*60 + "\n")
    
    # Check if models exist
    if os.path.exists('trained_models.pkl'):
        print("Found existing trained models. Loading...")
        if load_models():
            print(f"✓ Successfully loaded models for {len(available_cities)} cities\n")
        else:
            print("✗ Failed to load models. Will attempt to train new models.\n")
    
    # If no models loaded, try to train from JSON file
    if not available_cities:
        print("No models loaded. Checking for training data...")
        
        # Check for JSON data file
        json_files = ['nasa_weather_data.json', 'data/nasa_weather_data.json']
        data_file = None
        
        for file_path in json_files:
            if os.path.exists(file_path):
                data_file = file_path
                print(f"✓ Found data file: {file_path}\n")
                break
        
        if data_file:
            print("Starting model training...")
            print("This may take a few minutes depending on data size...\n")
            
            success = load_and_train_models(data_file)
            
            if success:
                print("\n" + "="*60)
                print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"✓ Trained models for {len(available_cities)} cities")
                print("✓ Models saved to 'trained_models.pkl'")
                print("\nAvailable cities:")
                for i, city in enumerate(available_cities, 1):
                    print(f"  {i}. {city}")
                print("\n")
            else:
                print("\n" + "="*60)
                print("MODEL TRAINING FAILED!")
                print("="*60)
                print("Please check:")
                print("  1. JSON file format is correct")
                print("  2. File contains sufficient data (100+ points per city)")
                print("  3. Check error messages above for details\n")
        else:
            print("\n" + "="*60)
            print("NO TRAINING DATA FOUND!")
            print("="*60)
            print("Please provide NASA POWER data:")
            print("  1. Place 'nasa_weather_data.json' in the project root, OR")
            print("  2. Place it in 'data/nasa_weather_data.json', OR")
            print("  3. Run 'python prepare_data.py' to fetch data from NASA API")
            print("\nThe app will start but predictions won't work without trained models.\n")
    
    # Start the Flask application
    print("="*60)
    print("STARTING FLASK SERVER")
    print("="*60)
    print("Dashboard URL: http://localhost:5000")
    print("Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)