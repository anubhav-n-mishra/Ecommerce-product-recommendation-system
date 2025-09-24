#!/usr/bin/env python3
"""
🚀 Advanced E-Commerce Recommendation System - Flask Web Application
Author: Anubhav Mishra
Date: September 2025
GitHub: https://github.com/anubhav-n-mishra/Ecommerce-product-recommendation-system

Flask web interface for the 4 advanced recommendation models:
- EPR1: SVD Matrix Factorization (99.66% accuracy)
- EPR2: Popularity-Based Ranking (Cold-start solution)
- EPR3: User-Based Collaborative Filtering (75.16% accuracy)
- EPR4: Comprehensive Hybrid System (77.35% accuracy)
"""

from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

class RecommendationSystem:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained recommendation models"""
        model_files = ['epr1', 'epr2', 'epr3', 'epr4']
        
        for model_name in model_files:
            try:
                # Try loading .joblib first, then .pkl
                if os.path.exists(f'{model_name}.joblib'):
                    with open(f'{model_name}.joblib', 'rb') as f:
                        self.models[model_name] = joblib.load(f)
                    print(f"✅ Loaded {model_name}.joblib successfully")
                elif os.path.exists(f'{model_name}.pkl'):
                    with open(f'{model_name}.pkl', 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✅ Loaded {model_name}.pkl successfully")
                else:
                    print(f"⚠️ Model file {model_name} not found. Please run the notebooks first.")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {str(e)}")
    
    def get_recommendations(self, model_name, user_id, num_recommendations=5):
        """Get recommendations from specified model"""
        if model_name not in self.models:
            return {"error": f"Model {model_name} not loaded"}
        
        try:
            model = self.models[model_name]
            
            # This would contain the actual prediction logic based on model type
            # For now, returning mock data structure
            recommendations = {
                "model": model_name,
                "user_id": user_id,
                "recommendations": [
                    {"product_id": f"P{i+1000}", "predicted_rating": 4.5 - i*0.1, "confidence": 0.9 - i*0.05}
                    for i in range(num_recommendations)
                ],
                "model_accuracy": self.get_model_accuracy(model_name),
                "timestamp": datetime.now().isoformat()
            }
            
            return recommendations
            
        except Exception as e:
            return {"error": f"Error generating recommendations: {str(e)}"}
    
    def get_model_accuracy(self, model_name):
        """Return model accuracy based on training results"""
        accuracies = {
            'epr1': 99.66,
            'epr2': 'N/A (Popularity-based)',
            'epr3': 75.16,
            'epr4': 77.35
        }
        return accuracies.get(model_name, 'Unknown')

# Initialize recommendation system
rec_system = RecommendationSystem()

@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html', models=list(rec_system.models.keys()))

@app.route('/api/models/status')
def model_status():
    """Get status of all loaded models"""
    status = {}
    for model_name in ['epr1', 'epr2', 'epr3', 'epr4']:
        status[model_name] = {
            'loaded': model_name in rec_system.models,
            'accuracy': rec_system.get_model_accuracy(model_name),
            'description': get_model_description(model_name)
        }
    return jsonify(status)

@app.route('/api/recommend/<model_name>', methods=['POST'])
def get_recommendations(model_name):
    """API endpoint for getting recommendations"""
    data = request.get_json()
    user_id = data.get('user_id', 1)
    num_recommendations = data.get('num_recommendations', 5)
    
    recommendations = rec_system.get_recommendations(model_name, user_id, num_recommendations)
    return jsonify(recommendations)

@app.route('/api/models/performance')
def model_performance():
    """Get performance metrics for all models"""
    performance = {
        'epr1': {
            'name': 'SVD Matrix Factorization',
            'accuracy': 99.66,
            'rmse': 0.045,
            'features': 'High precision, Fast processing',
            'best_for': 'Users with sufficient rating history'
        },
        'epr2': {
            'name': 'Popularity-Based Ranking',
            'accuracy': 'N/A',
            'rmse': 'N/A',
            'features': 'Cold-start solution, Trending products',
            'best_for': 'New users, Popular items discovery'
        },
        'epr3': {
            'name': 'User-Based Collaborative Filtering',
            'accuracy': 75.16,
            'rmse': 1.247,
            'features': 'Social recommendations, Interpretable',
            'best_for': 'Users with similar preferences'
        },
        'epr4': {
            'name': 'Comprehensive Hybrid System',
            'accuracy': 77.35,
            'rmse': 0.906,
            'features': 'Mean-centered SVD, Best overall',
            'best_for': 'Production deployment, All users'
        }
    }
    return jsonify(performance)

def get_model_description(model_name):
    """Get model description"""
    descriptions = {
        'epr1': 'SVD Matrix Factorization - High precision collaborative filtering',
        'epr2': 'Popularity-Based Ranking - Cold-start problem solution',
        'epr3': 'User-Based Collaborative Filtering - Social recommendation approach',
        'epr4': 'Comprehensive Hybrid System - State-of-the-art performance'
    }
    return descriptions.get(model_name, 'Unknown model')

if __name__ == '__main__':
    print("🚀 Starting Advanced E-Commerce Recommendation System")
    print("📊 Models Available:", list(rec_system.models.keys()))
    print("🌐 Web Interface: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000/api/models/status")
    print("✨ Created by Anubhav Mishra - September 2025")
    
    app.run(debug=True, host='0.0.0.0', port=5000)