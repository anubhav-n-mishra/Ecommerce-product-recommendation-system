#!/usr/bin/env python3
"""
🚀 Unified Recommendation System Interface
Author: Anubhav Mishra
Date: September 2025
GitHub: https://github.com/anubhav-n-mishra/Ecommerce-product-recommendation-system

Unified interface for all 4 recommendation models with ensemble capabilities.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class UnifiedRecommendationSystem:
    """
    Unified interface for all recommendation models with ensemble capabilities
    """
    
    def __init__(self):
        self.models = {}
        self.model_performances = {
            'epr1': {'accuracy': 99.66, 'rmse': 0.045, 'type': 'svd'},
            'epr2': {'accuracy': None, 'rmse': None, 'type': 'popularity'},
            'epr3': {'accuracy': 75.16, 'rmse': 1.247, 'type': 'user_cf'},
            'epr4': {'accuracy': 77.35, 'rmse': 0.906, 'type': 'hybrid_svd'}
        }
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available trained models"""
        for model_name in ['epr1', 'epr2', 'epr3', 'epr4']:
            try:
                # Try joblib first, then pickle
                if self.load_model_file(model_name, 'joblib'):
                    print(f"✅ Loaded {model_name}.joblib")
                elif self.load_model_file(model_name, 'pkl'):
                    print(f"✅ Loaded {model_name}.pkl")
                else:
                    print(f"⚠️ {model_name} model not found")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
    
    def load_model_file(self, model_name, file_type):
        """Load individual model file"""
        filename = f"{model_name}.{file_type}"
        try:
            if file_type == 'joblib':
                with open(filename, 'rb') as f:
                    self.models[model_name] = joblib.load(f)
            else:  # pkl
                with open(filename, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def get_single_model_recommendations(self, model_name, user_id, num_recommendations=5):
        """Get recommendations from a single model"""
        if model_name not in self.models:
            return {"error": f"Model {model_name} not available"}
        
        model = self.models[model_name]
        model_info = self.model_performances[model_name]
        
        # Mock recommendation generation (would be replaced with actual model logic)
        recommendations = []
        for i in range(num_recommendations):
            rec = {
                'product_id': f"PROD_{1000 + i}",
                'predicted_rating': round(4.8 - i * 0.15, 2),
                'confidence': round(0.95 - i * 0.05, 2),
                'rank': i + 1
            }
            recommendations.append(rec)
        
        return {
            'model': model_name,
            'model_type': model_info['type'],
            'accuracy': model_info['accuracy'],
            'rmse': model_info['rmse'],
            'user_id': user_id,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_ensemble_recommendations(self, user_id, num_recommendations=5, weights=None):
        """
        Get ensemble recommendations combining multiple models
        """
        if weights is None:
            # Default weights based on model performance
            weights = {
                'epr1': 0.35,  # Highest accuracy
                'epr2': 0.15,  # Popularity baseline
                'epr3': 0.25,  # User-based CF
                'epr4': 0.25   # Hybrid approach
            }
        
        all_recommendations = {}
        total_weight = 0
        
        # Get recommendations from each available model
        for model_name, weight in weights.items():
            if model_name in self.models:
                recs = self.get_single_model_recommendations(model_name, user_id, num_recommendations * 2)
                if 'error' not in recs:
                    all_recommendations[model_name] = {
                        'recommendations': recs['recommendations'],
                        'weight': weight,
                        'accuracy': recs['accuracy']
                    }
                    total_weight += weight
        
        if not all_recommendations:
            return {"error": "No models available for ensemble"}
        
        # Combine recommendations using weighted voting
        product_scores = {}
        
        for model_name, model_data in all_recommendations.items():
            weight = model_data['weight'] / total_weight  # Normalize weights
            
            for rec in model_data['recommendations']:
                product_id = rec['product_id']
                score = rec['predicted_rating'] * rec['confidence'] * weight
                
                if product_id not in product_scores:
                    product_scores[product_id] = {
                        'total_score': 0,
                        'vote_count': 0,
                        'models_voted': []
                    }
                
                product_scores[product_id]['total_score'] += score
                product_scores[product_id]['vote_count'] += 1
                product_scores[product_id]['models_voted'].append(model_name)
        
        # Rank products by ensemble score
        ensemble_recommendations = []
        for product_id, score_data in product_scores.items():
            ensemble_score = score_data['total_score'] / len(all_recommendations)
            ensemble_recommendations.append({
                'product_id': product_id,
                'ensemble_score': round(ensemble_score, 3),
                'vote_count': score_data['vote_count'],
                'models_voted': score_data['models_voted'],
                'consensus_strength': score_data['vote_count'] / len(all_recommendations)
            })
        
        # Sort by ensemble score and return top N
        ensemble_recommendations.sort(key=lambda x: x['ensemble_score'], reverse=True)
        top_recommendations = ensemble_recommendations[:num_recommendations]
        
        # Add ranking
        for i, rec in enumerate(top_recommendations):
            rec['rank'] = i + 1
        
        return {
            'method': 'ensemble',
            'models_used': list(all_recommendations.keys()),
            'weights_used': {k: v['weight']/total_weight for k, v in all_recommendations.items()},
            'user_id': user_id,
            'recommendations': top_recommendations,
            'total_candidates': len(product_scores),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_model_comparison(self, user_id, num_recommendations=5):
        """Compare recommendations across all available models"""
        comparison = {
            'user_id': user_id,
            'models': {},
            'summary': {
                'total_models': len(self.models),
                'best_accuracy': 0,
                'best_model': None
            }
        }
        
        for model_name in self.models.keys():
            recs = self.get_single_model_recommendations(model_name, user_id, num_recommendations)
            if 'error' not in recs:
                comparison['models'][model_name] = recs
                
                # Update summary
                if recs.get('accuracy') and recs['accuracy'] > comparison['summary']['best_accuracy']:
                    comparison['summary']['best_accuracy'] = recs['accuracy']
                    comparison['summary']['best_model'] = model_name
        
        return comparison
    
    def get_system_status(self):
        """Get overall system status and capabilities"""
        return {
            'system': 'Unified Recommendation System',
            'author': 'Anubhav Mishra',
            'version': '1.0',
            'date': 'September 2025',
            'models_loaded': list(self.models.keys()),
            'total_models': len(self.models),
            'capabilities': [
                'Single model recommendations',
                'Ensemble recommendations',
                'Model comparison',
                'Performance benchmarking'
            ],
            'performance_summary': self.model_performances
        }

def main():
    """Demo of the unified recommendation system"""
    print("🚀 Unified Recommendation System - Demo")
    print("=" * 50)
    
    # Initialize system
    rec_system = UnifiedRecommendationSystem()
    
    # Show system status
    status = rec_system.get_system_status()
    print(f"📊 Models loaded: {status['models_loaded']}")
    print(f"⚡ Total capabilities: {len(status['capabilities'])}")
    
    # Demo user
    user_id = 42
    
    if rec_system.models:
        print(f"\n🎯 Generating recommendations for User {user_id}...")
        
        # Single model recommendation
        if 'epr4' in rec_system.models:
            epr4_recs = rec_system.get_single_model_recommendations('epr4', user_id)
            print(f"📈 EPR4 Recommendations: {len(epr4_recs['recommendations'])} items")
        
        # Ensemble recommendation
        ensemble_recs = rec_system.get_ensemble_recommendations(user_id)
        if 'error' not in ensemble_recs:
            print(f"🌟 Ensemble Recommendations: {len(ensemble_recs['recommendations'])} items")
            print(f"🤖 Models used: {ensemble_recs['models_used']}")
    else:
        print("⚠️ No models loaded. Please run the training notebooks first.")
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    main()