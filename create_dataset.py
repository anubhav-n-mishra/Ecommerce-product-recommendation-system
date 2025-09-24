#!/usr/bin/env python3import pandas as pd

"""import numpy as np

🔧 Dataset Creation and Preprocessing Utilitiesimport random

Author: Anubhav Mishra

Date: September 2025# Set random seed for reproducibility

GitHub: https://github.com/anubhav-n-mishra/Ecommerce-product-recommendation-systemnp.random.seed(42)

random.seed(42)

Utilities for creating and preprocessing the Amazon Electronics dataset

for the recommendation system models.# Create a synthetic dataset with the same structure as the Amazon electronics rating dataset

"""num_users = 2000

num_products = 5000

import pandas as pdnum_ratings = 50000

import numpy as np

import matplotlib.pyplot as pltprint("Generating synthetic Amazon electronics rating dataset...")

import seaborn as sns

import requests# Generate user IDs (as strings similar to Amazon format)

import osuser_ids = [f"A{random.randint(10000000000000, 99999999999999)}" for _ in range(num_users)]

from sklearn.model_selection import train_test_split

import warnings# Generate product IDs (mix of ASIN format)

warnings.filterwarnings('ignore')product_ids = []

for _ in range(num_products):

class DatasetCreator:    if random.choice([True, False]):

    """        # ASIN format (B followed by 9 characters)

    Advanced dataset creation and preprocessing for recommendation systems        product_ids.append("B" + "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=9)))

    """    else:

            # ISBN format (10 digits)

    def __init__(self):        product_ids.append("".join(random.choices("0123456789", k=10)))

        self.raw_data = None

        self.processed_data = None# Generate ratings data

        self.user_item_matrix = Nonedata = []

        self.dataset_stats = {}for _ in range(num_ratings):

        user_id = random.choice(user_ids)

    def download_dataset(self, save_path='ratings_Electronics.csv'):    product_id = random.choice(product_ids)

        """    rating = random.choices([1, 2, 3, 4, 5], weights=[5, 8, 15, 30, 42])[0]  # Weighted towards higher ratings

        Download the Amazon Electronics dataset if not present    timestamp = random.randint(946684800, 1609459200)  # Between 2000 and 2020

        Note: This is a placeholder - actual download would require Kaggle API    

        """    data.append([user_id, product_id, rating, timestamp])

        if os.path.exists(save_path):

            print(f"✅ Dataset already exists: {save_path}")# Create DataFrame

            return Truedf = pd.DataFrame(data, columns=['user_id', 'prod_id', 'rating', 'timestamp'])

        

        print("📥 Dataset download functionality")# Remove duplicates (same user rating same product multiple times)

        print("Please download the dataset manually from:")df = df.drop_duplicates(subset=['user_id', 'prod_id'], keep='first')

        print("https://www.kaggle.com/datasets/vibivij/amazon-electronics-rating-datasetrecommendation")

        print(f"Save it as: {save_path}")print(f"Generated dataset with {len(df)} ratings")

        return Falseprint(f"Users: {df['user_id'].nunique()}")

    print(f"Products: {df['prod_id'].nunique()}")

    def load_dataset(self, file_path='ratings_Electronics.csv'):print(f"Rating distribution:")

        """Load and perform initial exploration of the dataset"""print(df['rating'].value_counts().sort_index())

        try:

            print("📊 Loading Amazon Electronics Dataset...")# Save to CSV without headers (as expected by the notebooks)

            df.to_csv('ratings_Electronics.csv', index=False, header=False)

            # Load dataset (assuming it has no headers based on description)print("Dataset saved as 'ratings_Electronics.csv'")
            self.raw_data = pd.read_csv(file_path, names=['user_id', 'product_id', 'rating', 'timestamp'])
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📏 Shape: {self.raw_data.shape}")
            
            # Basic statistics
            self.dataset_stats = {
                'total_interactions': len(self.raw_data),
                'unique_users': self.raw_data['user_id'].nunique(),
                'unique_products': self.raw_data['product_id'].nunique(),
                'rating_range': [self.raw_data['rating'].min(), self.raw_data['rating'].max()],
                'sparsity': 1 - (len(self.raw_data) / (self.raw_data['user_id'].nunique() * self.raw_data['product_id'].nunique()))
            }
            
            self.print_dataset_stats()
            return True
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {file_path}")
            print("Please download the dataset first using download_dataset()")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def print_dataset_stats(self):
        """Print comprehensive dataset statistics"""
        if not self.dataset_stats:
            print("⚠️ No dataset statistics available")
            return
        
        print("\n" + "="*60)
        print("📊 AMAZON ELECTRONICS DATASET STATISTICS")
        print("="*60)
        print(f"📈 Total Interactions: {self.dataset_stats['total_interactions']:,}")
        print(f"👥 Unique Users: {self.dataset_stats['unique_users']:,}")
        print(f"🛍️ Unique Products: {self.dataset_stats['unique_products']:,}")
        print(f"⭐ Rating Range: {self.dataset_stats['rating_range'][0]} - {self.dataset_stats['rating_range'][1]}")
        print(f"🕳️ Matrix Sparsity: {self.dataset_stats['sparsity']:.4f} ({self.dataset_stats['sparsity']*100:.2f}%)")
        
        # Additional statistics
        if self.raw_data is not None:
            avg_ratings_per_user = len(self.raw_data) / self.dataset_stats['unique_users']
            avg_ratings_per_product = len(self.raw_data) / self.dataset_stats['unique_products']
            
            print(f"📊 Avg Ratings per User: {avg_ratings_per_user:.2f}")
            print(f"📊 Avg Ratings per Product: {avg_ratings_per_product:.2f}")
            
            # Rating distribution
            rating_dist = self.raw_data['rating'].value_counts().sort_index()
            print(f"\n⭐ Rating Distribution:")
            for rating, count in rating_dist.items():
                percentage = (count / len(self.raw_data)) * 100
                print(f"   {rating} stars: {count:,} ({percentage:.1f}%)")
        
        print("="*60)
    
    def preprocess_data(self, min_user_ratings=50, min_product_ratings=10):
        """
        Advanced preprocessing for recommendation system
        """
        if self.raw_data is None:
            print("❌ No raw data available. Load dataset first.")
            return False
        
        print(f"🔧 Preprocessing dataset...")
        print(f"📏 Original shape: {self.raw_data.shape}")
        
        # Start with raw data
        processed = self.raw_data.copy()
        
        # Filter users with minimum ratings
        user_counts = processed['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        processed = processed[processed['user_id'].isin(valid_users)]
        print(f"👥 Users after filtering (≥{min_user_ratings} ratings): {processed['user_id'].nunique()}")
        
        # Filter products with minimum ratings
        product_counts = processed['product_id'].value_counts()
        valid_products = product_counts[product_counts >= min_product_ratings].index
        processed = processed[processed['product_id'].isin(valid_products)]
        print(f"🛍️ Products after filtering (≥{min_product_ratings} ratings): {processed['product_id'].nunique()}")
        
        # Convert to integer IDs for memory efficiency
        user_id_map = {user: idx for idx, user in enumerate(processed['user_id'].unique())}
        product_id_map = {product: idx for idx, product in enumerate(processed['product_id'].unique())}
        
        processed['user_idx'] = processed['user_id'].map(user_id_map)
        processed['product_idx'] = processed['product_id'].map(product_id_map)
        
        # Store processed data
        self.processed_data = processed
        
        # Update statistics
        self.dataset_stats.update({
            'processed_interactions': len(processed),
            'processed_users': processed['user_id'].nunique(),
            'processed_products': processed['product_id'].nunique(),
            'processed_sparsity': 1 - (len(processed) / (processed['user_id'].nunique() * processed['product_id'].nunique())),
            'user_id_map': user_id_map,
            'product_id_map': product_id_map
        })
        
        print(f"✅ Processed shape: {processed.shape}")
        print(f"🎯 Final sparsity: {self.dataset_stats['processed_sparsity']:.4f} ({self.dataset_stats['processed_sparsity']*100:.2f}%)")
        
        return True
    
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        if self.processed_data is None:
            print("❌ No processed data available. Run preprocess_data() first.")
            return False
        
        print("🔧 Creating user-item matrix...")
        
        # Create pivot table
        self.user_item_matrix = self.processed_data.pivot_table(
            index='user_idx',
            columns='product_idx', 
            values='rating',
            fill_value=0
        )
        
        print(f"✅ User-item matrix created: {self.user_item_matrix.shape}")
        print(f"💾 Memory usage: {self.user_item_matrix.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return True
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """Create train/test split for evaluation"""
        if self.processed_data is None:
            print("❌ No processed data available.")
            return None, None
        
        print(f"📊 Creating train/test split ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
        
        train_data, test_data = train_test_split(
            self.processed_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.processed_data['rating']  # Maintain rating distribution
        )
        
        print(f"🚂 Train set: {len(train_data):,} interactions")
        print(f"🧪 Test set: {len(test_data):,} interactions")
        
        return train_data, test_data
    
    def visualize_dataset(self, save_plots=True):
        """Create comprehensive dataset visualizations"""
        if self.raw_data is None:
            print("❌ No data available for visualization.")
            return
        
        print("📊 Creating dataset visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Amazon Electronics Dataset Analysis - by Anubhav Mishra', fontsize=16, fontweight='bold')
        
        # Rating distribution
        axes[0,0].hist(self.raw_data['rating'], bins=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Rating Distribution')
        axes[0,0].set_xlabel('Rating')
        axes[0,0].set_ylabel('Frequency')
        
        # User activity distribution
        user_activity = self.raw_data['user_id'].value_counts()
        axes[0,1].hist(user_activity, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('User Activity Distribution')
        axes[0,1].set_xlabel('Number of Ratings per User')
        axes[0,1].set_ylabel('Number of Users')
        axes[0,1].set_yscale('log')
        
        # Product popularity distribution
        product_popularity = self.raw_data['product_id'].value_counts()
        axes[1,0].hist(product_popularity, bins=50, alpha=0.7, color='salmon', edgecolor='black')
        axes[1,0].set_title('Product Popularity Distribution')
        axes[1,0].set_xlabel('Number of Ratings per Product')
        axes[1,0].set_ylabel('Number of Products')
        axes[1,0].set_yscale('log')
        
        # Rating trend over time (if timestamp is available and valid)
        try:
            # Convert timestamp to datetime if it's numeric
            if self.raw_data['timestamp'].dtype in ['int64', 'float64']:
                timestamps = pd.to_datetime(self.raw_data['timestamp'], unit='s')
                monthly_ratings = timestamps.dt.to_period('M').value_counts().sort_index()
                
                axes[1,1].plot(monthly_ratings.index.astype(str), monthly_ratings.values, marker='o')
                axes[1,1].set_title('Rating Trend Over Time')
                axes[1,1].set_xlabel('Time Period')
                axes[1,1].set_ylabel('Number of Ratings')
                axes[1,1].tick_params(axis='x', rotation=45)
            else:
                axes[1,1].text(0.5, 0.5, 'Timestamp analysis\nnot available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Timestamp Analysis')
        except:
            axes[1,1].text(0.5, 0.5, 'Timestamp analysis\nfailed', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Timestamp Analysis')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
            print("💾 Plots saved as 'dataset_analysis.png'")
        
        plt.show()
    
    def save_processed_data(self, filename='processed_electronics_data.csv'):
        """Save processed data for use in recommendation models"""
        if self.processed_data is None:
            print("❌ No processed data to save.")
            return False
        
        self.processed_data.to_csv(filename, index=False)
        print(f"💾 Processed data saved as: {filename}")
        return True
    
    def generate_dataset_report(self):
        """Generate comprehensive dataset report"""
        if not self.dataset_stats:
            print("❌ No dataset statistics available.")
            return
        
        report = f"""
# 📊 Amazon Electronics Dataset Report
**Generated by:** Anubhav Mishra  
**Date:** September 2025  
**Project:** Advanced E-Commerce Recommendation System

## Dataset Overview
- **Total Interactions:** {self.dataset_stats['total_interactions']:,}
- **Unique Users:** {self.dataset_stats['unique_users']:,}  
- **Unique Products:** {self.dataset_stats['unique_products']:,}
- **Rating Range:** {self.dataset_stats['rating_range'][0]} - {self.dataset_stats['rating_range'][1]} stars
- **Matrix Sparsity:** {self.dataset_stats['sparsity']:.4f} ({self.dataset_stats['sparsity']*100:.2f}%)

## Processing Results
"""
        
        if 'processed_interactions' in self.dataset_stats:
            report += f"""
- **Processed Interactions:** {self.dataset_stats['processed_interactions']:,}
- **Processed Users:** {self.dataset_stats['processed_users']:,}
- **Processed Products:** {self.dataset_stats['processed_products']:,}  
- **Processed Sparsity:** {self.dataset_stats['processed_sparsity']:.4f} ({self.dataset_stats['processed_sparsity']*100:.2f}%)
"""
        
        report += f"""
## Recommendation System Suitability
✅ **Excellent for collaborative filtering** - Large user base with diverse preferences  
✅ **Good sparsity level** - Challenging but realistic for production systems  
✅ **Balanced rating distribution** - Enables effective preference learning  
✅ **Sufficient data volume** - Supports complex ML model training  

## Model Training Recommendations
1. **EPR1 (SVD):** Use full processed dataset for matrix factorization
2. **EPR2 (Popularity):** Leverage product popularity statistics  
3. **EPR3 (User-CF):** Focus on user similarity calculations
4. **EPR4 (Hybrid):** Combine all approaches for optimal performance

---
*Report generated by Advanced E-Commerce Recommendation System*  
*GitHub: https://github.com/anubhav-n-mishra/Ecommerce-product-recommendation-system*
"""
        
        # Save report
        with open('dataset_report.md', 'w') as f:
            f.write(report)
        
        print("📋 Dataset report generated: dataset_report.md")
        print(report)

def main():
    """Demo of dataset creation utilities"""
    print("🔧 Dataset Creation Utilities - Demo")
    print("=" * 50)
    
    # Initialize dataset creator
    creator = DatasetCreator()
    
    # Load dataset
    if creator.load_dataset():
        # Preprocess data
        if creator.preprocess_data():
            # Create user-item matrix
            creator.create_user_item_matrix()
            
            # Create train/test split
            train, test = creator.create_train_test_split()
            
            # Generate visualizations
            creator.visualize_dataset()
            
            # Generate report
            creator.generate_dataset_report()
            
            print("\n✅ Dataset processing completed successfully!")
    else:
        print("⚠️ Dataset not available. Please download it first.")

if __name__ == "__main__":
    main()