#!/usr/bin/env python3
"""
Advanced Fairness Analysis using Fairlearn
Healthcare Chatbot Bias and Fairness Audit
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate, count, false_positive_rate, false_negative_rate
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(dataset_path, text_col=None, label_col=None, sensitive_col=None, threshold=None):
    """Load and prepare the dataset for fairness analysis."""
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Handle the specific dataset structure
    if label_col == 'CES-D' and 'cesd' in df.columns:
        # Create binary labels based on threshold
        y = (df['cesd'] >= threshold).astype(int)
        print(f"Created binary labels using {label_col} >= {threshold}")
    elif label_col in df.columns:
        y = df[label_col]
    else:
        raise ValueError(f"Label column '{label_col}' not found in dataset")
    
    # Handle sensitive attribute
    if sensitive_col == 'gender' and 'sex' in df.columns:
        sensitive_attr = df['sex']
        print(f"Using 'sex' column as sensitive attribute for gender")
    elif sensitive_col in df.columns:
        sensitive_attr = df[sensitive_col]
    else:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataset")
    
    # Prepare features (exclude target and sensitive attributes)
    exclude_cols = [label_col, sensitive_col, 'cesd', 'sex', 'id']
    if text_col and text_col in df.columns:
        exclude_cols.append(text_col)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])  # Only numeric features
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    print(f"Sensitive attribute distribution: {sensitive_attr.value_counts()}")
    
    return X, y, sensitive_attr

def compute_fairness_metrics(y_true, y_pred, sensitive_attr, threshold=0.5):
    """Compute comprehensive fairness metrics using Fairlearn."""
    print("\n=== Fairness Metrics Analysis ===")
    
    # Convert probabilities to binary predictions if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_binary = (y_pred[:, 1] >= threshold).astype(int)
    else:
        y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Define metrics to compute
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'selection_rate': selection_rate,
        'count': count,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
    }
    
    # Compute metrics by group
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred_binary,
        sensitive_features=sensitive_attr
    )
    
    print("\nFairness Metrics by Group:")
    print(metric_frame.by_group)
    
    # Compute disparities
    print("\nDisparities (max - min across groups):")
    disparities = metric_frame.difference()
    print(disparities)
    
    return metric_frame, disparities

def apply_bias_mitigation(X_train, y_train, X_test, sensitive_attr_train, sensitive_attr_test, method='threshold'):
    """Apply bias mitigation techniques."""
    print(f"\n=== Applying Bias Mitigation: {method} ===")
    
    # Train baseline model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    if method == 'threshold':
        # Post-processing: Threshold optimization
        mitigator = ThresholdOptimizer(
            estimator=model,
            constraints="equalized_odds",
            prefit=True
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_attr_train)
        y_pred_mitigated = mitigator.predict(X_test, sensitive_features=sensitive_attr_test)
        y_pred_proba = model.predict_proba(X_test)
        
    elif method == 'exponentiated_gradient':
        # In-processing: Exponentiated Gradient
        mitigator = ExponentiatedGradient(
            LogisticRegression(random_state=42, max_iter=1000),
            constraints="equalized_odds",
            eps=0.01
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_attr_train)
        y_pred_mitigated = mitigator.predict(X_test)
        y_pred_proba = mitigator.predict_proba(X_test)
        
    elif method == 'grid_search':
        # In-processing: Grid Search
        mitigator = GridSearch(
            LogisticRegression(random_state=42, max_iter=1000),
            constraints="equalized_odds",
            grid_size=10
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_attr_train)
        y_pred_mitigated = mitigator.predict(X_test)
        y_pred_proba = mitigator.predict_proba(X_test)
    
    else:
        # Baseline (no mitigation)
        y_pred_mitigated = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    return y_pred_mitigated, y_pred_proba, model

def create_fairness_plots(metric_frames, disparities, output_dir):
    """Create visualization plots for fairness analysis."""
    print(f"\n=== Creating Fairness Visualizations ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Selection Rate by Group
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    selection_rates = metric_frames['baseline'].by_group['selection_rate']
    selection_rates.plot(kind='bar')
    plt.title('Selection Rate by Group (Baseline)')
    plt.ylabel('Selection Rate')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    # Fix: disparities is a Series, not a DataFrame
    disparity_values = [disparities['selection_rate']]
    disparity_labels = ['Selection Rate Disparity']
    plt.bar(disparity_labels, disparity_values)
    plt.title('Selection Rate Disparity (Baseline)')
    plt.ylabel('Disparity')
    plt.xticks(rotation=45)
    
    # Plot 2: Accuracy by Group
    plt.subplot(2, 2, 3)
    accuracies = metric_frames['baseline'].by_group['accuracy']
    accuracies.plot(kind='bar')
    plt.title('Accuracy by Group (Baseline)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    # Fix: disparities is a Series, not a DataFrame
    disparity_values = [disparities['accuracy']]
    disparity_labels = ['Accuracy Disparity']
    plt.bar(disparity_labels, disparity_values)
    plt.title('Accuracy Disparity (Baseline)')
    plt.ylabel('Disparity')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fairness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Comparison across methods (if multiple methods available)
    if len(metric_frames) > 1:
        plt.figure(figsize=(15, 5))
        
        methods = list(metric_frames.keys())
        
        # Only plot if we have multiple methods with valid disparities
        try:
            selection_rate_disparities = [metric_frames[method].difference()['selection_rate'] for method in methods]
            
            plt.subplot(1, 3, 1)
            plt.bar(methods, selection_rate_disparities)
            plt.title('Selection Rate Disparity Comparison')
            plt.ylabel('Disparity')
            plt.xticks(rotation=45)
            
            accuracy_disparities = [metric_frames[method].difference()['accuracy'] for method in methods]
            plt.subplot(1, 3, 2)
            plt.bar(methods, accuracy_disparities)
            plt.title('Accuracy Disparity Comparison')
            plt.ylabel('Disparity')
            plt.xticks(rotation=45)
            
            precision_disparities = [metric_frames[method].difference()['precision'] for method in methods]
            plt.subplot(1, 3, 3)
            plt.bar(methods, precision_disparities)
            plt.title('Precision Disparity Comparison')
            plt.ylabel('Disparity')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/mitigation_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create comparison plot: {e}")
    
    print(f"Visualizations saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Advanced Fairness Analysis using Fairlearn')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV file')
    parser.add_argument('--text-col', help='Name of text column (optional)')
    parser.add_argument('--label-col', required=True, help='Name of label column')
    parser.add_argument('--threshold', type=float, default=16, help='Threshold for binary classification')
    parser.add_argument('--sensitive-col', required=True, help='Name of sensitive attribute column')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    print("=== Healthcare Chatbot Fairness Analysis ===")
    print(f"Dataset: {args.dataset}")
    print(f"Label column: {args.label_col}")
    print(f"Sensitive attribute: {args.sensitive_col}")
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {args.output_dir}")
    
    # Load and prepare data
    X, y, sensitive_attr = load_and_prepare_data(
        args.dataset, 
        args.text_col, 
        args.label_col, 
        args.sensitive_col, 
        args.threshold
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=y
    )
    
    # Split sensitive attributes accordingly
    sensitive_attr_train = sensitive_attr.iloc[X_train.index]
    sensitive_attr_test = sensitive_attr.iloc[X_test.index]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store results
    results = {}
    metric_frames = {}
    
    # Baseline analysis
    print("\n=== Baseline Analysis ===")
    y_pred_baseline, y_pred_proba_baseline, baseline_model = apply_bias_mitigation(
        X_train_scaled, y_train, X_test_scaled, 
        sensitive_attr_train, sensitive_attr_test, 
        method='baseline'
    )
    
    metric_frame_baseline, disparities_baseline = compute_fairness_metrics(
        y_test, y_pred_proba_baseline, sensitive_attr_test
    )
    
    results['baseline'] = {
        'accuracy': accuracy_score(y_test, y_pred_baseline),
        'precision': precision_score(y_test, y_pred_baseline),
        'recall': recall_score(y_test, y_pred_baseline),
        'auc': roc_auc_score(y_test, y_pred_proba_baseline[:, 1])
    }
    metric_frames['baseline'] = metric_frame_baseline
    
    # Apply different mitigation methods (simplified for compatibility)
    mitigation_methods = ['threshold']  # Only use threshold for now
    
    for method in mitigation_methods:
        try:
            print(f"\n=== {method.upper()} Mitigation ===")
            y_pred_mitigated, y_pred_proba_mitigated, mitigated_model = apply_bias_mitigation(
                X_train_scaled, y_train, X_test_scaled,
                sensitive_attr_train, sensitive_attr_test,
                method=method
            )
            
            metric_frame_mitigated, disparities_mitigated = compute_fairness_metrics(
                y_test, y_pred_proba_mitigated, sensitive_attr_test
            )
            
            results[method] = {
                'accuracy': accuracy_score(y_test, y_pred_mitigated),
                'precision': precision_score(y_test, y_pred_mitigated),
                'recall': recall_score(y_test, y_pred_mitigated),
                'auc': roc_auc_score(y_test, y_pred_proba_mitigated[:, 1])
            }
            metric_frames[method] = metric_frame_mitigated
            
        except Exception as e:
            print(f"Error with {method}: {e}")
            continue
    
    # Create visualizations
    create_fairness_plots(metric_frames, disparities_baseline, args.output_dir)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'{args.output_dir}/fairness_results_summary.csv')
    
    # Save detailed metrics
    with pd.ExcelWriter(f'{args.output_dir}/fairness_detailed_metrics.xlsx') as writer:
        for method, metric_frame in metric_frames.items():
            metric_frame.by_group.to_excel(writer, sheet_name=f'{method}_by_group')
            metric_frame.difference().to_excel(writer, sheet_name=f'{method}_disparities')
    
    print(f"\n=== Results Summary ===")
    print(results_df)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("Files created:")
    print("- fairness_results_summary.csv")
    print("- fairness_detailed_metrics.xlsx")
    print("- fairness_analysis.png")
    print("- mitigation_comparison.png")

if __name__ == "__main__":
    main()
