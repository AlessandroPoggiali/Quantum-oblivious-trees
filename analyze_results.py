#!/usr/bin/env python3
"""
Analysis script for Quantum vs Classical Oblivious Trees testing results.
Run this after test.py to analyze and visualize results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def analyze_results(csv_file: str):
    """Load and analyze results from CSV"""
    
    if not Path(csv_file).exists():
        print(f"File {csv_file} not found")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} results from {csv_file}\n")
    
    return df

def print_statistics(df: pd.DataFrame):
    """Print summary statistics"""
    print("="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    # Count outcomes
    quantum_better = len(df[df['better'] == 'QUANTUM'])
    classical_better = len(df[df['better'] == 'CLASSICAL'])
    similar = len(df[df['better'] == 'SIMILAR'])
    failed = len(df[df['better'] == 'QUANTUM_FAILED'])
    total = len(df)
    
    print(f"\nOutcome Distribution:")
    print(f"  Quantum Better:    {quantum_better:3d} ({100*quantum_better/total:5.1f}%)")
    print(f"  Classical Better:  {classical_better:3d} ({100*classical_better/total:5.1f}%)")
    print(f"  Similar:           {similar:3d} ({100*similar/total:5.1f}%)")
    print(f"  Quantum Failed:    {failed:3d} ({100*failed/total:5.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total:             {total:3d}")
    
    # Accuracy statistics
    print(f"\n\nAccuracy Statistics (test set):")
    
    # Classical
    class_acc = df['classical_test_acc'].dropna()
    print(f"\n  Classical Model:")
    print(f"    Mean:   {class_acc.mean():.4f}")
    print(f"    Median: {class_acc.median():.4f}")
    print(f"    Std:    {class_acc.std():.4f}")
    print(f"    Min:    {class_acc.min():.4f}")
    print(f"    Max:    {class_acc.max():.4f}")
    
    # Quantum (only for successful runs)
    quantum_acc = df[df['better'] != 'QUANTUM_FAILED']['quantum_test_acc'].dropna()
    if len(quantum_acc) > 0:
        print(f"\n  Quantum Model (successful runs):")
        print(f"    Mean:   {quantum_acc.mean():.4f}")
        print(f"    Median: {quantum_acc.median():.4f}")
        print(f"    Std:    {quantum_acc.std():.4f}")
        print(f"    Min:    {quantum_acc.min():.4f}")
        print(f"    Max:    {quantum_acc.max():.4f}")
    
    # Accuracy difference
    df_success = df[df['better'] != 'QUANTUM_FAILED'].copy()
    df_success['acc_diff'] = df_success['quantum_test_acc'] - df_success['classical_test_acc']
    
    print(f"\n  Quantum - Classical Difference:")
    print(f"    Mean:   {df_success['acc_diff'].mean():+.4f}")
    print(f"    Median: {df_success['acc_diff'].median():+.4f}")
    print(f"    Std:    {df_success['acc_diff'].std():.4f}")
    print(f"    Min:    {df_success['acc_diff'].min():+.4f}")
    print(f"    Max:    {df_success['acc_diff'].max():+.4f}")
    
    # Dataset statistics
    print(f"\n\nDataset Characteristics:")
    print(f"  Avg features:      {df['num_features'].mean():.1f}")
    print(f"  Avg classes:       {df['num_classes'].mean():.1f}")
    print(f"  Avg train samples: {df['num_samples_train'].mean():.0f}")
    
    # Parameter analysis
    print(f"\n\nParameter Counts:")
    print(f"  Classical params (mean):  {df['classical_params'].mean():.0f}")
    print(f"  Quantum params (mean):    {df['quantum_params'].dropna().mean():.0f}")
    
    return df_success

def print_quantum_better_datasets(df: pd.DataFrame):
    """List datasets where quantum performs better"""
    print("\n" + "="*80)
    print("QUANTUM ADVANTAGE DATASETS")
    print("="*80)
    
    quantum_wins = df[df['better'] == 'QUANTUM'].sort_values('quantum_test_acc', ascending=False)
    
    if len(quantum_wins) == 0:
        print("No datasets where quantum performed better")
        return
    
    print(f"\n{len(quantum_wins)} datasets show quantum advantage:\n")
    print(f"{'Dataset':<20} {'Classical':<12} {'Quantum':<12} {'Improvement':<12} {'Features':<10}")
    print("─" * 70)
    
    for _, row in quantum_wins.iterrows():
        dataset = row['dataset']
        class_acc = row['classical_test_acc']
        quant_acc = row['quantum_test_acc']
        improvement = quant_acc - class_acc
        features = row['num_features']
        print(f"{dataset:<20} {class_acc:>10.4f}  {quant_acc:>10.4f}  {improvement:>10.4f}  {features:>8.0f}")
    
    return quantum_wins

def plot_results(df: pd.DataFrame, output_prefix: str = 'analysis'):
    """Create visualizations"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantum vs Classical Oblivious Trees - Performance Comparison', fontsize=16)
    
    # Filter successful runs
    df_success = df[df['better'] != 'QUANTUM_FAILED'].copy()
    df_success['acc_diff'] = df_success['quantum_test_acc'] - df_success['classical_test_acc']
    
    # 1. Accuracy comparison scatter plot
    ax = axes[0, 0]
    ax.scatter(df_success['classical_test_acc'], df_success['quantum_test_acc'], alpha=0.6, s=100)
    
    # Add diagonal line (equal performance)
    min_acc = min(df_success['classical_test_acc'].min(), df_success['quantum_test_acc'].min())
    max_acc = max(df_success['classical_test_acc'].max(), df_success['quantum_test_acc'].max())
    ax.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', label='Equal performance', linewidth=2)
    
    ax.set_xlabel('Classical Test Accuracy', fontsize=11)
    ax.set_ylabel('Quantum Test Accuracy', fontsize=11)
    ax.set_title('Accuracy: Quantum vs Classical', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Outcome distribution
    ax = axes[0, 1]
    outcomes = df['better'].value_counts()
    colors = {'QUANTUM': 'green', 'CLASSICAL': 'red', 'SIMILAR': 'gray', 'QUANTUM_FAILED': 'orange'}
    colors_list = [colors.get(outcome, 'blue') for outcome in outcomes.index]
    
    bars = ax.bar(outcomes.index, outcomes.values, color=colors_list, alpha=0.7)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Model Performance Distribution', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Accuracy difference histogram
    ax = axes[1, 0]
    ax.hist(df_success['acc_diff'], bins=15, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.axvline(df_success['acc_diff'].mean(), color='green', linestyle='-', linewidth=2, label='Mean')
    ax.set_xlabel('Quantum Accuracy - Classical Accuracy', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Accuracy Difference Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Accuracy by dataset size
    ax = axes[1, 1]
    df_success['size_category'] = pd.cut(df_success['num_samples_train'], 
                                         bins=[0, 100, 500, float('inf')],
                                         labels=['Small (<100)', 'Medium (100-500)', 'Large (>500)'])
    
    size_groups = df_success.groupby('size_category', observed=True)[['classical_test_acc', 'quantum_test_acc']].mean()
    
    x_pos = np.arange(len(size_groups))
    width = 0.35
    
    ax.bar(x_pos - width/2, size_groups['classical_test_acc'], width, label='Classical', alpha=0.8)
    ax.bar(x_pos + width/2, size_groups['quantum_test_acc'], width, label='Quantum', alpha=0.8)
    
    ax.set_ylabel('Average Test Accuracy', fontsize=11)
    ax.set_xlabel('Dataset Size', fontsize=11)
    ax.set_title('Performance by Dataset Size', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(size_groups.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSummary saved to {output_prefix}_comparison.png")
    
    return fig

def main():
    # Get CSV file from command line or use default
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    
    # Load data
    df = analyze_results(csv_file)
    if df is None:
        return
    
    # Print statistics
    df_success = print_statistics(df)
    
    # Print quantum better datasets
    quantum_wins = print_quantum_better_datasets(df)
    
    # Create plots
    plot_results(df)
    
    # Save summary
    summary_file = 'analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("QUANTUM VS CLASSICAL OBLIVIOUS TREES - ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        total = len(df)
        quantum_better = len(df[df['better'] == 'QUANTUM'])
        classical_better = len(df[df['better'] == 'CLASSICAL'])
        
        f.write(f"Results from: {csv_file}\n")
        f.write(f"Total datasets tested: {total}\n")
        f.write(f"Quantum performed better: {quantum_better} ({100*quantum_better/total:.1f}%)\n")
        f.write(f"Classical performed better: {classical_better} ({100*classical_better/total:.1f}%)\n\n")
        
        if len(quantum_wins) > 0:
            f.write(f"Datasets with quantum advantage:\n")
            for _, row in quantum_wins.iterrows():
                improvement = row['quantum_test_acc'] - row['classical_test_acc']
                f.write(f"  - {row['dataset']}: +{improvement:.4f} ({row['quantum_test_acc']:.4f} vs {row['classical_test_acc']:.4f})\n")
    
    print(f"\nSummary saved to {summary_file}")

if __name__ == '__main__':
    main()
