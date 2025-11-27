"""
Visualize FHE Softmax Benchmark Results
Creates graphs comparing Taylor vs Chebyshev approximations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def load_results(csv_path='benchmark_results.csv'):
    """Load benchmark results from CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please run 'python benchmark.py' first to generate results.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} test results from {csv_path}")
    
    # Filter to successful tests only
    successful = df[df['success'] == True]
    print(f"Successful tests: {len(successful)}/{len(df)}")
    
    return successful


def plot_degree_vs_error(df, output_file='degree_vs_error.png'):
    """
    Create line graph showing Degree vs L1 Error.
    
    - X-axis: Polynomial degree (dynamic)
    - Y-axis: Mean L1 error
    - Two lines: Taylor (blue), Chebyshev (red dashed)
    - Error bars: Standard deviation
    """
    print("\n[GRAPH 1] Creating Degree vs L1 Error plot...")
    
    # Get all unique degrees present in the data
    all_degrees = sorted(df['degree'].unique())
    
    # Calculate mean and std for each method and degree
    stats = df.groupby(['method', 'degree'])['l1_error'].agg(['mean', 'std']).reset_index()
    
    taylor_stats = stats[stats['method'] == 'taylor'].sort_values('degree')
    cheby_stats = stats[stats['method'] == 'chebyshev'].sort_values('degree')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot Taylor series
    if not taylor_stats.empty:
        plt.errorbar(
            taylor_stats['degree'],
            taylor_stats['mean'],
            yerr=taylor_stats['std'],
            marker='o',
            markersize=8,
            linestyle='-',
            linewidth=2,
            capsize=5,
            color='#2E86AB',
            label='Taylor Series'
        )
    
    # Plot Chebyshev
    if not cheby_stats.empty:
        plt.errorbar(
            cheby_stats['degree'],
            cheby_stats['mean'],
            yerr=cheby_stats['std'],
            marker='s',
            markersize=8,
            linestyle='--',
            linewidth=2,
            capsize=5,
            color='#A23B72',
            label='Chebyshev Polynomials'
        )
    
    # Formatting
    plt.xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
    plt.ylabel('Mean L1 Error', fontsize=12, fontweight='bold')
    plt.title('FHE Softmax: Polynomial Degree vs Approximation Error', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Set x-ticks to be exactly the degrees we have
    plt.xticks(all_degrees)
    
    # Add minor grid
    plt.minorticks_on()
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[GRAPH 1] Saved to: {output_file}")
    
    # Print values
    print("\nL1 Error Statistics:")
    if not taylor_stats.empty:
        print("Taylor Series:")
        for _, row in taylor_stats.iterrows():
            print(f"  Degree {int(row['degree'])}: {row['mean']:.6f} ± {row['std']:.6f}")
            
    if not cheby_stats.empty:
        print("\nChebyshev Polynomials:")
        for _, row in cheby_stats.iterrows():
            print(f"  Degree {int(row['degree'])}: {row['mean']:.6f} ± {row['std']:.6f}")


def plot_argmax_hits_misses(df, output_file='argmax_hits_vs_misses.png'):
    """
    Create bar chart showing Argmax Hits vs Misses.
    
    Two subplots (side by side):
    - Left: Taylor Series
    - Right: Chebyshev Polynomials
    
    For each degree, show bars for Hit (green) and Miss (red) counts.
    """
    print("\n[GRAPH 2] Creating Argmax Hits vs Misses plot...")
    
    # Get unique degrees and methods
    degrees = sorted(df['degree'].unique())
    methods = sorted(df['method'].unique())
    
    # If we have fewer than 2 methods, adjust subplots
    num_methods = len(methods)
    if num_methods == 0:
        print("No methods found to plot.")
        return

    fig, axes = plt.subplots(1, num_methods, figsize=(7 * num_methods, 6))
    
    # Ensure axes is iterable even if only 1 subplot
    if num_methods == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        hits = []
        misses = []
        
        for degree in degrees:
            subset = df[(df['method'] == method) & (df['degree'] == degree)]
            if len(subset) == 0:
                hits.append(0)
                misses.append(0)
            else:
                hit_count = subset['argmax_match'].sum()
                miss_count = len(subset) - hit_count
                hits.append(hit_count)
                misses.append(miss_count)
        
        # Position of bars
        x = np.arange(len(degrees))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, hits, width, label='Hit', 
                       color='#06A77D', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, misses, width, label='Miss', 
                       color='#D62246', edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only label if non-zero
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Formatting
        method_name = method.upper() if method.upper() in ['TAYLOR', 'CHEBYSHEV'] else method.title()
        if method == 'taylor': method_name = 'Taylor Series'
        if method == 'chebyshev': method_name = 'Chebyshev Polynomials'
        
        ax.set_title(f'{method_name}', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count (out of 10 samples)', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(degrees)
        
        # Set Y limit to max samples + 1 for headroom
        max_samples = 10 # Default assumption
        if len(df) > 0:
             # Try to infer samples per degree/method
             counts = df.groupby(['method', 'degree']).size()
             if not counts.empty:
                 max_samples = counts.max()
        
        ax.set_ylim(0, max_samples + 1)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add horizontal line at max_samples (perfect score)
        ax.axhline(y=max_samples, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Overall title
    fig.suptitle('Classification Accuracy: Argmax Hit vs Miss', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[GRAPH 2] Saved to: {output_file}")
    
    # Print summary
    print("\nArgmax Preservation Summary:")
    for method in methods:
        method_name = method.title()
        print(f"\n{method_name}:")
        for degree in degrees:
            subset = df[(df['method'] == method) & (df['degree'] == degree)]
            if len(subset) > 0:
                hits = subset['argmax_match'].sum()
                total = len(subset)
                pct = (hits / total * 100) if total > 0 else 0
                print(f"  Degree {degree}: {hits}/{total} hits ({pct:.0f}% accuracy)")


def main():
    """Main visualization function."""
    print("=" * 70)
    print("FHE SOFTMAX BENCHMARK - VISUALIZATION")
    print("=" * 70)
    
    # Load results
    df = load_results('benchmark_results.csv')
    
    if len(df) == 0:
        print("Error: No successful test results found!")
        return
    
    # Create graphs
    plot_degree_vs_error(df)
    plot_argmax_hits_misses(df)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. degree_vs_error.png")
    print("  2. argmax_hits_vs_misses.png")
    print("\nAll done! Check the PNG files for results.")


if __name__ == "__main__":
    main()
