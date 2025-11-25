import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
import psutil
import os
import sys
from typing import Dict, List, Tuple
import pandas as pd
from memory_profiler import memory_usage

class AdvancedPerformanceAnalyzer:
    def __init__(self):
        self.results_dir = "results/graphs"
        os.makedirs(self.results_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Initialize metrics collection
        self.metrics_history = {
            'tenseal': [],
            'paillier': []
        }
    
    def measure_memory_consumption(self, func, *args, **kwargs) -> Tuple[float, any]:
        """Measure peak memory consumption of a function"""
        result = memory_usage((func, args, kwargs), max_usage=True, retval=True)
        peak_memory = result[0]
        func_result = result[1]
        return peak_memory, func_result
    
    def measure_ciphertext_sizes(self, tenseal_encrypted, paillier_encrypted) -> Dict:
        """Measure ciphertext sizes in bytes"""
        import pickle
        
        # Measure TenSEAL ciphertext size
        tenseal_size = 0
        if tenseal_encrypted is not None:
            try:
                # Check if it's a list/array with elements
                if isinstance(tenseal_encrypted, (list, tuple)) and len(tenseal_encrypted) > 0:
                    sample_ct = tenseal_encrypted[0]
                else:
                    sample_ct = tenseal_encrypted
                    
                # Use TenSEAL's serialize method instead of pickle
                if hasattr(sample_ct, 'serialize'):
                    tenseal_size = len(sample_ct.serialize())
                else:
                    tenseal_size = len(pickle.dumps(sample_ct))
            except Exception as e:
                print(f"Warning: Could not measure TenSEAL ciphertext size: {e}")
                tenseal_size = 0
        
        # Measure Paillier ciphertext size
        paillier_size = 0
        if paillier_encrypted is not None:
            try:
                # Check if it's a list/array with elements
                if isinstance(paillier_encrypted, (list, tuple)) and len(paillier_encrypted) > 0:
                    sample_ct = paillier_encrypted[0]
                else:
                    sample_ct = paillier_encrypted
                    
                paillier_size = len(pickle.dumps(sample_ct))
            except Exception as e:
                print(f"Warning: Could not measure Paillier ciphertext size: {e}")
                paillier_size = 0
        
        return {
            'tenseal_ciphertext_size': tenseal_size,
            'paillier_ciphertext_size': paillier_size,
            'size_ratio': paillier_size / tenseal_size if tenseal_size > 0 else float('inf')
        }
    
    def measure_throughput(self, operation_func, test_data, num_operations: int) -> float:
        """Measure operations per second"""
        start_time = time.time()
        
        for i in range(num_operations):
            operation_func(test_data)
        
        end_time = time.time()
        throughput = num_operations / (end_time - start_time)
        return throughput
    
    def measure_computation_error(self, original_data, decrypted_data) -> Dict:
        """Measure numerical error in computations"""
        if original_data is None or decrypted_data is None:
            return {}
        
        # Check if data is empty
        try:
            if len(original_data) == 0 or len(decrypted_data) == 0:
                return {}
        except TypeError:
            # Handle non-iterable types
            return {}
        
        original_array = np.array(original_data)
        decrypted_array = np.array(decrypted_data)
        
        # Calculate various error metrics
        absolute_error = np.abs(original_array - decrypted_array)
        relative_error = np.abs((original_array - decrypted_array) / (original_array + 1e-10))
        mse = np.mean((original_array - decrypted_array) ** 2)
        
        return {
            'max_absolute_error': np.max(absolute_error),
            'mean_absolute_error': np.mean(absolute_error),
            'max_relative_error': np.max(relative_error),
            'mean_squared_error': mse
        }
    
    def measure_key_sizes(self, tenseal_context, paillier_public_key) -> Dict:
        """Measure key sizes in bytes"""
        import pickle
        
        # Measure TenSEAL context size
        tenseal_key_size = 0
        if tenseal_context:
            try:
                # Use TenSEAL's serialize method instead of pickle
                if hasattr(tenseal_context, 'serialize'):
                    tenseal_key_size = len(tenseal_context.serialize())
                else:
                    tenseal_key_size = len(pickle.dumps(tenseal_context))
            except Exception as e:
                print(f"Warning: Could not measure TenSEAL context size: {e}")
                tenseal_key_size = 0
        
        # Measure Paillier public key size
        paillier_key_size = 0
        if paillier_public_key:
            try:
                paillier_key_size = len(pickle.dumps(paillier_public_key))
            except Exception as e:
                print(f"Warning: Could not measure Paillier key size: {e}")
                paillier_key_size = 0
        
        return {
            'tenseal_context_size': tenseal_key_size,
            'paillier_public_key_size': paillier_key_size
        }
    
    def analyze_noise_growth(self, tenseal_demo, base_data, operations_chain):
        """Analyze noise growth through operation chains"""
        print("Analyzing noise growth...")
        
        try:
            # Encrypt base data
            encrypted_base = tenseal_demo.encrypt_batch_data([base_data])[0]
            
            noise_metrics = []
            current_encrypted = encrypted_base
            
            for i, operation in enumerate(operations_chain):
                # Apply operation
                if operation == 'add':
                    current_encrypted = current_encrypted + current_encrypted
                elif operation == 'multiply':
                    current_encrypted = current_encrypted * 2.0
                elif operation == 'square':
                    current_encrypted = current_encrypted * current_encrypted
                
                # Decrypt and measure deviation from expected
                decrypted = current_encrypted.decrypt()
                expected = self._compute_expected_value(base_data, operations_chain[:i+1])
                
                error = np.mean(np.abs(np.array(decrypted) - np.array(expected)))
                noise_metrics.append({
                    'operation_chain_length': i + 1,
                    'operations': operations_chain[:i+1],
                    'mean_error': error,
                    'max_error': np.max(np.abs(np.array(decrypted) - np.array(expected)))
                })
            
            return noise_metrics
        except Exception as e:
            print(f"Warning: Could not analyze noise growth: {e}")
            return []
    
    def _compute_expected_value(self, base_data, operations):
        """Compute expected value after operation chain"""
        # Convert to list if it's a numpy array
        if isinstance(base_data, np.ndarray):
            result = base_data.tolist()
        else:
            result = list(base_data)  # Create a copy as a list
        
        for op in operations:
            if op == 'add':
                result = [x + x for x in result]
            elif op == 'multiply':
                result = [x * 2.0 for x in result]
            elif op == 'square':
                result = [x * x for x in result]
        
        return result
    
    def comprehensive_benchmark(self, tenseal_demo, paillier_demo, data_sizes: List[int]):
        """Run comprehensive benchmark with research-grade metrics"""
        print("🚀 Running Comprehensive FHE Benchmark...")
        print("=" * 60)
        
        all_metrics = {
            'data_sizes': data_sizes,
            'tenseal': [],
            'paillier': []
        }
        
        for size in data_sizes:
            print(f"\n📊 Testing with {size} records...")
            
            # Generate test data
            from src.data_simulator import MedicalDataSimulator
            records = MedicalDataSimulator.generate_patient_records(size)
            feature_matrix = MedicalDataSimulator.records_to_feature_matrix(records[:min(size, 10)])
            
            # TenSEAL metrics
            tenseal_metrics = self._benchmark_tenseal(tenseal_demo, feature_matrix, size)
            all_metrics['tenseal'].append(tenseal_metrics)
            
            # Paillier metrics  
            paillier_metrics = self._benchmark_paillier(paillier_demo, records, size)
            all_metrics['paillier'].append(paillier_metrics)
            
            self._print_metrics_comparison(tenseal_metrics, paillier_metrics)
        
        return all_metrics
    
    def _benchmark_tenseal(self, tenseal_demo, feature_matrix, data_size):
        """Benchmark TenSEAL with comprehensive metrics"""
        metrics = {}
        
        try:
            # Memory consumption for encryption
            mem_enc, encrypted_data = self.measure_memory_consumption(
                tenseal_demo.encrypt_batch_data, feature_matrix
            )
            metrics['encryption_memory_mb'] = mem_enc
            metrics['encryption_time'] = tenseal_demo.performance_metrics.get('encryption_time', 0)
            
            # Ciphertext size
            size_info = self.measure_ciphertext_sizes(encrypted_data, None)
            metrics['ciphertext_size_bytes'] = size_info['tenseal_ciphertext_size']
            
            # Key sizes
            key_sizes = self.measure_key_sizes(tenseal_demo.context, None)
            metrics['key_size_bytes'] = key_sizes['tenseal_context_size']
            
            # Throughput for operations
            def add_operation(data):
                return data[0] + data[0] if len(data) > 0 else None
            
            if encrypted_data is not None and len(encrypted_data) > 0:
                metrics['throughput_ops_sec'] = self.measure_throughput(
                    add_operation, encrypted_data, min(100, len(encrypted_data))
                )
            else:
                metrics['throughput_ops_sec'] = 0
            
            # Computation error
            if encrypted_data is not None and len(encrypted_data) > 0:
                original_sample = feature_matrix[0]
                decrypted_sample = encrypted_data[0].decrypt()
                error_metrics = self.measure_computation_error(original_sample, decrypted_sample)
                metrics.update(error_metrics)
            
            # Noise growth analysis
            if feature_matrix.size > 0:
                # Convert to list properly
                base_data = feature_matrix[0].tolist() if isinstance(feature_matrix[0], np.ndarray) else list(feature_matrix[0])
                noise_metrics = self.analyze_noise_growth(tenseal_demo, base_data, 
                                                        ['add', 'multiply', 'add', 'square'])
                metrics['noise_growth'] = noise_metrics
        
        except Exception as e:
            print(f"Error in TenSEAL benchmarking: {e}")
            # Set default values for failed metrics
            metrics.setdefault('encryption_memory_mb', 0)
            metrics.setdefault('encryption_time', 0)
            metrics.setdefault('ciphertext_size_bytes', 0)
            metrics.setdefault('key_size_bytes', 0)
            metrics.setdefault('throughput_ops_sec', 0)
        
        return metrics
    
    def _benchmark_paillier(self, paillier_demo, records, data_size):
        """Benchmark Paillier with comprehensive metrics"""
        metrics = {}
        
        try:
            # Memory consumption for encryption
            mem_enc, encrypted_data = self.measure_memory_consumption(
                paillier_demo.encrypt_patient_data, records, 'age'
            )
            metrics['encryption_memory_mb'] = mem_enc
            metrics['encryption_time'] = paillier_demo.performance_metrics.get('encryption_time', 0)
            
            # Ciphertext size
            size_info = self.measure_ciphertext_sizes(None, encrypted_data)
            metrics['ciphertext_size_bytes'] = size_info['paillier_ciphertext_size']
            
            # Key sizes
            key_sizes = self.measure_key_sizes(None, paillier_demo.public_key)
            metrics['key_size_bytes'] = key_sizes['paillier_public_key_size']
            
            # Throughput for operations
            def add_operation(data):
                if len(data) > 0:
                    return data[0] + data[1] if len(data) > 1 else data[0]
                return None
            
            if encrypted_data is not None and len(encrypted_data) > 0:
                metrics['throughput_ops_sec'] = self.measure_throughput(
                    add_operation, encrypted_data, min(100, len(encrypted_data))
                )
            else:
                metrics['throughput_ops_sec'] = 0
            
            # Computation error (Paillier should be exact for integers)
            if encrypted_data is not None and len(encrypted_data) > 0:
                original_ages = [rec['age'] for rec in records[:len(encrypted_data)]]
                decrypted_ages = [paillier_demo.private_key.decrypt(ct) / 100 for ct in encrypted_data]
                error_metrics = self.measure_computation_error(original_ages, decrypted_ages)
                metrics.update(error_metrics)
        
        except Exception as e:
            print(f"Error in Paillier benchmarking: {e}")
            # Set default values for failed metrics
            metrics.setdefault('encryption_memory_mb', 0)
            metrics.setdefault('encryption_time', 0)
            metrics.setdefault('ciphertext_size_bytes', 0)
            metrics.setdefault('key_size_bytes', 0)
            metrics.setdefault('throughput_ops_sec', 0)
        
        return metrics
    
    def _print_metrics_comparison(self, tenseal_metrics, paillier_metrics):
        """Print side-by-side metrics comparison"""
        print(f"  {'Metric':<25} {'TenSEAL':<15} {'Paillier':<15} {'Ratio':<10}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")
        
        metrics_to_compare = [
            ('Encryption Memory (MB)', 'encryption_memory_mb'),
            ('Ciphertext Size (B)', 'ciphertext_size_bytes'), 
            ('Key Size (B)', 'key_size_bytes'),
            ('Throughput (op/s)', 'throughput_ops_sec'),
            ('Mean Abs Error', 'mean_absolute_error')
        ]
        
        for display_name, metric_key in metrics_to_compare:
            t_val = tenseal_metrics.get(metric_key, 0)
            p_val = paillier_metrics.get(metric_key, 0)
            
            if t_val == 0 or p_val == 0:
                ratio = 'N/A'
            else:
                ratio = f"{p_val/t_val:.2f}x"
            
            print(f"  {display_name:<25} {t_val:<15.2f} {p_val:<15.2f} {ratio:<10}")
    
    def create_research_grade_graphs(self, benchmark_data):
        """Create publication-quality graphs with research metrics"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Memory Consumption Comparison
        ax1 = plt.subplot(3, 3, 1)
        self._plot_memory_comparison(ax1, benchmark_data)
        
        # 2. Ciphertext Size Comparison
        ax2 = plt.subplot(3, 3, 2)
        self._plot_ciphertext_size_comparison(ax2, benchmark_data)
        
        # 3. Throughput Comparison
        ax3 = plt.subplot(3, 3, 3)
        self._plot_throughput_comparison(ax3, benchmark_data)
        
        # 4. Key Size Comparison
        ax4 = plt.subplot(3, 3, 4)
        self._plot_key_size_comparison(ax4, benchmark_data)
        
        # 5. Computation Error
        ax5 = plt.subplot(3, 3, 5)
        self._plot_computation_error(ax5, benchmark_data)
        
        # 6. Noise Growth Analysis
        ax6 = plt.subplot(3, 3, 6)
        self._plot_noise_growth(ax6, benchmark_data)
        
        # 7. Comprehensive Radar Chart
        ax7 = plt.subplot(3, 3, (7, 9))
        self._plot_radar_chart(ax7, benchmark_data)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/research_grade_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary table
        self._create_metrics_table(benchmark_data)
        
        return fig
    
    def _plot_memory_comparison(self, ax, data):
        """Plot memory consumption comparison"""
        sizes = data['data_sizes']
        tenseal_mem = [m.get('encryption_memory_mb', 0) for m in data['tenseal']]
        paillier_mem = [m.get('encryption_memory_mb', 0) for m in data['paillier']]
        
        ax.plot(sizes, tenseal_mem, 'o-', linewidth=2, markersize=6, label='TenSEAL (FHE)')
        ax.plot(sizes, paillier_mem, 's-', linewidth=2, markersize=6, label='Paillier (PHE)')
        ax.set_xlabel('Data Size (records)')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Consumption\n(Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_ciphertext_size_comparison(self, ax, data):
        """Plot ciphertext size comparison"""
        sizes = data['data_sizes']
        tenseal_ct = [m.get('ciphertext_size_bytes', 0) / 1024 for m in data['tenseal']]  # KB
        paillier_ct = [m.get('ciphertext_size_bytes', 0) / 1024 for m in data['paillier']]  # KB
        
        ax.bar([x-0.2 for x in range(len(sizes))], tenseal_ct, width=0.4, label='TenSEAL', alpha=0.8)
        ax.bar([x+0.2 for x in range(len(sizes))], paillier_ct, width=0.4, label='Paillier', alpha=0.8)
        ax.set_xlabel('Data Size (records)')
        ax.set_ylabel('Ciphertext Size (KB)')
        ax.set_title('Ciphertext Size Comparison\n(Lower is Better)')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_throughput_comparison(self, ax, data):
        """Plot throughput comparison"""
        sizes = data['data_sizes']
        tenseal_tp = [m.get('throughput_ops_sec', 0) for m in data['tenseal']]
        paillier_tp = [m.get('throughput_ops_sec', 0) for m in data['paillier']]
        
        ax.semilogy(sizes, tenseal_tp, 'o-', linewidth=2, markersize=6, label='TenSEAL')
        ax.semilogy(sizes, paillier_tp, 's-', linewidth=2, markersize=6, label='Paillier')
        ax.set_xlabel('Data Size (records)')
        ax.set_ylabel('Throughput (operations/sec)')
        ax.set_title('Computational Throughput\n(Higher is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_key_size_comparison(self, ax, data):
        """Plot key size comparison"""
        if data['tenseal'] and data['paillier']:
            tenseal_key = data['tenseal'][0].get('key_size_bytes', 0) / 1024  # KB
            paillier_key = data['paillier'][0].get('key_size_bytes', 0) / 1024  # KB
            
            schemes = ['TenSEAL', 'Paillier']
            key_sizes = [tenseal_key, paillier_key]
            
            bars = ax.bar(schemes, key_sizes, color=['skyblue', 'lightcoral'], alpha=0.8)
            ax.set_ylabel('Key Size (KB)')
            ax.set_title('Key Size Comparison\n(Lower is Better)')
            
            # Add value labels on bars
            for bar, size in zip(bars, key_sizes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{size:.1f} KB', ha='center', va='bottom')
    
    def _plot_computation_error(self, ax, data):
        """Plot computation error comparison"""
        sizes = data['data_sizes']
        tenseal_error = [m.get('mean_absolute_error', 0) for m in data['tenseal']]
        paillier_error = [m.get('mean_absolute_error', 0) for m in data['paillier']]
        
        ax.plot(sizes, tenseal_error, 'o-', linewidth=2, markersize=6, label='TenSEAL (CKKS)')
        ax.plot(sizes, paillier_error, 's-', linewidth=2, markersize=6, label='Paillier (Exact)')
        ax.set_xlabel('Data Size (records)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Computation Error\n(Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_noise_growth(self, ax, data):
        """Plot noise growth analysis"""
        if data['tenseal'] and 'noise_growth' in data['tenseal'][0]:
            noise_data = data['tenseal'][0]['noise_growth']
            if noise_data:  # Check if noise_data is not empty
                chain_lengths = [ng['operation_chain_length'] for ng in noise_data]
                errors = [ng['mean_error'] for ng in noise_data]
                
                ax.plot(chain_lengths, errors, 'o-', linewidth=2, markersize=6, color='purple')
                ax.set_xlabel('Operation Chain Length')
                ax.set_ylabel('Mean Error')
                ax.set_title('CKKS Noise Growth\n(With Operation Chains)')
                ax.grid(True, alpha=0.3)
    
    def _plot_radar_chart(self, ax, data):
        """Create comprehensive radar chart"""
        if not data['tenseal'] or not data['paillier']:
            return
        
        # Normalize metrics for radar chart (higher is better for some, lower for others)
        categories = ['Speed', 'Memory', 'Size', 'Precision', 'Throughput']
        
        # Use data from medium-sized test for representative values
        mid_idx = len(data['data_sizes']) // 2
        t_metrics = data['tenseal'][mid_idx]
        p_metrics = data['paillier'][mid_idx]
        
        # Normalize values (this is simplified - in reality you'd want careful normalization)
        tenseal_values = [
            1/max(t_metrics.get('encryption_time', 1), 0.001) * 1000,  # Speed (inverse of time)
            1/max(t_metrics.get('encryption_memory_mb', 1), 0.001),    # Memory (inverse)
            1/max(t_metrics.get('ciphertext_size_bytes', 1)/1024, 0.001),  # Size (inverse)
            1/(t_metrics.get('mean_absolute_error', 1) + 0.001), # Precision (inverse error)
            t_metrics.get('throughput_ops_sec', 1)         # Throughput (direct)
        ]
        
        paillier_values = [
            1/max(p_metrics.get('encryption_time', 1), 0.001) * 1000,
            1/max(p_metrics.get('encryption_memory_mb', 1), 0.001),
            1/max(p_metrics.get('ciphertext_size_bytes', 1)/1024, 0.001),
            1/(p_metrics.get('mean_absolute_error', 1) + 0.001),
            p_metrics.get('throughput_ops_sec', 1)
        ]
        
        # Normalize to 0-1 scale for radar chart
        max_val = max(max(tenseal_values), max(paillier_values))
        if max_val > 0:
            tenseal_norm = [v/max_val for v in tenseal_values]
            paillier_norm = [v/max_val for v in paillier_values]
        else:
            tenseal_norm = [0] * len(tenseal_values)
            paillier_norm = [0] * len(paillier_values)
        
        # Complete the circle
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        tenseal_norm += tenseal_norm[:1]
        paillier_norm += paillier_norm[:1]
        angles += angles[:1]
        
        ax.plot(angles, tenseal_norm, 'o-', linewidth=2, label='TenSEAL (FHE)')
        ax.fill(angles, tenseal_norm, alpha=0.25)
        ax.plot(angles, paillier_norm, 's-', linewidth=2, label='Paillier (PHE)')
        ax.fill(angles, paillier_norm, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Scheme Comparison\n(Radar Chart)')
        ax.legend(loc='upper right')
        ax.grid(True)
    
    def _create_metrics_table(self, data):
        """Create a summary metrics table"""
        if not data['tenseal'] or not data['paillier']:
            return
        
        # Use medium-sized test for representative values
        mid_idx = len(data['data_sizes']) // 2
        t_metrics = data['tenseal'][mid_idx]
        p_metrics = data['paillier'][mid_idx]
        
        summary_data = {
            'Metric': [
                'Encryption Time (s)',
                'Memory Usage (MB)', 
                'Ciphertext Size (KB)',
                'Key Size (KB)',
                'Throughput (op/s)',
                'Mean Absolute Error',
                'Scheme Type',
                'Floating Point Support',
                'Unlimited Multiplications'
            ],
            'TenSEAL (CKKS)': [
                f"{t_metrics.get('encryption_time', 0):.4f}",
                f"{t_metrics.get('encryption_memory_mb', 0):.2f}",
                f"{t_metrics.get('ciphertext_size_bytes', 0)/1024:.2f}",
                f"{t_metrics.get('key_size_bytes', 0)/1024:.2f}",
                f"{t_metrics.get('throughput_ops_sec', 0):.2f}",
                f"{t_metrics.get('mean_absolute_error', 0):.6f}",
                "Fully Homomorphic",
                "✅ Yes",
                "✅ Yes"
            ],
            'Paillier': [
                f"{p_metrics.get('encryption_time', 0):.4f}",
                f"{p_metrics.get('encryption_memory_mb', 0):.2f}",
                f"{p_metrics.get('ciphertext_size_bytes', 0)/1024:.2f}",
                f"{p_metrics.get('key_size_bytes', 0)/1024:.2f}",
                f"{p_metrics.get('throughput_ops_sec', 0):.2f}",
                f"{p_metrics.get('mean_absolute_error', 0):.6f}",
                "Partially Homomorphic", 
                "❌ No",
                "❌ No"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        table_path = f'{self.results_dir}/metrics_summary.csv'
        df.to_csv(table_path, index=False)
        print(f"\n📋 Summary table saved to: {table_path}")
        
        # Print nice table to console
        print("\n" + "="*80)
        print("COMPREHENSIVE METRICS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        
        return df