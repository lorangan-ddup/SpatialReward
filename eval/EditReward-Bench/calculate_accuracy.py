import os
import json
import argparse
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with 'pip install datasets' to use HF datasets.")


PROMPT_FOLLOWING = "prompt_following"
CONSISTENCY = "consistency"
OVERALL = "overall"
SCORE_CATEGORIES = [PROMPT_FOLLOWING, CONSISTENCY, OVERALL]


# Default HuggingFace dataset
DEFAULT_HF_DATASET = "EditScore/EditReward-Bench"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True, help="Path to the result JSON file")
    parser.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET, help="HuggingFace dataset name (default: EditScore/EditReward-Bench)")
    return parser.parse_args()


def load_benchmark_from_hf(dataset_name):
    """Load benchmark metadata from HuggingFace dataset"""
    if not HF_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    print(f"Loading benchmark from HuggingFace: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    
    # Convert HF dataset to benchmark format
    benchmark_dataset = []
    for item in dataset:
        benchmark_dataset.append({
            "key": item["key"],  # [key1, key2]
            "task_type": item["task_type"],
            "dimension": item["dimension"]
        })
    
    print(f"Loaded {len(benchmark_dataset)} benchmark samples from HuggingFace")
    return benchmark_dataset


def load_results(result_file):
    """Load results from JSON file"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {result_file}")
    return results


def calculate_statistics(results, benchmark_dataset):
    """
    Calculate statistics based on the EditScore benchmark format - optimized with parallel processing
    Each sample in benchmark has 2 keys, and we compare their scores ONLY on the dimension specified in the benchmark
    """
    
    # Group results by task_type and dimension
    task_results = defaultdict(lambda: defaultdict(list))
    
    # Pre-allocate lists for better performance
    missing_keys = []
    
    # Process samples in parallel for faster computation
    def process_benchmark_sample(sample):
        """Process a single benchmark sample - parallelizable"""
        key1, key2 = sample["key"]
        task_type = sample["task_type"]
        dimension = sample["dimension"]  # The dimension to compare on
        
        # Get scores for both outputs
        if key1 not in results or key2 not in results:
            return None, (key1, key2)  # Missing keys
        
        scores1 = results[key1]["scores"]
        scores2 = results[key2]["scores"]
        
        # IMPORTANT: Only compare on the dimension specified in the benchmark
        # This is the correct logic matching the official EditScore implementation
        if dimension not in scores1 or dimension not in scores2:
            return None, (key1, key2)  # Missing dimension
        
        score1 = scores1[dimension]
        score2 = scores2[dimension]
        
        # Record: 1 if correct (score1 > score2), 0 otherwise
        correct = 1 if score1 > score2 else 0
        
        sample_result = {
            "correct": correct,
            "score1": score1,
            "score2": score2,
            "key1": key1,
            "key2": key2,
        }
        
        return (task_type, dimension, sample_result), None
    
    # Parallel processing with ThreadPoolExecutor
    print("Calculating statistics with parallel workers...")
    with ThreadPoolExecutor(max_workers=min(16, len(benchmark_dataset))) as executor:
        futures = [executor.submit(process_benchmark_sample, sample) 
                  for sample in benchmark_dataset]
        
        for future in as_completed(futures):
            result, missing = future.result()
            if missing:
                missing_keys.append(missing)
                print(f"Warning: Missing results for keys {missing[0]} or {missing[1]}")
            elif result:
                task_type, dimension, sample_result = result
                task_results[task_type][dimension].append(sample_result)
    
    if missing_keys:
        print(f"Total missing key pairs: {len(missing_keys)}")
    
    # Calculate accuracy for each task_type and dimension using numpy
    accuracies = defaultdict(dict)
    all_scores = defaultdict(list)
    
    for task_type in task_results:
        for dim in SCORE_CATEGORIES:
            results_list = task_results[task_type][dim]
            if len(results_list) == 0:
                accuracies[task_type][dim] = 0.0
                continue
            
            # Vectorized accuracy calculation
            correct_array = np.array([r["correct"] for r in results_list])
            accuracy = float(np.mean(correct_array))
            accuracies[task_type][dim] = accuracy
            
            # Collect all scores for overall statistics
            all_scores[dim].extend([r["score1"] for r in results_list])
            all_scores[dim].extend([r["score2"] for r in results_list])
    
    return accuracies, all_scores, task_results


def print_results(accuracies, all_scores):
    """Print results in a formatted table"""
    
    # Get all task types
    task_types = sorted(accuracies.keys())
    
    # Calculate averages
    avg_accuracies = {}
    for dim in SCORE_CATEGORIES:
        scores = [accuracies[task][dim] for task in task_types if dim in accuracies[task]]
        avg_accuracies[dim] = np.mean(scores) if scores else 0.0
    
    # Print header
    print("\n" + "="*100)
    print("EditScore Benchmark Results")
    print("="*100)
    
    # Print per-task results
    print("\nPer-Task Accuracies:")
    print("-"*100)
    print(f"{'Task Type':<25} {'Prompt Following':>15} {'Consistency':>15} {'Overall':>15}")
    print("-"*100)
    
    for task in task_types:
        pf = accuracies[task].get(PROMPT_FOLLOWING, 0.0)
        cons = accuracies[task].get(CONSISTENCY, 0.0)
        overall = accuracies[task].get(OVERALL, 0.0)
        print(f"{task:<25} {pf:>15.3f} {cons:>15.3f} {overall:>15.3f}")
    
    print("-"*100)
    print(f"{'Average':<25} {avg_accuracies[PROMPT_FOLLOWING]:>15.3f} {avg_accuracies[CONSISTENCY]:>15.3f} {avg_accuracies[OVERALL]:>15.3f}")
    print("-"*100)
    
    # Print grouped results
    groups = {
        'object': ['subject-add', 'subject-remove', 'subject-replace'],
        'appearance': ['color_alter', 'material_alter', 'style_change', 'tone_transfer'],
        'scene': ['background_change', 'extract'],
        'advanced': ['ps_human', 'text_change', 'motion_change', 'compose'],
    }
    
    print("\nGrouped Accuracies:")
    print("-"*100)
    print(f"{'Group':<25} {'Prompt Following':>15} {'Consistency':>15} {'Overall':>15}")
    print("-"*100)
    
    for group_name, group_tasks in groups.items():
        # Filter tasks that exist in our results
        valid_tasks = [t for t in group_tasks if t in accuracies]
        
        if not valid_tasks:
            continue
        
        pf_scores = [accuracies[t].get(PROMPT_FOLLOWING, 0.0) for t in valid_tasks]
        cons_scores = [accuracies[t].get(CONSISTENCY, 0.0) for t in valid_tasks]
        overall_scores = [accuracies[t].get(OVERALL, 0.0) for t in valid_tasks]
        
        pf_mean = np.mean(pf_scores)
        cons_mean = np.mean(cons_scores)
        overall_mean = np.mean(overall_scores)
        
        print(f"{group_name:<25} {pf_mean:>15.3f} {cons_mean:>15.3f} {overall_mean:>15.3f}")
    
    print("-"*100)
    
    # Print score statistics
    print("\nScore Statistics (0-10 scale):")
    print("-"*100)
    print(f"{'Dimension':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-"*100)
    
    for dim in SCORE_CATEGORIES:
        scores = all_scores[dim]
        if len(scores) == 0:
            continue
        print(f"{dim:<25} {np.min(scores):>10.3f} {np.max(scores):>10.3f} {np.mean(scores):>10.3f} {np.std(scores):>10.3f}")
    
    print("-"*100)
    print("\n")


def save_detailed_results(result_file, accuracies, all_scores, task_results):
    """Save detailed results to a summary file"""
    output_file = result_file.replace('.json', '_summary.json')
    
    summary = {
        "accuracies": {task: dict(accs) for task, accs in accuracies.items()},
        "score_statistics": {
            dim: {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            } for dim, scores in all_scores.items() if len(scores) > 0
        },
        "average_accuracies": {
            dim: float(np.mean([accuracies[task].get(dim, 0.0) for task in accuracies]))
            for dim in SCORE_CATEGORIES
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Detailed results saved to {output_file}")


def main():
    args = parse_args()
    
    # Load results
    results = load_results(args.result_file)
    
    # Load benchmark from HuggingFace
    benchmark_dataset = load_benchmark_from_hf(args.hf_dataset)
    
    # Calculate statistics
    accuracies, all_scores, task_results = calculate_statistics(results, benchmark_dataset)
    
    print_results(accuracies, all_scores)
    
    # Save detailed results
    save_detailed_results(args.result_file, accuracies, all_scores, task_results)


if __name__ == "__main__":
    main()

