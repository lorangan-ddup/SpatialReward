import os
import json
import argparse
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True, help="Path to the result JSON file")
    parser.add_argument("--benchmark_file", type=str, required=True, help="Path to the MMRB2 edit.json file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save evaluation results")
    args = parser.parse_args()
    return args


def load_results(result_file):
    """Load inference results"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} inference results from {result_file}")
    return results


def load_benchmark(benchmark_file):
    """Load MMRB2 benchmark - supports both HuggingFace dataset and JSON file"""
    import os
    
    # Check if it's a HuggingFace dataset path
    if not os.path.exists(benchmark_file) and '/' in benchmark_file:
        from datasets import load_dataset
        print(f"Loading benchmark from HuggingFace dataset: {benchmark_file}")
        hf_dataset = load_dataset(benchmark_file, 'edit', split='test')
        pairs = list(hf_dataset)
        print(f"Loaded {len(pairs)} pairs from HuggingFace dataset")
    elif os.path.isfile(benchmark_file):
        print(f"Loading benchmark from JSON file: {benchmark_file}")
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pairs = data["pairs"]
        print(f"Loaded {len(pairs)} pairs from JSON file")
    else:
        raise ValueError(f"Invalid benchmark_file: {benchmark_file}")
    
    return pairs


def calculate_accuracy(results, benchmark_pairs):
    """
    Calculate accuracy by comparing model predictions with ground truth
    
    For each pair:
    - Get scores for response A and response B
    - Determine which one has higher overall score
    - Compare with ground truth (pair["chosen"])
    - Calculate accuracy (overall + by image count + by source)
    """
    
    total_pairs = 0
    correct_predictions = 0
    missing_pairs = []
    per_source_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    per_image_count_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for pair in benchmark_pairs:
        pair_id = pair["id"]
        ground_truth = pair["chosen"]  # "A" or "B"
        source = pair.get("prompt_source", "unknown")
        
        # Count number of input images
        num_input_images = sum(1 for item in pair["prompt_content"] if item[0] == "image")
        
        key_a = f"{pair_id}_A"
        key_b = f"{pair_id}_B"
        
        # Check if both results exist
        if key_a not in results or key_b not in results:
            missing_pairs.append(pair_id)
            continue
        
        # Get overall scores
        score_a = results[key_a]["scores"]["overall"]
        score_b = results[key_b]["scores"]["overall"]
        
        # Determine model's prediction
        if score_a > score_b:
            predicted = "A"
        elif score_b > score_a:
            predicted = "B"
        else:
            # Tie - mark as "tie" (will be incorrect since ground_truth is always A or B)
            predicted = "tie"
        
        # Check if prediction matches ground truth
        is_correct = (predicted == ground_truth)
        
        total_pairs += 1
        per_source_stats[source]["total"] += 1
        per_image_count_stats[num_input_images]["total"] += 1
        
        if is_correct:
            correct_predictions += 1
            per_source_stats[source]["correct"] += 1
            per_image_count_stats[num_input_images]["correct"] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0.0
    
    # Calculate per-source accuracy
    per_source_accuracy = {}
    for source, stats in per_source_stats.items():
        per_source_accuracy[source] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    # Calculate per-image-count accuracy
    per_image_count_accuracy = {}
    for count, stats in per_image_count_stats.items():
        per_image_count_accuracy[count] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    return {
        "overall_accuracy": overall_accuracy,
        "total_pairs": total_pairs,
        "correct_predictions": correct_predictions,
        "missing_pairs": len(missing_pairs),
        "per_source_accuracy": per_source_accuracy,
        "per_source_stats": dict(per_source_stats),
        "per_image_count_accuracy": per_image_count_accuracy,
        "per_image_count_stats": dict(per_image_count_stats),
        "missing_pair_ids": missing_pairs,
    }


def print_results(evaluation_results):
    """Print evaluation results in a formatted table"""
    
    print("\n" + "="*80)
    print("MMRB2 Image Editing Evaluation Results")
    print("="*80)
    
    # Overall results
    print("\nOverall Performance:")
    print("-"*80)
    print(f"Total Pairs:         {evaluation_results['total_pairs']}")
    print(f"Correct Predictions: {evaluation_results['correct_predictions']}")
    print(f"Missing Pairs:       {evaluation_results['missing_pairs']}")
    print(f"Overall Accuracy:    {evaluation_results['overall_accuracy']:.2%}")
    print("-"*80)
    
    # Per-image-count breakdown (NEW!)
    if evaluation_results.get('per_image_count_accuracy'):
        print("\n📊 Per-Image-Count Accuracy:")
        print("-"*80)
        print(f"{'Input Images':<20} {'Accuracy':>12} {'Correct':>10} {'Total':>10} {'Type':>15}")
        print("-"*80)
        
        # Sort by number of images
        for count in sorted(evaluation_results['per_image_count_accuracy'].keys()):
            acc = evaluation_results['per_image_count_accuracy'][count]
            stats = evaluation_results['per_image_count_stats'][count]
            task_type = "Single-Image" if count == 1 else f"Multi-Image ({count})"
            print(f"{count:<20} {acc:>11.2%} {stats['correct']:>10} {stats['total']:>10} {task_type:>15}")
        
        print("-"*80)
    
    # Per-source breakdown
    if evaluation_results.get('per_source_accuracy'):
        print("\n📁 Per-Source Accuracy:")
        print("-"*80)
        print(f"{'Source':<30} {'Accuracy':>12} {'Correct':>10} {'Total':>10}")
        print("-"*80)
        
        # Sort by source name
        for source in sorted(evaluation_results['per_source_accuracy'].keys()):
            acc = evaluation_results['per_source_accuracy'][source]
            stats = evaluation_results['per_source_stats'][source]
            print(f"{source:<30} {acc:>11.2%} {stats['correct']:>10} {stats['total']:>10}")
        
        print("-"*80)
    
    # Missing pairs warning
    if evaluation_results['missing_pairs'] > 0:
        print(f"\n⚠️  Warning: {evaluation_results['missing_pairs']} pairs are missing results")
        if len(evaluation_results['missing_pair_ids']) <= 10:
            print(f"Missing pair IDs: {', '.join(evaluation_results['missing_pair_ids'])}")
    
    print("\n" + "="*80 + "\n")


def save_results(evaluation_results, output_file):
    """Save evaluation results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to {output_file}")


def main():
    args = parse_args()
    
    # Load data
    results = load_results(args.result_file)
    benchmark_pairs = load_benchmark(args.benchmark_file)
    
    # Calculate accuracy
    print("\nCalculating accuracy...")
    evaluation_results = calculate_accuracy(results, benchmark_pairs)
    
    # Print results
    print_results(evaluation_results)
    
    # Save results
    if args.output_file:
        save_results(evaluation_results, args.output_file)
    else:
        # Auto-generate output filename
        output_file = args.result_file.replace('.json', '_evaluation.json')
        save_results(evaluation_results, output_file)


if __name__ == "__main__":
    main()

