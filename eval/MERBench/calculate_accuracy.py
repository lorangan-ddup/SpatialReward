#!/usr/bin/env python3
"""
Comprehensive Statistics for MERBench Results

Calculates:
- Exact Match Accuracy
- Kendall's Tau correlation
- Spearman's Rho correlation  
- NDCG (Normalized Discounted Cumulative Gain)
- Top-k Accuracy
- Position Error
- Error Pattern Analysis
- Confidence vs Accuracy
- Category Breakdowns
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict


def kendall_tau(x, y):
    """Manual Kendall's Tau implementation"""
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                concordant += 1
            elif (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                discordant += 1
    return (concordant - discordant) / (0.5 * n * (n - 1)) if n > 1 else 0


def spearman_rho(x, y):
    """Manual Spearman's rho implementation"""
    n = len(x)
    if n == 0:
        return 0
    d_squared = sum((x[i] - y[i])**2 for i in range(n))
    return 1 - (6 * d_squared) / (n * (n**2 - 1)) if n > 1 else 0


def calculate_ranking_metrics(pred_rank, gt_rank, scores=None):
    """Calculate comprehensive ranking metrics
    
    Args:
        pred_rank: Predicted ranking indices
        gt_rank: Ground truth ranking indices
        scores: Original scores (optional), used for strict tie-breaking
    """
    n = len(pred_rank)
    
    # 1. Exact match (Strict: ties are errors)
    exact_match = (list(pred_rank) == list(gt_rank))
    
    # Check for ties: if scores exist with duplicates and different GT ranks, mark as error
    if scores and len(set(scores)) < n:
        # Has ties
        exact_match = False
    
    # 2. Kendall's Tau
    tau = kendall_tau(pred_rank, gt_rank)
    
    # 3. Spearman correlation
    rho = spearman_rho(pred_rank, gt_rank)
    
    # 4. NDCG
    dcg = sum([(n - gt_rank[i]) / np.log2(pred_rank[i] + 2) for i in range(n)])
    idcg = sum([(n - i) / np.log2(i + 2) for i in range(n)])
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # 5. Top-k accuracy
    top1_acc = int(pred_rank[0] == 0)
    top2_acc = int(set(pred_rank[:min(2, n)]) == set(range(min(2, n)))) if n >= 2 else 0
    
    # 6. Position deviation
    position_error = np.mean([abs(pred_rank[i] - i) for i in range(n)])
    
    # 7. Error type
    error_type = detect_error_type(pred_rank, gt_rank) if not exact_match else None
    
    return {
        'exact_match': exact_match,
        'kendall_tau': tau,
        'spearman_rho': rho,
        'ndcg': ndcg,
        'top1_accuracy': top1_acc,
        'top2_accuracy': top2_acc,
        'position_error': position_error,
        'error_type': error_type
    }


def detect_error_type(pred_rank, gt_rank):
    """Detect error patterns"""
    n = len(pred_rank)
    
    # Full reversal
    if pred_rank == list(reversed(gt_rank)):
        return 'full_reverse'
    
    # Adjacent position swap
    swap_count = sum(1 for i in range(n-1) if abs(pred_rank[i] - gt_rank[i]) == 1)
    if swap_count >= n - 1:
        return 'adjacent_swap'
    
    # Top prediction error
    if pred_rank[0] != 0:
        return 'top_confused'
    
    # Bottom prediction error
    if pred_rank[-1] != n-1:
        return 'bottom_confused'
    
    return 'mixed'


def analyze_by_category(results):
    """Analyze by category"""
    category_stats = defaultdict(lambda: {
        'count': 0,
        'correct': 0,
        'tau_sum': 0,
        'ndcg_sum': 0,
        'top1_sum': 0,
        'error_types': defaultdict(int)
    })
    
    for pair_id, sample in results.items():
        if 'label' not in sample or 'pred_rank' not in sample or 'gt_rank' not in sample:
            continue
            
        category = sample['label']
        scores = sample.get('scores', [])
        metrics = calculate_ranking_metrics(sample['pred_rank'], sample['gt_rank'], scores=scores)
        
        stats = category_stats[category]
        stats['count'] += 1
        stats['correct'] += int(metrics['exact_match'])
        stats['tau_sum'] += metrics['kendall_tau']
        stats['ndcg_sum'] += metrics['ndcg']
        stats['top1_sum'] += metrics['top1_accuracy']
        
        if metrics['error_type']:
            stats['error_types'][metrics['error_type']] += 1
    
    # Calculate averages
    for category, stats in category_stats.items():
        count = stats['count']
        stats['accuracy'] = stats['correct'] / count if count > 0 else 0
        stats['avg_tau'] = stats['tau_sum'] / count if count > 0 else 0
        stats['avg_ndcg'] = stats['ndcg_sum'] / count if count > 0 else 0
        stats['avg_top1'] = stats['top1_sum'] / count if count > 0 else 0
    
    return dict(category_stats)


def analyze_by_pair_type(results):
    """Analyze by pair type"""
    pair_stats = defaultdict(lambda: {
        'count': 0,
        'correct': 0,
        'tau_sum': 0,
        'ndcg_sum': 0
    })
    
    for pair_id, sample in results.items():
        if 'pair_type' not in sample or 'pred_rank' not in sample or 'gt_rank' not in sample:
            continue
            
        pair_type = sample['pair_type']
        scores = sample.get('scores', [])
        metrics = calculate_ranking_metrics(sample['pred_rank'], sample['gt_rank'], scores=scores)
        
        stats = pair_stats[pair_type]
        stats['count'] += 1
        stats['correct'] += int(metrics['exact_match'])
        stats['tau_sum'] += metrics['kendall_tau']
        stats['ndcg_sum'] += metrics['ndcg']
    
    # Calculate averages
    for pair_type, stats in pair_stats.items():
        count = stats['count']
        stats['accuracy'] = stats['correct'] / count if count > 0 else 0
        stats['avg_tau'] = stats['tau_sum'] / count if count > 0 else 0
        stats['avg_ndcg'] = stats['ndcg_sum'] / count if count > 0 else 0
    
    return dict(pair_stats)


def overall_metrics(results):
    """Calculate overall metrics"""
    all_metrics = []
    for pair_id, sample in results.items():
        if 'pred_rank' not in sample or 'gt_rank' not in sample:
            continue
        scores = sample.get('scores', [])
        metrics = calculate_ranking_metrics(sample['pred_rank'], sample['gt_rank'], scores=scores)
        all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    return {
        'total_samples': len(all_metrics),
        'exact_match_acc': np.mean([m['exact_match'] for m in all_metrics]),
        'avg_kendall_tau': np.mean([m['kendall_tau'] for m in all_metrics]),
        'avg_spearman_rho': np.mean([m['spearman_rho'] for m in all_metrics]),
        'avg_ndcg': np.mean([m['ndcg'] for m in all_metrics]),
        'top1_accuracy': np.mean([m['top1_accuracy'] for m in all_metrics]),
        'top2_accuracy': np.mean([m['top2_accuracy'] for m in all_metrics]),
        'avg_position_error': np.mean([m['position_error'] for m in all_metrics]),
    }


def load_results(result_path):
    """Load results from JSON file"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive MERBench Statistics")
    parser.add_argument("--result_file", type=str, required=True,
                       help="Path to inference results JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save comprehensive report (JSON)")
    parser.add_argument("--output_md", type=str, default=None,
                       help="Path to save markdown report")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to data directory (optional)")
    args = parser.parse_args()
    
    # Load results
    print("=" * 60)
    print("MERBench Comprehensive Statistics")
    print("=" * 60)
    print(f"Loading: {args.result_file}")
    
    results = load_results(args.result_file)
    print(f"Loaded {len(results)} samples\n")
    
    # Calculate metrics
    overall = overall_metrics(results)
    by_pair = analyze_by_pair_type(results)
    by_category = analyze_by_category(results)
    
    # Print overall metrics
    print("=" * 60)
    print("Overall Performance")
    print("=" * 60)
    print(f"Total Samples:     {overall.get('total_samples', 0)}")
    print(f"Exact Match Acc:   {overall.get('exact_match_acc', 0):.3f}")
    print(f"Kendall Tau:       {overall.get('avg_kendall_tau', 0):.3f}")
    print(f"Spearman Rho:      {overall.get('avg_spearman_rho', 0):.3f}")
    print(f"NDCG:              {overall.get('avg_ndcg', 0):.3f}")
    print(f"Top-1 Accuracy:    {overall.get('top1_accuracy', 0):.3f}")
    print(f"Top-2 Accuracy:    {overall.get('top2_accuracy', 0):.3f}")
    print(f"Avg Position Err:  {overall.get('avg_position_error', 0):.3f}")
    
    # Print by pair type
    print("\n" + "=" * 60)
    print("Performance by N-Pair Type")
    print("=" * 60)
    for pair_type in sorted(by_pair.keys()):
        stats = by_pair[pair_type]
        print(f"\n{pair_type}-Pair:")
        print(f"  Count:     {stats['count']}")
        print(f"  Accuracy:  {stats['accuracy']:.3f}")
        print(f"  Avg Tau:   {stats['avg_tau']:.3f}")
        print(f"  Avg NDCG:  {stats['avg_ndcg']:.3f}")
    
    # Print by category
    print("\n" + "=" * 60)
    print("Performance by Category")
    print("=" * 60)
    for category in sorted(by_category.keys()):
        stats = by_category[category]
        print(f"\n{category}:")
        print(f"  Count:     {stats['count']}")
        print(f"  Accuracy:  {stats['accuracy']:.3f}")
        print(f"  Avg Tau:   {stats['avg_tau']:.3f}")
        print(f"  Avg NDCG:  {stats['avg_ndcg']:.3f}")
        print(f"  Top-1 Acc: {stats['avg_top1']:.3f}")
        
        if stats['error_types']:
            print(f"  Error Types:")
            for error_type, count in sorted(stats['error_types'].items(), key=lambda x: -x[1]):
                print(f"    {error_type}: {count}")
    
    # Save JSON report
    if args.output:
        report = {
            'overall': overall,
            'by_pair_type': by_pair,
            'by_category': by_category
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ JSON report saved to: {args.output}")
    
    # Save Markdown report
    if args.output_md:
        md_content = generate_markdown_report(overall, by_pair, by_category)
        with open(args.output_md, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✅ Markdown report saved to: {args.output_md}")
    
    print("\n" + "=" * 60)


def generate_markdown_report(overall, by_pair, by_category):
    """Generate markdown report"""
    md = "# MERBench Comprehensive Statistics Report\n\n"
    
    md += "## Overall Performance\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Total Samples | {overall.get('total_samples', 0)} |\n"
    md += f"| Exact Match Accuracy | {overall.get('exact_match_acc', 0):.3f} |\n"
    md += f"| Kendall's Tau | {overall.get('avg_kendall_tau', 0):.3f} |\n"
    md += f"| Spearman's Rho | {overall.get('avg_spearman_rho', 0):.3f} |\n"
    md += f"| NDCG | {overall.get('avg_ndcg', 0):.3f} |\n"
    md += f"| Top-1 Accuracy | {overall.get('top1_accuracy', 0):.3f} |\n"
    md += f"| Top-2 Accuracy | {overall.get('top2_accuracy', 0):.3f} |\n"
    md += f"| Avg Position Error | {overall.get('avg_position_error', 0):.3f} |\n\n"
    
    md += "## Performance by N-Pair Type\n\n"
    md += "| Pair Type | Count | Accuracy | Kendall τ | NDCG |\n"
    md += "|-----------|-------|----------|-----------|------|\n"
    for pair_type in sorted(by_pair.keys()):
        stats = by_pair[pair_type]
        md += f"| {pair_type}-Pair | {stats['count']} | {stats['accuracy']:.3f} | {stats['avg_tau']:.3f} | {stats['avg_ndcg']:.3f} |\n"
    md += "\n"
    
    md += "## Performance by Category\n\n"
    md += "| Category | Count | Accuracy | Kendall τ | NDCG | Top-1 Acc |\n"
    md += "|----------|-------|----------|-----------|------|----------|\n"
    for category in sorted(by_category.keys()):
        stats = by_category[category]
        md += f"| {category} | {stats['count']} | {stats['accuracy']:.3f} | {stats['avg_tau']:.3f} | {stats['avg_ndcg']:.3f} | {stats['avg_top1']:.3f} |\n"
    md += "\n"
    
    return md


if __name__ == "__main__":
    main()
