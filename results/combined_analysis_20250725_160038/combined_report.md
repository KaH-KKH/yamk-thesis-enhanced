# Combined LLM Evaluation Results
Generated: 2025-07-25 16:00:41

## Executive Summary

### Model Rankings

| Rank | Model | Composite Score | Runs | Best Features |
|------|-------|-----------------|------|---------------|
| 1 | stablelm_3b | 0.830 | 1 | Completeness, Test Validity |
| 2 | Meta-Llama-3-8B-Instruct | 0.819 | 1 | Completeness, Test Validity |
| 3 | tinyllama | 0.809 | 1 | Completeness, Test Validity |
| 4 | gemma_7b_it_4bit | 0.809 | 1 | Completeness, Test Validity |
| 5 | Falcon3-7B-Base | 0.748 | 1 | Completeness, Test Validity |
| 6 | phi2 | 0.734 | 1 | Completeness, Test Validity |
| 7 | mistral | 0.721 | 1 | Completeness, Test Validity |
| 8 | Qwen2-7B-Instruct | 0.696 | 1 | Completeness, Test Validity |
| 9 | opt_1.3b | 0.649 | 1 | Completeness, Test Validity |
| 10 | pythia_1b | 0.552 | 1 | Completeness, Test Validity |

## Detailed Metrics

### Raw Metrics Table

| Model | Completeness | Test Validity | Gen Time (s) | Memory (MB) | LLM Score |
|-------|--------------|---------------|--------------|-------------|-----------|
| stablelm_3b | 1.00 | 1.00 | 82.7 | 47.4 | 10.0 |
| Meta-Llama-3-8B-Instruct | 1.00 | 1.00 | 761.4 | -286.3 | 10.0 |
| tinyllama | 1.00 | 1.00 | 88.4 | 1421.4 | 8.3 |
| gemma_7b_it_4bit | 1.00 | 1.00 | 1295.9 | -185.5 | 10.0 |
| Falcon3-7B-Base | 1.00 | 1.00 | 2778.5 | -129.1 | 9.7 |
| phi2 | 1.00 | 1.00 | 426.5 | 9711.5 | 10.0 |
| mistral | 1.00 | 1.00 | 1333.9 | 897.5 | 10.0 |
| Qwen2-7B-Instruct | 1.00 | 1.00 | 1023.9 | -65.2 | 10.0 |
| opt_1.3b | 1.00 | 1.00 | 65.1 | 71.1 | 3.6 |
| pythia_1b | 1.00 | 1.00 | 73.5 | 23.3 | 3.6 |

## Visualizations

- `radar_chart.html` - Multi-dimensional comparison (interactive)
- `scatter_plot.png` - Quality vs Performance trade-off
- `comparison_heatmap.png` - Pairwise win/loss matrix
- `raw_metrics.csv` - Raw data for further analysis
- `normalized_metrics.csv` - Normalized data (0-1 scale)

## Recommendations

1. **Best Overall**: stablelm_3b - Highest composite score (0.830)
2. **Best Quality**: stablelm_3b - Highest completeness score (1.00)
3. **Fastest**: opt_1.3b - Lowest generation time (65.1s)
4. **Most Efficient**: opt_1.3b - Best quality/speed balance

## Notes

- Composite score is the average of all normalized metrics
- LLM scores are normalized to 0-1 range (original scale: 0-10)
- Generation time and memory usage are inverted for normalization (lower is better)
- Rankings are based on the composite score
