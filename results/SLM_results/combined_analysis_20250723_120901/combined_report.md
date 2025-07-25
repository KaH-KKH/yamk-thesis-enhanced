# Combined LLM Evaluation Results
Generated: 2025-07-23 12:09:04

## Executive Summary

### Model Rankings

| Rank | Model | Composite Score | Runs | Best Features |
|------|-------|-----------------|------|---------------|
| 1 | stablelm_3b | 0.843 | 1 | Completeness, Test Validity |
| 2 | tinyllama | 0.818 | 1 | Completeness, Test Validity |
| 3 | opt_1.3b | 0.660 | 1 | Completeness, Test Validity |
| 4 | phi2 | 0.651 | 1 | Completeness, Test Validity |
| 5 | pythia_1b | 0.553 | 1 | Completeness, Test Validity |

## Detailed Metrics

### Raw Metrics Table

| Model | Completeness | Test Validity | Gen Time (s) | Memory (MB) | LLM Score |
|-------|--------------|---------------|--------------|-------------|-----------|
| stablelm_3b | 1.00 | 1.00 | 82.7 | 47.4 | 10.0 |
| tinyllama | 1.00 | 1.00 | 88.4 | 1421.4 | 8.3 |
| opt_1.3b | 1.00 | 1.00 | 65.1 | 71.1 | 3.6 |
| phi2 | 1.00 | 1.00 | 426.5 | 9711.5 | 10.0 |
| pythia_1b | 1.00 | 1.00 | 73.5 | 23.3 | 3.6 |

## Visualizations

- `radar_chart.html` - Multi-dimensional comparison (interactive)
- `scatter_plot.png` - Quality vs Performance trade-off
- `comparison_heatmap.png` - Pairwise win/loss matrix
- `raw_metrics.csv` - Raw data for further analysis
- `normalized_metrics.csv` - Normalized data (0-1 scale)

## Recommendations

1. **Best Overall**: stablelm_3b - Highest composite score (0.843)
2. **Best Quality**: stablelm_3b - Highest completeness score (1.00)
3. **Fastest**: opt_1.3b - Lowest generation time (65.1s)
4. **Most Efficient**: opt_1.3b - Best quality/speed balance

## Notes

- Composite score is the average of all normalized metrics
- LLM scores are normalized to 0-1 range (original scale: 0-10)
- Generation time and memory usage are inverted for normalization (lower is better)
- Rankings are based on the composite score
