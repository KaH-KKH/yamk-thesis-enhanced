# Combined LLM Evaluation Results
Generated: 2025-06-20 20:55:29

## Executive Summary

### Model Rankings

| Rank | Model | Composite Score | Runs | Best Features |
|------|-------|-----------------|------|---------------|
| 1 | mistral | 0.798 | 2 | Completeness, Test Validity |
| 2 | gemma_7b_it_4bit | 0.787 | 2 | Completeness, Test Validity |
| 3 | Falcon3-7B-Base | 0.728 | 2 | Completeness, Test Validity |
| 4 | Meta-Llama-3-8B-Instruct | 0.724 | 2 | Completeness, Test Validity |
| 5 | Qwen2-7B-Instruct | 0.499 | 2 | Completeness, Test Validity |

## Detailed Metrics

### Raw Metrics Table

| Model | Completeness | Test Validity | Gen Time (s) | Memory (MB) | LLM Score |
|-------|--------------|---------------|--------------|-------------|-----------|
| mistral | 1.00 | 1.00 | 1248.0 | 520.5 | 10.0 |
| gemma_7b_it_4bit | 1.00 | 1.00 | 1402.0 | 254.5 | 9.9 |
| Falcon3-7B-Base | 1.00 | 1.00 | 3156.4 | 66.1 | 9.2 |
| Meta-Llama-3-8B-Instruct | 1.00 | 1.00 | 1431.4 | 1582.9 | 10.0 |
| Qwen2-7B-Instruct | 1.00 | 1.00 | 1414.4 | 822.8 | 9.9 |

## Visualizations

- `radar_chart.html` - Multi-dimensional comparison (interactive)
- `scatter_plot.png` - Quality vs Performance trade-off
- `comparison_heatmap.png` - Pairwise win/loss matrix
- `raw_metrics.csv` - Raw data for further analysis
- `normalized_metrics.csv` - Normalized data (0-1 scale)

## Recommendations

1. **Best Overall**: mistral - Highest composite score (0.798)
2. **Best Quality**: gemma_7b_it_4bit - Highest completeness score (1.00)
3. **Fastest**: mistral - Lowest generation time (1248.0s)
4. **Most Efficient**: mistral - Best quality/speed balance

## Notes

- Composite score is the average of all normalized metrics
- LLM scores are normalized to 0-1 range (original scale: 0-10)
- Generation time and memory usage are inverted for normalization (lower is better)
- Rankings are based on the composite score
