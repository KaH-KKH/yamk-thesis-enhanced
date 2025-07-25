# Combined LLM Evaluation Results
Generated: 2025-07-25 15:57:50

## Executive Summary

### Model Rankings

| Rank | Model | Composite Score | Runs | Best Features |
|------|-------|-----------------|------|---------------|
| 1 | gemma_7b_it_4bit | 0.815 | 1 | Completeness, Test Validity |
| 2 | Meta-Llama-3-8B-Instruct | 0.731 | 1 | Completeness, Test Validity |
| 3 | mistral | 0.662 | 1 | Completeness, Test Validity |
| 4 | Qwen2-7B-Instruct | 0.626 | 1 | Completeness, Test Validity |
| 5 | Falcon3-7B-Base | 0.599 | 1 | Completeness, Test Validity |

## Detailed Metrics

### Raw Metrics Table

| Model | Completeness | Test Validity | Gen Time (s) | Memory (MB) | LLM Score |
|-------|--------------|---------------|--------------|-------------|-----------|
| gemma_7b_it_4bit | 1.00 | 1.00 | 1295.9 | -185.5 | 10.0 |
| Meta-Llama-3-8B-Instruct | 1.00 | 1.00 | 761.4 | -286.3 | 10.0 |
| mistral | 1.00 | 1.00 | 1333.9 | 897.5 | 10.0 |
| Qwen2-7B-Instruct | 1.00 | 1.00 | 1023.9 | -65.2 | 10.0 |
| Falcon3-7B-Base | 1.00 | 1.00 | 2778.5 | -129.1 | 9.7 |

## Visualizations

- `radar_chart.html` - Multi-dimensional comparison (interactive)
- `scatter_plot.png` - Quality vs Performance trade-off
- `comparison_heatmap.png` - Pairwise win/loss matrix
- `raw_metrics.csv` - Raw data for further analysis
- `normalized_metrics.csv` - Normalized data (0-1 scale)

## Recommendations

1. **Best Overall**: gemma_7b_it_4bit - Highest composite score (0.815)
2. **Best Quality**: Qwen2-7B-Instruct - Highest completeness score (1.00)
3. **Fastest**: Meta-Llama-3-8B-Instruct - Lowest generation time (761.4s)
4. **Most Efficient**: Meta-Llama-3-8B-Instruct - Best quality/speed balance

## Notes

- Composite score is the average of all normalized metrics
- LLM scores are normalized to 0-1 range (original scale: 0-10)
- Generation time and memory usage are inverted for normalization (lower is better)
- Rankings are based on the composite score
