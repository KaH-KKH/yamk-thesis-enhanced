# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-20T13:48:04.518844

## Executive Summary

- **Total Models Analyzed:** 2
- **Total Test Files:** 4
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 4
- **Best Performing Model:** Meta-Llama-3-8B-Instruct
- **Worst Performing Model:** Meta-Llama-3-8B-Instruct

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| Meta-Llama-3-8B-Instruct | 2 | 0 | 2 | 0.0% | 0.869s |
| mistral | 2 | 0 | 2 | 0.0% | 0.540s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|

### Top 10 Most Common Errors


## Model-Specific Insights

### Meta-Llama-3-8B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: None

### mistral
- Success Rate: 0.0%
- Most Common Error Types: None

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** mistral - These models consistently generate syntactically correct Robot Framework tests.

## Technical Details

### Dryrun Command Used
```bash
robot --dryrun --output NONE --report NONE --log NONE <test_file>
```

### Error Categories
- **syntax_error**: Invalid Robot Framework syntax
- **missing_keyword**: Referenced keyword not found
- **invalid_argument**: Wrong number or format of arguments
- **missing_library**: Required library not imported
- **invalid_variable**: Variable syntax or reference errors
- **structural_error**: Test structure issues (empty tests, setup/teardown problems)
