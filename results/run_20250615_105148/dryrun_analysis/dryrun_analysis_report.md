# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-15T11:34:54.329445

## Executive Summary

- **Total Models Analyzed:** 2
- **Total Test Files:** 2
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 2
- **Best Performing Model:** mistral
- **Worst Performing Model:** mistral

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| mistral | 1 | 0 | 1 | 0.0% | 0.585s |
| Falcon3-7B-Base | 1 | 0 | 1 | 0.0% | 0.553s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|

### Top 10 Most Common Errors


## Model-Specific Insights

### mistral
- Success Rate: 0.0%
- Most Common Error Types: None

### Falcon3-7B-Base
- Success Rate: 0.0%
- Most Common Error Types: None

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** Falcon3-7B-Base - These models consistently generate syntactically correct Robot Framework tests.

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
