# Robot Framework Dryrun Analysis Report
**Generated:** 2025-07-22T17:14:49.226489

## Executive Summary

- **Total Models Analyzed:** 1
- **Total Test Files:** 2
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 2
- **Best Performing Model:** pythia_1b
- **Worst Performing Model:** pythia_1b

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| pythia_1b | 2 | 0 | 2 | 0.0% | 0.758s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|

### Top 10 Most Common Errors


## Model-Specific Insights

### pythia_1b
- Success Rate: 0.0%
- Most Common Error Types: None

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** pythia_1b - These models consistently generate syntactically correct Robot Framework tests.

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
