# Robot Framework Dryrun Analysis Report
**Generated:** 2025-07-22T20:34:47.194429

## Executive Summary

- **Total Models Analyzed:** 1
- **Total Test Files:** 0
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 0
- **Best Performing Model:** opt_1.3b
- **Worst Performing Model:** opt_1.3b

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| opt_1.3b | 0 | 0 | 0 | 0.0% | 0.000s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|

### Top 10 Most Common Errors


## Model-Specific Insights

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** opt_1.3b - These models consistently generate syntactically correct Robot Framework tests.

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
