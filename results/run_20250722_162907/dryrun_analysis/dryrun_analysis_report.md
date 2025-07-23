# Robot Framework Dryrun Analysis Report
**Generated:** 2025-07-22T16:44:37.490498

## Executive Summary

- **Total Models Analyzed:** 1
- **Total Test Files:** 2
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 2
- **Best Performing Model:** tinyllama
- **Worst Performing Model:** tinyllama

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| tinyllama | 2 | 0 | 2 | 0.0% | 0.887s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 1 | 100.0% |

### Top 10 Most Common Errors

1. **No keyword with name 'Custom Login Keyword' found** (occurred 1 times)

## Model-Specific Insights

### tinyllama
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (1)

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** tinyllama - These models consistently generate syntactically correct Robot Framework tests.
4. **Missing Keywords:** Models are using non-existent keywords. Ensure training/prompts include valid Browser library keywords.

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
