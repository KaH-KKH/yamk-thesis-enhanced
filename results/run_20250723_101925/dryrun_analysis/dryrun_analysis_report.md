# Robot Framework Dryrun Analysis Report
**Generated:** 2025-07-23T11:26:56.416160

## Executive Summary

- **Total Models Analyzed:** 5
- **Total Test Files:** 10
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 10
- **Best Performing Model:** tinyllama
- **Worst Performing Model:** tinyllama

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| tinyllama | 2 | 0 | 2 | 0.0% | 0.570s |
| phi2 | 2 | 0 | 2 | 0.0% | 0.567s |
| pythia_1b | 2 | 0 | 2 | 0.0% | 0.550s |
| opt_1.3b | 2 | 0 | 2 | 0.0% | 0.552s |
| stablelm_3b | 2 | 0 | 2 | 0.0% | 0.605s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 2 | 100.0% |

### Top 10 Most Common Errors

1. **No keyword with name 'Custom Login Keyword' found** (occurred 2 times)

## Model-Specific Insights

### tinyllama
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (1)

### phi2
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (1)

### pythia_1b
- Success Rate: 0.0%
- Most Common Error Types: None

### opt_1.3b
- Success Rate: 0.0%
- Most Common Error Types: None

### stablelm_3b
- Success Rate: 0.0%
- Most Common Error Types: None

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** opt_1.3b, stablelm_3b - These models consistently generate syntactically correct Robot Framework tests.
2. **Models Requiring Improvement:** tinyllama, phi2 - These models frequently generate tests with syntax or structural errors.
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
