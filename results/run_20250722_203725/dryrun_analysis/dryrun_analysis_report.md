# Robot Framework Dryrun Analysis Report
**Generated:** 2025-07-22T20:55:18.771004

## Executive Summary

- **Total Models Analyzed:** 1
- **Total Test Files:** 2
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 2
- **Best Performing Model:** stablelm_3b
- **Worst Performing Model:** stablelm_3b

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| stablelm_3b | 2 | 0 | 2 | 0.0% | 0.772s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 17 | 100.0% |

### Top 10 Most Common Errors

1. **No keyword with name 'Click Button: Sign Out' found** (occurred 1 times)
2. **No keyword with name '[Test Teardown]' found** (occurred 1 times)
3. **No keyword with name 'Assert Header: Basic Auth' found** (occurred 1 times)
4. **No keyword with name 'Press Sign In button' found** (occurred 1 times)
5. **No keyword with name 'Input username and password' found** (occurred 1 times)
6. **No keyword with name 'If error message, Repeat Test Actions until successful' found** (occurred 1 times)
7. **No keyword with name 'Check the header' found** (occurred 1 times)
8. **No keyword with name 'Click Button: Sign In' found** (occurred 1 times)
9. **No keyword with name 'Check the two headers' found** (occurred 1 times)
10. **No keyword with name 'The browser page opens successfully' found** (occurred 1 times)

## Model-Specific Insights

### stablelm_3b
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (17)

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** stablelm_3b - These models consistently generate syntactically correct Robot Framework tests.
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
