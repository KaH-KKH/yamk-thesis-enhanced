# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-20T15:22:13.428631

## Executive Summary

- **Total Models Analyzed:** 2
- **Total Test Files:** 4
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 4
- **Best Performing Model:** Qwen2-7B-Instruct
- **Worst Performing Model:** Qwen2-7B-Instruct

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| Qwen2-7B-Instruct | 2 | 0 | 2 | 0.0% | 0.639s |
| Falcon3-7B-Base | 2 | 0 | 2 | 0.0% | 0.542s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 8 | 100.0% |

### Top 10 Most Common Errors

1. **No keyword with name 'The URL of the page is "https://the-internet.herokuapp.com/"' found** (occurred 1 times)
2. **No keyword with name '```' found** (occurred 1 times)
3. **No keyword with name 'These additional steps help to ensure that the test case not only verifies the main functionality but also checks other aspects of the page to confirm that the system is behaving as expected.' found** (occurred 1 times)
4. **No keyword with name '- Open The Internet Site URL' found** (occurred 1 times)
5. **No keyword with name 'The title of the page is "The Internet Site"' found** (occurred 1 times)
6. **No keyword with name 'In this extended example, additional verification steps have been added to ensure that:' found** (occurred 1 times)
7. **No keyword with name '- Initialize Browser library' found** (occurred 1 times)
8. **No keyword with name '- Close the browser' found** (occurred 1 times)

## Model-Specific Insights

### Qwen2-7B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: None

### Falcon3-7B-Base
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (8)

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** Falcon3-7B-Base - These models consistently generate syntactically correct Robot Framework tests.
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
