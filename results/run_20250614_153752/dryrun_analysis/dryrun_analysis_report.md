# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-14T16:28:03.399147

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
| mistral | 1 | 0 | 1 | 0.0% | 1.837s |
| gemma_7b_it_4bit | 1 | 0 | 1 | 0.0% | 0.144s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| other | 2 | 100.0% |

### Top 10 Most Common Errors

1. **Suite 'Example Login Use Case' contains no tests or tasks.** (occurred 1 times)
2. **Error in file '/home/kkhalttunen/yamk_thesis_enhanced/data/test_cases/gemma_7b_it_4bit/example_login_use_case.robot' on line 17: Unrecognized section header '**'. Valid sections: 'Settings', 'Variables', 'Test Cases', 'Tasks', 'Keywords' and 'Comments'.** (occurred 1 times)

## Model-Specific Insights

### mistral
- Success Rate: 0.0%
- Most Common Error Types: None

### gemma_7b_it_4bit
- Success Rate: 0.0%
- Most Common Error Types: other (2)

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** gemma_7b_it_4bit - These models consistently generate syntactically correct Robot Framework tests.

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
