# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-20T17:21:21.521274

## Executive Summary

- **Total Models Analyzed:** 3
- **Total Test Files:** 6
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 6
- **Best Performing Model:** mistral
- **Worst Performing Model:** mistral

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| mistral | 2 | 0 | 2 | 0.0% | 0.570s |
| Qwen2-7B-Instruct | 2 | 0 | 2 | 0.0% | 0.537s |
| gemma_7b_it_4bit | 2 | 0 | 2 | 0.0% | 0.312s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 3 | 60.0% |
| other | 2 | 40.0% |

### Top 10 Most Common Errors

1. **Suite 'Basic Auth PASS use case' contains no tests or tasks.** (occurred 1 times)
2. **Error in file '/home/kkhalttunen/yamk_thesis_enhanced/data/test_cases/gemma_7b_it_4bit/Basic_Auth_PASS_use_case.robot' on line 17: Unrecognized section header '**'. Valid sections: 'Settings', 'Variables', 'Test Cases', 'Tasks', 'Keywords' and 'Comments'.** (occurred 1 times)
3. **No keyword with name 'Navigate to the website' found** (occurred 1 times)
4. **No keyword with name '```' found** (occurred 1 times)
5. **No keyword with name 'Please note that this is an example, and the generated test case may need to be adjusted based on the specific requirements of the use case.**' found** (occurred 1 times)

## Model-Specific Insights

### mistral
- Success Rate: 0.0%
- Most Common Error Types: None

### Qwen2-7B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: None

### gemma_7b_it_4bit
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (3), other (2)

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** Qwen2-7B-Instruct, gemma_7b_it_4bit - These models consistently generate syntactically correct Robot Framework tests.
2. **Models Requiring Improvement:** mistral - These models frequently generate tests with syntax or structural errors.
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
