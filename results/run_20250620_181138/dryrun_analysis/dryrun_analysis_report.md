# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-20T20:32:11.915746

## Executive Summary

- **Total Models Analyzed:** 3
- **Total Test Files:** 6
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 6
- **Best Performing Model:** Meta-Llama-3-8B-Instruct
- **Worst Performing Model:** Meta-Llama-3-8B-Instruct

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| Meta-Llama-3-8B-Instruct | 2 | 0 | 2 | 0.0% | 0.456s |
| Falcon3-7B-Base | 2 | 0 | 2 | 0.0% | 0.572s |
| gemma_7b_it_4bit | 2 | 0 | 2 | 0.0% | 0.328s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 14 | 77.8% |
| other | 4 | 22.2% |

### Top 10 Most Common Errors

1. **Suite 'Basic Auth PASS use case' contains no tests or tasks.** (occurred 2 times)
2. **Error in file '/home/kkhalttunen/yamk_thesis_enhanced/data/test_cases/Meta-Llama-3-8B-Instruct/Basic_Auth_PASS_use_case.robot' on line 17: Unrecognized section header '** UC-LOGIN-001: Basic Authentication'. Valid sections: 'Settings', 'Variables', 'Test Cases', 'Tasks', 'Keywords' and 'Comments'.** (occurred 1 times)
3. **No keyword with name 'Note: You can run this test case using Robot Framework, and it should pass if the cancel button functionality is working correctly. If the test case fails, it indicates that there is an issue with the cancel button functionality. You can modify the test case as needed to better test the functionality.' found** (occurred 1 times)
4. **No keyword with name 'Click Button' found** (occurred 1 times)
5. **No keyword with name 'Keywords Used:' found** (occurred 1 times)
6. **No keyword with name 'https://the-internet.herokuapp.com/basic_auth' found** (occurred 1 times)
7. **No keyword with name 'Verify Page Contains' found** (occurred 1 times)
8. **No keyword with name 'chrome' found** (occurred 1 times)
9. **No keyword with name 'Wait Until Page Contains' found** (occurred 1 times)
10. **No keyword with name 'Maximize Browser Window' found** (occurred 1 times)

## Model-Specific Insights

### Meta-Llama-3-8B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (8), other (2)

### Falcon3-7B-Base
- Success Rate: 0.0%
- Most Common Error Types: None

### gemma_7b_it_4bit
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (6), other (2)

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** Falcon3-7B-Base, gemma_7b_it_4bit - These models consistently generate syntactically correct Robot Framework tests.
2. **Models Requiring Improvement:** Meta-Llama-3-8B-Instruct - These models frequently generate tests with syntax or structural errors.
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
