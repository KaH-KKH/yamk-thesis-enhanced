# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-14T14:07:20.709329

## Executive Summary

- **Total Models Analyzed:** 5
- **Total Test Files:** 5
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 5
- **Best Performing Model:** Meta-Llama-3-8B-Instruct
- **Worst Performing Model:** Meta-Llama-3-8B-Instruct

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| Meta-Llama-3-8B-Instruct | 1 | 0 | 1 | 0.0% | 1.352s |
| Qwen2-7B-Instruct | 1 | 0 | 1 | 0.0% | 0.581s |
| Falcon3-7B-Base | 1 | 0 | 1 | 0.0% | 0.617s |
| gemma_7b_it_4bit | 1 | 0 | 1 | 0.0% | 0.124s |
| mistral | 1 | 0 | 1 | 0.0% | 0.538s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 3 | 60.0% |
| other | 2 | 40.0% |

### Top 10 Most Common Errors

1. **No keyword with name 'This test case uses Browser library keywords, tests the secure login functionality, and includes proper setup and teardown. The test case name, documentation, tags, setup, main test steps, and teardown are clearly defined. The test case uses appropriate selectors and waits, and includes a timeout for the Wait Until Page Contains keyword. The test case is executable and can be run to verify the secure login functionality of the system.' found** (occurred 1 times)
2. **No keyword with name 'https://the-internet.herokuapp.com/login' found** (occurred 1 times)
3. **No keyword with name '```' found** (occurred 1 times)
4. **Suite 'Example Login Use Case' contains no tests or tasks.** (occurred 1 times)
5. **Error in file '/home/kkhalttunen/yamk_thesis_enhanced/data/test_cases/gemma_7b_it_4bit/example_login_use_case.robot' on line 17: Unrecognized section header '**'. Valid sections: 'Settings', 'Variables', 'Test Cases', 'Tasks', 'Keywords' and 'Comments'.** (occurred 1 times)

## Model-Specific Insights

### Meta-Llama-3-8B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (3)

### Qwen2-7B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: None

### Falcon3-7B-Base
- Success Rate: 0.0%
- Most Common Error Types: None

### gemma_7b_it_4bit
- Success Rate: 0.0%
- Most Common Error Types: other (2)

### mistral
- Success Rate: 0.0%
- Most Common Error Types: None

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** gemma_7b_it_4bit, mistral - These models consistently generate syntactically correct Robot Framework tests.
2. **Models Requiring Improvement:** Meta-Llama-3-8B-Instruct, Qwen2-7B-Instruct - These models frequently generate tests with syntax or structural errors.
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
