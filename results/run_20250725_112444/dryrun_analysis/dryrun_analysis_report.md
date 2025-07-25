# Robot Framework Dryrun Analysis Report
**Generated:** 2025-07-25T14:27:20.339166

## Executive Summary

- **Total Models Analyzed:** 5
- **Total Test Files:** 10
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 10
- **Best Performing Model:** mistral
- **Worst Performing Model:** mistral

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| mistral | 2 | 0 | 2 | 0.0% | 0.565s |
| gemma_7b_it_4bit | 2 | 0 | 2 | 0.0% | 0.314s |
| Falcon3-7B-Base | 2 | 0 | 2 | 0.0% | 0.324s |
| Meta-Llama-3-8B-Instruct | 2 | 0 | 2 | 0.0% | 0.523s |
| Qwen2-7B-Instruct | 2 | 0 | 2 | 0.0% | 0.524s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 21 | 84.0% |
| other | 4 | 16.0% |

### Top 10 Most Common Errors

1. **No keyword with name '```' found** (occurred 2 times)
2. **Suite 'Basic Auth PASS use case' contains no tests or tasks.** (occurred 2 times)
3. **No keyword with name 'Go To Page' found** (occurred 1 times)
4. **No keyword with name 'Test Unauthorized Access:' found** (occurred 1 times)
5. **No keyword with name '[Documentation]' found** (occurred 1 times)
6. **No keyword with name 'https://the-internet.herokuapp.com/basic_auth' found** (occurred 1 times)
7. **No keyword with name 'Verify user cannot access protected content without credentials' found** (occurred 1 times)
8. **No keyword with name 'Maximize Browser Window' found** (occurred 1 times)
9. **No keyword with name 'Test Steps ***' found** (occurred 1 times)
10. **No keyword with name 'new_window=yes' found** (occurred 1 times)

## Model-Specific Insights

### mistral
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (17)

### gemma_7b_it_4bit
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (4), other (2)

### Falcon3-7B-Base
- Success Rate: 0.0%
- Most Common Error Types: other (2)

### Meta-Llama-3-8B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: None

### Qwen2-7B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: None

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** Meta-Llama-3-8B-Instruct, Qwen2-7B-Instruct - These models consistently generate syntactically correct Robot Framework tests.
2. **Models Requiring Improvement:** mistral, gemma_7b_it_4bit - These models frequently generate tests with syntax or structural errors.
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
