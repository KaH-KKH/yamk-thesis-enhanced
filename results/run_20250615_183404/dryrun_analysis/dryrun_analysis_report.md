# Robot Framework Dryrun Analysis Report
**Generated:** 2025-06-15T19:58:29.213788

## Executive Summary

- **Total Models Analyzed:** 3
- **Total Test Files:** 3
- **Successful Tests:** 0 (0.0%)
- **Failed Tests:** 3
- **Best Performing Model:** Meta-Llama-3-8B-Instruct
- **Worst Performing Model:** Meta-Llama-3-8B-Instruct

## Model Performance Comparison

| Model | Total Tests | Successful | Failed | Success Rate | Avg Execution Time |
|-------|-------------|------------|--------|--------------|-------------------|
| Meta-Llama-3-8B-Instruct | 1 | 0 | 1 | 0.0% | 1.249s |
| mistral | 1 | 0 | 1 | 0.0% | 0.700s |
| Qwen2-7B-Instruct | 1 | 0 | 1 | 0.0% | 0.795s |

## Error Analysis

### Error Type Distribution

| Error Type | Occurrences | Percentage |
|------------|-------------|------------|
| missing_keyword | 7 | 100.0% |

### Top 10 Most Common Errors

1. **No keyword with name 'Open Browser [URL=https://the-internet.herokuapp.com] [Title=The Internet]' found** (occurred 1 times)
2. **No keyword with name 'The user should remain on the login page without accessing the secure area.' found** (occurred 1 times)
3. **No keyword with name 'Click Element With XPath=//a[@href='/login'] [Wait=1s]' found** (occurred 1 times)
4. **No keyword with name 'Expected Outcomes:' found** (occurred 1 times)
5. **No keyword with name 'Wait Until Text Present [Text=Welcome] [Timeout=10s]' found** (occurred 1 times)
6. **No keyword with name 'The system should display an error message indicating invalid credentials.' found** (occurred 1 times)
7. **No keyword with name '```' found** (occurred 1 times)

## Model-Specific Insights

### Meta-Llama-3-8B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: None

### mistral
- Success Rate: 0.0%
- Most Common Error Types: None

### Qwen2-7B-Instruct
- Success Rate: 0.0%
- Most Common Error Types: missing_keyword (7)

## Recommendations

Based on the dryrun analysis:

1. **Most Stable Models:** mistral, Qwen2-7B-Instruct - These models consistently generate syntactically correct Robot Framework tests.
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
