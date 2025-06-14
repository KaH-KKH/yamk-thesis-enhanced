# src/evaluators/llm_evaluator.py
"""
LLM-based evaluation for generated content
"""

import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger
import json

from ..models.model_loader import ModelLoader
from ..utils.file_handler import FileHandler


class LLMEvaluator:
    """LLM-based evaluation for use cases and test cases"""
    
    def __init__(self, evaluator_model: str = "mistral", config_path: str = "configs/config.yaml"):
        self.config = FileHandler.load_yaml(config_path)
        self.model_loader = ModelLoader(config_path)
        self.evaluator_model = evaluator_model
        
        # Evaluation prompts
        self.use_case_eval_prompt = """
You are an expert software requirements analyst. Evaluate the following use case based on these criteria:

1. **Completeness** (0-10): Are all necessary sections present (actors, preconditions, main flow, postconditions)?
2. **Clarity** (0-10): Is the use case clear and unambiguous?
3. **Testability** (0-10): Can this use case be easily converted to test cases?
4. **Technical Accuracy** (0-10): Are the technical details correct and feasible?
5. **Structure** (0-10): Is the format consistent and well-organized?

Use Case to evaluate:
{content}

Provide your evaluation in JSON format:
{{
    "completeness": <score>,
    "clarity": <score>,
    "testability": <score>,
    "technical_accuracy": <score>,
    "structure": <score>,
    "overall_score": <average score>,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "suggestions": ["improvement suggestions"]
}}
"""

        self.test_case_eval_prompt = """
You are a Robot Framework expert. Evaluate the following test case based on these criteria:

1. **Syntax Correctness** (0-10): Is the Robot Framework syntax correct?
2. **Test Coverage** (0-10): Does it adequately test the functionality?
3. **Best Practices** (0-10): Does it follow RF best practices (waits, selectors, error handling)?
4. **Maintainability** (0-10): Is the test maintainable and reusable?
5. **Executability** (0-10): Will this test actually run successfully?

Test Case to evaluate:
{content}

Provide your evaluation in JSON format:
{{
    "syntax_correctness": <score>,
    "test_coverage": <score>,
    "best_practices": <score>,
    "maintainability": <score>,
    "executability": <score>,
    "overall_score": <average score>,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "suggestions": ["improvement suggestions"]
}}
"""

    async def evaluate_use_case(self, use_case_content: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single use case using LLM"""
        prompt = self.use_case_eval_prompt.format(content=use_case_content)
        
        try:
            # Generate evaluation
            response = self.model_loader.generate(
                self.evaluator_model,
                prompt,
                max_new_tokens=512,
                temperature=0.3  # Lower temperature for consistent evaluation
            )
            
            # Parse JSON response
            evaluation = self._parse_json_response(response)
            evaluation["evaluated_model"] = model_name
            evaluation["content_type"] = "use_case"
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating use case: {e}")
            return self._get_default_evaluation("use_case", model_name)
    
    async def evaluate_test_case(self, test_case_content: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single test case using LLM"""
        prompt = self.test_case_eval_prompt.format(content=test_case_content)
        
        try:
            # Generate evaluation
            response = self.model_loader.generate(
                self.evaluator_model,
                prompt,
                max_new_tokens=512,
                temperature=0.3
            )
            
            # Parse JSON response
            evaluation = self._parse_json_response(response)
            evaluation["evaluated_model"] = model_name
            evaluation["content_type"] = "test_case"
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating test case: {e}")
            return self._get_default_evaluation("test_case", model_name)
    
    async def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models using LLM-based evaluation"""
        comparison_prompt = """
You are an expert in LLM evaluation. Compare the following models based on their generated outputs:

{model_summaries}

Provide a comprehensive comparison including:
1. Ranking of models from best to worst
2. Key strengths and weaknesses of each model
3. Recommendations for which model to use in different scenarios

Format your response as JSON:
{{
    "ranking": [
        {{"rank": 1, "model": "model_name", "score": 0-100, "reason": "why it's ranked here"}}
    ],
    "model_analysis": {{
        "model_name": {{
            "strengths": ["list"],
            "weaknesses": ["list"],
            "best_use_cases": ["scenarios where this model excels"]
        }}
    }},
    "recommendations": {{
        "best_overall": "model_name",
        "best_for_clarity": "model_name",
        "best_for_completeness": "model_name",
        "best_for_test_generation": "model_name",
        "reasoning": "explanation of recommendations"
    }}
}}
"""
        
        # Prepare model summaries
        summaries = []
        for model, results in model_results.items():
            summary = f"Model: {model}\n"
            if "llm_evaluation" in results:
                eval_data = results["llm_evaluation"]
                summary += f"Average Use Case Score: {eval_data.get('avg_use_case_score', 0):.1f}/10\n"
                summary += f"Average Test Case Score: {eval_data.get('avg_test_case_score', 0):.1f}/10\n"
            summaries.append(summary)
        
        prompt = comparison_prompt.format(model_summaries="\n\n".join(summaries))
        
        response = self.model_loader.generate(
            self.evaluator_model,
            prompt,
            max_new_tokens=1024,
            temperature=0.3
        )
        
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Try to parse the entire response
                return json.loads(response)
        except:
            logger.warning("Failed to parse JSON response, using fallback")
            return self._extract_structured_data(response)
    
    def _extract_structured_data(self, response: str) -> Dict[str, Any]:
        """Extract structured data from non-JSON response"""
        # Fallback parsing logic
        result = {
            "overall_score": 5.0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        # Try to extract scores
        import re
        score_patterns = {
            "completeness": r"completeness[:\s]+(\d+)",
            "clarity": r"clarity[:\s]+(\d+)",
            "testability": r"testability[:\s]+(\d+)",
            "technical_accuracy": r"technical[:\s]+(\d+)",
            "structure": r"structure[:\s]+(\d+)"
        }
        
        scores = []
        for key, pattern in score_patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                result[key] = score
                scores.append(score)
        
        if scores:
            result["overall_score"] = sum(scores) / len(scores)
        
        return result
    
    def _get_default_evaluation(self, content_type: str, model_name: str) -> Dict[str, Any]:
        """Return default evaluation when parsing fails"""
        return {
            "evaluated_model": model_name,
            "content_type": content_type,
            "overall_score": 0,
            "error": "Failed to evaluate"
        }