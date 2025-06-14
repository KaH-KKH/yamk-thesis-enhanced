"""
RF Agent - Generates Robot Framework test cases from use cases
Fixed version without pydantic-ai HuggingFaceModel
"""

import asyncio
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re

from pydantic import BaseModel, Field
from loguru import logger

from ..models.model_loader import ModelLoader
from ..tools.rf_tools import RobotFrameworkTools
from ..utils.file_handler import FileHandler


class TestCase(BaseModel):
    """Robot Framework test case structure"""
    name: str = Field(default="Test Case", description="Test case name")
    documentation: str = Field(default="", description="Test documentation")
    tags: List[str] = Field(default_factory=list, description="Test tags")
    setup: Optional[List[str]] = Field(default=None, description="Test setup steps")
    steps: List[Dict[str, str]] = Field(default_factory=list, description="Test steps")
    teardown: Optional[List[str]] = Field(default=None, description="Test teardown steps")
    variables: Optional[Dict[str, str]] = Field(default=None, description="Test variables")


class RFAgent:
    """Agent for generating Robot Framework test cases from use cases"""
    
    def __init__(self, model_name: str, config_path: str = "configs/config.yaml"):
        self.model_name = model_name
        self.config = FileHandler.load_yaml(config_path)
        self.model_loader = ModelLoader(config_path)
        
        # Load model
        logger.info(f"Loading model for RF Agent: {model_name}")
        self.model = self.model_loader.load_model(model_name)

        # Tyhjennä muisti mallin latauksen jälkeen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get system prompt from config
        self.system_prompt = self.config["agents"]["rf_agent"]["system_prompt"]
        
        # Initialize tools
        self.rf_tools = RobotFrameworkTools()
        
        logger.info(f"RF Agent initialized with model: {model_name}")
    
    async def generate_test_case(self, use_case_file: str) -> str:
        """Generate Robot Framework test case from use case file"""
        try:
            # Read use case
            use_case_text = FileHandler.read_text_file(use_case_file)
            
            # Parse use case if it's JSON
            if use_case_file.endswith('.json'):
                use_case_data = json.loads(use_case_text)
                # Convert to text format for processing
                use_case_text = self._format_use_case_for_processing(use_case_data)
            
            # Create prompt
            prompt = f"""{self.system_prompt}

Target URL: {self.config["paths"]["base_url"]}

Use Case:
{use_case_text}

Generate a Robot Framework test case that:
1. Uses Browser library keywords
2. Tests the functionality described in the use case
3. Includes proper setup and teardown
4. Has clear step-by-step test actions
5. Uses appropriate selectors and waits

Format the test case with:
- Test case name
- Documentation
- Tags
- Setup steps
- Main test steps
- Teardown steps
"""

            # Generate response
            logger.info(f"Generating test case for: {use_case_file}")
            start_time = datetime.now()
            
            response = self.model_loader.generate(
                self.model_name,
                prompt,
                max_new_tokens=1024,
                temperature=0.7
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Test case generated in {generation_time:.2f} seconds")
            
            # Parse response into TestCase object
            test_case = self._parse_response_to_test_case(response, use_case_text)
            
            # Convert to Robot Framework format
            robot_content = self._convert_to_robot_format(test_case)
            
            return robot_content
        
        except Exception as e:
            logger.error(f"Error generating test case for {use_case_file}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _format_use_case_for_processing(self, use_case_data: Dict[str, Any]) -> str:
        """Format use case data for processing"""
        lines = []
        
        if "title" in use_case_data:
            lines.append(f"USE CASE: {use_case_data['title']}")
        
        if "id" in use_case_data:
            lines.append(f"ID: {use_case_data['id']}")
            
        if "actors" in use_case_data:
            lines.extend(["", "ACTORS:"])
            lines.extend([f"- {actor}" for actor in use_case_data.get('actors', [])])
        
        if "preconditions" in use_case_data:
            lines.extend(["", "PRECONDITIONS:"])
            lines.extend([f"{i+1}. {pc}" for i, pc in enumerate(use_case_data.get('preconditions', []))])
            
        if "main_flow" in use_case_data:
            lines.extend(["", "MAIN FLOW:"])
            lines.extend([f"{i+1}. {step}" for i, step in enumerate(use_case_data.get('main_flow', []))])
            
        if "postconditions" in use_case_data:
            lines.extend(["", "POSTCONDITIONS:"])
            lines.extend([f"{i+1}. {pc}" for i, pc in enumerate(use_case_data.get('postconditions', []))])
        
        return "\n".join(lines)
    
    def _parse_response_to_test_case(self, response: str, use_case_text: str) -> TestCase:
        """Parse LLM response into TestCase object"""
        # Initialize test case with defaults
        test_case_data = {
            "name": "Test Login Functionality",
            "documentation": "Automated test for login functionality",
            "tags": ["login", "smoke"],
            "setup": None,
            "steps": [],
            "teardown": None,
            "variables": None
        }
        
        # Extract test scenarios from use case
        scenarios = self.rf_tools.extract_test_scenarios(use_case_text)
        
        # Parse the response for Robot Framework elements
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_upper = line.upper()
            
            # Detect sections
            if "TEST CASE:" in line_upper or "TEST NAME:" in line_upper:
                name_match = re.search(r'[:\-]\s*(.+)$', line)
                if name_match:
                    test_case_data["name"] = name_match.group(1).strip()
                    
            elif "DOCUMENTATION:" in line_upper or "DESCRIPTION:" in line_upper:
                doc_match = re.search(r'[:\-]\s*(.+)$', line)
                if doc_match:
                    test_case_data["documentation"] = doc_match.group(1).strip()
                    
            elif "TAGS:" in line_upper:
                tags_match = re.search(r'[:\-]\s*(.+)$', line)
                if tags_match:
                    test_case_data["tags"] = [t.strip() for t in tags_match.group(1).split(',')]
                    
            elif "SETUP:" in line_upper or "TEST SETUP:" in line_upper:
                current_section = "setup"
                test_case_data["setup"] = []
                
            elif "STEPS:" in line_upper or "TEST STEPS:" in line_upper or "MAIN STEPS:" in line_upper:
                current_section = "steps"
                
            elif "TEARDOWN:" in line_upper or "TEST TEARDOWN:" in line_upper:
                current_section = "teardown"
                test_case_data["teardown"] = []
                
            elif current_section and line:
                # Process content based on section
                content = re.sub(r'^[-•*\d]+[\s.)]*', '', line).strip()
                
                if content:
                    if current_section == "setup" and test_case_data["setup"] is not None:
                        test_case_data["setup"].append(content)
                        
                    elif current_section == "teardown" and test_case_data["teardown"] is not None:
                        test_case_data["teardown"].append(content)
                        
                    elif current_section == "steps":
                        # Convert action to Robot Framework keyword
                        keyword_info = self.rf_tools.map_action_to_keyword(
                            content, 
                            self.config["paths"]["base_url"]
                        )
                        test_case_data["steps"].append(keyword_info)
        
        # If no steps were parsed, create default steps based on use case
        if not test_case_data["steps"]:
            test_case_data["steps"] = self._generate_default_steps(use_case_text)
        
        # Set default variables
        test_case_data["variables"] = {
            "BASE_URL": self.config["paths"]["base_url"],
            "BROWSER": "chromium",
            "HEADLESS": "false"
        }
        
        return TestCase(**test_case_data)
    
    def _generate_default_steps(self, use_case_text: str) -> List[Dict[str, str]]:
        """Generate default test steps based on use case"""
        steps = []
        
        # Look for login-related content
        if "login" in use_case_text.lower():
            steps.extend([
                {"keyword": "New Browser", "args": "${BROWSER}    headless=${HEADLESS}"},
                {"keyword": "New Page", "args": "${BASE_URL}/login"},
                {"keyword": "Type Text", "args": "id=username    tomsmith"},
                {"keyword": "Type Text", "args": "id=password    SuperSecretPassword!"},
                {"keyword": "Click", "args": "css=button[type='submit']"},
                {"keyword": "Wait For Elements State", "args": "text=You logged into a secure area!    visible"},
                {"keyword": "Take Screenshot", "args": ""}
            ])
        else:
            # Generic steps
            steps.extend([
                {"keyword": "New Browser", "args": "${BROWSER}    headless=${HEADLESS}"},
                {"keyword": "New Page", "args": "${BASE_URL}"},
                {"keyword": "Log", "args": "Executing test steps"},
                {"keyword": "Take Screenshot", "args": ""}
            ])
        
        return steps
    
    def _convert_to_robot_format(self, test_case: TestCase) -> str:
        """Convert TestCase to Robot Framework format"""
        lines = []
        
        # Settings section
        lines.extend([
            "*** Settings ***",
            f"Documentation    {test_case.documentation}",
            "Library          Browser",
            "Library          OperatingSystem",
            "Library          DateTime",
            "Test Setup       Setup Browser",
            "Test Teardown    Close Browser",
            ""
        ])
        
        # Add tags if present
        if test_case.tags:
            lines.insert(3, f"Test Tags        {' '.join(test_case.tags)}")
        
        # Variables section
        lines.extend([
            "*** Variables ***",
            "${BASE_URL}      https://the-internet.herokuapp.com",
            "${BROWSER}       chromium",
            "${HEADLESS}      false",
            "${TIMEOUT}       10s",
            ""
        ])
        
        # Add custom variables if any
        if test_case.variables:
            for name, value in test_case.variables.items():
                if name not in ["BASE_URL", "BROWSER", "HEADLESS"]:
                    lines.insert(-1, f"${{{name}}}    {value}")
        
        # Test Cases section
        lines.extend([
            "*** Test Cases ***",
            test_case.name,
            f"    [Documentation]    {test_case.documentation}",
        ])
        
        # Add tags to test case
        if test_case.tags:
            lines.append(f"    [Tags]    {' '.join(test_case.tags)}")
        
        # Add custom setup if specified
        if test_case.setup:
            lines.append("    [Setup]    Run Keywords")
            for step in test_case.setup:
                lines.append(f"    ...    {step}")
        
        # Add test steps
        lines.append("    ")
        for step in test_case.steps:
            if isinstance(step, dict):
                keyword = step.get("keyword", "Log")
                args = step.get("args", "")
                if args:
                    lines.append(f"    {keyword}    {args}")
                else:
                    lines.append(f"    {keyword}")
            else:
                lines.append(f"    {step}")
        
        # Add custom teardown if specified
        if test_case.teardown:
            lines.append("    ")
            lines.append("    [Teardown]    Run Keywords")
            for step in test_case.teardown:
                lines.append(f"    ...    {step}")
        
        # Keywords section
        lines.extend([
            "",
            "*** Keywords ***",
            "Setup Browser",
            "    New Browser    ${BROWSER}    headless=${HEADLESS}",
            "    Set Browser Timeout    ${TIMEOUT}",
            "    New Context    viewport={'width': 1280, 'height': 720}",
            "",
            "Close Browser",
            "    Take Screenshot    fullPage=True",
            "    Close Browser    ALL",
            "",
            "Login To Application",
            "    [Arguments]    ${username}    ${password}",
            "    Go To    ${BASE_URL}/login",
            "    Type Text    id=username    ${username}",
            "    Type Text    id=password    ${password}",
            "    Click    css=button[type='submit']",
            "    Wait For Elements State    css=.flash    visible    timeout=${TIMEOUT}",
            ""
        ])
        
        return "\n".join(lines)
    
    async def batch_generate(self, use_case_dir: str, output_dir: str) -> Dict[str, Any]:
        """Generate test cases for all use cases in directory"""
        use_case_path = Path(use_case_dir)
        output_path = Path(output_dir) / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all use case files
        use_case_files = list(use_case_path.glob("*.txt")) + list(use_case_path.glob("*.json"))
        logger.info(f"Found {len(use_case_files)} use case files")

        # KORJAUS: Sample size rajoitus - käytä oikeaa muuttujan nimeä
        if not self.config.get("evaluation", {}).get("full_evaluation", True):
            sample_size = self.config.get("evaluation", {}).get("sample_size", 3)
            if len(use_case_files) > sample_size:  # KORJATTU: use_case_files
                use_case_files = use_case_files[:sample_size]  # KORJATTU: use_case_files
                logger.info(f"Limited to {sample_size} files for evaluation")
        
        results = []
        for uc_file in use_case_files:
            try:
                # Generate test case
                robot_content = await self.generate_test_case(str(uc_file))
                
                # Save test case
                output_file = output_path / f"{uc_file.stem}.robot"
                FileHandler.save_text_file(robot_content, str(output_file))
                
                results.append({
                    "use_case": uc_file.name,
                    "test_case": output_file.name,
                    "status": "success"
                })
                
                logger.success(f"Generated test case: {output_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {uc_file.name}: {str(e)}")
                results.append({
                    "use_case": uc_file.name,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save generation report
        report = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(use_case_files),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        }
        
        report_file = output_path / "generation_report.json"
        FileHandler.save_json(report, str(report_file))
        
        return report
    
    def __del__(self):
        """Cleanup when agent is destroyed"""
        # Käytä model_loaderin unload_model metodia
        if hasattr(self, 'model_loader') and hasattr(self, 'model_name'):
            self.model_loader.unload_model(self.model_name)
        
        # Varmista että muisti tyhjennetään
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU memory cleared for model: {self.model_name}")


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RF Agent - Generate Robot Framework tests")
    parser.add_argument("--model", required=True, help="Model name (mistral, gemma_7b_it_4bit)")
    parser.add_argument("--input", required=True, help="Input use case file or directory")
    parser.add_argument("--output", default="data/test_cases", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Process directory of files")
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = RFAgent(args.model)
        
        if args.batch:
            # Batch processing
            report = await agent.batch_generate(args.input, args.output)
            print(f"\nGeneration complete!")
            print(f"Successful: {report['successful']}")
            print(f"Failed: {report['failed']}")
        else:
            # Single file processing
            robot_content = await agent.generate_test_case(args.input)
            
            output_path = Path(args.output) / args.model
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            input_name = Path(args.input).stem
            output_file = output_path / f"{input_name}.robot"
            
            FileHandler.save_text_file(robot_content, str(output_file))
            
            print(f"\nTest case generated successfully!")
            print(f"Output: {output_file}")
            
            # Print first few lines of the generated test
            print(f"\n--- Generated Test Case Preview ---")
            lines = robot_content.split('\n')[:20]
            for line in lines:
                print(line)
            print("...")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())