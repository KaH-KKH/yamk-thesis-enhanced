"""
RF Agent - Generates Robot Framework test cases from use cases
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.huggingface import HuggingFaceModel
from loguru import logger

from ..models.model_loader import ModelLoader
from ..tools.rf_tools import RobotFrameworkTools
from ..utils.file_handler import FileHandler


class TestCase(BaseModel):
    """Robot Framework test case structure"""
    name: str = Field(description="Test case name")
    documentation: str = Field(description="Test documentation")
    tags: List[str] = Field(description="Test tags")
    setup: Optional[List[str]] = Field(default=None, description="Test setup steps")
    steps: List[Dict[str, str]] = Field(description="Test steps with keywords and arguments")
    teardown: Optional[List[str]] = Field(default=None, description="Test teardown steps")
    variables: Optional[Dict[str, str]] = Field(default=None, description="Test variables")


class RFAgentContext(BaseModel):
    """Context for RF Agent"""
    use_case_text: str
    model_name: str
    base_url: str = "https://the-internet.herokuapp.com/"
    timestamp: datetime = Field(default_factory=datetime.now)


class RFAgent:
    """Agent for generating Robot Framework test cases from use cases"""
    
    def __init__(self, model_name: str, config_path: str = "configs/config.yaml"):
        self.model_name = model_name
        self.config = FileHandler.load_yaml(config_path)
        self.model_loader = ModelLoader(config_path)
        
        # Load model
        self.model = self.model_loader.load_model(model_name)
        
        # Get system prompt from config
        self.system_prompt = self.config["agents"]["rf_agent"]["system_prompt"]
        
        # Initialize tools
        self.rf_tools = RobotFrameworkTools()
        
        # Initialize Pydantic AI agent
        self.agent = Agent(
            model=HuggingFaceModel(
                model=self.model,
                tokenizer=self.model_loader.tokenizer
            ),
            result_type=TestCase,
            system_prompt=self.system_prompt,
            deps_type=RFAgentContext,
        )
        
        # Add tools
        self._register_tools()
        
        logger.info(f"RF Agent initialized with model: {model_name}")
    
    def _register_tools(self):
        """Register tools for the agent"""
        
        @self.agent.tool
        async def extract_test_scenarios(ctx: RunContext[RFAgentContext]) -> List[str]:
            """Extract test scenarios from use case"""
            use_case = ctx.deps.use_case_text
            return self.rf_tools.extract_test_scenarios(use_case)
        
        @self.agent.tool
        async def generate_browser_keywords(ctx: RunContext[RFAgentContext], action: str) -> Dict[str, str]:
            """Generate appropriate Browser library keywords for an action"""
            return self.rf_tools.map_action_to_keyword(action, ctx.deps.base_url)
        
        @self.agent.tool
        async def create_test_structure(ctx: RunContext[RFAgentContext]) -> Dict[str, Any]:
            """Create Robot Framework test structure"""
            return {
                "settings": self.rf_tools.generate_settings(),
                "variables": self.rf_tools.generate_variables(ctx.deps.base_url),
                "keywords": []
            }
    
    async def generate_test_case(self, use_case_file: str) -> str:
        """Generate Robot Framework test case from use case file"""
        # Read use case
        use_case_text = FileHandler.read_text_file(use_case_file)
        
        # Parse use case if it's JSON
        if use_case_file.endswith('.json'):
            use_case_data = json.loads(use_case_text)
            # Convert to text format for processing
            use_case_text = self._format_use_case_for_processing(use_case_data)
        
        # Create context
        context = RFAgentContext(
            use_case_text=use_case_text,
            model_name=self.model_name,
            base_url=self.config["paths"]["base_url"]
        )
        
        # Generate test case
        logger.info(f"Generating test case for: {use_case_file}")
        start_time = datetime.now()
        
        result = await self.agent.run(
            f"Generate a Robot Framework test case from the following use case:\n\n{use_case_text}",
            deps=context
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Test case generated in {generation_time:.2f} seconds")
        
        # Convert to Robot Framework format
        robot_content = self._convert_to_robot_format(result.data)
        
        return robot_content
    
    def _format_use_case_for_processing(self, use_case_data: Dict[str, Any]) -> str:
        """Format use case data for processing"""
        lines = []
        
        if "title" in use_case_data:
            lines.append(f"USE CASE: {use_case_data['title']}")
        
        if "actors" in use_case_data:
            lines.extend(["ACTORS:", *[f"- {actor}" for actor in use_case_data['actors']]])
        
        if "main_flow" in use_case_data:
            lines.extend(["MAIN FLOW:", *[f"{i+1}. {step}" for i, step in enumerate(use_case_data['main_flow'])]])
        
        return "\n".join(lines)
    
    def _convert_to_robot_format(self, test_case: TestCase) -> str:
        """Convert TestCase to Robot Framework format"""
        lines = []
        
        # Settings section
        lines.extend([
            "*** Settings ***",
            f"Documentation    {test_case.documentation}",
            "Library          Browser",
            "Test Tags        " + "    ".join(test_case.tags) if test_case.tags else "",
            ""
        ])
        
        # Variables section
        if test_case.variables:
            lines.extend([
                "*** Variables ***",
                *[f"${{{name}}}    {value}" for name, value in test_case.variables.items()],
                ""
            ])
        else:
            lines.extend([
                "*** Variables ***",
                "${BASE_URL}    https://the-internet.herokuapp.com",
                "${BROWSER}     chromium",
                "${HEADLESS}    false",
                ""
            ])
        
        # Test Cases section
        lines.extend([
            "*** Test Cases ***",
            test_case.name,
            f"    [Documentation]    {test_case.documentation}",
            f"    [Tags]    {' '.join(test_case.tags)}" if test_case.tags else ""
        ])
        
        # Setup
        if test_case.setup:
            lines.append("    [Setup]    Run Keywords")
            for step in test_case.setup:
                lines.append(f"    ...    {step}")
        
        # Test steps
        for step in test_case.steps:
            if isinstance(step, dict):
                keyword = step.get("keyword", "")
                args = step.get("args", "")
                lines.append(f"    {keyword}    {args}")
            else:
                lines.append(f"    {step}")
        
        # Teardown
        if test_case.teardown:
            lines.append("    [Teardown]    Run Keywords")
            for step in test_case.teardown:
                lines.append(f"    ...    {step}")
        
        # Keywords section (if needed)
        lines.extend([
            "",
            "*** Keywords ***",
            "Setup Browser",
            "    New Browser    ${BROWSER}    headless=${HEADLESS}",
            "    New Page    ${BASE_URL}",
            "",
            "Close Browser",
            "    Close Browser    ALL",
            ""
        ])
        
        return "\n".join(lines)
    
    async def batch_generate(self, use_case_dir: str, output_dir: str):
        """Generate test cases for all use cases in directory"""
        use_case_path = Path(use_case_dir)
        output_path = Path(output_dir) / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all use case files
        use_case_files = list(use_case_path.glob("*.txt")) + list(use_case_path.glob("*.json"))
        logger.info(f"Found {len(use_case_files)} use case files")
        
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
    
    async def generate_advanced_test(self, use_case_file: str) -> str:
        """Generate advanced test case with data-driven testing and better structure"""
        # Read use case
        use_case_text = FileHandler.read_text_file(use_case_file)
        
        # Analyze use case for data-driven opportunities
        data_driven_scenarios = self.rf_tools.identify_data_driven_scenarios(use_case_text)
        
        # Generate enhanced test case
        prompt = f"""
        Generate an advanced Robot Framework test case with:
        1. Data-driven testing where applicable
        2. Proper error handling
        3. Screenshots on failure
        4. Detailed logging
        5. Reusable keywords
        
        Use case:
        {use_case_text}
        
        Data-driven scenarios identified:
        {data_driven_scenarios}
        """
        
        context = RFAgentContext(
            use_case_text=use_case_text,
            model_name=self.model_name
        )
        
        result = await self.agent.run(prompt, deps=context)
        
        return self._convert_to_advanced_robot_format(result.data, data_driven_scenarios)
    
    def _convert_to_advanced_robot_format(self, test_case: TestCase, data_scenarios: List[Dict]) -> str:
        """Convert to advanced Robot Framework format with templates"""
        lines = []
        
        # Enhanced settings
        lines.extend([
            "*** Settings ***",
            f"Documentation    {test_case.documentation}",
            "Library          Browser",
            "Library          OperatingSystem",
            "Library          DateTime",
            "Library          Collections",
            "Resource         common_keywords.robot",
            "Test Setup       Setup Test",
            "Test Teardown    Teardown Test",
            f"Test Tags        {' '.join(test_case.tags)}" if test_case.tags else "",
            "Test Timeout     5 minutes",
            ""
        ])
        
        # Variables with test data
        lines.extend([
            "*** Variables ***",
            "${BASE_URL}          https://the-internet.herokuapp.com",
            "${BROWSER}           chromium",
            "${HEADLESS}          false",
            "${SCREENSHOT_DIR}    ${OUTPUT_DIR}/screenshots",
            "@{TEST_DATA}        " + "    ".join([str(s) for s in data_scenarios]) if data_scenarios else "",
            ""
        ])
        
        # Test cases with templates if data-driven
        if data_scenarios:
            lines.extend([
                "*** Test Cases ***",
                f"{test_case.name} With Multiple Data Sets",
                "    [Template]    Execute Test With Data",
                "    FOR    ${data}    IN    @{TEST_DATA}",
                "    \\    ${data}",
                "    END",
                ""
            ])
        
        # Regular test case
        lines.extend([
            test_case.name,
            f"    [Documentation]    {test_case.documentation}",
            "    TRY",
            *[f"        {self._format_step(step)}" for step in test_case.steps],
            "    EXCEPT",
            "        Capture Page Screenshot    ${SCREENSHOT_DIR}/${TEST_NAME}_failure.png",
            "        Fail    Test failed: ${ERROR_MESSAGE}",
            "    END",
            ""
        ])
        
        # Enhanced keywords
        lines.extend([
            "*** Keywords ***",
            "Setup Test",
            "    Create Directory    ${SCREENSHOT_DIR}",
            "    Setup Browser",
            "    Set Browser Timeout    30s",
            "",
            "Teardown Test", 
            "    ${status}=    Get Test Status",
            "    Run Keyword If    '${status}' == 'FAIL'",
            "    ...    Capture Page Screenshot    ${SCREENSHOT_DIR}/${TEST_NAME}_final.png",
            "    Close Browser",
            "",
            "Execute Test With Data",
            "    [Arguments]    ${test_data}",
            "    Log    Executing test with data: ${test_data}",
            "    # Implementation based on test data",
            ""
        ])
        
        return "\n".join(lines)
    
    def _format_step(self, step: Dict[str, str]) -> str:
        """Format a test step with proper Browser library syntax"""
        if isinstance(step, dict):
            keyword = step.get("keyword", "")
            args = step.get("args", "")
            
            # Add proper Browser library syntax
            if keyword == "Click":
                return f"{keyword}    {args}"
            elif keyword == "Type Text":
                return f"{keyword}    {args}"
            elif keyword == "Wait For Elements State":
                return f"{keyword}    {args}    visible    timeout=10s"
            else:
                return f"{keyword}    {args}"
        return str(step)


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RF Agent - Generate Robot Framework tests")
    parser.add_argument("--model", required=True, help="Model name (mistral, gemma_7b_it_4bit)")
    parser.add_argument("--input", required=True, help="Input use case file or directory")
    parser.add_argument("--output", default="data/test_cases", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Process directory of files")
    parser.add_argument("--advanced", action="store_true", help="Generate advanced test cases")
    
    args = parser.parse_args()
    
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
        if args.advanced:
            robot_content = await agent.generate_advanced_test(args.input)
        else:
            robot_content = await agent.generate_test_case(args.input)
        
        output_path = Path(args.output) / args.model
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        input_name = Path(args.input).stem
        output_file = output_path / f"{input_name}.robot"
        
        FileHandler.save_text_file(robot_content, str(output_file))
        
        print(f"\nTest case generated: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())