"""
UC Agent - Generates use cases from requirements and user stories
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
from ..tools.text_tools import TextProcessor
from ..utils.file_handler import FileHandler


class UseCase(BaseModel):
    """Use case structure"""
    id: str = Field(description="Unique identifier")
    title: str = Field(description="Use case title")
    actors: List[str] = Field(description="Actors involved")
    preconditions: List[str] = Field(description="Preconditions")
    main_flow: List[str] = Field(description="Main flow steps")
    alternative_flows: Optional[List[Dict[str, List[str]]]] = Field(
        default=None, description="Alternative flows"
    )
    postconditions: List[str] = Field(description="Postconditions")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class UCAgentContext(BaseModel):
    """Context for UC Agent"""
    requirement_text: str
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)


class UCAgent:
    """Agent for generating use cases from requirements"""
    
    def __init__(self, model_name: str, config_path: str = "configs/config.yaml"):
        self.model_name = model_name
        self.config = FileHandler.load_yaml(config_path)
        self.model_loader = ModelLoader(config_path)
        
        # Load model
        self.model = self.model_loader.load_model(model_name)
        
        # Get system prompt from config
        self.system_prompt = self.config["agents"]["uc_agent"]["system_prompt"]
        
        # Initialize Pydantic AI agent
        self.agent = Agent(
            model=HuggingFaceModel(
                model=self.model,
                tokenizer=self.model_loader.tokenizer
            ),
            result_type=UseCase,
            system_prompt=self.system_prompt,
            deps_type=UCAgentContext,
        )
        
        # Add tools
        self._register_tools()
        
        logger.info(f"UC Agent initialized with model: {model_name}")
    
    def _register_tools(self):
        """Register tools for the agent"""
        
        @self.agent.tool
        async def extract_actors(ctx: RunContext[UCAgentContext]) -> List[str]:
            """Extract actors from requirement text"""
            text = ctx.deps.requirement_text
            processor = TextProcessor()
            return processor.extract_actors(text)
        
        @self.agent.tool
        async def identify_actions(ctx: RunContext[UCAgentContext]) -> List[str]:
            """Identify key actions from requirement text"""
            text = ctx.deps.requirement_text
            processor = TextProcessor()
            return processor.extract_actions(text)
        
        @self.agent.tool
        async def format_use_case(ctx: RunContext[UCAgentContext], use_case_data: dict) -> str:
            """Format use case in standard template"""
            processor = TextProcessor()
            return processor.format_use_case(use_case_data)
    
    async def generate_use_case(self, requirement_file: str) -> UseCase:
        """Generate use case from requirement file"""
        # Read requirement
        requirement_text = FileHandler.read_text_file(requirement_file)
        
        # Create context
        context = UCAgentContext(
            requirement_text=requirement_text,
            model_name=self.model_name
        )
        
        # Generate use case
        logger.info(f"Generating use case for: {requirement_file}")
        start_time = datetime.now()
        
        result = await self.agent.run(
            f"Generate a detailed use case from the following requirement:\n\n{requirement_text}",
            deps=context
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Use case generated in {generation_time:.2f} seconds")
        
        return result.data
    
    async def batch_generate(self, requirement_dir: str, output_dir: str):
        """Generate use cases for all requirements in directory"""
        requirement_path = Path(requirement_dir)
        output_path = Path(output_dir) / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all requirement files
        requirement_files = list(requirement_path.glob("*.txt"))
        logger.info(f"Found {len(requirement_files)} requirement files")
        
        results = []
        for req_file in requirement_files:
            try:
                # Generate use case
                use_case = await self.generate_use_case(str(req_file))
                
                # Save use case
                output_file = output_path / f"{req_file.stem}_use_case.json"
                FileHandler.save_json(use_case.model_dump(), str(output_file))
                
                # Save as formatted text
                text_output = self._format_use_case_text(use_case)
                text_file = output_path / f"{req_file.stem}_use_case.txt"
                FileHandler.save_text_file(text_output, str(text_file))
                
                results.append({
                    "requirement": req_file.name,
                    "use_case": output_file.name,
                    "status": "success"
                })
                
                logger.success(f"Generated use case: {output_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {req_file.name}: {str(e)}")
                results.append({
                    "requirement": req_file.name,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save generation report
        report = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(requirement_files),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        }
        
        report_file = output_path / "generation_report.json"
        FileHandler.save_json(report, str(report_file))
        
        return report
    
    def _format_use_case_text(self, use_case: UseCase) -> str:
        """Format use case as readable text"""
        lines = [
            f"USE CASE: {use_case.title}",
            f"ID: {use_case.id}",
            "",
            "ACTORS:",
            *[f"  - {actor}" for actor in use_case.actors],
            "",
            "PRECONDITIONS:",
            *[f"  {i+1}. {pc}" for i, pc in enumerate(use_case.preconditions)],
            "",
            "MAIN FLOW:",
            *[f"  {i+1}. {step}" for i, step in enumerate(use_case.main_flow)],
            ""
        ]
        
        if use_case.alternative_flows:
            lines.extend([
                "ALTERNATIVE FLOWS:",
                *[f"  {flow['name']}:" for flow in use_case.alternative_flows],
                *[f"    {i+1}. {step}" for flow in use_case.alternative_flows 
                  for i, step in enumerate(flow.get('steps', []))],
                ""
            ])
        
        lines.extend([
            "POSTCONDITIONS:",
            *[f"  {i+1}. {pc}" for i, pc in enumerate(use_case.postconditions)]
        ])
        
        if use_case.notes:
            lines.extend(["", "NOTES:", use_case.notes])
        
        return "\n".join(lines)


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="UC Agent - Generate use cases")
    parser.add_argument("--model", required=True, help="Model name (mistral, gemma_7b_it_4bit)")
    parser.add_argument("--input", required=True, help="Input requirement file or directory")
    parser.add_argument("--output", default="data/user_stories", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Process directory of files")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = UCAgent(args.model)
    
    if args.batch:
        # Batch processing
        report = await agent.batch_generate(args.input, args.output)
        print(f"\nGeneration complete!")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")
    else:
        # Single file processing
        use_case = await agent.generate_use_case(args.input)
        output_path = Path(args.output) / args.model
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        input_name = Path(args.input).stem
        json_file = output_path / f"{input_name}_use_case.json"
        text_file = output_path / f"{input_name}_use_case.txt"
        
        FileHandler.save_json(use_case.model_dump(), str(json_file))
        FileHandler.save_text_file(
            agent._format_use_case_text(use_case), 
            str(text_file)
        )
        
        print(f"\nUse case generated:")
        print(f"JSON: {json_file}")
        print(f"Text: {text_file}")


if __name__ == "__main__":
    asyncio.run(main())