"""
UC Agent - Generates use cases from requirements
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from pydantic import BaseModel, Field
from loguru import logger

from ..models.model_loader import ModelLoader
from ..tools.text_tools import TextProcessor
from ..utils.file_handler import FileHandler


class UseCase(BaseModel):
    """Use case structure"""
    id: str = Field(default="UC001", description="Unique identifier")
    title: str = Field(default="Use Case", description="Use case title")
    actors: List[str] = Field(default_factory=list, description="Actors involved")
    preconditions: List[str] = Field(default_factory=list, description="Preconditions")
    main_flow: List[str] = Field(default_factory=list, description="Main flow steps")
    alternative_flows: Optional[List[Dict[str, List[str]]]] = Field(default=None)
    postconditions: List[str] = Field(default_factory=list, description="Postconditions")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class UCAgent:
    """Agent for generating use cases from requirements"""
    
    def __init__(self, model_name: str, config_path: str = "configs/config.yaml"):
        self.model_name = model_name
        self.config = FileHandler.load_yaml(config_path)
        self.model_loader = ModelLoader(config_path)
        self.text_processor = TextProcessor()
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = self.model_loader.load_model(model_name)
        
        # Get system prompt from config
        self.system_prompt = self.config["agents"]["uc_agent"]["system_prompt"]
        
        logger.info(f"UC Agent initialized with model: {model_name}")
    
    async def generate_use_case(self, requirement_file: str) -> UseCase:
        """Generate use case from requirement file"""
        # Read requirement
        requirement_text = FileHandler.read_text_file(requirement_file)
        
        # Create prompt
        prompt = f"""{self.system_prompt}

Requirement:
{requirement_text}

Generate a detailed use case with:
- ID (e.g., UC-LOGIN-001)
- Title
- Actors
- Preconditions
- Main flow (numbered steps)
- Alternative flows (if any)
- Postconditions

Format the response as a structured use case."""

        # Generate response using model
        logger.info(f"Generating use case for: {requirement_file}")
        start_time = datetime.now()
        
        response = self.model_loader.generate(
            self.model_name, 
            prompt,
            max_new_tokens=1024,
            temperature=0.7
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Use case generated in {generation_time:.2f} seconds")
        
        # Parse response into UseCase object
        use_case = self._parse_response_to_use_case(response, requirement_text)
        
        return use_case
    
    def _parse_response_to_use_case(self, response: str, requirement_text: str) -> UseCase:
        """Parse LLM response into UseCase object"""
        # Extract actors using text processor
        actors = self.text_processor.extract_actors(requirement_text)
        
        # Simple parsing logic
        use_case_data = {
            "id": "UC-001",
            "title": "Generated Use Case",
            "actors": actors,
            "preconditions": [],
            "main_flow": [],
            "postconditions": []
        }
        
        # Parse response sections
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if any(marker in line.upper() for marker in ["ID:", "IDENTIFIER:"]):
                use_case_data["id"] = line.split(":", 1)[-1].strip()
            elif "TITLE:" in line.upper():
                use_case_data["title"] = line.split(":", 1)[-1].strip()
            elif "ACTORS:" in line.upper():
                current_section = "actors"
            elif "PRECONDITIONS:" in line.upper():
                current_section = "preconditions"
            elif "MAIN FLOW:" in line.upper():
                current_section = "main_flow"
            elif "POSTCONDITIONS:" in line.upper():
                current_section = "postconditions"
            elif current_section and line:
                # Add to current section
                if line.startswith(("- ", "• ", "* ", "1.", "2.", "3.")):
                    line = line.lstrip("- •*123456789. ")
                if current_section in use_case_data:
                    use_case_data[current_section].append(line)
        
        return UseCase(**use_case_data)
    
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
        
        return results
    
    def _format_use_case_text(self, use_case: UseCase) -> str:
        """Format use case as readable text"""
        return self.text_processor.format_use_case(use_case.model_dump())


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="UC Agent - Generate use cases")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", default="data/user_stories", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Batch processing")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = UCAgent(args.model)
    
    if args.batch:
        # Batch processing
        results = await agent.batch_generate(args.input, args.output)
        print(f"\nGeneration complete!")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
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