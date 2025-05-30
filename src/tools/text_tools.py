# src/tools/text_tools.py
"""
Text processing tools for agents
"""

import re
from typing import List, Dict, Any


class TextProcessor:
    """Process text for agent tools"""
    
    def extract_actors(self, text: str) -> List[str]:
        """Extract actors from requirement text"""
        actors = []
        
        # Look for common actor patterns
        patterns = [
            r'(?:user|customer|admin|system|operator|visitor|guest)s?',
            r'(?:the\s+)?(\w+)\s+(?:can|must|should|shall)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            actors.extend(matches)
        
        # Clean and deduplicate
        actors = list(set([a.strip() for a in actors if a.strip()]))
        
        # Default actors if none found
        if not actors:
            actors = ["User", "System"]
        
        return actors
    
    def extract_actions(self, text: str) -> List[str]:
        """Extract key actions from text"""
        actions = []
        
        # Action verbs pattern
        action_verbs = [
            'login', 'logout', 'click', 'enter', 'submit', 'view', 'display',
            'validate', 'verify', 'check', 'create', 'update', 'delete',
            'select', 'navigate', 'upload', 'download', 'search'
        ]
        
        for verb in action_verbs:
            if verb in text.lower():
                # Find context around verb
                pattern = rf'\b{verb}\b[^.]*'
                matches = re.findall(pattern, text, re.IGNORECASE)
                actions.extend(matches)
        
        return actions
    
    def format_use_case(self, use_case_data: Dict[str, Any]) -> str:
        """Format use case in standard template"""
        template = """
USE CASE: {title}
ID: {id}

ACTORS:
{actors}

PRECONDITIONS:
{preconditions}

MAIN FLOW:
{main_flow}

ALTERNATIVE FLOWS:
{alt_flows}

POSTCONDITIONS:
{postconditions}

NOTES:
{notes}
"""
        
        formatted_actors = "\n".join([f"- {actor}" for actor in use_case_data.get('actors', [])])
        formatted_preconditions = "\n".join([f"{i+1}. {pc}" for i, pc in enumerate(use_case_data.get('preconditions', []))])
        formatted_main_flow = "\n".join([f"{i+1}. {step}" for i, step in enumerate(use_case_data.get('main_flow', []))])
        formatted_postconditions = "\n".join([f"{i+1}. {pc}" for i, pc in enumerate(use_case_data.get('postconditions', []))])
        
        # Format alternative flows
        alt_flows = []
        for flow in use_case_data.get('alternative_flows', []):
            alt_flows.append(f"\n{flow.get('name', 'Alternative')}:")
            alt_flows.extend([f"  {i+1}. {step}" for i, step in enumerate(flow.get('steps', []))])
        formatted_alt_flows = "\n".join(alt_flows) if alt_flows else "None"
        
        return template.format(
            title=use_case_data.get('title', 'Untitled'),
            id=use_case_data.get('id', 'UC001'),
            actors=formatted_actors,
            preconditions=formatted_preconditions,
            main_flow=formatted_main_flow,
            alt_flows=formatted_alt_flows,
            postconditions=formatted_postconditions,
            notes=use_case_data.get('notes', 'None')
        )