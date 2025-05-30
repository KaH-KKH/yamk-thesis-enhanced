# src/tools/rf_tools.py
"""
Robot Framework specific tools
"""

from typing import List, Dict, Any
import re


class RobotFrameworkTools:
    """Tools for Robot Framework test generation"""
    
    def __init__(self):
        self.browser_keywords = {
            "navigate": "Go To",
            "click": "Click",
            "type": "Type Text",
            "enter": "Type Text",
            "verify": "Get Text",
            "check": "Get Element States",
            "wait": "Wait For Elements State",
            "select": "Select Options By",
            "submit": "Click",
            "login": "Custom Login Keyword",
            "logout": "Custom Logout Keyword"
        }
    
    def extract_test_scenarios(self, use_case: str) -> List[str]:
        """Extract test scenarios from use case"""
        scenarios = []
        
        # Extract main flow
        main_flow_match = re.search(r'MAIN FLOW:(.*?)(?:ALTERNATIVE|POSTCONDITIONS|$)', use_case, re.DOTALL)
        if main_flow_match:
            main_steps = main_flow_match.group(1).strip()
            scenarios.append(f"Main Flow: {main_steps[:100]}...")
        
        # Extract alternative flows
        alt_flow_matches = re.findall(r'(Alternative.*?):(.*?)(?:Alternative|POSTCONDITIONS|$)', use_case, re.DOTALL)
        for name, steps in alt_flow_matches:
            scenarios.append(f"{name}: {steps.strip()[:100]}...")
        
        return scenarios
    
    def map_action_to_keyword(self, action: str, base_url: str) -> Dict[str, str]:
        """Map action to Browser library keyword"""
        action_lower = action.lower()
        
        # Find matching keyword
        for key, keyword in self.browser_keywords.items():
            if key in action_lower:
                return {
                    "keyword": keyword,
                    "args": self._generate_args_for_keyword(keyword, action, base_url)
                }
        
        # Default
        return {
            "keyword": "Log",
            "args": f"Action: {action}"
        }
    
    def _generate_args_for_keyword(self, keyword: str, action: str, base_url: str) -> str:
        """Generate arguments for keyword based on action"""
        if keyword == "Go To":
            # Extract URL if mentioned
            url_match = re.search(r'(?:to|at)\s+([^\s]+)', action)
            if url_match:
                path = url_match.group(1)
                return f"{base_url}/{path}" if not path.startswith('http') else path
            return base_url
        
        elif keyword == "Click":
            # Extract element
            element_patterns = [
                r'(?:click|press)\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button|\s+link)?',
                r'button\s+(?:named|labeled|with)\s+"([^"]+)"'
            ]
            for pattern in element_patterns:
                match = re.search(pattern, action, re.IGNORECASE)
                if match:
                    element = match.group(1).strip()
                    return f'text="{element}"'
            return 'text="Submit"'
        
        elif keyword == "Type Text":
            # Extract field and value
            field_match = re.search(r'(?:type|enter)\s+(?:in\s+)?(?:the\s+)?(\w+)', action, re.IGNORECASE)
            value_match = re.search(r'"([^"]+)"', action)
            
            field = field_match.group(1) if field_match else "field"
            value = value_match.group(1) if value_match else "test value"
            
            return f'id="{field}"    {value}'
        
        return ""
    
    def generate_settings(self) -> Dict[str, List[str]]:
        """Generate Robot Framework settings"""
        return {
            "Library": ["Browser"],
            "Documentation": ["Automated test case generated from use case"],
            "Test Setup": ["Setup Browser"],
            "Test Teardown": ["Close Browser"]
        }
    
    def generate_variables(self, base_url: str) -> Dict[str, str]:
        """Generate common variables"""
        return {
            "BASE_URL": base_url,
            "BROWSER": "chromium",
            "TIMEOUT": "30s",
            "SCREENSHOT_DIR": "${OUTPUT_DIR}/screenshots"
        }
    
    def identify_data_driven_scenarios(self, use_case: str) -> List[Dict[str, Any]]:
        """Identify scenarios suitable for data-driven testing"""
        scenarios = []
        
        # Look for patterns indicating multiple data sets
        patterns = [
            r'(?:valid|invalid)\s+(?:credentials|data|input)',
            r'(?:different|multiple|various)\s+(?:users|scenarios|cases)',
            r'(?:test|verify)\s+with\s+(?:different|multiple)'
        ]
        
        for pattern in patterns:
            if re.search(pattern, use_case, re.IGNORECASE):
                # Extract potential test data
                if 'credentials' in use_case.lower():
                    scenarios.append({
                        "type": "login",
                        "data": [
                            {"username": "tomsmith", "password": "SuperSecretPassword!", "expected": "success"},
                            {"username": "invalid", "password": "wrong", "expected": "error"}
                        ]
                    })
                break
        
        return scenarios