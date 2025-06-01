# src/evaluators/metrics/rf_metrics.py
"""
Robot Framework specific metrics for test case quality evaluation
"""

import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter, defaultdict
from loguru import logger


class RobotFrameworkMetrics:
    """Calculate Robot Framework specific quality metrics"""
    
    def __init__(self):
        # Common Browser library keywords
        self.browser_keywords = {
            'navigation': ['New Browser', 'New Page', 'Go To', 'Reload'],
            'interaction': ['Click', 'Type Text', 'Press Keys', 'Select Options By'],
            'validation': ['Get Text', 'Get Element State', 'Get Element Count'],
            'wait': ['Wait For Elements State', 'Wait For Function', 'Wait For Navigation'],
            'screenshot': ['Take Screenshot'],
            'browser_control': ['Close Browser', 'Close Page', 'Switch Page']
        }
        
        # Best practice patterns
        self.selector_patterns = {
            'id': r'id=[\w-]+',
            'css': r'css=[^\\s]+',
            'xpath': r'xpath=[^\\s]+',
            'text': r'text="[^"]+"|text=[^\\s]+',
            'data_test': r'data-test-?id=[^\\s]+',
            'aria': r'aria-label=[^\\s]+'
        }
    
    def calculate_all_metrics(self, test_files: List[Path]) -> Dict[str, Any]:
        """Calculate all Robot Framework metrics"""
        metrics = {
            'overall_quality': {},
            'keyword_analysis': {},
            'structure_analysis': {},
            'best_practices': {},
            'individual_files': {}
        }
        
        # Analyze each file
        all_contents = []
        for test_file in test_files:
            content = test_file.read_text(encoding='utf-8')
            all_contents.append(content)
            
            file_metrics = self._analyze_single_file(content)
            metrics['individual_files'][test_file.name] = file_metrics
        
        # Aggregate metrics
        if all_contents:
            metrics['overall_quality'] = self._calculate_overall_quality(all_contents)
            metrics['keyword_analysis'] = self._analyze_keyword_usage(all_contents)
            metrics['structure_analysis'] = self._analyze_test_structure(all_contents)
            metrics['best_practices'] = self._analyze_best_practices(all_contents)
        
        return metrics
    
    def _analyze_single_file(self, content: str) -> Dict[str, Any]:
        """Analyze a single Robot Framework file"""
        lines = content.split('\n')
        
        metrics = {
            'syntax_valid': self._validate_syntax(content),
            'sections': self._identify_sections(lines),
            'test_count': self._count_test_cases(lines),
            'keyword_count': self._count_keywords(lines),
            'documentation_present': self._check_documentation(lines),
            'tags_present': self._check_tags(lines),
            'setup_teardown': self._check_setup_teardown(lines),
            'line_count': len(lines),
            'non_empty_lines': len([l for l in lines if l.strip()])
        }
        
        return metrics
    
    def _validate_syntax(self, content: str) -> bool:
        """Basic syntax validation"""
        required_sections = ['*** Settings ***', '*** Test Cases ***']
        return all(section in content for section in required_sections)
    
    def _identify_sections(self, lines: List[str]) -> List[str]:
        """Identify which sections are present"""
        sections = []
        section_pattern = r'^\*\*\*\s*(\w+)\s*\*\*\*'
        
        for line in lines:
            match = re.match(section_pattern, line)
            if match:
                sections.append(match.group(1))
        
        return sections
    
    def _count_test_cases(self, lines: List[str]) -> int:
        """Count number of test cases"""
        in_test_section = False
        test_count = 0
        
        for line in lines:
            if '*** Test Cases ***' in line:
                in_test_section = True
                continue
            elif line.startswith('***') and in_test_section:
                in_test_section = False
                continue
            
            if in_test_section and line.strip() and not line.startswith(' '):
                test_count += 1
        
        return test_count
    
    def _count_keywords(self, lines: List[str]) -> int:
        """Count number of custom keywords"""
        in_keyword_section = False
        keyword_count = 0
        
        for line in lines:
            if '*** Keywords ***' in line:
                in_keyword_section = True
                continue
            elif line.startswith('***') and in_keyword_section:
                in_keyword_section = False
                continue
            
            if in_keyword_section and line.strip() and not line.startswith(' '):
                keyword_count += 1
        
        return keyword_count
    
    def _check_documentation(self, lines: List[str]) -> bool:
        """Check if documentation is present"""
        return any('Documentation' in line for line in lines)
    
    def _check_tags(self, lines: List[str]) -> bool:
        """Check if tags are used"""
        return any('[Tags]' in line or 'Test Tags' in line for line in lines)
    
    def _check_setup_teardown(self, lines: List[str]) -> Dict[str, bool]:
        """Check for setup and teardown"""
        return {
            'test_setup': any('Test Setup' in line or '[Setup]' in line for line in lines),
            'test_teardown': any('Test Teardown' in line or '[Teardown]' in line for line in lines),
            'suite_setup': any('Suite Setup' in line for line in lines),
            'suite_teardown': any('Suite Teardown' in line for line in lines)
        }
    
    def _calculate_overall_quality(self, contents: List[str]) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        total_tests = 0
        documented_tests = 0
        tagged_tests = 0
        tests_with_verification = 0
        
        for content in contents:
            # Count tests and their properties
            test_blocks = re.findall(
                r'(?:^|\n)([^\s].*?)\n((?:    .*\n)*)',
                content[content.find('*** Test Cases ***'):] if '*** Test Cases ***' in content else '',
                re.MULTILINE
            )
            
            for test_name, test_body in test_blocks:
                if test_name.strip() and not test_name.startswith('*'):
                    total_tests += 1
                    
                    if '[Documentation]' in test_body:
                        documented_tests += 1
                    if '[Tags]' in test_body:
                        tagged_tests += 1
                    if any(kw in test_body for kw in ['Should', 'Wait For', 'Get Text']):
                        tests_with_verification += 1
        
        return {
            'total_tests': total_tests,
            'documentation_coverage': documented_tests / total_tests if total_tests > 0 else 0,
            'tag_coverage': tagged_tests / total_tests if total_tests > 0 else 0,
            'verification_coverage': tests_with_verification / total_tests if total_tests > 0 else 0
        }
    
    def _analyze_keyword_usage(self, contents: List[str]) -> Dict[str, Any]:
        """Analyze keyword usage patterns"""
        all_keywords = []
        keyword_definitions = set()
        keyword_calls = []
        
        for content in contents:
            # Extract all keyword usages
            lines = content.split('\n')
            in_keywords_section = False
            
            for line in lines:
                if '*** Keywords ***' in line:
                    in_keywords_section = True
                    continue
                elif line.startswith('***'):
                    in_keywords_section = False
                
                # Count keyword definitions
                if in_keywords_section and line.strip() and not line.startswith(' '):
                    keyword_definitions.add(line.strip())
                
                # Count keyword calls (4 spaces indentation)
                if line.startswith('    ') and not line.startswith('        '):
                    keyword = line.strip().split()[0] if line.strip() else ''
                    if keyword and not keyword.startswith('['):
                        keyword_calls.append(keyword)
                        all_keywords.append(keyword)
        
        # Analyze Browser library usage
        browser_usage = defaultdict(int)
        for category, keywords in self.browser_keywords.items():
            for keyword in keywords:
                count = sum(1 for k in all_keywords if k == keyword)
                if count > 0:
                    browser_usage[category] += count
        
        # Calculate reuse metrics
        keyword_call_counts = Counter(keyword_calls)
        
        return {
            'total_keywords_used': len(all_keywords),
            'unique_keywords': len(set(all_keywords)),
            'custom_keywords_defined': len(keyword_definitions),
            'keyword_reuse_ratio': len(keyword_definitions) / len(set(keyword_calls)) if keyword_calls else 0,
            'browser_keyword_usage': dict(browser_usage),
            'most_used_keywords': dict(keyword_call_counts.most_common(10))
        }
    
    def _analyze_test_structure(self, contents: List[str]) -> Dict[str, Any]:
        """Analyze test structure and complexity"""
        test_complexities = []
        test_lengths = []
        
        for content in contents:
            # Extract test cases
            if '*** Test Cases ***' in content:
                test_section = content[content.find('*** Test Cases ***'):]
                test_blocks = re.split(r'\n(?=[^\s])', test_section)
                
                for block in test_blocks:
                    if block.strip() and not block.startswith('*'):
                        lines = block.split('\n')
                        test_name = lines[0].strip()
                        test_steps = [l for l in lines[1:] if l.strip() and l.startswith('    ')]
                        
                        if test_name:
                            test_lengths.append(len(test_steps))
                            
                            # Calculate complexity based on control structures
                            complexity = 1  # Base complexity
                            for step in test_steps:
                                if 'IF' in step or 'FOR' in step:
                                    complexity += 1
                                if 'Run Keyword' in step:
                                    complexity += 0.5
                            
                            test_complexities.append(complexity)
        
        return {
            'avg_test_length': sum(test_lengths) / len(test_lengths) if test_lengths else 0,
            'max_test_length': max(test_lengths) if test_lengths else 0,
            'min_test_length': min(test_lengths) if test_lengths else 0,
            'avg_complexity': sum(test_complexities) / len(test_complexities) if test_complexities else 0,
            'max_complexity': max(test_complexities) if test_complexities else 0
        }
    
    def _analyze_best_practices(self, contents: List[str]) -> Dict[str, Any]:
        """Analyze adherence to best practices"""
        all_content = '\n'.join(contents)
        
        # Count selector types
        selector_usage = {}
        for selector_type, pattern in self.selector_patterns.items():
            matches = re.findall(pattern, all_content)
            selector_usage[selector_type] = len(matches)
        
        total_selectors = sum(selector_usage.values())
        
        # Check for wait strategies
        explicit_waits = len(re.findall(r'Wait For', all_content))
        implicit_waits = len(re.findall(r'Sleep', all_content))
        
        # Check for error handling
        try_except = len(re.findall(r'TRY|EXCEPT', all_content))
        run_keyword_and_expect = len(re.findall(r'Run Keyword And Expect Error', all_content))
        
        # Check for screenshots
        screenshots = len(re.findall(r'Take Screenshot', all_content))
        
        # Data-driven testing
        templates = len(re.findall(r'\[Template\]', all_content))
        for_loops = len(re.findall(r'FOR', all_content))
        
        return {
            'selector_usage': selector_usage,
            'best_selector_ratio': (selector_usage.get('id', 0) + selector_usage.get('data_test', 0)) / total_selectors if total_selectors > 0 else 0,
            'explicit_wait_usage': explicit_waits,
            'implicit_wait_usage': implicit_waits,
            'wait_strategy_score': explicit_waits / (explicit_waits + implicit_waits) if (explicit_waits + implicit_waits) > 0 else 0,
            'error_handling_present': (try_except + run_keyword_and_expect) > 0,
            'screenshot_usage': screenshots,
            'data_driven_testing': (templates + for_loops) > 0,
            'template_usage': templates,
            'loop_usage': for_loops
        }
    
    def generate_quality_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a quality report from metrics"""
        report = []
        report.append("# Robot Framework Test Quality Report\n")
        
        # Overall quality
        if 'overall_quality' in metrics:
            report.append("## Overall Quality Metrics")
            oq = metrics['overall_quality']
            report.append(f"- Total Tests: {oq.get('total_tests', 0)}")
            report.append(f"- Documentation Coverage: {oq.get('documentation_coverage', 0):.1%}")
            report.append(f"- Tag Coverage: {oq.get('tag_coverage', 0):.1%}")
            report.append(f"- Verification Coverage: {oq.get('verification_coverage', 0):.1%}")
            report.append("")
        
        # Best practices
        if 'best_practices' in metrics:
            report.append("## Best Practices Analysis")
            bp = metrics['best_practices']
            report.append(f"- Explicit Waits: {bp.get('explicit_wait_usage', 0)}")
            report.append(f"- Implicit Waits (Sleep): {bp.get('implicit_wait_usage', 0)}")
            report.append(f"- Wait Strategy Score: {bp.get('wait_strategy_score', 0):.1%}")
            report.append(f"- Error Handling: {'Yes' if bp.get('error_handling_present', False) else 'No'}")
            report.append(f"- Data-Driven Testing: {'Yes' if bp.get('data_driven_testing', False) else 'No'}")
            report.append("")
        
        return '\n'.join(report)