"""
Advanced Executor for RCA Agent with enhanced code generation and execution.

Key improvements over basic executor:
- Enhanced error recovery and self-correction
- Memory-aware execution with resource monitoring
- Code validation and optimization
- Context-aware code generation
- Intelligent retry mechanisms
"""

import re
import time
import psutil
import traceback
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from rca.api_router import get_chat_completion
import tiktoken
from rca.baseline.rca_agent_advanced.data_utils import (
    TelemetryDataDetector, 
    extract_components_from_instruction,
    extract_time_range_from_instruction
)


class AdvancedExecutor:
    """Advanced executor with enhanced code generation and execution capabilities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.memory_threshold = config.get('memory_mb', 2048) * 1024 * 1024  # Convert to bytes
        self.execution_history = []
        self.common_errors = {}  # Track common errors for better recovery
        self.data_detector = TelemetryDataDetector()  # Add data detection capability
        
    def execute_with_recovery(self, instruction: str, context: Dict, kernel, logger) -> Tuple[str, str, bool]:
        """
        Execute instruction with enhanced error recovery and optimization.
        
        Args:
            instruction: The instruction to execute
            context: Analysis context
            kernel: IPython kernel
            logger: Logger instance
            
        Returns:
            Tuple of (code, result, success)
        """
        logger.debug(f"🔧 Executing: {instruction}")
        
        # Pre-execution checks
        if self._should_optimize_memory():
            self._optimize_memory(kernel, logger)
        
        # Generate and execute code
        max_attempts = self.config.get('max_retries', 3)
        
        for attempt in range(max_attempts):
            try:
                # Generate code with context awareness
                code = self._generate_context_aware_code(instruction, context, attempt, logger)
                
                if not code:
                    logger.warning(f"⚠️ Failed to generate code for: {instruction}")
                    continue
                
                # Validate code before execution
                validation_result = self._validate_code(code, logger)
                if not validation_result['valid']:
                    logger.warning(f"⚠️ Code validation failed: {validation_result['reason']}")
                    if attempt < max_attempts - 1:
                        continue
                
                # Execute with monitoring
                result, success = self._execute_with_monitoring(code, kernel, logger)
                
                # Record execution history for learning
                self._record_execution(instruction, code, result, success)
                
                if success:
                    logger.debug(f"✅ Execution successful")
                    return code, result, True
                else:
                    logger.warning(f"⚠️ Execution failed (attempt {attempt + 1}/{max_attempts})")
                    
                    # Learn from error for next attempt
                    self._learn_from_error(instruction, code, result)
                    
            except Exception as e:
                logger.error(f"❌ Execution error (attempt {attempt + 1}/{max_attempts}): {str(e)}")
                result = str(e)
        
        # If all attempts failed, return the last attempt's result
        return code if 'code' in locals() else "# Failed to generate code", result, False
    
    def _generate_context_aware_code(self, instruction: str, context: Dict, attempt: int, logger) -> str:
        """Generate code with context awareness and attempt-specific improvements"""
        
        # Check if this is a data loading instruction and handle specially
        if self._is_data_loading_instruction(instruction):
            return self._generate_intelligent_data_loading_code(instruction, context, logger)
        
        # Build enhanced prompt with context
        system_prompt = self._build_enhanced_system_prompt(context, attempt)
        
        # Add specific guidance based on instruction type
        instruction_guidance = self._get_instruction_guidance(instruction)
        
        # Include error learning from previous attempts
        error_guidance = self._get_error_guidance(instruction, attempt)
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"{instruction_guidance}\n\n{error_guidance}\n\nInstruction: {instruction}"}
        ]
        
        try:
            response = get_chat_completion(messages=messages)
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            
            if code:
                # Apply post-processing improvements
                code = self._post_process_code(code, context)
            
            return code
            
        except Exception as e:
            logger.error(f"❌ Failed to generate code: {str(e)}")
            return ""
    
    def _build_enhanced_system_prompt(self, context: Dict, attempt: int) -> str:
        """Build enhanced system prompt with context and attempt information"""
        
        base_prompt = """You are an Data Scientist and should Generate Code for RCA analysis. Generate robust, efficient Python code for telemetry data analysis.

ENHANCED CAPABILITIES:
- Memory-efficient data processing
- Robust error handling
- Optimized pandas operations
- Context-aware analysis
- Domain-specific optimizations

EXECUTION ENVIRONMENT:
- IPython Kernel with stateful variables
- Pandas, NumPy, PyTZ available
- UTC+8 timezone (Asia/Shanghai)
- Memory-optimized display settings
"""
        
        # Add context-specific guidance
        domain_guidance = {
            'Banking': "Focus on transaction patterns, financial metrics, and service availability.",
            'Telecom': "Focus on network metrics, database performance, and connectivity issues.",
            'Market': "Focus on microservice interactions, container metrics, and distributed traces.",
            'Generic': "Apply general RCA principles with broad telemetry analysis."
        }
        
        domain = context.get('domain', 'Generic')
        complexity = context.get('complexity', 'Medium')
        
        enhanced_prompt = f"""{base_prompt}

ANALYSIS CONTEXT:
- Domain: {domain}
- Complexity: {complexity}
- Analysis Stage: {context.get('current_stage', 'Unknown')}

DOMAIN GUIDANCE:
{domain_guidance.get(domain, domain_guidance['Generic'])}

MEMORY OPTIMIZATION:
- Use chunked processing for large datasets
- Clear intermediate variables when done
- Use memory-efficient data types
- Avoid creating unnecessary copies

CODE GENERATION RULES:
1. Generate ONLY executable Python code in ```python code blocks
2. Use existing variables when possible (stateful kernel)
3. Display results with variable names, not print()
4. Handle edge cases (empty data, missing columns, etc.)
5. Use UTC+8 timezone for all datetime operations
6. Include brief comments for complex operations
7. Optimize for memory efficiency
8. Validate data before processing

ATTEMPT: {attempt + 1}"""
        
        return enhanced_prompt
    
    def _get_instruction_guidance(self, instruction: str) -> str:
        """Get specific guidance based on instruction type"""
        
        instruction_lower = instruction.lower()
        
        if 'threshold' in instruction_lower:
            return """THRESHOLD CALCULATION GUIDANCE:
- Calculate global thresholds using entire dataset BEFORE filtering by time
- Use percentile-based thresholds (P95, P99) for most KPIs
- Handle edge cases where data might be empty or sparse
- Consider different threshold methods for different KPI types
- CRITICAL: Load actual telemetry data files first - don't create dummy data!"""
        
        elif 'anomaly' in instruction_lower or 'detect' in instruction_lower:
            return """ANOMALY DETECTION GUIDANCE:
- Use the detect_anomalies() helper function when available
- Apply thresholds consistently across similar KPIs
- Filter out isolated spikes that might be noise
- Focus on consecutive anomalous points (faults)
- CRITICAL: Ensure real telemetry data is loaded from CSV files!"""
        
        elif 'correlation' in instruction_lower:
            return """CORRELATION ANALYSIS GUIDANCE:
- Cross-reference timestamps across different data sources
- Look for temporal relationships and causal patterns
- Consider service dependencies and call chains
- Build timelines of related events
- CRITICAL: Work with actual loaded telemetry data, not simulated data!"""
        
        elif 'load' in instruction_lower or 'read' in instruction_lower or 'retrieve' in instruction_lower:
            return """DATA LOADING GUIDANCE:
- ALWAYS load actual CSV files from dataset/{domain}/telemetry/ directories
- Use proper file paths like dataset/Bank/telemetry/2021_03_05/metric/
- Load metric_app.csv, metric_container.csv, trace_span.csv, log_service.csv files
- Validate data structure after loading and show samples
- Handle timestamp conversion correctly (seconds vs milliseconds)
- NEVER create dummy/random data - always use real files!"""
        
        else:
            return """GENERAL RCA GUIDANCE:
- Follow the standard RCA workflow
- Validate inputs and handle edge cases
- Use clear variable names for intermediate results
- Document significant findings
- CRITICAL: Always work with real telemetry data from CSV files!"""
    
    def _get_error_guidance(self, instruction: str, attempt: int) -> str:
        """Get guidance based on common errors for this instruction type"""
        
        if attempt == 0:
            return ""  # No error guidance for first attempt
        
        # Check if we've seen similar errors before
        similar_errors = self._get_similar_errors(instruction)
        
        if similar_errors:
            guidance = "COMMON ERROR PREVENTION:\n"
            for error_type, solution in similar_errors.items():
                guidance += f"- {error_type}: {solution}\n"
            return guidance
        
        # General retry guidance
        return f"""RETRY GUIDANCE (Attempt {attempt + 1}):
- Previous attempts failed, try alternative approaches
- Simplify complex operations into smaller steps
- Add more error checking and validation
- Consider edge cases that might cause failures"""
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from AI response"""
        
        # Pattern for ```python code blocks
        python_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
        match = python_pattern.search(response)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: look for any code block
        code_pattern = re.compile(r"```\n(.*?)\n```", re.DOTALL)
        match = code_pattern.search(response)
        
        if match:
            code = match.group(1).strip()
            # Check if it looks like Python code
            if any(keyword in code for keyword in ['import ', 'def ', 'pd.', 'np.', '=']):
                return code
        
        # Final fallback: return response if it looks like code
        if any(keyword in response for keyword in ['import ', 'pd.', 'np.', '=']):
            return response.strip()
        
        return ""
    
    def _post_process_code(self, code: str, context: Dict) -> str:
        """Apply post-processing improvements to generated code"""
        
        # Add memory optimization if dealing with large data
        if 'read_csv' in code and 'chunksize' not in code:
            # Could add chunked reading for very large files
            pass
        
        # Ensure timezone handling
        if 'datetime' in code and 'Asia/Shanghai' not in code:
            # Add timezone context if working with timestamps
            if not code.startswith('# Timezone:'):
                code = f"# Timezone: UTC+8 (Asia/Shanghai)\n{code}"
        
        # Add error handling for critical operations
        if 'pd.read_csv' in code and 'try:' not in code:
            # Could wrap critical operations in try-catch
            pass
        
        return code
    
    def _validate_code(self, code: str, logger) -> Dict[str, Any]:
        """Validate code before execution"""
        
        validation_result = {'valid': True, 'reason': ''}
        
        # Check for dangerous operations
        dangerous_patterns = [
            r'rm\s+-rf',  # Dangerous shell commands
            r'os\.system\(',  # System calls
            r'subprocess\.',  # Subprocess calls
            r'open\(.+[\'"]w[\'"]',  # File writing
            r'\.to_file\(',  # File output
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                validation_result['valid'] = False
                validation_result['reason'] = f"Contains potentially dangerous operation: {pattern}"
                return validation_result
        
        # Check for matplotlib/seaborn (not allowed)
        if any(lib in code.lower() for lib in ['matplotlib', 'seaborn', 'plt.show']):
            validation_result['valid'] = False
            validation_result['reason'] = "Visualization libraries not allowed"
            return validation_result
        
        # Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            validation_result['valid'] = False
            validation_result['reason'] = f"Syntax error: {str(e)}"
            return validation_result
        
        return validation_result
    
    def _execute_with_monitoring(self, code: str, kernel, logger) -> Tuple[str, bool]:
        """Execute code with resource monitoring"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Execute the code
            exec_result = kernel.run_cell(code)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Log resource usage
            logger.debug(f"⏱️ Execution time: {execution_time:.2f}s")
            logger.debug(f"💾 Memory delta: {memory_delta / 1024 / 1024:.2f}MB")
            
            # Check execution success
            if exec_result.success:
                result = str(exec_result.result) if exec_result.result is not None else "Execution completed successfully"
                
                # Validate result size
                if len(result) > 32768:  # 32KB limit
                    logger.warning("⚠️ Result truncated due to size")
                    result = result[:32768] + "...\n[Result truncated due to size]"
                
                return result, True
            else:
                # Execution failed
                error_msg = ''.join(traceback.format_exception(
                    type(exec_result.error_in_exec),
                    exec_result.error_in_exec,
                    exec_result.error_in_exec.__traceback__
                ))
                return error_msg, False
                
        except Exception as e:
            logger.error(f"❌ Execution monitoring error: {str(e)}")
            return str(e), False
    
    def _should_optimize_memory(self) -> bool:
        """Check if memory optimization is needed"""
        if not self.config.get('memory_optimization', True):
            return False
        
        current_memory = self._get_memory_usage()
        return current_memory > (self.memory_threshold * 0.8)  # 80% threshold
    
    def _optimize_memory(self, kernel, logger):
        """Perform memory optimization"""
        logger.info("🧹 Optimizing memory usage...")
        
        # Clear unused variables
        cleanup_code = """
# Memory optimization
import gc
import sys

# Get current variables
current_vars = list(locals().keys())
large_vars = []

for var_name in current_vars:
    if not var_name.startswith('_'):  # Skip special variables
        try:
            var_obj = locals()[var_name]
            if hasattr(var_obj, '__sizeof__'):
                size = sys.getsizeof(var_obj)
                if size > 10 * 1024 * 1024:  # 10MB threshold
                    large_vars.append((var_name, size))
        except:
            pass

print(f"Large variables found: {len(large_vars)}")
for var_name, size in large_vars:
    print(f"  {var_name}: {size / 1024 / 1024:.2f}MB")

# Force garbage collection
gc.collect()
print("Memory optimization completed")
"""
        
        try:
            kernel.run_cell(cleanup_code)
            logger.info("✅ Memory optimization completed")
        except Exception as e:
            logger.warning(f"⚠️ Memory optimization failed: {str(e)}")
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def _record_execution(self, instruction: str, code: str, result: str, success: bool):
        """Record execution for learning and analysis"""
        execution_record = {
            'timestamp': time.time(),
            'instruction': instruction,
            'code': code,
            'result': result,
            'success': success,
            'code_length': len(code),
            'result_length': len(result)
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only recent history to manage memory
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]
    
    def _learn_from_error(self, instruction: str, code: str, error: str):
        """Learn from errors to improve future attempts"""
        
        # Classify error type
        error_type = self._classify_error(error)
        
        # Store solution guidance
        if error_type not in self.common_errors:
            self.common_errors[error_type] = []
        
        # Generate solution guidance based on error type
        solution = self._generate_error_solution(error_type, error)
        
        self.common_errors[error_type].append({
            'instruction_type': self._classify_instruction(instruction),
            'solution': solution,
            'timestamp': time.time()
        })
        
        # Keep only recent errors
        if len(self.common_errors[error_type]) > 5:
            self.common_errors[error_type] = self.common_errors[error_type][-3:]
    
    def _classify_error(self, error: str) -> str:
        """Classify error into categories"""
        error_lower = error.lower()
        
        if 'keyerror' in error_lower:
            return 'KeyError'
        elif 'filenotfounderror' in error_lower:
            return 'FileNotFound'
        elif 'memoryerror' in error_lower:
            return 'MemoryError'
        elif 'valueerror' in error_lower:
            return 'ValueError'
        elif 'typeerror' in error_lower:
            return 'TypeError'
        elif 'indexerror' in error_lower:
            return 'IndexError'
        else:
            return 'UnknownError'
    
    def _classify_instruction(self, instruction: str) -> str:
        """Classify instruction type"""
        instruction_lower = instruction.lower()
        
        if 'load' in instruction_lower or 'read' in instruction_lower:
            return 'data_loading'
        elif 'threshold' in instruction_lower:
            return 'threshold_calculation'
        elif 'anomaly' in instruction_lower:
            return 'anomaly_detection'
        elif 'correlation' in instruction_lower:
            return 'correlation_analysis'
        else:
            return 'general'
    
    def _generate_error_solution(self, error_type: str, error: str) -> str:
        """Generate solution guidance for error type"""
        
        solutions = {
            'KeyError': 'Check if column/key exists before accessing. Use .get() method or validate data structure.',
            'FileNotFound': 'Verify file path exists. Check if data is already loaded in variables.',
            'MemoryError': 'Use chunked processing or reduce data size. Clear intermediate variables.',
            'ValueError': 'Validate input data types and ranges. Handle edge cases.',
            'TypeError': 'Check data types match expected operations. Convert types as needed.',
            'IndexError': 'Validate array/list bounds. Check if data is empty before indexing.',
            'UnknownError': 'Add more error handling and validation steps.'
        }
        
        return solutions.get(error_type, solutions['UnknownError'])
    
    def _get_similar_errors(self, instruction: str) -> Dict[str, str]:
        """Get similar errors and their solutions for instruction type"""
        
        instruction_type = self._classify_instruction(instruction)
        similar_errors = {}
        
        for error_type, error_list in self.common_errors.items():
            for error_record in error_list:
                if error_record['instruction_type'] == instruction_type:
                    similar_errors[error_type] = error_record['solution']
                    break  # Take most recent solution
        
        return similar_errors
    
    def _is_data_loading_instruction(self, instruction: str) -> bool:
        """Check if this instruction is about data loading/retrieval"""
        data_loading_keywords = [
            'load', 'read', 'retrieve', 'get', 'fetch', 'access',
            'telemetry', 'data', 'csv', 'metric', 'trace', 'log',
            'initialize', 'setup', 'explore'
        ]
        
        instruction_lower = instruction.lower()
        return any(keyword in instruction_lower for keyword in data_loading_keywords)
    
    def _generate_intelligent_data_loading_code(self, instruction: str, context: Dict, logger) -> str:
        """Generate intelligent data loading code using data detector"""
        logger.info("🧠 Using intelligent data loading code generation")
        
        # Extract components and time range from instruction
        target_components = extract_components_from_instruction(instruction)
        time_range = extract_time_range_from_instruction(instruction)
        
        # Get dataset from context
        dataset = context.get('domain', 'Bank')  # Default to Bank if not specified
        
        # Map domain to dataset path
        dataset_mapping = {
            'Banking': 'Bank',
            'Telecom': 'Telecom', 
            'Market': 'Market/cloudbed-1',
            'Generic': 'Bank'  # Fallback
        }
        dataset = dataset_mapping.get(dataset, dataset)
        
        logger.info(f"📊 Generating data loading code for dataset: {dataset}")
        logger.info(f"🏷️  Target components: {target_components}")
        logger.info(f"⏰ Time range: {time_range}")
        
        # Generate intelligent loading code
        try:
            code = self.data_detector.generate_data_loading_code(
                dataset=dataset,
                instruction=instruction,
                target_components=target_components,
                time_range=time_range
            )
            
            logger.info("✅ Generated intelligent data loading code")
            return code
            
        except Exception as e:
            logger.error(f"❌ Failed to generate intelligent data loading code: {str(e)}")
            # Fallback to basic loading code
            return self._generate_basic_data_loading_code(dataset, instruction)
    
    def _generate_basic_data_loading_code(self, dataset: str, instruction: str) -> str:
        """Generate basic data loading code as fallback"""
        return f'''
# Basic data loading fallback
import pandas as pd
import os
import glob
from datetime import datetime
import pytz

# Set timezone
TIMEZONE = pytz.timezone('Asia/Shanghai')

print("🔍 Basic data loading for dataset: {dataset}")
print(f"📁 Looking for files in dataset/{dataset}/telemetry/")

# Initialize data containers
telemetry_data = {{}}
loaded_files = {{}}

# Check if dataset exists
dataset_path = "dataset/{dataset}/telemetry"
if os.path.exists(dataset_path):
    print("✅ Dataset path found")
    
    # Get available dates
    dates = [d for d in os.listdir(dataset_path) 
            if os.path.isdir(f"{{dataset_path}}/{{d}}") and d != '.DS_Store']
    print(f"📅 Available dates: {{dates}}")
    
    # Load from first available date
    if dates:
        date = dates[0]
        date_path = f"{{dataset_path}}/{{date}}"
        
        # Load metric files
        metric_path = f"{{date_path}}/metric"
        if os.path.exists(metric_path):
            metric_files = glob.glob(f"{{metric_path}}/*.csv")
            print(f"📈 Found metric files: {{[os.path.basename(f) for f in metric_files]}}")
            
            for file_path in metric_files:
                try:
                    filename = os.path.basename(file_path)
                    df = pd.read_csv(file_path)
                    print(f"  📊 Loaded {{filename}}: {{len(df)}} records")
                    print(f"    Columns: {{list(df.columns)}}")
                    
                    # Store in telemetry_data
                    key = filename.replace('.csv', '')
                    telemetry_data[key] = df
                    loaded_files[filename] = file_path
                    
                except Exception as e:
                    print(f"  ❌ Error loading {{filename}}: {{str(e)}}")
else:
    print(f"❌ Dataset path not found: {{dataset_path}}")
    print("📂 Current directory:", os.getcwd())
    if os.path.exists('dataset'):
        print("📁 Dataset directory contents:", os.listdir('dataset'))

print(f"\n📊 Loaded data keys: {{list(telemetry_data.keys())}}")
print(f"📁 Loaded files: {{list(loaded_files.keys())}}")
'''
