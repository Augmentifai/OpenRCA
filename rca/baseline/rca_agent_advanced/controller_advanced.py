"""
Advanced Controller for RCA Agent with enhanced reasoning capabilities.

Key improvements over basic controller:
- Stage-aware execution with context management
- Enhanced error handling and recovery
- Adaptive reasoning based on analysis progress
- Memory-efficient prompt management
- Multi-hypothesis evaluation
"""

import json
import re
import time
from typing import Dict, List, Tuple, Optional, Any
from IPython.terminal.embed import InteractiveShellEmbed

from rca.baseline.rca_agent_advanced.executor_advanced import AdvancedExecutor
from rca.api_router import get_chat_completion


class AdvancedController:
    """Advanced controller with stage-aware reasoning and enhanced error handling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.executor = AdvancedExecutor(config)
        self.context_memory = {}
        self.error_patterns = []
        
    def execute_stage(self, stage, context: Dict, agent_prompt, basic_prompt, logger, 
                     max_step: int = 10, max_turn: int = 3) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Execute a specific analysis stage with enhanced reasoning.
        
        Args:
            stage: The analysis stage to execute
            context: Analysis context and state
            agent_prompt: Agent prompt module
            basic_prompt: Basic prompt module  
            logger: Logger instance
            max_step: Maximum steps for this stage
            max_turn: Maximum retry turns
            
        Returns:
            Tuple of (prediction, trajectory, prompts)
        """
        logger.info(f"🎯 Executing stage: {stage.value}")
        
        # Create stage-specific system prompt
        system_prompt = self._create_stage_system_prompt(stage, context, agent_prompt, basic_prompt)
        
        # Initialize stage execution
        prompt_history = [{'role': 'system', 'content': system_prompt}]
        trajectory = []
        
        # Create IPython kernel with enhanced initialization
        kernel = self._initialize_enhanced_kernel()
        
        # Execute stage with adaptive reasoning
        try:
            prediction = self._execute_stage_reasoning(
                stage, context, prompt_history, trajectory, kernel, logger, max_step, max_turn
            )
            
            return prediction, trajectory, prompt_history
            
        except Exception as e:
            logger.error(f"❌ Stage execution failed: {str(e)}")
            
            # Attempt recovery if enabled
            if self.config.get('error_recovery', True):
                return self._handle_stage_error(stage, context, str(e), logger)
            else:
                raise
        finally:
            # Clean up kernel
            if 'kernel' in locals():
                kernel.reset()
    
    def _create_stage_system_prompt(self, stage, context: Dict, agent_prompt, basic_prompt) -> str:
        """Create stage-specific system prompt with enhanced context"""
        
        stage_instructions = {
            'initialization': """
You are starting the RCA analysis. Focus on:
1. Understanding the problem scope and timeframe
2. Identifying available data sources
3. Setting up the analysis environment
4. Planning the investigation approach
""",
            'data_exploration': """
You are exploring the available telemetry data. Focus on:
1. CRITICAL: Load actual CSV files from dataset/{domain}/telemetry/ directories
2. Load metric_app.csv, metric_container.csv from metric/ subdirectory  
3. Load trace_span.csv from trace/ subdirectory
4. Load log_service.csv from log/ subdirectory
5. Examine data structure, columns, and sample records
6. Identify available components, KPIs, and time ranges
7. NEVER create dummy/simulated data - always use real files!
""",
            'anomaly_detection': """
You are detecting anomalies in the system. Focus on:
1. Calculating appropriate thresholds for each KPI
2. Identifying anomalous data points and patterns
3. Filtering out noise and false positives
4. Documenting significant anomalies found
""",
            'correlation_analysis': """
You are analyzing correlations between different signals. Focus on:
1. Cross-referencing anomalies across different data sources
2. Identifying temporal correlations and causal relationships
3. Analyzing service dependencies and call chains
4. Building a timeline of events
""",
            'root_cause_localization': """
You are localizing the root cause. Focus on:
1. Identifying the most likely root cause from detected faults
2. Validating the root cause hypothesis with evidence
3. Determining the exact component, time, and reason
4. Ensuring the analysis follows the specified workflow
""",
            'validation': """
You are validating the identified root cause. Focus on:
1. Cross-checking findings against multiple data sources
2. Verifying the timeline and causal chain
3. Ensuring the root cause explains all observed symptoms
4. Confirming the analysis meets quality standards
""",
            'finalization': """
You are finalizing the RCA analysis. Focus on:
1. Summarizing key findings and evidence
2. Formatting the final answer according to requirements
3. Ensuring all required fields are included
4. Providing a clear, actionable conclusion
"""
        }
        
        stage_instruction = stage_instructions.get(stage.value, "Continue with the RCA analysis.")
        
        # Enhanced system prompt with stage context
        enhanced_system = f"""You are an Advanced Software Engineer specialized in RCA (Root Cause Analysis) with vast experience.

CURRENT ANALYSIS STAGE: {stage.value.upper()}
DETECTED DOMAIN: {context.get('domain', 'Unknown')}
COMPLEXITY LEVEL: {context.get('complexity', 'Unknown')}

{stage_instruction}

DOMAIN KNOWLEDGE:
{basic_prompt.schema}

RCA METHODOLOGY RULES:
{agent_prompt.rules}

ANALYSIS CONTEXT:
- Problem: {context.get('instruction', 'Not specified')}
- Stage: {stage.value}
- Expected Duration: {context.get('resource_requirements', {}).get('max_time_minutes', 15)} minutes
- Memory Limit: {context.get('resource_requirements', {}).get('memory_mb', 2048)} MB

Your response should follow the JSON format below:
{{
    "analysis": (Your analysis of the current situation and what needs to be done next),
    "completed": ("True" if this stage is complete and you have sufficient findings, otherwise "False"),
    "instruction": (Your instruction for the Executor, or final summary if completed),
    "confidence": (Your confidence level 0.0-1.0 in the current findings),
    "evidence": (List of key evidence found so far),
    "next_stage_ready": ("True" if ready to proceed to next stage, otherwise "False")
}}

Begin the {stage.value} analysis now."""

        return enhanced_system
    
    def _initialize_enhanced_kernel(self) -> InteractiveShellEmbed:
        """Initialize IPython kernel with enhanced setup"""
        kernel = InteractiveShellEmbed()
        
        # Enhanced initialization code with better memory management
        init_code = """
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optimize pandas for memory efficiency
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 50)
pd.set_option('mode.chained_assignment', None)

# Set timezone for consistent time handling
TIMEZONE = pytz.timezone('Asia/Shanghai')

# Helper functions for common RCA tasks
def safe_percentile(data, percentile):
    \"\"\"Safely calculate percentile handling edge cases\"\"\"
    try:
        if len(data) == 0:
            return None
        return np.percentile(data, percentile)
    except:
        return None

def detect_anomalies(series, threshold_method='percentile', threshold=95):
    \"\"\"Enhanced anomaly detection with multiple methods\"\"\"
    if len(series) == 0:
        return []
    
    if threshold_method == 'percentile':
        threshold_value = safe_percentile(series, threshold)
        if threshold_value is None:
            return []
        return series > threshold_value
    elif threshold_method == 'std':
        mean_val = series.mean()
        std_val = series.std()
        return abs(series - mean_val) > (threshold * std_val)
    else:
        return []

print("🔧 Advanced RCA environment initialized")
"""
        
        kernel.run_cell(init_code)
        return kernel
    
    def _execute_stage_reasoning(self, stage, context: Dict, prompt_history: List[Dict], 
                                trajectory: List[Dict], kernel, logger, max_step: int, max_turn: int) -> str:
        """Execute reasoning for a specific stage with enhanced logic"""
        
        # Start the conversation
        initial_prompt = f"Begin {stage.value} analysis for the problem: {context['instruction']}"
        prompt_history.append({'role': 'user', 'content': initial_prompt})
        
        step_count = 0
        retry_count = 0
        
        while step_count < max_step and retry_count < max_turn:
            try:
                # Get AI response with enhanced error handling
                response_raw = self._get_enhanced_ai_response(prompt_history, logger)
                
                # Parse and validate response
                response = self._parse_and_validate_response(response_raw, logger)
                
                if not response:
                    retry_count += 1
                    continue
                
                analysis = response.get('analysis', '')
                instruction = response.get('instruction', '')
                completed = response.get('completed', 'False')
                confidence = response.get('confidence', 0.5)
                evidence = response.get('evidence', [])
                
                logger.info(f"📊 Step {step_count + 1} - Confidence: {confidence:.2f}")
                logger.info(f"🔍 Analysis: {analysis}")
                logger.info(f"📝 Instruction: {instruction}")
                
                # Add to prompt history
                prompt_history.append({'role': 'assistant', 'content': response_raw})
                
                # Check if stage is completed
                if completed == "True":
                    logger.info(f"✅ Stage {stage.value} completed successfully")
                    return instruction
                
                # Execute the instruction with enhanced executor
                execution_result = self.executor.execute_with_recovery(
                    instruction, context, kernel, logger
                )
                
                code, result, success = execution_result
                
                # Record in trajectory
                trajectory.append({
                    'step': step_count + 1,
                    'stage': stage.value,
                    'code': code,
                    'result': result,
                    'success': success,
                    'confidence': confidence,
                    'evidence': evidence
                })
                
                # Provide feedback to AI
                if success:
                    feedback = f"Execution successful. Result:\n{result}"
                else:
                    feedback = f"Execution failed. Error:\n{result}"
                    retry_count += 1
                
                prompt_history.append({'role': 'user', 'content': feedback})
                step_count += 1
                
            except Exception as e:
                logger.error(f"🚨 Error in step {step_count + 1}: {str(e)}")
                retry_count += 1
                
                if retry_count >= max_turn:
                    raise Exception(f"Max retries exceeded in stage {stage.value}")
                
                # Add error feedback
                prompt_history.append({
                    'role': 'user', 
                    'content': f"An error occurred: {str(e)}. Please adjust your approach."
                })
        
        # If we reach here, we've exhausted steps without completion
        logger.warning(f"⚠️ Stage {stage.value} reached maximum steps without completion")
        
        # Generate summary from what we have
        summary_prompt = f"""
The {stage.value} stage has reached maximum steps. Based on the analysis so far, 
provide your best assessment and findings. Focus on:
1. Key insights discovered
2. Evidence gathered
3. Confidence in findings
4. Recommendations for next steps

Provide a final summary for this stage.
"""
        
        prompt_history.append({'role': 'user', 'content': summary_prompt})
        
        try:
            final_response = self._get_enhanced_ai_response(prompt_history, logger)
            return final_response
        except:
            return f"Stage {stage.value} completed with partial results"
    
    def _get_enhanced_ai_response(self, prompt_history: List[Dict], logger) -> str:
        """Get AI response with enhanced error handling and retries"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = get_chat_completion(messages=prompt_history)
                
                # Basic validation
                if not response or len(response.strip()) < 10:
                    raise ValueError("Response too short or empty")
                
                return response
                
            except Exception as e:
                logger.warning(f"⚠️ AI request attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"All AI request attempts failed: {str(e)}")
    
    def _parse_and_validate_response(self, response_raw: str, logger) -> Optional[Dict]:
        """Parse and validate AI response with enhanced error handling"""
        try:
            # Extract JSON from response
            if "```json" in response_raw:
                json_match = re.search(r"```json\n(.*?)\n```", response_raw, re.DOTALL)
                if json_match:
                    response_raw = json_match.group(1).strip()
            
            # Clean up response
            response_raw = response_raw.strip()
            if not response_raw.startswith('{'):
                # Try to find JSON-like content
                json_start = response_raw.find('{')
                if json_start != -1:
                    response_raw = response_raw[json_start:]
            
            # Parse JSON
            response = json.loads(response_raw)
            
            # Validate required fields
            required_fields = ['analysis', 'completed', 'instruction']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                logger.warning(f"⚠️ Response missing fields: {missing_fields}")
                # Try to construct a valid response
                response.update({
                    field: f"Missing {field}" for field in missing_fields
                })
            
            # Ensure confidence is numeric
            if 'confidence' in response:
                try:
                    response['confidence'] = float(response['confidence'])
                except:
                    response['confidence'] = 0.5
            
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON response: {str(e)}")
            logger.debug(f"Raw response: {response_raw[:500]}...")
            return None
        except Exception as e:
            logger.error(f"❌ Response validation failed: {str(e)}")
            return None
    
    def _handle_stage_error(self, stage, context: Dict, error: str, logger) -> Tuple[str, List[Dict], List[Dict]]:
        """Handle stage execution errors with recovery strategies"""
        logger.info(f"🔧 Attempting error recovery for stage {stage.value}")
        
        # Create a simplified fallback response
        fallback_prediction = f"Stage {stage.value} encountered an error: {error}. Proceeding with best available information."
        
        fallback_trajectory = [{
            'step': 1,
            'stage': stage.value,
            'code': '# Error recovery mode',
            'result': f'Stage failed: {error}',
            'success': False,
            'confidence': 0.1,
            'evidence': [f'Error in {stage.value}: {error}']
        }]
        
        fallback_prompts = [{
            'role': 'system',
            'content': f'Error recovery for stage {stage.value}'
        }]
        
        return fallback_prediction, fallback_trajectory, fallback_prompts
