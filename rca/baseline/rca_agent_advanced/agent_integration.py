"""
Integration module for the advanced agent system with stage agents.

This module provides the interface between the existing controller and the new 
stage-based agent system, enabling seamless integration into the existing RCA pipeline.
"""

import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

from rca.baseline.rca_agent_advanced.agents import StageAgentCoordinator
from rca.baseline.rca_agent_advanced.executor_advanced import AdvancedExecutor


class StageType(Enum):
    """RCA analysis stages matching the existing controller's stages"""
    INITIALIZATION = "initialization"
    DATA_EXPLORATION = "data_exploration"
    ANOMALY_DETECTION = "anomaly_detection"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ROOT_CAUSE_LOCALIZATION = "root_cause_localization"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class AgentIntegration:
    """
    Integrates the stage-based agent system with the existing controller
    
    This class provides a wrapper around the stage agent coordinator to make it
    compatible with the existing controller interface, allowing gradual transition
    to the new agent system.
    """
    
    def __init__(self, config: Dict, logger=None):
        """Initialize the integration module"""
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # Initialize executor and agent coordinator
        self.executor = AdvancedExecutor(config)
        self.agent_coordinator = StageAgentCoordinator(config)
        
        # Shared context for coordination
        self.shared_context = {}
        
        # Map controller stages to agent stages
        self.stage_mapping = {
            StageType.INITIALIZATION: "dataset_exploration",
            StageType.DATA_EXPLORATION: "dataset_exploration", 
            StageType.ANOMALY_DETECTION: "anomaly_detection",
            StageType.CORRELATION_ANALYSIS: "correlation_analysis",
            StageType.ROOT_CAUSE_LOCALIZATION: "root_cause_localization",
            StageType.VALIDATION: "root_cause_localization",
            StageType.FINALIZATION: "root_cause_localization"
        }
        
        self.logger.info("🔗 Agent Integration System initialized")
    
    def execute_stage(self, stage: StageType, context: Dict, instruction: str, kernel) -> Tuple[str, List[Dict], Dict]:
        """
        Execute a stage using the appropriate specialized agent
        
        Args:
            stage: The analysis stage to execute
            context: Analysis context and state
            instruction: The instruction to process
            kernel: IPython kernel for execution
            
        Returns:
            Tuple of (result, trajectory, metadata)
        """
        self.logger.info(f"🚀 Executing {stage.value} with agent system")
        
        # Update shared context with controller context
        self._update_shared_context(context)
        
        # Map controller stage to agent stage
        agent_stage = self.stage_mapping.get(stage, "dataset_exploration")
        
        # Check if this is a data loading stage that needs direct executor
        if self._is_data_loading_stage(stage, instruction):
            return self._execute_data_loading(instruction, context, kernel)
        
        # Use agent coordinator for this stage
        try:
            result, trajectory, confidence = self.agent_coordinator.execute_stage(
                agent_stage,
                self.shared_context,
                instruction,
                self.logger
            )
            
            # Format result for controller compatibility
            formatted_result = self._format_stage_result(
                stage=stage,
                agent_stage=agent_stage,
                raw_result=result,
                confidence=confidence
            )
            
            # Update shared context
            self.shared_context[f"{agent_stage}_result"] = {
                "result": result,
                "confidence": confidence
            }
            
            return formatted_result, trajectory, {"confidence": confidence, "stage": agent_stage}
            
        except Exception as e:
            self.logger.error(f"❌ Agent stage execution failed: {str(e)}")
            return self._handle_agent_error(stage, str(e))
    
    def _update_shared_context(self, context: Dict) -> None:
        """Update shared context with controller context"""
        # Copy relevant context fields
        relevant_keys = [
            'domain', 'instruction', 'dataset_path', 'telemetry_files',
            'findings', 'root_cause_candidates'
        ]
        
        for key in relevant_keys:
            if key in context:
                self.shared_context[key] = context[key]
        
        # Add domain name if available
        if 'domain' not in self.shared_context and 'domain' in context:
            self.shared_context['domain'] = context['domain']
    
    def _is_data_loading_stage(self, stage: StageType, instruction: str) -> bool:
        """Check if this is a data loading stage needing direct execution"""
        data_loading_stages = [StageType.INITIALIZATION, StageType.DATA_EXPLORATION]
        
        if stage not in data_loading_stages:
            return False
        
        # Check instruction for data loading keywords
        data_loading_keywords = [
            'load', 'read', 'csv', 'data', 'file', 'telemetry', 
            'fetch', 'import', 'dataset'
        ]
        
        instruction_lower = instruction.lower()
        return any(keyword in instruction_lower for keyword in data_loading_keywords)
    
    def _execute_data_loading(self, instruction: str, context: Dict, kernel) -> Tuple[str, List[Dict], Dict]:
        """Execute data loading using the executor directly"""
        self.logger.info("📊 Executing data loading with executor")
        
        # Execute with enhanced executor
        code, result, success = self.executor.execute_with_recovery(
            instruction, 
            context, 
            kernel, 
            self.logger
        )
        
        confidence = 0.9 if success else 0.2
        
        # Format trajectory
        trajectory = [{
            'step': 1,
            'action': 'data_loading',
            'code': code,
            'result': result[:500] + '...' if len(result) > 500 else result,
            'success': success,
            'confidence': confidence
        }]
        
        # Format result for controller compatibility
        formatted_result = {
            "analysis": f"Data loading {'successful' if success else 'failed'}",
            "completed": str(success),
            "instruction": result[:1000] + '...' if len(result) > 1000 else result,
            "confidence": confidence,
            "evidence": [f"Data loading {'succeeded' if success else 'failed'}"],
            "next_stage_ready": str(success)
        }
        
        # Convert to JSON string for controller compatibility
        import json
        formatted_result_str = json.dumps(formatted_result, indent=2)
        
        return formatted_result_str, trajectory, {"confidence": confidence, "stage": "data_loading"}
    
    def _format_stage_result(self, stage: StageType, agent_stage: str, raw_result: str, confidence: float) -> str:
        """Format stage result for controller compatibility"""
        # Convert raw result to controller-expected JSON format
        formatted_result = {
            "analysis": f"Advanced agent completed {agent_stage} analysis",
            "completed": "True" if confidence > 0.6 else "False",
            "instruction": raw_result[:1500] + '...' if len(raw_result) > 1500 else raw_result,
            "confidence": confidence,
            "evidence": [f"Agent analysis with confidence {confidence:.2f}"],
            "next_stage_ready": "True" if confidence > 0.5 else "False"
        }
        
        # Convert to JSON string for controller compatibility
        import json
        return json.dumps(formatted_result, indent=2)
    
    def _handle_agent_error(self, stage: StageType, error: str) -> Tuple[str, List[Dict], Dict]:
        """Handle agent execution errors with recovery"""
        self.logger.warning(f"⚠️ Handling error in {stage.value}: {error}")
        
        # Create fallback response
        import json
        fallback_result = {
            "analysis": f"Agent encountered an error in {stage.value} stage",
            "completed": "False",
            "instruction": f"Error occurred: {error}. Please try an alternative approach.",
            "confidence": 0.1,
            "evidence": [f"Error in {stage.value}: {error}"],
            "next_stage_ready": "False"
        }
        
        fallback_trajectory = [{
            'step': 1,
            'stage': stage.value,
            'action': 'error_recovery',
            'result': f'Stage failed: {error}',
            'success': False,
            'confidence': 0.1
        }]
        
        return json.dumps(fallback_result, indent=2), fallback_trajectory, {"confidence": 0.1, "stage": "error"}
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the integration module"""
        logger = logging.getLogger("AgentIntegration")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        return logger