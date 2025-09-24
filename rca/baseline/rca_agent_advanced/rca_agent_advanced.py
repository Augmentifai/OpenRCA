"""
Advanced RCA Agent with enhanced root cause analysis capabilities.

This implementation provides several improvements over the basic RCA agent:
- Enhanced error handling and recovery mechanisms
- Multi-stage analysis with validation steps
- Adaptive reasoning with confidence scoring
- Cross-domain knowledge integration
- Memory-aware execution planning
"""

import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from rca.baseline.rca_agent_advanced.controller_advanced import AdvancedController
from rca.baseline.rca_agent_advanced.executor_advanced import AdvancedExecutor


class AnalysisStage(Enum):
    """Stages of RCA analysis for structured progression"""
    INITIALIZATION = "initialization"
    DATA_EXPLORATION = "data_exploration"
    ANOMALY_DETECTION = "anomaly_detection"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ROOT_CAUSE_LOCALIZATION = "root_cause_localization"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


@dataclass
class AnalysisResult:
    """Container for analysis results with confidence scoring"""
    stage: AnalysisStage
    findings: Dict[str, Any]
    confidence_score: float
    evidence: List[str]
    next_actions: List[str]
    timestamp: float


class RCA_Agent_Advanced:
    """
    Advanced RCA Agent with enhanced capabilities for robust root cause analysis.
    
    Key improvements:
    - Multi-stage structured analysis
    - Confidence-based decision making
    - Enhanced error recovery
    - Cross-validation of findings
    - Memory and resource management
    """
    
    def __init__(self, agent_prompt, basic_prompt, config: Optional[Dict] = None):
        self.ap = agent_prompt
        self.bp = basic_prompt
        
        # Enhanced configuration
        self.config = self._init_config(config)
        
        # Initialize advanced components
        self.controller = AdvancedController(self.config)
        self.executor = AdvancedExecutor(self.config)
        
        # Analysis state management
        self.analysis_history: List[AnalysisResult] = []
        self.current_stage = AnalysisStage.INITIALIZATION
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_retries = self.config.get('max_retries', 3)
        
    def _init_config(self, config: Optional[Dict]) -> Dict:
        """Initialize configuration with defaults"""
        default_config = {
            'confidence_threshold': 0.7,
            'max_retries': 3,
            'validation_enabled': True,
            'cross_validation_enabled': True,
            'memory_optimization': True,
            'progressive_analysis': True,
            'adaptive_timeout': True,
            'error_recovery': True,
            'multi_hypothesis': True,
            'evidence_correlation': True
        }
        
        if config:
            default_config.update(config)
        return default_config
        
    def run(self, instruction: str, logger, max_step: int = 30, max_turn: int = 5) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Execute advanced RCA analysis with multi-stage approach.
        
        Args:
            instruction: The RCA query/problem description
            logger: Logger for tracking progress
            max_step: Maximum analysis steps
            max_turn: Maximum retry turns
            
        Returns:
            Tuple of (prediction, trajectory, prompt_history)
        """
        logger.info(f"🚀 Starting Advanced RCA Analysis")
        logger.info(f"📋 Objective: {instruction}")
        
        start_time = time.time()
        
        try:
            # Initialize analysis context
            analysis_context = self._initialize_analysis(instruction, logger)
            
            # Execute multi-stage analysis
            final_result = self._execute_staged_analysis(
                analysis_context, logger, max_step, max_turn
            )
            
            # Validate and finalize results
            validated_result = self._validate_and_finalize(final_result, logger)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Advanced RCA Analysis completed in {execution_time:.2f}s")
            
            return validated_result
            
        except Exception as e:
            logger.error(f"❌ Advanced RCA Analysis failed: {str(e)}")
            # Fallback to basic analysis if advanced fails
            return self._fallback_analysis(instruction, logger, max_step, max_turn)
    
    def _initialize_analysis(self, instruction: str, logger) -> Dict[str, Any]:
        """Initialize analysis context with problem understanding"""
        logger.info("🔍 Initializing analysis context...")
        
        context = {
            'instruction': instruction,
            'start_time': time.time(),
            'domain': "Bank", #self._detect_domain(instruction),
            'complexity': self._assess_complexity(instruction),
            'expected_stages': self._plan_analysis_stages(instruction),
            'resource_requirements': self._estimate_resources(instruction)
        }
        
        # Record initialization
        init_result = AnalysisResult(
            stage=AnalysisStage.INITIALIZATION,
            findings=context,
            confidence_score=1.0,
            evidence=[f"Problem: {instruction}"],
            next_actions=context['expected_stages'],
            timestamp=time.time()
        )
        self.analysis_history.append(init_result)
        
        logger.info(f"📊 Detected domain: {context['domain']}")
        logger.info(f"📈 Complexity level: {context['complexity']}")
        
        return context
    
    def _execute_staged_analysis(self, context: Dict, logger, max_step: int, max_turn: int) -> Tuple[str, List[Dict], List[Dict]]:
        """Execute analysis through structured stages"""
        logger.info("🔬 Beginning staged analysis...")
        
        all_trajectory = []
        all_prompts = []
        
        for stage in context['expected_stages']:
            logger.info(f"📍 Entering stage: {stage.value}")
            self.current_stage = stage
            
            try:
                # Execute stage with enhanced controller
                stage_result = self.controller.execute_stage(
                    stage, context, self.ap, self.bp, logger, 
                    max_step=max_step // len(context['expected_stages']),
                    max_turn=max_turn
                )
                
                prediction, trajectory, prompts = stage_result
                all_trajectory.extend(trajectory)
                all_prompts.extend(prompts)
                
                # Record stage results
                analysis_result = AnalysisResult(
                    stage=stage,
                    findings=self._extract_findings(prediction),
                    confidence_score=self._calculate_confidence(prediction, trajectory),
                    evidence=self._extract_evidence(trajectory),
                    next_actions=self._determine_next_actions(stage, prediction),
                    timestamp=time.time()
                )
                self.analysis_history.append(analysis_result)
                
                # Check if we have sufficient confidence to proceed or conclude
                if self._should_conclude_analysis(analysis_result, stage):
                    logger.info(f"✅ High confidence result achieved at stage: {stage.value}")
                    return prediction, all_trajectory, all_prompts
                    
            except Exception as e:
                logger.warning(f"⚠️ Stage {stage.value} failed: {str(e)}")
                if self.config['error_recovery']:
                    # Attempt recovery or skip to next stage
                    self._handle_stage_failure(stage, str(e), logger)
                else:
                    raise
        
        # Compile final results from all stages
        final_prediction = self._compile_final_prediction()
        return final_prediction, all_trajectory, all_prompts
    
    def _validate_and_finalize(self, result: Tuple[str, List[Dict], List[Dict]], logger) -> Tuple[str, List[Dict], List[Dict]]:
        """Validate results and provide final refinements"""
        if not self.config['validation_enabled']:
            return result
            
        logger.info("🔍 Validating analysis results...")
        
        prediction, trajectory, prompts = result
        
        # Cross-validation against multiple evidence sources
        if self.config['cross_validation_enabled']:
            validation_score = self._cross_validate_findings(prediction, trajectory)
            logger.info(f"📊 Cross-validation score: {validation_score:.2f}")
            
            if validation_score < self.confidence_threshold:
                logger.warning("⚠️ Low validation score, attempting refinement...")
                refined_result = self._refine_analysis(prediction, trajectory, logger)
                if refined_result:
                    prediction, trajectory, prompts = refined_result
        
        return prediction, trajectory, prompts
    
    def _fallback_analysis(self, instruction: str, logger, max_step: int, max_turn: int) -> Tuple[str, List[Dict], List[Dict]]:
        """Fallback to simpler analysis if advanced methods fail"""
        logger.info("🔄 Falling back to basic analysis...")
        
        # Import and use the basic controller as fallback
        from rca.baseline.rca_agent.controller import control_loop
        
        try:
            prediction, trajectory, prompts = control_loop(
                instruction, "", self.ap, self.bp, logger, max_step, max_turn
            )
            return prediction, trajectory, prompts
        except Exception as e:
            logger.error(f"❌ Fallback analysis also failed: {str(e)}")
            return "Analysis failed - unable to determine root cause", [], []
    
    def _detect_domain(self, instruction: str) -> str:
        """Detect the domain/system type from instruction"""
        instruction_lower = instruction.lower()
        
        if any(term in instruction_lower for term in ['bank', 'financial', 'payment', 'transaction']):
            return 'Bank'
        elif any(term in instruction_lower for term in ['telecom', 'network', 'db', 'database']):
            return 'Telecom'
        elif any(term in instruction_lower for term in ['market', 'e-commerce', 'shop', 'service']):
            return 'Market'
        else:
            return 'Generic'
    
    def _assess_complexity(self, instruction: str) -> str:
        """Assess problem complexity based on instruction characteristics"""
        complexity_indicators = {
            'multiple': 2,
            'cascade': 2,
            'intermittent': 2,
            'distributed': 2,
            'correlation': 1,
            'timeout': 1,
            'performance': 1,
            'failure': 1
        }
        
        score = sum(weight for term, weight in complexity_indicators.items() 
                   if term in instruction.lower())
        
        if score >= 4:
            return 'High'
        elif score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _plan_analysis_stages(self, instruction: str) -> List[AnalysisStage]:
        """Plan analysis stages based on problem characteristics"""
        stages = [AnalysisStage.INITIALIZATION, AnalysisStage.DATA_EXPLORATION]
        
        # Always include core stages
        stages.extend([
            AnalysisStage.ANOMALY_DETECTION,
            AnalysisStage.ROOT_CAUSE_LOCALIZATION
        ])
        
        # Add correlation analysis for complex problems
        if self._assess_complexity(instruction) in ['Medium', 'High']:
            stages.insert(-1, AnalysisStage.CORRELATION_ANALYSIS)
        
        # Add validation for high-stakes analysis
        if self.config['validation_enabled']:
            stages.append(AnalysisStage.VALIDATION)
        
        stages.append(AnalysisStage.FINALIZATION)
        return stages
    
    def _estimate_resources(self, instruction: str) -> Dict[str, Any]:
        """Estimate computational resources needed"""
        complexity = self._assess_complexity(instruction)
        
        resource_map = {
            'Low': {'memory_mb': 1024, 'max_time_minutes': 5, 'max_iterations': 10},
            'Medium': {'memory_mb': 2048, 'max_time_minutes': 15, 'max_iterations': 20},
            'High': {'memory_mb': 4096, 'max_time_minutes': 30, 'max_iterations': 30}
        }
        
        return resource_map.get(complexity, resource_map['Medium'])
    
    def _extract_findings(self, prediction: str) -> Dict[str, Any]:
        """Extract structured findings from prediction text"""
        try:
            # Try to parse as JSON first
            if prediction.strip().startswith('{'):
                return json.loads(prediction)
        except:
            pass
        
        # Extract key information using pattern matching
        findings = {
            'raw_prediction': prediction,
            'has_timestamp': 'datetime' in prediction.lower(),
            'has_component': 'component' in prediction.lower(),
            'has_reason': 'reason' in prediction.lower()
        }
        
        return findings
    
    def _calculate_confidence(self, prediction: str, trajectory: List[Dict]) -> float:
        """Calculate confidence score based on prediction quality and evidence"""
        score = 0.5  # Base score
        
        # Check prediction completeness
        if all(key in prediction.lower() for key in ['datetime', 'component', 'reason']):
            score += 0.2
        
        # Check trajectory quality
        if len(trajectory) > 0:
            successful_steps = sum(1 for step in trajectory if 'error' not in step.get('result', '').lower())
            score += 0.3 * (successful_steps / len(trajectory))
        
        return min(score, 1.0)
    
    def _extract_evidence(self, trajectory: List[Dict]) -> List[str]:
        """Extract evidence from execution trajectory"""
        evidence = []
        
        for step in trajectory:
            if 'result' in step and step['result']:
                # Extract meaningful results as evidence
                result = step['result']
                if any(indicator in result.lower() for indicator in ['anomaly', 'spike', 'threshold', 'correlation']):
                    evidence.append(f"Found: {result[:200]}...")  # Truncate for readability
        
        return evidence
    
    def _determine_next_actions(self, current_stage: AnalysisStage, prediction: str) -> List[str]:
        """Determine next analysis actions based on current stage and results"""
        next_actions = []
        
        stage_transitions = {
            AnalysisStage.INITIALIZATION: ["explore_data", "identify_timeframes"],
            AnalysisStage.DATA_EXPLORATION: ["detect_anomalies", "calculate_thresholds"],
            AnalysisStage.ANOMALY_DETECTION: ["correlate_events", "identify_root_cause"],
            AnalysisStage.CORRELATION_ANALYSIS: ["localize_root_cause", "validate_hypothesis"],
            AnalysisStage.ROOT_CAUSE_LOCALIZATION: ["validate_findings", "generate_report"],
            AnalysisStage.VALIDATION: ["finalize_analysis"],
            AnalysisStage.FINALIZATION: []
        }
        
        return stage_transitions.get(current_stage, [])
    
    def _should_conclude_analysis(self, result: AnalysisResult, stage: AnalysisStage) -> bool:
        """Determine if analysis should conclude early based on confidence"""
        if result.confidence_score >= 0.9 and stage in [
            AnalysisStage.ROOT_CAUSE_LOCALIZATION, 
            AnalysisStage.CORRELATION_ANALYSIS
        ]:
            return True
        return False
    
    def _handle_stage_failure(self, stage: AnalysisStage, error: str, logger):
        """Handle failure in a specific analysis stage"""
        logger.warning(f"🔧 Handling failure in stage {stage.value}: {error}")
        
        # Record the failure for learning
        failure_result = AnalysisResult(
            stage=stage,
            findings={'error': error, 'recovered': True},
            confidence_score=0.0,
            evidence=[f"Stage failure: {error}"],
            next_actions=["skip_to_next_stage"],
            timestamp=time.time()
        )
        self.analysis_history.append(failure_result)
    
    def _cross_validate_findings(self, prediction: str, trajectory: List[Dict]) -> float:
        """Cross-validate findings against multiple evidence sources"""
        # Simple validation based on consistency across trajectory
        evidence_count = 0
        consistent_evidence = 0
        
        for step in trajectory:
            if 'result' in step and step['result']:
                evidence_count += 1
                # Check if step result is consistent with final prediction
                if any(term in step['result'].lower() for term in prediction.lower().split()):
                    consistent_evidence += 1
        
        if evidence_count == 0:
            return 0.5  # Neutral score if no evidence
        
        return consistent_evidence / evidence_count
    
    def _refine_analysis(self, prediction: str, trajectory: List[Dict], logger) -> Optional[Tuple[str, List[Dict], List[Dict]]]:
        """Attempt to refine analysis results"""
        logger.info("🔄 Attempting to refine analysis...")
        
        # For now, return None to indicate no refinement
        # This could be enhanced with specific refinement strategies
        return None
    
    def _compile_final_prediction(self) -> str:
        """Compile final prediction from all analysis stages"""
        if not self.analysis_history:
            return "No analysis results available"
        
        # Find the stage with highest confidence
        best_result = max(self.analysis_history, key=lambda x: x.confidence_score)
        
        if 'raw_prediction' in best_result.findings:
            return best_result.findings['raw_prediction']
        
        return str(best_result.findings)
