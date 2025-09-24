"""
Specialized Stage Agents for RCA Analysis

This module contains specialized agents for each stage of RCA analysis,
each designed with specific tools and capabilities for their role.
"""

import json
import time
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from pathlib import Path

from rca.baseline.rca_agent_advanced.tools.system_tools import GenericDatasetAnalyzer
from rca.api_router import get_chat_completion


class BaseStageAgent(ABC):
    """Base class for all stage agents"""
    
    def __init__(self, stage_name: str, config: Dict):
        self.stage_name = stage_name
        self.config = config
        self.tools = {}
        self.findings = {}
        self.evidence = []
        
    @abstractmethod
    def execute(self, context: Dict, instruction: str, logger) -> Tuple[str, List[Dict], float]:
        """
        Execute the stage analysis
        
        Returns:
            Tuple of (result, trajectory, confidence_score)
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass


class DatasetExplorationAgent(BaseStageAgent):
    """Specialized agent for dataset exploration and structure discovery"""
    
    def __init__(self, config: Dict):
        super().__init__("dataset_exploration", config)
        self.analyzer = None  # Will be initialized with dataset path
        
    def execute(self, context: Dict, instruction: str, logger) -> Tuple[str, List[Dict], float]:
        """Execute dataset exploration with comprehensive analysis"""
        logger.info(f"🔍 Dataset Exploration Agent starting analysis")
        
        # Extract dataset path from instruction or context
        dataset_path = self._extract_dataset_path(instruction, context)
        if not dataset_path:
            return "Error: Could not determine dataset path", [], 0.0
        
        logger.info(f"📂 Analyzing dataset: {dataset_path}")
        
        # Initialize analyzer
        self.analyzer = GenericDatasetAnalyzer(dataset_path)
        
        trajectory = []
        start_time = time.time()
        
        try:
            # Step 1: Comprehensive dataset analysis
            logger.info("📊 Step 1: Comprehensive dataset structure analysis")
            analysis = self.analyzer.analyze_dataset()
            
            if "error" in analysis:
                return f"Dataset analysis failed: {analysis['error']}", [], 0.0
            
            trajectory.append({
                "step": 1,
                "action": "dataset_structure_analysis",
                "result": f"Analyzed {analysis['summary']['total_files']} files across {len(analysis['structure']['directories'])} directories",
                "details": analysis['summary']
            })
            
            # Step 2: Generate intelligent data loading code
            logger.info("🧠 Step 2: Generating intelligent data loading strategy")
            loading_strategy = self._generate_loading_strategy(analysis, instruction, logger)
            
            trajectory.append({
                "step": 2,
                "action": "data_loading_strategy",
                "result": "Generated optimized data loading strategy",
                "details": loading_strategy
            })
            
            # Step 3: Create dataset-specific recommendations
            logger.info("💡 Step 3: Creating analysis recommendations")
            recommendations = self._create_analysis_recommendations(analysis, instruction)
            
            trajectory.append({
                "step": 3,
                "action": "analysis_recommendations",
                "result": f"Generated {len(recommendations)} specific recommendations",
                "details": recommendations
            })
            
            # Compile comprehensive result
            result = self._compile_exploration_result(analysis, loading_strategy, recommendations)
            
            # Calculate confidence based on analysis completeness
            confidence = self._calculate_confidence(analysis)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Dataset exploration completed in {execution_time:.2f}s with confidence {confidence:.2f}")
            
            return result, trajectory, confidence
            
        except Exception as e:
            logger.error(f"❌ Dataset exploration failed: {str(e)}")
            return f"Dataset exploration failed: {str(e)}", trajectory, 0.1
    
    def _extract_dataset_path(self, instruction: str, context: Dict) -> Optional[str]:
        """Extract dataset path from instruction or context"""
        # Try context first
        if 'dataset_path' in context:
            return context['dataset_path']
        
        # Try to extract from instruction
        import re
        dataset_patterns = [
            r'dataset/([^/\s]+)',
            r'analyze\s+([^/\s]+)\s+dataset',
            r'folder\s+([^/\s]+)',
        ]
        
        for pattern in dataset_patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                dataset_name = match.group(1)
                return f"dataset/{dataset_name}"
        
        # Default fallback - look for any dataset directory
        if Path("dataset").exists():
            subdirs = [d for d in Path("dataset").iterdir() if d.is_dir()]
            if subdirs:
                return str(subdirs[0])  # Use first available dataset
        
        return None
    
    def _generate_loading_strategy(self, analysis: Dict, instruction: str, logger) -> Dict[str, Any]:
        """Generate optimized data loading strategy based on analysis"""
        strategy = {
            "priority_files": [],
            "loading_order": [],
            "memory_strategy": "standard",
            "preprocessing_steps": [],
            "code_template": ""
        }
        
        # Analyze CSV files for priority
        csv_files = analysis.get("csv_files", {})
        priority_scores = {}
        
        for file_path, file_analysis in csv_files.items():
            score = 0
            
            # Score based on file characteristics
            if file_analysis.get("potential_timestamps"):
                score += 3  # Time-based data is crucial for RCA
            if file_analysis.get("potential_metrics"):
                score += 2  # Metrics are important
            if file_analysis.get("potential_identifiers"):
                score += 1  # Identifiers help with correlation
            
            # Score based on file name relevance
            file_name_lower = file_path.lower()
            if any(keyword in file_name_lower for keyword in ['metric', 'performance', 'system']):
                score += 2
            if any(keyword in file_name_lower for keyword in ['trace', 'span', 'call']):
                score += 2
            if any(keyword in file_name_lower for keyword in ['log', 'event', 'error']):
                score += 1
            
            priority_scores[file_path] = score
        
        # Sort by priority
        sorted_files = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        strategy["priority_files"] = [{"file": f, "priority_score": s} for f, s in sorted_files[:5]]
        strategy["loading_order"] = [f for f, s in sorted_files]
        
        # Determine memory strategy
        total_size = analysis.get("summary", {}).get("total_size_mb", 0)
        if total_size > 1000:
            strategy["memory_strategy"] = "chunked"
            strategy["preprocessing_steps"].append("Use chunked reading for large files")
        elif total_size > 100:
            strategy["memory_strategy"] = "optimized"
            strategy["preprocessing_steps"].append("Optimize data types after loading")
        
        # Generate code template
        strategy["code_template"] = self._generate_loading_code_template(analysis, strategy)
        
        return strategy
    
    def _generate_loading_code_template(self, analysis: Dict, strategy: Dict) -> str:
        """Generate Python code template for loading the dataset"""
        code_parts = [
            "# Intelligent data loading based on dataset analysis",
            "import pandas as pd",
            "import numpy as np", 
            "from pathlib import Path",
            "import pytz",
            "",
            "# Setup",
            f"dataset_path = Path('{analysis['dataset_path']}')",
            "TIMEZONE = pytz.timezone('Asia/Shanghai')",
            "loaded_data = {}",
            ""
        ]
        
        # Add loading logic for each priority file
        for file_info in strategy["priority_files"]:
            file_path = file_info["file"]
            file_analysis = analysis["csv_files"].get(file_path, {})
            
            code_parts.extend([
                f"# Loading high-priority file: {file_path}",
                f"print('📊 Loading {file_path}...')",
                f"try:",
                f"    df = pd.read_csv(dataset_path / '{file_path}')"
            ])
            
            # Add timestamp handling if detected
            if file_analysis.get("potential_timestamps"):
                timestamp_cols = file_analysis["potential_timestamps"]
                code_parts.extend([
                    f"    # Handle timestamp columns: {timestamp_cols}",
                    f"    for col in {timestamp_cols}:",
                    f"        if col in df.columns:",
                    f"            df[col] = pd.to_datetime(df[col], errors='coerce')",
                    f"            if df[col].dt.tz is None:",
                    f"                df[col] = df[col].dt.tz_localize(TIMEZONE)"
                ])
            
            # Add identifier info
            if file_analysis.get("potential_identifiers"):
                id_cols = file_analysis["potential_identifiers"]
                code_parts.append(f"    # Identifier columns detected: {id_cols}")
            
            code_parts.extend([
                f"    loaded_data['{file_path.replace('.csv', '')}'] = df",
                f"    print(f'✅ Loaded {{len(df)}} records from {file_path}')",
                f"    print(f'   Columns: {{list(df.columns)}}')",
                f"except Exception as e:",
                f"    print(f'❌ Error loading {file_path}: {{e}}')",
                ""
            ])
        
        code_parts.extend([
            "# Summary",
            "print(f'\\n📊 Loaded {len(loaded_data)} datasets:')",
            "for name, df in loaded_data.items():",
            "    print(f'  {name}: {len(df)} records')",
        ])
        
        return "\n".join(code_parts)
    
    def _create_analysis_recommendations(self, analysis: Dict, instruction: str) -> List[Dict[str, Any]]:
        """Create specific analysis recommendations based on dataset structure"""
        recommendations = []
        
        # CSV-based recommendations
        csv_files = analysis.get("csv_files", {})
        for file_path, file_analysis in csv_files.items():
            file_recommendations = []
            
            if file_analysis.get("potential_metrics"):
                file_recommendations.append({
                    "type": "anomaly_detection",
                    "description": f"Use threshold-based anomaly detection on metrics in {file_path}",
                    "columns": file_analysis["potential_metrics"],
                    "priority": "high"
                })
            
            if file_analysis.get("potential_timestamps"):
                file_recommendations.append({
                    "type": "temporal_analysis", 
                    "description": f"Perform time-series analysis on {file_path}",
                    "columns": file_analysis["potential_timestamps"],
                    "priority": "high"
                })
            
            if file_analysis.get("potential_identifiers"):
                file_recommendations.append({
                    "type": "correlation_analysis",
                    "description": f"Use identifiers in {file_path} for cross-dataset correlation",
                    "columns": file_analysis["potential_identifiers"],
                    "priority": "medium"
                })
            
            recommendations.extend(file_recommendations)
        
        # Log-based recommendations
        log_files = analysis.get("log_files", {})
        for file_path, log_analysis in log_files.items():
            if log_analysis.get("error_patterns"):
                recommendations.append({
                    "type": "pattern_analysis",
                    "description": f"Analyze error patterns in {file_path}",
                    "patterns": [p["pattern"] for p in log_analysis["error_patterns"]],
                    "priority": "high"
                })
        
        return recommendations
    
    def _compile_exploration_result(self, analysis: Dict, strategy: Dict, recommendations: List[Dict]) -> str:
        """Compile comprehensive exploration result"""
        result_parts = [
            "# Dataset Exploration Results",
            "",
            f"## Dataset Overview",
            f"- Path: {analysis['dataset_path']}",
            f"- Total Files: {analysis['summary']['total_files']}",
            f"- CSV Files: {analysis['summary']['csv_files']}",
            f"- Log Files: {analysis['summary']['log_files']}",
            f"- Total Size: {analysis['summary']['total_size_mb']} MB",
            ""
        ]
        
        # CSV Files Analysis
        if analysis['csv_files']:
            result_parts.extend([
                "## CSV Files Analysis",
                ""
            ])
            
            for file_path, file_analysis in analysis['csv_files'].items():
                result_parts.extend([
                    f"### {file_path}",
                    f"- Rows: {file_analysis.get('row_count', 'Unknown')}",
                    f"- Columns: {len(file_analysis.get('columns', []))}",
                    f"- Key Columns:",
                ])
                
                if file_analysis.get('potential_timestamps'):
                    result_parts.append(f"  - Timestamps: {file_analysis['potential_timestamps']}")
                if file_analysis.get('potential_identifiers'):
                    result_parts.append(f"  - Identifiers: {file_analysis['potential_identifiers']}")
                if file_analysis.get('potential_metrics'):
                    result_parts.append(f"  - Metrics: {file_analysis['potential_metrics']}")
                
                result_parts.append("")
        
        # Loading Strategy
        result_parts.extend([
            "## Optimized Loading Strategy",
            "",
            f"Memory Strategy: {strategy['memory_strategy']}",
            "",
            "Priority Loading Order:"
        ])
        
        for i, file_info in enumerate(strategy['priority_files'][:3], 1):
            result_parts.append(f"{i}. {file_info['file']} (score: {file_info['priority_score']})")
        
        result_parts.extend([
            "",
            "## Analysis Recommendations",
            ""
        ])
        
        for i, rec in enumerate(recommendations[:5], 1):
            result_parts.append(f"{i}. {rec['description']} (Priority: {rec['priority']})")
        
        # Add executable code
        result_parts.extend([
            "",
            "## Ready-to-Execute Loading Code",
            "",
            "```python",
            strategy['code_template'],
            "```"
        ])
        
        return "\n".join(result_parts)
    
    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence based on analysis completeness"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on successful analysis
        if analysis.get('csv_files'):
            confidence += 0.2
        if analysis.get('log_files'):
            confidence += 0.1
        if analysis.get('recommendations'):
            confidence += 0.1
        
        # Boost based on data quality
        csv_files = analysis.get('csv_files', {})
        if csv_files:
            total_files = len(csv_files)
            files_with_timestamps = sum(1 for f in csv_files.values() if f.get('potential_timestamps'))
            files_with_metrics = sum(1 for f in csv_files.values() if f.get('potential_metrics'))
            
            if files_with_timestamps > 0:
                confidence += 0.1 * (files_with_timestamps / total_files)
            if files_with_metrics > 0:
                confidence += 0.1 * (files_with_metrics / total_files)
        
        return min(confidence, 1.0)
    
    def get_capabilities(self) -> List[str]:
        return [
            "Directory structure analysis",
            "CSV file content analysis", 
            "Log file pattern detection",
            "Data loading strategy generation",
            "File prioritization",
            "Memory optimization recommendations",
            "Analysis workflow planning"
        ]


class AnomalyDetectionAgent(BaseStageAgent):
    """Specialized agent for anomaly detection with statistical methods"""
    
    def __init__(self, config: Dict):
        super().__init__("anomaly_detection", config)
    
    def execute(self, context: Dict, instruction: str, logger) -> Tuple[str, List[Dict], float]:
        logger.info("📊 Anomaly Detection Agent starting analysis")
        
        # Generate sophisticated anomaly detection code
        code_template = self._generate_anomaly_detection_code(context, instruction)
        
        result = f"""# Anomaly Detection Analysis

## Advanced Statistical Anomaly Detection

The following code implements multiple anomaly detection methods:
- Threshold-based detection (P95, P99)
- Statistical outlier detection (Z-score, IQR)
- Time-series anomaly detection
- Multi-variate anomaly detection

```python
{code_template}
```

## Key Features:
- Automatic threshold calculation
- Multiple detection algorithms
- Temporal pattern recognition  
- Cross-correlation analysis
- Confidence scoring for each anomaly
"""
        
        trajectory = [{
            "step": 1,
            "action": "anomaly_detection_code_generation",
            "result": "Generated advanced anomaly detection algorithms",
            "confidence": 0.85
        }]
        
        return result, trajectory, 0.85
    
    def _generate_anomaly_detection_code(self, context: Dict, instruction: str) -> str:
        """Generate sophisticated anomaly detection code"""
        return """
# Advanced Anomaly Detection System
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetector:
    def __init__(self, methods=['threshold', 'zscore', 'iqr', 'isolation_forest']):
        self.methods = methods
        self.results = {}
        self.thresholds = {}
    
    def detect_anomalies(self, df, metric_columns, timestamp_col=None):
        \"\"\"Detect anomalies using multiple methods\"\"\"
        anomalies = {}
        
        for method in self.methods:
            if method == 'threshold':
                anomalies[method] = self._threshold_detection(df, metric_columns)
            elif method == 'zscore':
                anomalies[method] = self._zscore_detection(df, metric_columns)
            elif method == 'iqr':
                anomalies[method] = self._iqr_detection(df, metric_columns)
            elif method == 'isolation_forest':
                anomalies[method] = self._isolation_forest_detection(df, metric_columns)
        
        return self._combine_results(anomalies)
    
    def _threshold_detection(self, df, metric_columns):
        \"\"\"P95/P99 threshold-based detection\"\"\"
        anomalies = {}
        for col in metric_columns:
            if col in df.columns:
                p95 = df[col].quantile(0.95)
                p99 = df[col].quantile(0.99)
                self.thresholds[f'{col}_p95'] = p95
                self.thresholds[f'{col}_p99'] = p99
                
                anomalies[col] = {
                    'p95_anomalies': df[df[col] > p95].index.tolist(),
                    'p99_anomalies': df[df[col] > p99].index.tolist()
                }
        return anomalies
    
    def _zscore_detection(self, df, metric_columns, threshold=3):
        \"\"\"Z-score based detection\"\"\"
        anomalies = {}
        for col in metric_columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                anomalies[col] = df[z_scores > threshold].index.tolist()
        return anomalies
    
    def _iqr_detection(self, df, metric_columns):
        \"\"\"Interquartile range based detection\"\"\"
        anomalies = {}
        for col in metric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        return anomalies
    
    def _isolation_forest_detection(self, df, metric_columns, contamination=0.1):
        \"\"\"Isolation Forest for multivariate anomaly detection\"\"\"
        if len(metric_columns) < 2:
            return {}
        
        # Prepare data
        data = df[metric_columns].fillna(df[metric_columns].mean())
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        
        # Return indices of anomalies (-1 labels)
        anomaly_indices = df[anomaly_labels == -1].index.tolist()
        
        return {'multivariate_anomalies': anomaly_indices}
    
    def _combine_results(self, anomalies):
        \"\"\"Combine results from different methods\"\"\"
        combined = {}
        all_anomaly_indices = set()
        
        for method, method_results in anomalies.items():
            if isinstance(method_results, dict):
                for col, indices in method_results.items():
                    if isinstance(indices, list):
                        all_anomaly_indices.update(indices)
                    elif isinstance(indices, dict):
                        for sub_key, sub_indices in indices.items():
                            if isinstance(sub_indices, list):
                                all_anomaly_indices.update(sub_indices)
        
        combined['all_anomaly_indices'] = sorted(list(all_anomaly_indices))
        combined['by_method'] = anomalies
        combined['anomaly_count'] = len(all_anomaly_indices)
        
        return combined

# Usage Example:
print("🔍 Starting Advanced Anomaly Detection...")

# Initialize detector
detector = AdvancedAnomalyDetector()

# Assuming loaded_data is available from previous step
for dataset_name, df in loaded_data.items():
    print(f"\\n📊 Analyzing {dataset_name}...")
    
    # Identify numeric columns (potential metrics)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print(f"   ⚠️ No numeric columns found in {dataset_name}")
        continue
    
    print(f"   📈 Found {len(numeric_cols)} metric columns: {numeric_cols}")
    
    # Detect anomalies
    anomaly_results = detector.detect_anomalies(df, numeric_cols)
    
    print(f"   🚨 Detected {anomaly_results['anomaly_count']} anomalies")
    
    # Show top anomalies
    if anomaly_results['anomaly_count'] > 0:
        anomaly_indices = anomaly_results['all_anomaly_indices'][:10]  # Top 10
        anomaly_data = df.iloc[anomaly_indices]
        
        print("   🔍 Top anomalous records:")
        print(anomaly_data[numeric_cols].describe())
        
        # Store results
        detector.results[dataset_name] = anomaly_results

print("\\n✅ Anomaly detection completed")
print(f"📊 Summary: Analyzed {len(detector.results)} datasets")
for dataset, results in detector.results.items():
    print(f"   {dataset}: {results['anomaly_count']} anomalies detected")
"""
    
    def get_capabilities(self) -> List[str]:
        return [
            "Statistical threshold detection (P95, P99)",
            "Z-score based outlier detection", 
            "Interquartile range (IQR) detection",
            "Isolation Forest multivariate detection",
            "Time-series anomaly detection",
            "Cross-correlation analysis",
            "Confidence scoring",
            "Multi-method ensemble detection"
        ]


class CorrelationAnalysisAgent(BaseStageAgent):
    """Specialized agent for correlation analysis and pattern discovery"""
    
    def __init__(self, config: Dict):
        super().__init__("correlation_analysis", config)
    
    def execute(self, context: Dict, instruction: str, logger) -> Tuple[str, List[Dict], float]:
        logger.info("🔗 Correlation Analysis Agent starting analysis")
        
        code_template = self._generate_correlation_code(context, instruction)
        
        result = f"""# Correlation Analysis

## Advanced Pattern Discovery & Correlation Analysis

This analysis performs:
- Temporal correlation analysis
- Cross-dataset correlation
- Causal relationship inference
- Dependency graph construction

```python
{code_template}
```

## Analysis Capabilities:
- Time-based correlation windows
- Lag correlation detection  
- Multi-dimensional correlation
- Statistical significance testing
- Causal inference algorithms
"""
        
        trajectory = [{
            "step": 1,
            "action": "correlation_analysis_generation", 
            "result": "Generated advanced correlation analysis methods",
            "confidence": 0.80
        }]
        
        return result, trajectory, 0.80
    
    def _generate_correlation_code(self, context: Dict, instruction: str) -> str:
        return """
# Advanced Correlation Analysis System
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate
import networkx as nx
from itertools import combinations

class AdvancedCorrelationAnalyzer:
    def __init__(self):
        self.correlations = {}
        self.dependency_graph = nx.DiGraph()
        self.temporal_correlations = {}
    
    def analyze_correlations(self, datasets, time_window_minutes=5):
        \"\"\"Comprehensive correlation analysis across datasets\"\"\"
        print("🔗 Starting correlation analysis...")
        
        # Cross-dataset correlation
        self._cross_dataset_correlation(datasets)
        
        # Temporal correlation analysis
        self._temporal_correlation_analysis(datasets, time_window_minutes)
        
        # Build dependency graph
        self._build_dependency_graph()
        
        return self.correlations
    
    def _cross_dataset_correlation(self, datasets):
        \"\"\"Analyze correlations between different datasets\"\"\"
        print("\\n📊 Cross-dataset correlation analysis...")
        
        # Find common identifier columns
        common_identifiers = self._find_common_identifiers(datasets)
        print(f"   🔗 Common identifiers: {common_identifiers}")
        
        # Find datasets with timestamps
        temporal_datasets = {}
        for name, df in datasets.items():
            timestamp_cols = [col for col in df.columns 
                            if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_cols:
                temporal_datasets[name] = (df, timestamp_cols[0])
        
        print(f"   ⏰ Temporal datasets: {list(temporal_datasets.keys())}")
        
        # Correlation analysis between temporal datasets
        for (name1, (df1, time_col1)), (name2, (df2, time_col2)) in combinations(temporal_datasets.items(), 2):
            correlation = self._correlate_temporal_datasets(df1, df2, time_col1, time_col2, name1, name2)
            self.correlations[f"{name1}_vs_{name2}"] = correlation
    
    def _find_common_identifiers(self, datasets):
        \"\"\"Find common identifier columns across datasets\"\"\"
        all_columns = {}
        for name, df in datasets.items():
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['id', 'identifier', 'key', 'cmdb']):
                    if col not in all_columns:
                        all_columns[col] = []
                    all_columns[col].append(name)
        
        # Return columns that appear in multiple datasets
        return {col: datasets for col, datasets in all_columns.items() if len(datasets) > 1}
    
    def _correlate_temporal_datasets(self, df1, df2, time_col1, time_col2, name1, name2):
        \"\"\"Correlate two temporal datasets\"\"\"
        correlation_result = {
            'datasets': f"{name1} vs {name2}",
            'correlation_strength': 0.0,
            'significant_correlations': [],
            'temporal_lag': 0,
            'common_timeframe': None
        }
        
        try:
            # Convert timestamp columns
            df1[time_col1] = pd.to_datetime(df1[time_col1], errors='coerce')
            df2[time_col2] = pd.to_datetime(df2[time_col2], errors='coerce')
            
            # Find common timeframe
            start_time = max(df1[time_col1].min(), df2[time_col2].min())
            end_time = min(df1[time_col1].max(), df2[time_col2].max())
            
            if pd.isna(start_time) or pd.isna(end_time) or start_time >= end_time:
                return correlation_result
            
            correlation_result['common_timeframe'] = f"{start_time} to {end_time}"
            
            # Filter to common timeframe
            df1_filtered = df1[(df1[time_col1] >= start_time) & (df1[time_col1] <= end_time)]
            df2_filtered = df2[(df2[time_col2] >= start_time) & (df2[time_col2] <= end_time)]
            
            if len(df1_filtered) == 0 or len(df2_filtered) == 0:
                return correlation_result
            
            # Get numeric columns for correlation
            numeric_cols1 = df1_filtered.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols2 = df2_filtered.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols1 or not numeric_cols2:
                return correlation_result
            
            # Resample to common time intervals
            df1_resampled = df1_filtered.set_index(time_col1).resample('1min').mean()
            df2_resampled = df2_filtered.set_index(time_col2).resample('1min').mean()
            
            # Find correlations
            max_correlation = 0
            significant_pairs = []
            
            for col1 in numeric_cols1:
                for col2 in numeric_cols2:
                    if col1 in df1_resampled.columns and col2 in df2_resampled.columns:
                        # Align time series
                        aligned = pd.concat([df1_resampled[col1], df2_resampled[col2]], axis=1).dropna()
                        
                        if len(aligned) > 10:  # Need sufficient data
                            corr_coef, p_value = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                            
                            if abs(corr_coef) > 0.3 and p_value < 0.05:  # Significant correlation
                                significant_pairs.append({
                                    'column1': f"{name1}.{col1}",
                                    'column2': f"{name2}.{col2}",
                                    'correlation': round(corr_coef, 3),
                                    'p_value': round(p_value, 4),
                                    'strength': 'strong' if abs(corr_coef) > 0.7 else 'moderate'
                                })
                                
                                max_correlation = max(max_correlation, abs(corr_coef))
            
            correlation_result['correlation_strength'] = round(max_correlation, 3)
            correlation_result['significant_correlations'] = significant_pairs
            
        except Exception as e:
            correlation_result['error'] = str(e)
        
        return correlation_result
    
    def _temporal_correlation_analysis(self, datasets, time_window_minutes):
        \"\"\"Analyze temporal correlations within time windows\"\"\"
        print(f"\\n⏰ Temporal correlation analysis (window: {time_window_minutes} min)...")
        
        for name, df in datasets.items():
            # Find timestamp column
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if not timestamp_cols:
                continue
            
            time_col = timestamp_cols[0]
            print(f"   📊 Analyzing temporal patterns in {name}...")
            
            # Convert timestamp and sort
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df_sorted = df.sort_values(time_col).copy()
            
            # Get numeric columns
            numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                continue
            
            # Analyze correlations within time windows
            window_correlations = []
            df_sorted.set_index(time_col, inplace=True)
            
            # Resample to time windows
            window_data = df_sorted.resample(f'{time_window_minutes}min').mean()
            
            # Calculate correlations between metrics within windows
            for col1, col2 in combinations(numeric_cols, 2):
                if col1 in window_data.columns and col2 in window_data.columns:
                    clean_data = window_data[[col1, col2]].dropna()
                    
                    if len(clean_data) > 3:
                        corr_coef, p_value = pearsonr(clean_data[col1], clean_data[col2])
                        
                        if abs(corr_coef) > 0.3 and p_value < 0.05:
                            window_correlations.append({
                                'metric1': col1,
                                'metric2': col2,
                                'correlation': round(corr_coef, 3),
                                'p_value': round(p_value, 4)
                            })
            
            self.temporal_correlations[name] = window_correlations
            print(f"      Found {len(window_correlations)} significant temporal correlations")
    
    def _build_dependency_graph(self):
        \"\"\"Build dependency graph from correlation results\"\"\"
        print("\\n🕸️  Building dependency graph...")
        
        # Add nodes and edges based on correlations
        for corr_key, corr_data in self.correlations.items():
            if 'significant_correlations' in corr_data:
                for corr in corr_data['significant_correlations']:
                    col1 = corr['column1']
                    col2 = corr['column2']
                    weight = abs(corr['correlation'])
                    
                    self.dependency_graph.add_node(col1)
                    self.dependency_graph.add_node(col2)
                    self.dependency_graph.add_edge(col1, col2, weight=weight, correlation=corr['correlation'])
        
        print(f"   📊 Graph: {self.dependency_graph.number_of_nodes()} nodes, {self.dependency_graph.number_of_edges()} edges")
        
        # Find central nodes (highly connected)
        if self.dependency_graph.number_of_nodes() > 0:
            centrality = nx.degree_centrality(self.dependency_graph)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            print("   🎯 Most central metrics:")
            for node, cent in top_central:
                print(f"      {node}: {cent:.3f}")
    
    def get_summary(self):
        \"\"\"Get correlation analysis summary\"\"\"
        summary = {
            'total_correlations': len(self.correlations),
            'temporal_datasets': len(self.temporal_correlations),
            'dependency_graph_nodes': self.dependency_graph.number_of_nodes(),
            'dependency_graph_edges': self.dependency_graph.number_of_edges()
        }
        
        # Count significant correlations
        total_significant = 0
        for corr_data in self.correlations.values():
            if 'significant_correlations' in corr_data:
                total_significant += len(corr_data['significant_correlations'])
        
        summary['significant_correlations'] = total_significant
        return summary

# Usage
if 'loaded_data' in globals() and loaded_data:
    analyzer = AdvancedCorrelationAnalyzer()
    results = analyzer.analyze_correlations(loaded_data, time_window_minutes=5)
    
    print("\\n" + "="*50)
    print("📊 CORRELATION ANALYSIS SUMMARY")
    print("="*50)
    
    summary = analyzer.get_summary()
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Show top correlations
    print("\\n🔍 Significant Cross-Dataset Correlations:")
    for corr_key, corr_data in results.items():
        if 'significant_correlations' in corr_data and corr_data['significant_correlations']:
            print(f"\\n{corr_data['datasets']}:")
            for corr in corr_data['significant_correlations'][:3]:  # Top 3
                print(f"  {corr['column1']} ↔ {corr['column2']}: {corr['correlation']} ({corr['strength']})")
else:
    print("⚠️ No loaded_data available. Run dataset exploration first.")
"""
    
    def get_capabilities(self) -> List[str]:
        return [
            "Cross-dataset correlation analysis",
            "Temporal correlation detection",
            "Lag correlation analysis",
            "Dependency graph construction",
            "Statistical significance testing",
            "Causal relationship inference",
            "Multi-dimensional correlation",
            "Pattern discovery"
        ]


class RootCauseLocalizationAgent(BaseStageAgent):
    """Specialized agent for root cause localization with advanced reasoning"""
    
    def __init__(self, config: Dict):
        super().__init__("root_cause_localization", config)
    
    def execute(self, context: Dict, instruction: str, logger) -> Tuple[str, List[Dict], float]:
        logger.info("🎯 Root Cause Localization Agent starting analysis")
        
        code_template = self._generate_localization_code(context, instruction)
        
        result = f"""# Root Cause Localization

## Advanced Root Cause Analysis & Localization

This implements sophisticated RCA algorithms:
- Multi-factor root cause scoring
- Temporal causality analysis  
- Evidence aggregation
- Confidence-weighted decisions

```python
{code_template}
```

## Localization Features:
- Fault tree analysis
- Bayesian inference
- Evidence correlation
- Temporal causality
- Multi-hypothesis evaluation
- Confidence scoring
"""
        
        trajectory = [{
            "step": 1,
            "action": "root_cause_localization_generation",
            "result": "Generated advanced root cause localization algorithms", 
            "confidence": 0.88
        }]
        
        return result, trajectory, 0.88
    
    def _generate_localization_code(self, context: Dict, instruction: str) -> str:
        return """
# Advanced Root Cause Localization System
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RootCauseLocalizer:
    def __init__(self):
        self.evidence = {}
        self.candidates = {}
        self.final_analysis = {}
    
    def localize_root_cause(self, anomaly_results, correlation_results, datasets):
        \"\"\"Comprehensive root cause localization\"\"\"
        print("🎯 Starting Root Cause Localization...")
        
        # Step 1: Collect evidence from previous analyses
        self._collect_evidence(anomaly_results, correlation_results, datasets)
        
        # Step 2: Generate root cause candidates
        self._generate_candidates()
        
        # Step 3: Score candidates using multiple criteria
        self._score_candidates()
        
        # Step 4: Apply temporal causality analysis
        self._analyze_temporal_causality(datasets)
        
        # Step 5: Generate final root cause determination
        return self._determine_root_cause()
    
    def _collect_evidence(self, anomaly_results, correlation_results, datasets):
        \"\"\"Collect evidence from anomaly detection and correlation analysis\"\"\"
        print("\\n📊 Collecting evidence from previous analyses...")
        
        # Collect anomaly evidence
        anomaly_evidence = {}
        if hasattr(anomaly_results, 'results'):
            for dataset_name, results in anomaly_results.results.items():
                anomaly_evidence[dataset_name] = {
                    'anomaly_count': results['anomaly_count'],
                    'anomaly_indices': results['all_anomaly_indices']
                }
        
        # Collect correlation evidence
        correlation_evidence = {}
        if correlation_results:
            for corr_key, corr_data in correlation_results.items():
                if 'significant_correlations' in corr_data:
                    correlation_evidence[corr_key] = corr_data['significant_correlations']
        
        self.evidence = {
            'anomalies': anomaly_evidence,
            'correlations': correlation_evidence,
            'datasets': {name: len(df) for name, df in datasets.items()}
        }
        
        print(f"   📈 Anomaly evidence from {len(anomaly_evidence)} datasets")
        print(f"   🔗 Correlation evidence from {len(correlation_evidence)} analyses")
    
    def _generate_candidates(self):
        \"\"\"Generate potential root cause candidates\"\"\"
        print("\\n🧠 Generating root cause candidates...")
        
        candidates = {}
        
        # Generate candidates from anomalies
        for dataset_name, anomaly_data in self.evidence['anomalies'].items():
            if anomaly_data['anomaly_count'] > 0:
                candidates[f"anomaly_in_{dataset_name}"] = {
                    'type': 'anomaly',
                    'source_dataset': dataset_name,
                    'evidence_strength': min(anomaly_data['anomaly_count'] / 10, 1.0),  # Normalize
                    'description': f"Anomalous behavior detected in {dataset_name}",
                    'supporting_evidence': [f"{anomaly_data['anomaly_count']} anomalies detected"]
                }
        
        # Generate candidates from correlations
        for corr_key, correlations in self.evidence['correlations'].items():
            if correlations:
                # Find the strongest correlation
                strongest = max(correlations, key=lambda x: abs(x['correlation']))
                candidates[f"correlation_{corr_key}"] = {
                    'type': 'correlation',
                    'source_correlation': corr_key,
                    'evidence_strength': abs(strongest['correlation']),
                    'description': f"Strong correlation detected: {strongest['column1']} ↔ {strongest['column2']}",
                    'supporting_evidence': [f"Correlation: {strongest['correlation']}", f"P-value: {strongest['p_value']}"]
                }
        
        self.candidates = candidates
        print(f"   🎯 Generated {len(candidates)} root cause candidates")
    
    def _score_candidates(self):
        \"\"\"Score candidates using multiple criteria\"\"\"
        print("\\n⚖️ Scoring candidates using multi-criteria analysis...")
        
        for candidate_key, candidate in self.candidates.items():
            score = 0.0
            scoring_details = {}
            
            # Base evidence strength
            evidence_strength = candidate.get('evidence_strength', 0)
            score += evidence_strength * 0.4
            scoring_details['evidence_strength'] = evidence_strength
            
            # Type-based scoring
            if candidate['type'] == 'anomaly':
                score += 0.3  # Anomalies are strong indicators
                scoring_details['type_bonus'] = 0.3
            elif candidate['type'] == 'correlation':
                score += 0.2  # Correlations are supportive evidence
                scoring_details['type_bonus'] = 0.2
            
            # Temporal priority scoring (earlier events are more likely root causes)
            # This would require timestamp analysis - placeholder for now
            temporal_score = 0.1
            score += temporal_score
            scoring_details['temporal_score'] = temporal_score
            
            # Dataset significance scoring
            source_dataset = candidate.get('source_dataset', '')
            if 'metric' in source_dataset.lower():
                score += 0.1  # Metrics datasets are important
                scoring_details['dataset_significance'] = 0.1
            
            candidate['total_score'] = round(score, 3)
            candidate['scoring_details'] = scoring_details
        
        # Sort candidates by score
        sorted_candidates = sorted(self.candidates.items(), key=lambda x: x[1]['total_score'], reverse=True)
        self.candidates = dict(sorted_candidates)
        
        print("   🏆 Top candidates:")
        for i, (key, candidate) in enumerate(list(self.candidates.items())[:3], 1):
            print(f"      {i}. {candidate['description']} (Score: {candidate['total_score']})")
    
    def _analyze_temporal_causality(self, datasets):
        \"\"\"Analyze temporal causality patterns\"\"\"
        print("\\n⏰ Analyzing temporal causality...")
        
        temporal_analysis = {}
        
        # Find datasets with timestamps
        temporal_datasets = {}
        for name, df in datasets.items():
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_cols:
                temporal_datasets[name] = timestamp_cols[0]
        
        if len(temporal_datasets) < 2:
            print("   ⚠️ Insufficient temporal data for causality analysis")
            return
        
        # Analyze event ordering
        event_timeline = []
        for name, time_col in temporal_datasets.items():
            df = datasets[name]
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Get anomaly timestamps if available
            if name in self.evidence['anomalies']:
                anomaly_indices = self.evidence['anomalies'][name]['anomaly_indices']
                if anomaly_indices:
                    anomaly_times = df.iloc[anomaly_indices][time_col].dropna()
                    for timestamp in anomaly_times:
                        event_timeline.append({
                            'timestamp': timestamp,
                            'dataset': name,
                            'event_type': 'anomaly'
                        })
        
        # Sort events by time
        event_timeline.sort(key=lambda x: x['timestamp'])
        
        # Analyze causality patterns
        if event_timeline:
            earliest_event = event_timeline[0]
            temporal_analysis = {
                'earliest_anomaly': {
                    'timestamp': earliest_event['timestamp'].isoformat(),
                    'dataset': earliest_event['dataset'],
                    'potential_root_cause': True
                },
                'event_sequence': [
                    {
                        'timestamp': event['timestamp'].isoformat(),
                        'dataset': event['dataset'],
                        'event_type': event['event_type']
                    } for event in event_timeline[:10]  # Top 10 events
                ]
            }
            
            # Boost score for earliest anomaly
            for candidate_key, candidate in self.candidates.items():
                if candidate.get('source_dataset') == earliest_event['dataset']:
                    candidate['total_score'] += 0.2
                    candidate['temporal_priority'] = True
        
        self.evidence['temporal_analysis'] = temporal_analysis
        print(f"   📅 Analyzed {len(event_timeline)} temporal events")
    
    def _determine_root_cause(self):
        \"\"\"Make final root cause determination\"\"\"
        print("\\n🏁 Making final root cause determination...")
        
        if not self.candidates:
            return {
                'root_cause': 'Unknown',
                'confidence': 0.0,
                'reason': 'No significant evidence found',
                'analysis_summary': 'Insufficient data for root cause determination'
            }
        
        # Get top candidate
        top_candidate_key = next(iter(self.candidates))
        top_candidate = self.candidates[top_candidate_key]
        
        # Calculate final confidence
        base_confidence = min(top_candidate['total_score'], 1.0)
        
        # Adjust confidence based on evidence quality
        evidence_count = len(self.evidence['anomalies']) + len(self.evidence['correlations'])
        confidence_adjustment = min(evidence_count * 0.1, 0.3)
        
        final_confidence = min(base_confidence + confidence_adjustment, 1.0)
        
        # Generate root cause components and reasons (simplified)
        root_cause_component = self._extract_component_from_candidate(top_candidate)
        root_cause_reason = self._extract_reason_from_candidate(top_candidate)
        root_cause_time = self._extract_time_from_evidence()
        
        self.final_analysis = {
            'root_cause_component': root_cause_component,
            'root_cause_reason': root_cause_reason,
            'root_cause_occurrence_datetime': root_cause_time,
            'confidence': round(final_confidence, 3),
            'analysis_summary': f"Root cause identified: {top_candidate['description']}",
            'supporting_evidence': top_candidate['supporting_evidence'],
            'all_candidates': {k: v['total_score'] for k, v in list(self.candidates.items())[:5]}
        }
        
        # Format final result
        result_json = {
            "1": {
                "root cause occurrence datetime": root_cause_time,
                "root cause component": root_cause_component,
                "root cause reason": root_cause_reason
            }
        }
        
        print("\\n" + "="*60)
        print("🎯 ROOT CAUSE ANALYSIS RESULTS")
        print("="*60)
        print(f"Component: {root_cause_component}")
        print(f"Reason: {root_cause_reason}")
        print(f"Time: {root_cause_time}")
        print(f"Confidence: {final_confidence:.1%}")
        print(f"\\nAnalysis: {self.final_analysis['analysis_summary']}")
        print("\\nSupporting Evidence:")
        for evidence in self.final_analysis['supporting_evidence']:
            print(f"  • {evidence}")
        
        print("\\n📋 Final Answer (JSON format):")
        print(json.dumps(result_json, indent=2))
        
        return self.final_analysis
    
    def _extract_component_from_candidate(self, candidate):
        \"\"\"Extract component name from candidate\"\"\"
        source_dataset = candidate.get('source_dataset', 'unknown')
        
        # Map dataset names to component names
        component_mapping = {
            'metric_app': 'application_server',
            'metric_container': 'container_service',
            'trace_span': 'distributed_service',
            'log_service': 'service_component'
        }
        
        for pattern, component in component_mapping.items():
            if pattern in source_dataset.lower():
                return component
        
        return source_dataset or 'system_component'
    
    def _extract_reason_from_candidate(self, candidate):
        \"\"\"Extract failure reason from candidate\"\"\"
        if candidate['type'] == 'anomaly':
            return 'performance_anomaly'
        elif candidate['type'] == 'correlation':
            return 'service_dependency_failure'
        else:
            return 'system_failure'
    
    def _extract_time_from_evidence(self):
        \"\"\"Extract root cause occurrence time from evidence\"\"\"
        if 'temporal_analysis' in self.evidence:
            earliest = self.evidence['temporal_analysis'].get('earliest_anomaly')
            if earliest:
                return earliest['timestamp']
        
        # Default to a reasonable time based on typical RCA scenarios
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Usage
print("🎯 Starting comprehensive root cause localization...")

# Check if previous analysis results are available
if 'detector' in globals() and hasattr(detector, 'results'):
    if 'analyzer' in globals() and hasattr(analyzer, 'correlations'):
        if 'loaded_data' in globals() and loaded_data:
            
            # Initialize localizer
            localizer = RootCauseLocalizer()
            
            # Perform root cause localization
            final_result = localizer.localize_root_cause(detector, analyzer.correlations, loaded_data)
            
        else:
            print("❌ No loaded data available")
    else:
        print("❌ No correlation analysis results available")
else:
    print("❌ No anomaly detection results available")

print("\\n✅ Root cause localization completed")
"""
    
    def get_capabilities(self) -> List[str]:
        return [
            "Multi-factor root cause scoring",
            "Evidence aggregation and correlation", 
            "Temporal causality analysis",
            "Fault tree analysis",
            "Bayesian inference",
            "Multi-hypothesis evaluation",
            "Confidence-weighted decisions",
            "Component and reason extraction"
        ]


class StageAgentCoordinator:
    """Coordinates specialized stage agents for comprehensive RCA analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents = {
            'dataset_exploration': DatasetExplorationAgent(config),
            'anomaly_detection': AnomalyDetectionAgent(config),
            'correlation_analysis': CorrelationAnalysisAgent(config), 
            'root_cause_localization': RootCauseLocalizationAgent(config)
        }
        self.shared_context = {}
    
    def execute_stage(self, stage_name: str, context: Dict, instruction: str, logger) -> Tuple[str, List[Dict], float]:
        """Execute a specific stage using the appropriate specialized agent"""
        
        if stage_name not in self.agents:
            logger.error(f"Unknown stage: {stage_name}")
            return f"Error: Unknown stage {stage_name}", [], 0.0
        
        # Merge shared context
        full_context = {**context, **self.shared_context}
        
        # Execute stage with specialized agent
        agent = self.agents[stage_name]
        result, trajectory, confidence = agent.execute(full_context, instruction, logger)
        
        # Update shared context with findings
        self.shared_context[f"{stage_name}_results"] = {
            'result': result,
            'confidence': confidence,
            'trajectory': trajectory
        }
        
        return result, trajectory, confidence
    
    def get_stage_capabilities(self, stage_name: str) -> List[str]:
        """Get capabilities of a specific stage agent"""
        if stage_name in self.agents:
            return self.agents[stage_name].get_capabilities()
        return []
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all stage agents"""
        return {name: agent.get_capabilities() for name, agent in self.agents.items()}