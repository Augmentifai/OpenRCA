"""
Generic System Tools for Dataset Exploration and Analysis

This module provides intelligent tools for exploring and analyzing any dataset structure
without requiring hardcoded knowledge of specific formats or schemas.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime
# Optional import for character encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
from collections import Counter, defaultdict


class SystemExplorer:
    """System-level exploration tools using shell commands and file analysis"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        
    def explore_directory_structure(self) -> Dict[str, Any]:
        """
        Explore directory structure using tree-like analysis
        
        Returns:
            Comprehensive directory structure analysis
        """
        if not self.base_path.exists():
            return {"error": f"Path does not exist: {self.base_path}"}
        
        structure = {
            "base_path": str(self.base_path),
            "total_size_mb": 0,
            "directories": {},
            "files": {},
            "file_types": {},
            "summary": {}
        }
        
        try:
            # Walk through directory tree
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for root, dirs, files in os.walk(self.base_path):
                rel_root = Path(root).relative_to(self.base_path)
                
                # Directory info
                if str(rel_root) != '.':
                    structure["directories"][str(rel_root)] = {
                        "subdirs": dirs,
                        "file_count": len(files),
                        "files": []
                    }
                    dir_count += 1
                
                # File analysis
                for file in files:
                    file_path = Path(root) / file
                    try:
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        file_count += 1
                        
                        file_ext = file_path.suffix.lower()
                        if file_ext not in structure["file_types"]:
                            structure["file_types"][file_ext] = 0
                        structure["file_types"][file_ext] += 1
                        
                        file_info = {
                            "name": file,
                            "path": str(file_path.relative_to(self.base_path)),
                            "size_bytes": file_size,
                            "size_mb": round(file_size / (1024 * 1024), 2),
                            "extension": file_ext,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                        
                        structure["files"][str(file_path.relative_to(self.base_path))] = file_info
                        
                        # Add to directory listing
                        if str(rel_root) in structure["directories"]:
                            structure["directories"][str(rel_root)]["files"].append(file_info)
                        
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Cannot access file {file_path}: {e}")
            
            structure["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            structure["summary"] = {
                "total_files": file_count,
                "total_directories": dir_count,
                "largest_files": self._get_largest_files(structure["files"], 5),
                "file_type_distribution": structure["file_types"]
            }
            
        except Exception as e:
            structure["error"] = str(e)
        
        return structure
    
    def grep_files(self, pattern: str, file_types: List[str] = None, max_matches: int = 100) -> Dict[str, List[Dict]]:
        """
        Search for patterns in files using grep-like functionality
        
        Args:
            pattern: Regex pattern to search for
            file_types: File extensions to search in (e.g., ['.log', '.txt'])
            max_matches: Maximum number of matches to return
            
        Returns:
            Dictionary mapping file paths to matches
        """
        matches = {}
        match_count = 0
        
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if match_count >= max_matches:
                    break
                    
                file_path = Path(root) / file
                
                # Filter by file type if specified
                if file_types and file_path.suffix.lower() not in file_types:
                    continue
                
                try:
                    # Detect encoding
                    encoding = 'utf-8'  # Default encoding
                    if HAS_CHARDET:
                        try:
                            with open(file_path, 'rb') as f:
                                raw_data = f.read(10000)  # Sample first 10KB
                                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
                        except Exception:
                            pass
                    
                    # Search in file
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        file_matches = []
                        for line_num, line in enumerate(f, 1):
                            if match_count >= max_matches:
                                break
                            if re.search(pattern, line, re.IGNORECASE):
                                file_matches.append({
                                    "line_number": line_num,
                                    "line": line.strip(),
                                    "match": re.search(pattern, line, re.IGNORECASE).group(0)
                                })
                                match_count += 1
                        
                        if file_matches:
                            matches[str(file_path.relative_to(self.base_path))] = file_matches
                            
                except Exception as e:
                    self.logger.warning(f"Cannot search file {file_path}: {e}")
        
        return matches
    
    def _get_largest_files(self, files_dict: Dict, count: int = 5) -> List[Dict]:
        """Get the largest files by size"""
        sorted_files = sorted(files_dict.values(), key=lambda x: x['size_bytes'], reverse=True)
        return sorted_files[:count]


class CSVAnalyzer:
    """Intelligent CSV file analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_csv_structure(self, file_path: Union[str, Path], sample_rows: int = 5) -> Dict[str, Any]:
        """
        Analyze CSV file structure and content
        
        Args:
            file_path: Path to CSV file
            sample_rows: Number of sample rows to include
            
        Returns:
            Comprehensive CSV analysis
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File does not exist: {file_path}"}
        
        analysis = {
            "file_path": str(file_path),
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "columns": [],
            "column_count": 0,
            "row_count": 0,
            "data_types": {},
            "sample_data": [],
            "column_stats": {},
            "potential_identifiers": [],
            "potential_timestamps": [],
            "potential_metrics": [],
            "encoding": "utf-8"
        }
        
        try:
            # Detect encoding
            analysis["encoding"] = 'utf-8'  # Default encoding
            if HAS_CHARDET:
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read(10000)
                        encoding_info = chardet.detect(raw_data)
                        analysis["encoding"] = encoding_info['encoding'] or 'utf-8'
                except Exception:
                    pass
            
            # Read CSV with pandas
            df = pd.read_csv(file_path, encoding=analysis["encoding"], nrows=1000)  # Sample first 1000 rows
            
            analysis["columns"] = list(df.columns)
            analysis["column_count"] = len(df.columns)
            analysis["row_count"] = len(df)
            
            # Get sample data
            analysis["sample_data"] = df.head(sample_rows).to_dict('records')
            
            # Analyze each column
            for col in df.columns:
                col_analysis = self._analyze_column(df[col], col)
                analysis["column_stats"][col] = col_analysis
                analysis["data_types"][col] = str(df[col].dtype)
                
                # Categorize columns
                if col_analysis["likely_identifier"]:
                    analysis["potential_identifiers"].append(col)
                if col_analysis["likely_timestamp"]:
                    analysis["potential_timestamps"].append(col)
                if col_analysis["likely_metric"]:
                    analysis["potential_metrics"].append(col)
            
            # Get full row count (for large files, estimate)
            try:
                full_df = pd.read_csv(file_path, encoding=analysis["encoding"])
                analysis["row_count"] = len(full_df)
            except:
                # Estimate row count for very large files
                with open(file_path, 'r', encoding=analysis["encoding"]) as f:
                    analysis["row_count"] = sum(1 for line in f) - 1  # Subtract header
            
        except Exception as e:
            analysis["error"] = str(e)
            self.logger.error(f"Error analyzing CSV {file_path}: {e}")
        
        return analysis
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze individual column characteristics"""
        analysis = {
            "null_count": series.isnull().sum(),
            "null_percentage": round((series.isnull().sum() / len(series)) * 100, 2),
            "unique_count": series.nunique(),
            "unique_percentage": round((series.nunique() / len(series)) * 100, 2),
            "likely_identifier": False,
            "likely_timestamp": False,
            "likely_metric": False,
            "sample_values": []
        }
        
        # Sample values (non-null)
        non_null_values = series.dropna()
        if len(non_null_values) > 0:
            sample_size = min(5, len(non_null_values))
            analysis["sample_values"] = non_null_values.head(sample_size).tolist()
        
        # Detect column types
        column_lower = column_name.lower()
        
        # Identifier detection
        if any(keyword in column_lower for keyword in ['id', 'identifier', 'key', 'cmdb']):
            analysis["likely_identifier"] = True
        elif analysis["unique_percentage"] > 80:  # High uniqueness suggests identifier
            analysis["likely_identifier"] = True
        
        # Timestamp detection
        if any(keyword in column_lower for keyword in ['time', 'date', 'timestamp', 'created', 'modified']):
            analysis["likely_timestamp"] = True
        elif series.dtype == 'object':
            # Try to parse some values as timestamps
            sample_vals = non_null_values.head(3).astype(str)
            timestamp_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{10}',             # Unix timestamp (seconds)
                r'\d{13}',             # Unix timestamp (milliseconds)
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            ]
            
            for val in sample_vals:
                if any(re.search(pattern, str(val)) for pattern in timestamp_patterns):
                    analysis["likely_timestamp"] = True
                    break
        
        # Metric detection
        if series.dtype in ['int64', 'float64']:
            analysis["likely_metric"] = True
        elif any(keyword in column_lower for keyword in ['count', 'rate', 'usage', 'cpu', 'memory', 'disk', 'network', 'value', 'metric']):
            analysis["likely_metric"] = True
        
        # Statistical analysis for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            analysis.update({
                "min": series.min(),
                "max": series.max(),
                "mean": round(series.mean(), 2),
                "median": series.median(),
                "std": round(series.std(), 2)
            })
        
        return analysis


class LogAnalyzer:
    """Intelligent log file analysis with pattern detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.log_level_patterns = {
            'ERROR': [r'\bERROR\b', r'\bErr\b', r'\bFAIL\b', r'\bFATAL\b'],
            'WARN': [r'\bWARN\b', r'\bWARNING\b'],
            'INFO': [r'\bINFO\b', r'\bInformation\b'],
            'DEBUG': [r'\bDEBUG\b', r'\bDBG\b'],
            'TRACE': [r'\bTRACE\b', r'\bTRC\b']
        }
    
    def analyze_log_file(self, file_path: Union[str, Path], sample_lines: int = 100) -> Dict[str, Any]:
        """
        Analyze log file structure and patterns
        
        Args:
            file_path: Path to log file
            sample_lines: Number of lines to sample for pattern analysis
            
        Returns:
            Comprehensive log file analysis
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File does not exist: {file_path}"}
        
        analysis = {
            "file_path": str(file_path),
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "total_lines": 0,
            "log_levels": {},
            "timestamp_patterns": [],
            "common_patterns": [],
            "unique_messages": [],
            "error_patterns": [],
            "sample_lines": [],
            "encoding": "utf-8"
        }
        
        try:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                encoding_info = chardet.detect(raw_data)
                analysis["encoding"] = encoding_info['encoding'] or 'utf-8'
            
            # Read and analyze log file
            with open(file_path, 'r', encoding=analysis["encoding"], errors='ignore') as f:
                lines = []
                line_count = 0
                
                # Collect sample lines and count total
                for line in f:
                    line_count += 1
                    if len(lines) < sample_lines:
                        lines.append(line.strip())
                
                analysis["total_lines"] = line_count
                analysis["sample_lines"] = lines[:10]  # Store first 10 for reference
            
            # Analyze patterns
            self._analyze_log_levels(lines, analysis)
            self._analyze_timestamp_patterns(lines, analysis)
            self._extract_common_patterns(lines, analysis)
            self._identify_error_patterns(lines, analysis)
            
        except Exception as e:
            analysis["error"] = str(e)
            self.logger.error(f"Error analyzing log file {file_path}: {e}")
        
        return analysis
    
    def _analyze_log_levels(self, lines: List[str], analysis: Dict):
        """Detect log levels in the sample lines"""
        level_counts = defaultdict(int)
        
        for line in lines:
            line_upper = line.upper()
            for level, patterns in self.log_level_patterns.items():
                if any(re.search(pattern, line_upper) for pattern in patterns):
                    level_counts[level] += 1
                    break
        
        analysis["log_levels"] = dict(level_counts)
    
    def _analyze_timestamp_patterns(self, lines: List[str], analysis: Dict):
        """Detect timestamp patterns in log lines"""
        timestamp_patterns = [
            (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'YYYY-MM-DD HH:MM:SS'),
            (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', 'MM/DD/YYYY HH:MM:SS'),
            (r'\d{10}', 'Unix timestamp (seconds)'),
            (r'\d{13}', 'Unix timestamp (milliseconds)'),
            (r'\w{3} \d{2} \d{2}:\d{2}:\d{2}', 'MMM DD HH:MM:SS'),
        ]
        
        detected_patterns = []
        for pattern, description in timestamp_patterns:
            matches = 0
            for line in lines[:20]:  # Check first 20 lines
                if re.search(pattern, line):
                    matches += 1
            
            if matches > 0:
                detected_patterns.append({
                    "pattern": pattern,
                    "description": description,
                    "matches": matches,
                    "confidence": round(matches / min(20, len(lines)), 2)
                })
        
        analysis["timestamp_patterns"] = sorted(detected_patterns, key=lambda x: x["confidence"], reverse=True)
    
    def _extract_common_patterns(self, lines: List[str], analysis: Dict):
        """Extract common message patterns"""
        # Simple pattern extraction - look for repeated structures
        pattern_counts = defaultdict(int)
        
        for line in lines:
            # Remove timestamps and log levels to focus on message structure
            cleaned_line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '[TIMESTAMP]', line)
            cleaned_line = re.sub(r'\b(ERROR|WARN|INFO|DEBUG|TRACE)\b', '[LEVEL]', cleaned_line, flags=re.IGNORECASE)
            
            # Extract pattern by replacing variable parts
            pattern = re.sub(r'\d+', '[NUMBER]', cleaned_line)
            pattern = re.sub(r'\b\w+@\w+\.\w+\b', '[EMAIL]', pattern)
            pattern = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '[IP]', pattern)
            
            pattern_counts[pattern] += 1
        
        # Get most common patterns
        common_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis["common_patterns"] = [{"pattern": p, "count": c} for p, c in common_patterns]
    
    def _identify_error_patterns(self, lines: List[str], analysis: Dict):
        """Identify error and exception patterns"""
        error_patterns = []
        
        for line in lines:
            line_upper = line.upper()
            if any(keyword in line_upper for keyword in ['ERROR', 'EXCEPTION', 'FAIL', 'FATAL', 'CRITICAL']):
                # Extract key parts of error message
                cleaned = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', line)
                cleaned = re.sub(r'\d+', '[NUM]', cleaned)
                error_patterns.append(cleaned.strip())
        
        # Count unique error patterns
        error_counts = Counter(error_patterns)
        analysis["error_patterns"] = [{"pattern": p, "count": c} for p, c in error_counts.most_common(5)]


class GenericDatasetAnalyzer:
    """High-level analyzer that combines all tools"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.explorer = SystemExplorer(dataset_path)
        self.csv_analyzer = CSVAnalyzer()
        self.log_analyzer = LogAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis
        
        Returns:
            Complete dataset analysis with structure, files, and content analysis
        """
        analysis = {
            "dataset_path": str(self.dataset_path),
            "timestamp": datetime.now().isoformat(),
            "structure": {},
            "csv_files": {},
            "log_files": {},
            "other_files": {},
            "summary": {},
            "recommendations": []
        }
        
        try:
            # Explore directory structure
            self.logger.info("🔍 Exploring directory structure...")
            structure = self.explorer.explore_directory_structure()
            analysis["structure"] = structure
            
            if "error" in structure:
                analysis["error"] = structure["error"]
                return analysis
            
            # Analyze files by type
            csv_count = 0
            log_count = 0
            other_count = 0
            
            for file_path, file_info in structure["files"].items():
                full_path = self.dataset_path / file_path
                
                if file_info["extension"] in ['.csv']:
                    self.logger.info(f"📊 Analyzing CSV: {file_path}")
                    csv_analysis = self.csv_analyzer.analyze_csv_structure(full_path)
                    analysis["csv_files"][file_path] = csv_analysis
                    csv_count += 1
                    
                elif file_info["extension"] in ['.log', '.txt'] or 'log' in file_info["name"].lower():
                    self.logger.info(f"📄 Analyzing log: {file_path}")
                    log_analysis = self.log_analyzer.analyze_log_file(full_path)
                    analysis["log_files"][file_path] = log_analysis
                    log_count += 1
                    
                else:
                    analysis["other_files"][file_path] = file_info
                    other_count += 1
            
            # Generate summary and recommendations
            analysis["summary"] = {
                "total_files": len(structure["files"]),
                "csv_files": csv_count,
                "log_files": log_count,
                "other_files": other_count,
                "total_size_mb": structure["total_size_mb"]
            }
            
            # Generate recommendations for analysis
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
        except Exception as e:
            analysis["error"] = str(e)
            self.logger.error(f"Error analyzing dataset: {e}")
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on dataset analysis"""
        recommendations = []
        
        # CSV file recommendations
        if analysis["csv_files"]:
            recommendations.append("📊 CSV files detected - suitable for metrics and structured data analysis")
            
            # Check for common RCA patterns
            csv_files = analysis["csv_files"]
            has_metrics = any("metric" in path.lower() for path in csv_files.keys())
            has_traces = any("trace" in path.lower() for path in csv_files.keys())
            
            if has_metrics:
                recommendations.append("📈 Metric files found - can perform threshold-based anomaly detection")
            if has_traces:
                recommendations.append("🔗 Trace files found - can perform distributed tracing analysis")
        
        # Log file recommendations
        if analysis["log_files"]:
            recommendations.append("📄 Log files detected - suitable for pattern-based anomaly detection")
            
            # Check log levels
            for file_path, log_analysis in analysis["log_files"].items():
                if "ERROR" in log_analysis.get("log_levels", {}):
                    recommendations.append(f"🚨 Error patterns found in {file_path} - priority for analysis")
        
        # Size-based recommendations
        total_size = analysis["summary"]["total_size_mb"]
        if total_size > 1000:  # > 1GB
            recommendations.append("💾 Large dataset detected - recommend chunked processing and memory optimization")
        elif total_size < 10:  # < 10MB
            recommendations.append("📦 Small dataset - can process entirely in memory")
        
        return recommendations