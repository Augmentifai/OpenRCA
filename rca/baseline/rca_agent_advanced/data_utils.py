"""
Enhanced Data Detection and Loading Utilities for Advanced RCA Agent.

This module provides intelligent data detection, loading, and preprocessing
capabilities to ensure the agent correctly identifies and loads telemetry data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional, Any
import glob
import logging


class TelemetryDataDetector:
    """Intelligent telemetry data detection and loading system"""
    
    def __init__(self):
        self.timezone = pytz.timezone('Asia/Shanghai')
        self.data_cache = {}
        self.available_files = {}
        
    def detect_available_files(self, dataset: str, target_date: str = None) -> Dict[str, List[str]]:
        """
        Detect all available telemetry files for a dataset.
        
        Args:
            dataset: Dataset name (e.g., 'Bank', 'Telecom', 'Market/cloudbed-1')
            target_date: Optional target date in YYYY_MM_DD format
            
        Returns:
            Dictionary mapping data types to available files
        """
        base_path = f"dataset/{dataset}/telemetry"
        
        if not os.path.exists(base_path):
            return {}
        
        available_files = {
            'metric': [],
            'trace': [],
            'log': []
        }
        
        # If target_date specified, look only in that directory
        if target_date:
            date_dirs = [target_date] if os.path.exists(f"{base_path}/{target_date}") else []
        else:
            # Find all available date directories
            date_dirs = [d for d in os.listdir(base_path) 
                        if os.path.isdir(f"{base_path}/{d}") and d != '.DS_Store']
        
        for date_dir in date_dirs:
            date_path = f"{base_path}/{date_dir}"
            
            for data_type in ['metric', 'trace', 'log']:
                type_path = f"{date_path}/{data_type}"
                if os.path.exists(type_path):
                    csv_files = glob.glob(f"{type_path}/*.csv")
                    for file_path in csv_files:
                        file_info = {
                            'path': file_path,
                            'date': date_dir,
                            'filename': os.path.basename(file_path),
                            'size_mb': os.path.getsize(file_path) / (1024 * 1024)
                        }
                        available_files[data_type].append(file_info)
        
        self.available_files[dataset] = available_files
        return available_files
    
    def generate_data_loading_code(self, dataset: str, instruction: str, 
                                 target_components: List[str] = None,
                                 time_range: Tuple[datetime, datetime] = None) -> str:
        """
        Generate intelligent code for loading relevant telemetry data.
        
        Args:
            dataset: Dataset name
            instruction: Analysis instruction to understand data requirements
            target_components: Specific components to focus on
            time_range: Time range for filtering data
            
        Returns:
            Python code for loading the data
        """
        
        # Detect available files
        available_files = self.detect_available_files(dataset)
        
        if not any(available_files.values()):
            return self._generate_fallback_code(dataset, instruction)
        
        # Generate smart loading code based on available files
        code_parts = []
        
        # Add imports and setup
        code_parts.append("""
import pandas as pd
import numpy as np
import pytz
import os
import glob
from datetime import datetime, timedelta

# Set up timezone
TIMEZONE = pytz.timezone('Asia/Shanghai')

# Initialize data containers
telemetry_data = {}
loaded_files = {}

print("🔍 Detecting available telemetry files...")
""")
        
        # Add file detection logic
        code_parts.append(f"""
dataset_path = "dataset/{dataset}/telemetry"
available_dates = []

if os.path.exists(dataset_path):
    available_dates = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(f"{{dataset_path}}/{{d}}") and d != '.DS_Store']
    available_dates.sort()
    print(f"📅 Available dates: {{available_dates}}")
else:
    print(f"❌ Dataset path not found: {{dataset_path}}")
    print("🔍 Let's check what's available in the current directory...")
    print(f"Current directory contents: {{os.listdir('.')}}")
    if os.path.exists('dataset'):
        print(f"Dataset directory contents: {{os.listdir('dataset')}}")
""")
        
        # Add intelligent data loading based on instruction analysis
        if 'metric' in available_files and available_files['metric']:
            code_parts.append(self._generate_metric_loading_code(available_files['metric'], target_components, time_range))
        
        if 'trace' in available_files and available_files['trace']:
            code_parts.append(self._generate_trace_loading_code(available_files['trace'], target_components, time_range))
        
        if 'log' in available_files and available_files['log']:
            code_parts.append(self._generate_log_loading_code(available_files['log'], target_components, time_range))
        
        # Add summary and validation
        code_parts.append("""
# Summary of loaded data
print("\\n📊 Data Loading Summary:")
for data_type, data in telemetry_data.items():
    if isinstance(data, pd.DataFrame) and not data.empty:
        print(f"  {data_type}: {len(data)} records, columns: {list(data.columns)}")
    elif isinstance(data, list):
        print(f"  {data_type}: {len(data)} items")
    else:
        print(f"  {data_type}: No data loaded")

print(f"\\n📁 Loaded files: {list(loaded_files.keys())}")
""")
        
        return '\n'.join(code_parts)
    
    def _generate_metric_loading_code(self, metric_files: List[Dict], 
                                    target_components: List[str] = None,
                                    time_range: Tuple[datetime, datetime] = None) -> str:
        """Generate code for loading metric files"""
        
        return f"""
# Load metric data
print("\\n📈 Loading metric data...")
metric_data_frames = []

for date in available_dates:
    metric_path = f"{{dataset_path}}/{{date}}/metric"
    if os.path.exists(metric_path):
        metric_files = glob.glob(f"{{metric_path}}/*.csv")
        print(f"  Found metric files for {{date}}: {{[os.path.basename(f) for f in metric_files]}}")
        
        for file_path in metric_files:
            try:
                filename = os.path.basename(file_path)
                print(f"    Loading {{filename}}...")
                
                df = pd.read_csv(file_path)
                
                # Standardize timestamp column
                timestamp_col = None
                for col in ['timestamp', 'startTime', 'time']:
                    if col in df.columns:
                        timestamp_col = col
                        break
                
                if timestamp_col:
                    # Handle timestamp conversion
                    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
                    
                    # Convert to datetime (handle both seconds and milliseconds)
                    if df[timestamp_col].max() > 2000000000:  # Likely milliseconds
                        df['datetime'] = pd.to_datetime(df[timestamp_col], unit='ms', utc=True)
                    else:  # Likely seconds
                        df['datetime'] = pd.to_datetime(df[timestamp_col], unit='s', utc=True)
                    
                    # Convert to Asia/Shanghai timezone
                    df['datetime'] = df['datetime'].dt.tz_convert(TIMEZONE)
                
                df['file_source'] = filename
                df['date'] = date
                
                # Filter by target components if specified
                #if target_components and 'cmdb_id' in df.columns:
                #    df = df[df['cmdb_id'].isin({target_components})]
                
                metric_data_frames.append(df)
                loaded_files[filename] = file_path
                
                print(f"      ✅ Loaded {{len(df)}} records from {{filename}}")
                
            except Exception as e:
                print(f"      ❌ Error loading {{filename}}: {{str(e)}}")

if metric_data_frames:
    telemetry_data['metrics'] = pd.concat(metric_data_frames, ignore_index=True)
    print(f"📈 Total metric records: {{len(telemetry_data['metrics'])}}")
    
    # Show sample of data structure
    if not telemetry_data['metrics'].empty:
        print("\\n📋 Metric data structure sample:")
        print(telemetry_data['metrics'].head())
        print(f"\\n📊 Available columns: {{list(telemetry_data['metrics'].columns)}}")
        
        if 'cmdb_id' in telemetry_data['metrics'].columns:
            unique_components = telemetry_data['metrics']['cmdb_id'].unique()
            print(f"🏷️  Available components: {{list(unique_components)}}")
        
        if 'kpi_name' in telemetry_data['metrics'].columns:
            unique_kpis = telemetry_data['metrics']['kpi_name'].unique()
            print(f"📊 Available KPIs: {{list(unique_kpis)}}")
else:
    print("⚠️ No metric data loaded")
"""
    
    def _generate_trace_loading_code(self, trace_files: List[Dict], 
                                   target_components: List[str] = None,
                                   time_range: Tuple[datetime, datetime] = None) -> str:
        """Generate code for loading trace files"""
        
        return f"""
# Load trace data
print("\\n🔗 Loading trace data...")
trace_data_frames = []

for date in available_dates:
    trace_path = f"{{dataset_path}}/{{date}}/trace"
    if os.path.exists(trace_path):
        trace_files = glob.glob(f"{{trace_path}}/*.csv")
        print(f"  Found trace files for {{date}}: {{[os.path.basename(f) for f in trace_files]}}")
        
        for file_path in trace_files:
            try:
                filename = os.path.basename(file_path)
                print(f"    Loading {{filename}}...")
                
                df = pd.read_csv(file_path)
                
                # Standardize timestamp column
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                    
                    # Trace timestamps are usually in milliseconds
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    df['datetime'] = df['datetime'].dt.tz_convert(TIMEZONE)
                
                df['file_source'] = filename
                df['date'] = date
                
                # Filter by target components if specified
                #if target_components and 'cmdb_id' in df.columns:
                #    df = df[df['cmdb_id'].isin({target_components})]
                
                trace_data_frames.append(df)
                loaded_files[filename] = file_path
                
                print(f"      ✅ Loaded {{len(df)}} records from {{filename}}")
                
            except Exception as e:
                print(f"      ❌ Error loading {{filename}}: {{str(e)}}")

if trace_data_frames:
    telemetry_data['traces'] = pd.concat(trace_data_frames, ignore_index=True)
    print(f"🔗 Total trace records: {{len(telemetry_data['traces'])}}")
    
    # Show sample of data structure
    if not telemetry_data['traces'].empty:
        print("\\n📋 Trace data structure sample:")
        print(telemetry_data['traces'].head())
else:
    print("⚠️ No trace data loaded")
"""
    
    def _generate_log_loading_code(self, log_files: List[Dict], 
                                 target_components: List[str] = None,
                                 time_range: Tuple[datetime, datetime] = None) -> str:
        """Generate code for loading log files"""
        
        return f"""
# Load log data  
print("\\n📄 Loading log data...")
log_data_frames = []

for date in available_dates:
    log_path = f"{{dataset_path}}/{{date}}/log"
    if os.path.exists(log_path):
        log_files = glob.glob(f"{{log_path}}/*.csv")
        print(f"  Found log files for {{date}}: {{[os.path.basename(f) for f in log_files]}}")
        
        for file_path in log_files:
            try:
                filename = os.path.basename(file_path)
                print(f"    Loading {{filename}}...")
                
                df = pd.read_csv(file_path)
                
                # Standardize timestamp column
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                    
                    # Log timestamps are usually in seconds
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                    df['datetime'] = df['datetime'].dt.tz_convert(TIMEZONE)
                
                df['file_source'] = filename
                df['date'] = date
                
                # Filter by target components if specified
                #if target_components and 'cmdb_id' in df.columns:
                #    df = df[df['cmdb_id'].isin({target_components})]
                
                log_data_frames.append(df)
                loaded_files[filename] = file_path
                
                print(f"      ✅ Loaded {{len(df)}} records from {{filename}}")
                
            except Exception as e:
                print(f"      ❌ Error loading {{filename}}: {{str(e)}}")

if log_data_frames:
    telemetry_data['logs'] = pd.concat(log_data_frames, ignore_index=True)
    print(f"📄 Total log records: {{len(telemetry_data['logs'])}}")
    
    # Show sample of data structure
    if not telemetry_data['logs'].empty:
        print("\\n📋 Log data structure sample:")
        print(telemetry_data['logs'].head())
else:
    print("⚠️ No log data loaded")
"""
    
    def _generate_fallback_code(self, dataset: str, instruction: str) -> str:
        """Generate fallback code when files are not detected"""
        
        return f"""
# Fallback: Dataset files not found, let's investigate
print("🔍 Dataset files not detected. Investigating available data...")
print(f"Current working directory: {{os.getcwd()}}")

# Check if dataset directory exists
if os.path.exists('dataset'):
    print(f"✅ Dataset directory exists")
    dataset_contents = os.listdir('dataset')
    print(f"📁 Dataset contents: {{dataset_contents}}")
    
    # Check specific dataset
    dataset_path = "dataset/{dataset}"
    if os.path.exists(dataset_path):
        print(f"✅ {dataset} dataset exists")
        contents = os.listdir(dataset_path)
        print(f"📁 {dataset} contents: {{contents}}")
        
        # Check telemetry directory
        telemetry_path = f"{{dataset_path}}/telemetry"
        if os.path.exists(telemetry_path):
            print(f"✅ Telemetry directory exists")
            dates = os.listdir(telemetry_path)
            print(f"📅 Available dates: {{dates}}")
            
            # Check first available date
            if dates:
                first_date = [d for d in dates if d != '.DS_Store'][0] if dates else None
                if first_date:
                    date_path = f"{{telemetry_path}}/{{first_date}}"
                    print(f"📂 Contents of {{first_date}}: {{os.listdir(date_path)}}")
                    
                    # Check each data type
                    for data_type in ['metric', 'trace', 'log']:
                        type_path = f"{{date_path}}/{{data_type}}"
                        if os.path.exists(type_path):
                            files = os.listdir(type_path)
                            print(f"  📊 {{data_type}} files: {{files}}")
        else:
            print(f"❌ Telemetry directory not found: {{telemetry_path}}")
    else:
        print(f"❌ Dataset not found: {{dataset_path}}")
else:
    print("❌ Dataset directory not found")
    print("📂 Available directories:", [d for d in os.listdir('.') if os.path.isdir(d)])

# Set up empty containers for now
telemetry_data = {{}}
loaded_files = {{}}
print("\\n⚠️ No telemetry data loaded. Please ensure dataset files are available.")
"""


def extract_components_from_instruction(instruction: str) -> List[str]:
    """Extract component names from the instruction text"""
    # Common component patterns in OpenRCA
    common_components = [
        'Tomcat01', 'Tomcat02', 'Tomcat03', 'Tomcat04',
        'apache01', 'apache02', 'MG01', 'MG02', 'IG01', 'IG02',
        'Mysql01', 'Mysql02', 'Redis01', 'Redis02',
        'os_001', 'os_002', 'os_003', 'docker_001', 'docker_002',
        'db_001', 'db_002', 'frontend', 'shippingservice', 'checkoutservice'
    ]
    
    found_components = []
    instruction_upper = instruction.upper()
    
    for component in common_components:
        if component.upper() in instruction_upper:
            found_components.append(component)
    
    return found_components


def extract_time_range_from_instruction(instruction: str) -> Optional[Tuple[datetime, datetime]]:
    """Extract time range from instruction text"""
    import re
    
    # Look for patterns like "14:30 to 15:00" or "March 4, 2021"
    time_patterns = [
        r'(\d{1,2}):(\d{2})\s*to\s*(\d{1,2}):(\d{2})',
        r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})',
    ]
    
    date_patterns = [
        r'March\s+(\d{1,2}),\s*(\d{4})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
    ]
    
    timezone = pytz.timezone('Asia/Shanghai')
    
    # Extract time range
    start_time = None
    end_time = None
    date_info = None
    
    for pattern in time_patterns:
        match = re.search(pattern, instruction)
        if match:
            start_hour, start_min, end_hour, end_min = map(int, match.groups())
            # We'll need date info to complete this
            break
    
    for pattern in date_patterns:
        match = re.search(pattern, instruction)
        if match:
            if 'March' in pattern:
                day, year = map(int, match.groups())
                month = 3
            else:
                year, month, day = map(int, match.groups())
            date_info = (year, month, day)
            break
    
    if date_info and 'start_hour' in locals():
        year, month, day = date_info
        start_time = datetime(year, month, day, start_hour, start_min, tzinfo=timezone)
        end_time = datetime(year, month, day, end_hour, end_min, tzinfo=timezone)
        return (start_time, end_time)
    
    return None
