# data_quality.py
import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
import json
import asyncio

class DataQualityAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the analyzer with OpenAI API key"""
        self.api_key = api_key
        openai.api_key = api_key

    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Analyze data quality metrics
        
        Args:
            df (pd.DataFrame): Input DataFrame to analyze
            
        Returns:
            Dict: Dictionary containing various quality metrics
        """
        try:
            quality_metrics = {
                "completeness": self._check_completeness(df),
                "uniqueness": self._check_uniqueness(df),
                "consistency": self._check_consistency(df),
                "statistics": self._get_basic_statistics(df),
                "patterns": self._detect_patterns(df),
                "sample_data": df.head(3).to_dict(orient='records')
            }
            return self._convert_to_native_types(quality_metrics)
        except Exception as e:
            return {"error": str(e)}

    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness metrics"""
        try:
            return {
                "missing_values": df.isnull().sum().to_dict(),
                "completion_rate": (1 - df.isnull().sum() / len(df)).to_dict(),
                "total_rows": len(df),
                "complete_rows": len(df.dropna())
            }
        except Exception as e:
            return {"error": str(e)}

    def _check_uniqueness(self, df: pd.DataFrame) -> Dict:
        """Check data uniqueness metrics"""
        try:
            return {
                "unique_counts": df.nunique().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "potential_keys": [
                    col for col in df.columns 
                    if df[col].nunique() == len(df)
                ],
                "uniqueness_ratio": {
                    col: float(df[col].nunique() / len(df))
                    for col in df.columns
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def _check_consistency(self, df: pd.DataFrame) -> Dict:
        """Check data consistency and identify outliers"""
        try:
            consistency = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    stats = pd.to_numeric(df[col], errors='coerce')
                    if not stats.empty:
                        consistency[col] = {
                            "outliers": self._detect_outliers(stats),
                            "value_range": {
                                "min": float(stats.min()),
                                "max": float(stats.max())
                            },
                            "distribution": {
                                "skewness": float(stats.skew()),
                                "kurtosis": float(stats.kurtosis())
                            }
                        }
            return consistency
        except Exception as e:
            return {"error": str(e)}

    def _detect_outliers(self, series: pd.Series) -> Dict:
        """Detect outliers using IQR method"""
        try:
            Q1 = float(series.quantile(0.25))
            Q3 = float(series.quantile(0.75))
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            return {
                "count": int(len(outliers)),
                "percentage": float(len(outliers) / len(series)),
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound)
                },
                "extreme_values": {
                    "min": float(outliers.min()) if not outliers.empty else None,
                    "max": float(outliers.max()) if not outliers.empty else None
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic statistical metrics"""
        try:
            stats = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    series = pd.to_numeric(df[col], errors='coerce')
                    if not series.empty:
                        stats[col] = {
                            "mean": float(series.mean()),
                            "median": float(series.median()),
                            "std": float(series.std()),
                            "skewness": float(series.skew()),
                            "quartiles": {
                                "25%": float(series.quantile(0.25)),
                                "50%": float(series.quantile(0.50)),
                                "75%": float(series.quantile(0.75))
                            }
                        }
            return stats
        except Exception as e:
            return {"error": str(e)}

    async def suggest_data_warehouse_structure(self, df: pd.DataFrame, data_analysis: Dict) -> Dict:
        """Suggest data warehouse structure using OpenAI API"""
        try:
            # Prepare the data for the prompt
            data_analysis_clean = self._convert_to_native_types(data_analysis)
            sample_data = df.head(3).to_dict(orient='records')

            prompt = f"""As a data warehouse architect, analyze this dataset and suggest an optimal structure:

            Data Analysis:
            {json.dumps(data_analysis_clean, indent=2)}

            Sample Data:
            {json.dumps(sample_data, indent=2)}

            Please provide a complete data warehouse design including:
            1. Fact tables (identify measures and grain)
            2. Dimension tables (with attributes and hierarchies)
            3. Relationships between tables
            4. Recommended indexes and partitioning strategy
            5. Data modeling best practices and optimization suggestions

            Format the response in a clear, structured way using Markdown."""

            # Make the API call
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data warehouse architect. Provide detailed, technical, and practical advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            if response and response.choices:
                suggestion = response.choices[0].message.content
                return {
                    "status": "success",
                    "content": suggestion,
                    "type": "markdown"
                }
            else:
                return {"error": "No response from OpenAI API"}

        except Exception as e:
            return {"error": str(e)}

    def _detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect patterns in string data"""
        try:
            patterns = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().sample(min(1000, len(df))).tolist()
                    patterns[col] = {
                        "common_prefixes": self._find_common_prefixes(sample),
                        "common_formats": self._detect_formats(sample),
                        "value_patterns": self._analyze_value_patterns(sample)
                    }
            return patterns
        except Exception as e:
            return {"error": str(e)}

    def _convert_to_native_types(self, obj: Any) -> Any:
        """Convert numpy and pandas types to native Python types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj

    def _find_common_prefixes(self, values: List[str], min_length: int = 2) -> List[str]:
        """Find common prefixes in string values"""
        try:
            if not values:
                return []
            
            prefixes = {}
            for value in values:
                if isinstance(value, str):
                    for i in range(min_length, len(value) + 1):
                        prefix = value[:i]
                        prefixes[prefix] = prefixes.get(prefix, 0) + 1
            
            threshold = len(values) * 0.1
            return [p for p, c in prefixes.items() if c > threshold]
        except Exception as e:
            return []

    def _detect_formats(self, values: List[str]) -> Dict:
        """Detect common string formats"""
        try:
            total = len(values)
            if total == 0:
                return {}
                
            formats = {
                "numeric_only": sum(1 for x in values if str(x).isdigit()),
                "alpha_only": sum(1 for x in values if str(x).isalpha()),
                "alphanumeric": sum(1 for x in values if str(x).isalnum()),
                "with_special_chars": sum(1 for x in values if not str(x).isalnum())
            }
            
            return {k: float(v)/total for k, v in formats.items()}
        except Exception as e:
            return {"error": str(e)}

    def _analyze_value_patterns(self, values: List[str]) -> Dict:
        """Analyze patterns in string values"""
        try:
            length_values = [len(str(x)) for x in values if x is not None]
            if not length_values:
                return {}

            return {
                "length_stats": {
                    "min": min(length_values),
                    "max": max(length_values),
                    "most_common": max(
                        set(length_values),
                        key=lambda l: length_values.count(l)
                    )
                },
                "format_consistency": {
                    "consistent_length": len(set(length_values)) == 1
                }
            }
        except Exception as e:
            return {"error": str(e)}