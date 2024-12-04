# dax_generator.py
import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
import json
import asyncio

class DAXGenerator:
    def __init__(self, api_key: str):
        """Initialize DAX Generator with OpenAI API key"""
        self.api_key = api_key
        openai.api_key = api_key
        
        # Enhanced system prompt for better DAX generation
        self.system_prompt = """You are an expert Power BI DAX developer specializing in optimized measures and calculations.
        Your responsibilities:
        1. Generate precise, optimized, and production-ready DAX code
        2. Provide clear, detailed explanations of the code
        3. Suggest performance optimizations and best practices
        4. Recommend appropriate visualizations
        5. Include error handling in complex measures
        
        Respond in JSON format with:
        {
            "dax_code": "The optimized DAX code",
            "explanation": "Detailed explanation of how the measure works",
            "optimization_tips": ["List of specific optimization suggestions"],
            "suggested_visuals": ["List of recommended visualizations"],
            "error_handling": "Description of error handling approach",
            "performance_considerations": "Performance impact and considerations"
        }"""

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

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Analyze dataset structure for DAX generation
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Comprehensive analysis of the dataset
        """
        try:
            analysis = {
                "columns": list(df.columns) if not df.empty else [],
                "data_types": df.dtypes.astype(str).to_dict() if not df.empty else {},
                "sample_data": df.head(3).to_dict(orient='records') if not df.empty else [],
                "statistics": {
                    "numerical": self._analyze_numerical(df),
                    "categorical": self._analyze_categorical(df),
                    "temporal": self._analyze_temporal(df)
                },
                "hierarchies": self._detect_hierarchies(df),
                "data_patterns": self._analyze_data_patterns(df)
            }
            return self._convert_to_native_types(analysis)
        except Exception as e:
            return {
                "error": f"Error analyzing dataset: {str(e)}",
                "columns": [],
                "data_types": {},
                "sample_data": [],
                "statistics": {
                    "numerical": {},
                    "categorical": {},
                    "temporal": {}
                },
                "hierarchies": [],
                "data_patterns": {}
            }

    def _analyze_numerical(self, df: pd.DataFrame) -> Dict:
        """Enhanced analysis of numerical columns"""
        try:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            return {
                "columns": list(numerical_cols),
                "statistics": {
                    col: {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                        "has_negatives": bool((df[col] < 0).any()) if not df[col].empty else None,
                        "distinct_values": int(df[col].nunique()),
                        "zero_count": int((df[col] == 0).sum()),
                        "null_count": int(df[col].isnull().sum())
                    } for col in numerical_cols
                }
            }
        except Exception as e:
            return {"error": f"Error analyzing numerical columns: {str(e)}"}

    def _analyze_categorical(self, df: pd.DataFrame) -> Dict:
        """Enhanced analysis of categorical columns"""
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            return {
                "columns": list(categorical_cols),
                "unique_values": {
                    col: {
                        "count": int(df[col].nunique()),
                        "top_values": df[col].value_counts().head(5).to_dict(),
                        "null_count": int(df[col].isnull().sum()),
                        "is_boolean_like": self._is_boolean_like(df[col])
                    } for col in categorical_cols
                }
            }
        except Exception as e:
            return {"error": f"Error analyzing categorical columns: {str(e)}"}

    def _analyze_temporal(self, df: pd.DataFrame) -> Dict:
        """Enhanced analysis of temporal columns"""
        try:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            temporal_analysis = {}
            
            for col in date_cols:
                temporal_analysis[col] = {
                    "min_date": str(df[col].min()),
                    "max_date": str(df[col].max()),
                    "total_days": int((df[col].max() - df[col].min()).days),
                    "date_parts": {
                        "has_time": bool((df[col].dt.time != pd.Timestamp('00:00:00').time()).any()),
                        "has_seconds": bool((df[col].dt.second != 0).any()),
                        "has_milliseconds": bool((df[col].dt.microsecond != 0).any())
                    }
                }
            
            return temporal_analysis
        except Exception as e:
            return {"error": f"Error analyzing temporal columns: {str(e)}"}

    def _detect_hierarchies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential hierarchies in the data"""
        try:
            hierarchies = []
            
            # Date hierarchies
            date_cols = df.select_dtypes(include=['datetime64']).columns
            for col in date_cols:
                hierarchies.append({
                    "name": f"{col}_hierarchy",
                    "type": "date",
                    "levels": ["Year", "Quarter", "Month", "Date"],
                    "source_column": col
                })
            
            # Geographic hierarchies
            geo_patterns = ['country', 'region', 'city', 'state', 'province']
            geo_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in geo_patterns)]
            if len(geo_cols) > 1:
                hierarchies.append({
                    "name": "geography_hierarchy",
                    "type": "geographic",
                    "levels": geo_cols,
                    "source_columns": geo_cols
                })
            
            return hierarchies
        except Exception as e:
            return []

    def _analyze_data_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in the data for better DAX generation"""
        try:
            patterns = {
                "incremental_columns": self._detect_incremental_columns(df),
                "percentage_columns": self._detect_percentage_columns(df),
                "amount_columns": self._detect_amount_columns(df),
                "flag_columns": self._detect_flag_columns(df)
            }
            return patterns
        except Exception as e:
            return {"error": str(e)}

    def _is_boolean_like(self, series: pd.Series) -> bool:
        """Check if a categorical column is boolean-like"""
        unique_values = series.dropna().unique()
        boolean_pairs = [
            {'yes', 'no'}, {'true', 'false'}, {'0', '1'},
            {'y', 'n'}, {'t', 'f'}, {0, 1}, {True, False}
        ]
        return any(set(map(str.lower, map(str, unique_values))) == pair for pair in boolean_pairs)

    async def generate_dax_query(self, request: str, context: Dict) -> Dict:
        """Generate optimized DAX query based on request and context"""
        try:
            # Check for an error in context before proceeding
            if "error" in context:
                return {"error": context["error"]}
            
            prompt = self._build_enhanced_prompt(request, context)
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            result['dax_code'] = self._validate_and_optimize_dax(result.get('dax_code', ''))
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "dax_code": None,
                "explanation": "Error generating DAX code",
                "optimization_tips": [],
                "suggested_visuals": []
            }

    def _build_enhanced_prompt(self, request: str, context: Dict) -> str:
        """Build enhanced prompt for better DAX generation"""
        return f"""Generate an optimized DAX measure for: {request}

        Dataset Context:
        - Columns: {context['columns']}
        - Data Types: {context['data_types']}
        
        Statistics:
        - Numerical Columns: {json.dumps(context['statistics']['numerical'], indent=2)}
        - Categorical Columns: {json.dumps(context['statistics']['categorical'], indent=2)}
        - Temporal Columns: {json.dumps(context['statistics']['temporal'], indent=2)}
        
        Hierarchies:
        {json.dumps(context.get('hierarchies', []), indent=2)}
        
        Data Patterns:
        {json.dumps(context.get('data_patterns', {}), indent=2)}
        
        Sample Data:
        {json.dumps(context['sample_data'], indent=2)}
        
        Please provide:
        1. Production-ready, optimized DAX code
        2. Detailed explanation of the measure
        3. Performance optimization tips
        4. Error handling considerations
        5. Appropriate visualization suggestions
        6. Performance impact analysis"""

    def _validate_and_optimize_dax(self, dax_code: str) -> str:
        """Validate and optimize DAX code"""
        if not dax_code:
            return "// Error: Empty DAX code"
        
        dax_code = dax_code.strip()
        
        # Add header comments
        if not dax_code.startswith("//"):
            dax_code = f"""// Auto-generated DAX measure
// Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
// Performance Impact: Review carefully for large datasets

{dax_code}"""
        
        return dax_code

    def suggest_common_measures(self, context: Dict) -> List[Dict]:
        """Suggest common DAX measures based on context"""
        try:
            suggestions = []
            
            # Basic measures for numerical columns
            for col in context['statistics']['numerical'].get('columns', []):
                measures = self._get_numerical_measures(col)
                for measure in measures:
                    measure.setdefault('name', f"Unnamed Measure for {col}")  # Set default name if missing
                    suggestions.append(measure)
            
            # Time intelligence measures
            if context['statistics']['temporal']:
                time_measures = self._get_time_intelligence_measures()
                for measure in time_measures:
                    measure.setdefault('name', "Unnamed Time Intelligence Measure")
                    suggestions.append(measure)
            
            # KPI measures
            kpi_measures = self._get_kpi_measures(context)
            for measure in kpi_measures:
                measure.setdefault('name', "Unnamed KPI Measure")
                suggestions.append(measure)
            
            return suggestions
        except Exception as e:
            return [{"error": str(e)}]

    def _get_numerical_measures(self, column: str) -> List[Dict]:
        """Get common numerical measures"""
        return [
            {
                "name": f"Total_{column}",
                "description": f"Total sum of {column}",
                "complexity": "Simple",
                "dax": f"Total_{column} = SUM({column})"
            },
            {
                "name": f"Avg_{column}",
                "description": f"Average of {column}",
                "complexity": "Simple",
                "dax": f"Avg_{column} = AVERAGE({column})"
            },
            {
                "name": f"YoY_Growth_{column}",
                "description": f"Year-over-year growth of {column}",
                "complexity": "Advanced",
                "dax": f"YoY_Growth_{column} = CALCULATE(SUM({column}))/CALCULATE(SUM({column}), SAMEPERIODLASTYEAR('Date'[Date])) - 1"
            }
        ]

    def _get_time_intelligence_measures(self) -> List[Dict]:
        """Get common time intelligence measures"""
        return [
            {
                "name": "MTD_Sales",
                "description": "Month-to-date sales",
                "complexity": "Intermediate",
                "dax": "MTD_Sales = CALCULATE(SUM(Sales), DATESMTD('Date'[Date]))"
            },
            {
                "name": "Rolling_12M",
                "description": "Rolling 12-month total",
                "complexity": "Advanced",
                "dax": "Rolling_12M = CALCULATE(SUM(Sales), DATESINPERIOD('Date'[Date], MAX('Date'[Date]), -12, MONTH))"
            }
        ]

    def _get_kpi_measures(self, context: Dict) -> List[Dict]:
        """Get common KPI measures"""
        return [
            {
                "name": "Sales_vs_Target",
                "description": "Sales performance against target",
                "complexity": "Advanced",
                "dax": "Sales_vs_Target = DIVIDE(SUM(Sales), SUM(Target), 0)"
            },
            {
                "name": "Profit_Margin",
                "description": "Profit margin percentage",
                "complexity": "Intermediate",
                "dax": "Profit_Margin = DIVIDE(SUM(Profit), SUM(Sales), 0)"
            }
        ]
