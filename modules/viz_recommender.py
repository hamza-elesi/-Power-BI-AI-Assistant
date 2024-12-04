# viz_recommender.py
import openai
import pandas as pd
from typing import Dict, List
import json

class VisualizationRecommender:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key

    def analyze_data_for_viz(self, df: pd.DataFrame) -> Dict:
        """Analyze data for visualization recommendations."""
        analysis = {
            "column_types": df.dtypes.astype(str).to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "numerical_cols": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object']).columns.tolist(),
            "date_cols": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "row_count": len(df),
            "sample_data": df.head(3).to_dict(orient='records')
        }
        return analysis

    async def get_visualization_recommendations(self, data_analysis: Dict) -> str:
        """Get visualization recommendations in a readable format via OpenAI."""
        prompt = f"""As a Power BI visualization expert, suggest the best visualizations for this dataset:

        Data Structure:
        - Numerical columns: {data_analysis['numerical_cols']}
        - Categorical columns: {data_analysis['categorical_cols']}
        - Date columns: {data_analysis['date_cols']}
        - Row count: {data_analysis['row_count']}
        
        Sample data:
        {json.dumps(data_analysis['sample_data'], indent=2)}
        
        For each suggested visualization, provide:
        1. Visualization type
        2. Columns to use
        3. Recommended configuration
        4. Expected business insight
        5. Recommended interactions

        Respond with well-formatted text suitable for direct presentation."""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Power BI data visualization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Return response content directly as formatted text
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error: {str(e)}"

    async def suggest_dashboard_layout(self, visualizations: str) -> str:
        """Suggest dashboard layout in a readable format."""
        prompt = f"""Based on the following visualizations, propose a dashboard layout:

        {visualizations}
        
        Consider:
        1. KPIs relative importance
        2. Visual hierarchy
        3. Logical analysis flow
        4. Visual interactions
        
        Respond with a well-formatted layout plan suitable for direct presentation."""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Power BI dashboard design expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Return response content directly as formatted text
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
