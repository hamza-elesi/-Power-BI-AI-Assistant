# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from modules.data_quality import DataQualityAnalyzer
from modules.dax_generator import DAXGenerator
import os
import asyncio
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="BI Assistant Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize session state
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'dw_structure' not in st.session_state:
    st.session_state.dw_structure = None
if 'dax_suggestions' not in st.session_state:
    st.session_state.dax_suggestions = None
if 'dataset_analysis' not in st.session_state:
    st.session_state.dataset_analysis = None

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dax-code {
        background-color: #272822;
        color: #f8f8f2;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar setup
with st.sidebar:
    st.image("Power-BI.png", caption="BI Assistant Pro")
    st.markdown("---")
    
    # API Key Status
    if API_KEY:
        st.success("✅ API Key configured")
    else:
        st.error("❌ API Key not found")
        st.warning("Please set OPENAI_API_KEY in your .env file")
    
    # Navigation
    st.header("Navigation")
    tool_choice = st.radio(
        "Choose Tool:",
        ["Data Quality Analysis", "DAX Generator", "Data Warehouse Designer"]
    )

# Main Area
st.title("🎯 BI Assistant Pro")

# File Upload Section
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx"],
    help="Upload your dataset to analyze and generate DAX measures"
)

if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df
        
        # Data Preview
        with st.expander("📊 Data Preview", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            st.dataframe(df.head())
        
        # Tool-specific functionality
        if tool_choice == "Data Quality Analysis":
            st.header("📈 Data Quality Analysis")
            
            if st.button("Analyze Data Quality"):
                with st.spinner("Analyzing data quality..."):
                    dq_analyzer = DataQualityAnalyzer(api_key=API_KEY)
                    quality_metrics = dq_analyzer.analyze_data_quality(df)
                    st.session_state.quality_metrics = quality_metrics
            
            if st.session_state.quality_metrics:
                metrics = st.session_state.quality_metrics
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Completeness", "Uniqueness", "Consistency", "Statistics"
                ])
                
                with tab1:
                    completeness = metrics['completeness']
                    completion_df = pd.DataFrame({
                        'Column': list(completeness['completion_rate'].keys()),
                        'Rate': list(completeness['completion_rate'].values())
                    })
                    fig = px.bar(
                        completion_df,
                        x='Column',
                        y='Rate',
                        title='Data Completeness',
                        color='Rate',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    uniqueness = metrics['uniqueness']
                    st.write("#### Unique Value Analysis")
                    st.json(uniqueness)
                
                with tab3:
                    consistency = metrics['consistency']
                    for col, details in consistency.items():
                        with st.expander(f"Column: {col}"):
                            st.json(details)
                
                with tab4:
                    statistics = metrics['statistics']
                    for col, stats in statistics.items():
                        with st.expander(f"Statistics for {col}"):
                            st.json(stats)
        
        elif tool_choice == "DAX Generator":
            st.header("🎯 DAX Measure Generator")
            
            # Initialize DAX Generator
            dax_gen = DAXGenerator(api_key=API_KEY)
            
            # Analyze dataset if not already done
            if not st.session_state.dataset_analysis:
                with st.spinner("Analyzing dataset structure..."):
                    st.session_state.dataset_analysis = dax_gen.analyze_dataset(df)
                    st.session_state.dax_suggestions = dax_gen.suggest_common_measures(
                        st.session_state.dataset_analysis
                    )
            
            # DAX Generation Interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Generate Custom DAX")
                dax_request = st.text_area(
                    "Describe the measure you want to create",
                    placeholder="E.g., Calculate total sales by region with year-over-year growth",
                    height=100
                )
                
                if st.button("Generate DAX") and dax_request:
                    with st.spinner("Generating optimized DAX measure..."):
                        result = asyncio.run(dax_gen.generate_dax_query(
                            dax_request,
                            st.session_state.dataset_analysis
                        ))
                        
                        if "error" not in result:
                            st.code(result['dax_code'], language="sql")
                            
                            with st.expander("Explanation"):
                                st.write(result['explanation'])
                            
                            with st.expander("Optimization Tips"):
                                for tip in result['optimization_tips']:
                                    st.info(tip)
                            
                            with st.expander("Suggested Visualizations"):
                                for viz in result['suggested_visuals']:
                                    st.write(f"- {viz}")
                        else:
                            st.error(f"Error: {result['error']}")
            
                with col2:
                    st.subheader("Suggested Measures")
                    if st.session_state.dax_suggestions:
                        for suggestion in st.session_state.dax_suggestions:
                            suggestion_name = suggestion.get('name', "Unnamed Measure")
                            with st.expander(f"📊 {suggestion_name}"):
                                st.write(f"**Description:** {suggestion.get('description', 'No description available')}")
                                st.write(f"**Complexity:** {suggestion.get('complexity', 'Not specified')}")
                                if 'dax' in suggestion:
                                    st.code(suggestion['dax'], language="sql")
                    else:
                        st.info("No measure suggestions available for this dataset")

        
        elif tool_choice == "Data Warehouse Designer":
            st.header("🏗️ Data Warehouse Designer")
            
            try:
                if st.button("Generate Data Warehouse Structure"):
                    with st.spinner("Analyzing and generating structure..."):
                        dq_analyzer = DataQualityAnalyzer(api_key=API_KEY)
                        quality_metrics = dq_analyzer.analyze_data_quality(df)
                        structure = asyncio.run(
                            dq_analyzer.suggest_data_warehouse_structure(df, quality_metrics)
                        )
                        
                        if 'error' not in structure:
                            st.markdown(structure['content'])
                        else:
                            st.error(f"Error: {structure['error']}")
            except Exception as e:
                st.error(f"Error in Data Warehouse Designer: {str(e)}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    # Welcome message and instructions
    st.info("👋 Welcome to BI Assistant Pro!")
    
    st.markdown("""
    ### Features:
    - 📊 **Data Quality Analysis**: Comprehensive data quality metrics and visualizations
    - 🎯 **DAX Generator**: AI-powered DAX measure generation and optimization
    - 🏗️ **Data Warehouse Designer**: Smart data warehouse structure suggestions
    
    ### How to start:
    1. Upload your CSV or Excel file using the uploader above
    2. Choose your desired tool from the sidebar
    3. Follow the tool-specific instructions
    
    ### Tips:
    - Ensure your data is properly formatted
    - Review the data preview before analysis
    - Check the suggested measures for quick insights
    """)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit | "
    "Power BI Assistant Pro v1.0"
)
