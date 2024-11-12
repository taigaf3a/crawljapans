import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from utils.data_processor import DataProcessor
from utils.visualizations import Visualizer
import io

# Page config
st.set_page_config(
    page_title="Crawler Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

def main():
    st.title("🕷️ Crawler Data Analyzer")
    st.markdown("""
    Upload your crawler data to analyze patterns, frequencies, and visualize crawling behavior.
    """)

    # File upload section
    uploaded_file = st.file_uploader("Upload Crawler Data (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            # Read and process data
            data_processor = DataProcessor()
            df = data_processor.load_data(uploaded_file)
            st.session_state.data = df
            
            # Display basic statistics
            st.header("📊 Overview Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total URLs", len(df['url'].unique()))
            with col2:
                st.metric("Total Crawls", len(df))
            with col3:
                st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Crawl Analysis", "🗺️ Heat Maps", "📑 Detailed Data"])
            
            visualizer = Visualizer()
            
            with tab1:
                st.subheader("Daily Crawl Frequency")
                daily_crawls = visualizer.plot_daily_crawls(df)
                st.plotly_chart(daily_crawls, use_container_width=True)
                
                st.subheader("Monthly Crawl Distribution")
                monthly_crawls = visualizer.plot_monthly_crawls(df)
                st.plotly_chart(monthly_crawls, use_container_width=True)

            with tab2:
                st.subheader("Crawl Frequency Heat Map")
                heatmap = visualizer.create_heatmap(df)
                st.plotly_chart(heatmap, use_container_width=True)

            with tab3:
                st.subheader("Detailed Crawl Data")
                # Configure AG-Grid
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_column("url", filter=True)
                gb.configure_column("date", filter=True)
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True)
                
                grid_options = gb.build()
                AgGrid(
                    df,
                    gridOptions=grid_options,
                    enable_enterprise_modules=True,
                    allow_unsafe_jscode=True,
                    theme='streamlit'
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
