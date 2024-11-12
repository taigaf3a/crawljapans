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
    uploaded_files = st.file_uploader(
        "Upload Crawler Data Files (CSV or GZ)",
        type=['csv', 'gz'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        try:
            # Initialize progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize empty DataFrame
            combined_df = pd.DataFrame()
            data_processor = DataProcessor()
            
            # Process each uploaded file
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing file {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Read and process data
                df = data_processor.load_data(uploaded_file)
                # Concatenate with existing data
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
            
            # Sort by date for consistency
            combined_df = combined_df.sort_values('date')
            st.session_state.data = combined_df
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.success(f"Successfully processed {len(uploaded_files)} file(s)")
            
            # Export section
            st.sidebar.header("📥 Export Data")
            export_format = st.sidebar.selectbox(
                "Choose export format",
                ["CSV", "Excel", "GZ (Compressed)"]
            )
            
            # Export buttons for different data views
            export_options = {
                "Raw Data": combined_df,
                "Daily Crawl Frequency": data_processor.calculate_crawl_frequency(combined_df),
                "Monthly Statistics": data_processor.calculate_monthly_stats(combined_df),
                "URL Patterns": data_processor.get_url_patterns(combined_df)
            }
            
            export_dataset = st.sidebar.selectbox(
                "Choose dataset to export",
                list(export_options.keys())
            )
            
            if st.sidebar.button("Export Data"):
                export_data = export_options[export_dataset]
                if export_format == "CSV":
                    data = data_processor.export_data(export_data, 'csv')
                    st.sidebar.download_button(
                        label="Download CSV",
                        data=data,
                        file_name=f'crawler_data_{export_dataset.lower().replace(" ", "_")}.csv',
                        mime='text/csv'
                    )
                elif export_format == "Excel":
                    data = data_processor.export_data(export_data, 'excel')
                    st.sidebar.download_button(
                        label="Download Excel",
                        data=data,
                        file_name=f'crawler_data_{export_dataset.lower().replace(" ", "_")}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                else:  # GZ (Compressed)
                    data = data_processor.export_data(export_data, 'gz')
                    st.sidebar.download_button(
                        label="Download GZ",
                        data=data,
                        file_name=f'crawler_data_{export_dataset.lower().replace(" ", "_")}.csv.gz',
                        mime='application/gzip'
                    )
            
            # Display basic statistics
            st.header("📊 Overview Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total URLs", len(combined_df['url'].unique()))
            with col2:
                st.metric("Total Crawls", len(combined_df))
            with col3:
                st.metric("Date Range", f"{combined_df['date'].min().date()} to {combined_df['date'].max().date()}")

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Crawl Analysis", "📊 Statistical Insights", "🗺️ Heat Maps", "📑 Detailed Data"])
            
            visualizer = Visualizer()
            
            with tab1:
                st.subheader("Daily Crawl Frequency")
                daily_crawls = visualizer.plot_daily_crawls(combined_df)
                st.plotly_chart(daily_crawls, use_container_width=True)
                
                st.subheader("Monthly Crawl Distribution")
                monthly_crawls = visualizer.plot_monthly_crawls(combined_df)
                st.plotly_chart(monthly_crawls, use_container_width=True)

            with tab2:
                st.subheader("Advanced Statistical Analysis")
                stats_results = data_processor.perform_statistical_analysis(combined_df)
                
                # Basic Statistics
                st.write("### Daily Crawl Statistics")
                basic_stats = stats_results['basic_stats']
                cols = st.columns(5)
                cols[0].metric("Mean Daily Crawls", f"{basic_stats['mean_daily_crawls']:.2f}")
                cols[1].metric("Median Daily Crawls", f"{basic_stats['median_daily_crawls']:.2f}")
                cols[2].metric("Std Dev", f"{basic_stats['std_daily_crawls']:.2f}")
                cols[3].metric("Skewness", f"{basic_stats['skewness']:.2f}")
                cols[4].metric("Kurtosis", f"{basic_stats['kurtosis']:.2f}")
                
                # Time Series Decomposition
                st.write("### Time Series Decomposition")
                decomp_plot = visualizer.plot_time_series_decomposition(
                    stats_results['trend'],
                    stats_results['seasonal'],
                    stats_results['residual']
                )
                st.plotly_chart(decomp_plot, use_container_width=True)
                
                # URL Distribution
                st.write("### URL Distribution Analysis")
                url_div = stats_results['url_diversity']
                cols = st.columns(2)
                cols[0].metric("Unique URLs", url_div['unique_urls'])
                cols[1].metric("Gini Coefficient", f"{url_div['gini_coefficient']:.3f}")
                
                url_dist_plot = visualizer.plot_url_distribution(url_div['top_urls'])
                st.plotly_chart(url_dist_plot, use_container_width=True)
                
                # Peak Hours
                st.write("### Peak Crawling Hours")
                st.write(f"Top 3 peak hours: {', '.join(map(str, stats_results['peak_hours']))}")

            with tab3:
                st.subheader("Crawl Frequency Heat Map")
                heatmap = visualizer.create_heatmap(combined_df)
                st.plotly_chart(heatmap, use_container_width=True)

            with tab4:
                st.subheader("Detailed Crawl Data")
                # Configure AG-Grid
                gb = GridOptionsBuilder.from_dataframe(combined_df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_column("url", filter=True)
                gb.configure_column("date", filter=True)
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True)
                
                grid_options = gb.build()
                AgGrid(
                    combined_df,
                    gridOptions=grid_options,
                    enable_enterprise_modules=True,
                    allow_unsafe_jscode=True,
                    theme='streamlit'
                )

        except Exception as e:
            st.error(f"Error processing file(s): {str(e)}")

if __name__ == "__main__":
    main()
