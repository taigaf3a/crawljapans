# Keeping the same imports and initial setup
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from utils.data_processor import DataProcessor
from utils.visualizations import Visualizer
import io
from datetime import timedelta, datetime

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

    # Updated file upload section to include .txt files
    uploaded_files = st.file_uploader(
        "Upload Log Files",
        type=['txt', 'log', 'gz'],
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
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Crawl Analysis", 
                "📊 Statistical Insights", 
                "🗺️ Heat Maps", 
                "🔄 Period Comparison",
                "📑 Detailed Data",
                "🎨 Visualization Presets"
            ])
            
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
                
                # Enhanced URL Distribution Analysis
                st.write("### URL Distribution Analysis")
                
                # Date range filter
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=combined_df['date'].min().date(),
                        min_value=combined_df['date'].min().date(),
                        max_value=combined_df['date'].max().date()
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=combined_df['date'].max().date(),
                        min_value=start_date,
                        max_value=combined_df['date'].max().date()
                    )

                # Sorting and filtering options
                col1, col2, col3 = st.columns(3)
                with col1:
                    sort_by = st.selectbox(
                        "Sort URLs by",
                        ["total_crawls", "months_active", "latest_crawl", "avg_daily_crawls", "success_rate"],
                        index=0
                    )
                with col2:
                    sort_order = st.radio("Sort Order", ["Descending", "Ascending"])
                with col3:
                    display_count = st.number_input("Number of URLs to display", min_value=5, max_value=50, value=10)

                # Get URL patterns with filters
                url_patterns = data_processor.get_url_patterns(
                    combined_df,
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                    sort_by,
                    sort_order == "Ascending"
                )

                # Display URL distribution chart
                url_dist_plot = visualizer.plot_url_distribution(url_patterns, sort_by, display_count)
                st.plotly_chart(url_dist_plot, use_container_width=True)

                # Display detailed URL data
                st.write("### Detailed URL Analysis")
                st.dataframe(
                    url_patterns.style.format({
                        'total_crawls': '{:,.0f}',
                        'avg_daily_crawls': '{:.2f}',
                        'success_rate': '{:.1f}%',
                        'first_crawl': lambda x: x.strftime('%Y-%m-%d'),
                        'latest_crawl': lambda x: x.strftime('%Y-%m-%d')
                    }),
                    height=400
                )

                # Download URL analysis data
                if st.button("Download URL Analysis Data"):
                    csv = url_patterns.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="url_analysis.csv",
                        mime="text/csv"
                    )

                # Peak Hours
                st.write("### Peak Crawling Hours")
                st.write(f"Top 3 peak hours: {', '.join(map(str, stats_results['peak_hours']))}")

            with tab3:
                st.subheader("Crawl Frequency Heat Map")
                heatmap = visualizer.create_heatmap(combined_df)
                st.plotly_chart(heatmap, use_container_width=True)

            with tab4:
                st.subheader("Compare Time Periods")
                
                # Date range selection for both periods
                col1, col2 = st.columns(2)
                
                min_date = combined_df['date'].min().date()
                max_date = combined_df['date'].max().date()
                default_end_date1 = min(max_date, min_date + timedelta(days=30))
                
                with col1:
                    st.write("#### Period 1")
                    start_date1 = st.date_input(
                        "Start Date (Period 1)",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    end_date1 = st.date_input(
                        "End Date (Period 1)",
                        value=default_end_date1,
                        min_value=start_date1,
                        max_value=max_date
                    )

                default_start_date2 = min(max_date, end_date1 + timedelta(days=1))
                default_end_date2 = min(max_date, default_start_date2 + timedelta(days=30))
                
                with col2:
                    st.write("#### Period 2")
                    start_date2 = st.date_input(
                        "Start Date (Period 2)",
                        value=default_start_date2,
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    end_date2 = st.date_input(
                        "End Date (Period 2)",
                        value=default_end_date2,
                        min_value=start_date2,
                        max_value=max_date
                    )

                if st.button("Compare Periods"):
                    # Convert dates to pandas Timestamps
                    start_ts1 = pd.Timestamp(datetime.combine(start_date1, datetime.min.time()))
                    end_ts1 = pd.Timestamp(datetime.combine(end_date1, datetime.max.time()))
                    start_ts2 = pd.Timestamp(datetime.combine(start_date2, datetime.min.time()))
                    end_ts2 = pd.Timestamp(datetime.combine(end_date2, datetime.max.time()))
                    
                    metrics, period1_df, period2_df = data_processor.compare_time_periods(
                        combined_df,
                        start_ts1,
                        end_ts1,
                        start_ts2,
                        end_ts2
                    )
                    
                    # Display comparison metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### Period 1 Metrics")
                        st.metric("Total Crawls", metrics['period1']['total_crawls'])
                        st.metric("Unique URLs", metrics['period1']['unique_urls'])
                        st.metric("Avg Daily Crawls", f"{metrics['period1']['avg_daily_crawls']:.2f}")
                        st.write(f"Peak Hours: {', '.join(map(str, metrics['period1']['peak_hours']))}")
                    
                    with col2:
                        st.write("#### Period 2 Metrics")
                        st.metric("Total Crawls", metrics['period2']['total_crawls'])
                        st.metric("Unique URLs", metrics['period2']['unique_urls'])
                        st.metric("Avg Daily Crawls", f"{metrics['period2']['avg_daily_crawls']:.2f}")
                        st.write(f"Peak Hours: {', '.join(map(str, metrics['period2']['peak_hours']))}")
                    
                    # Visualize comparisons
                    st.write("### Comparative Analysis")
                    
                    # Total crawls comparison
                    total_crawls_fig = visualizer.plot_period_comparison(
                        metrics['period1']['total_crawls'],
                        metrics['period2']['total_crawls'],
                        'total_crawls',
                        'Total Crawls Comparison'
                    )
                    st.plotly_chart(total_crawls_fig, use_container_width=True)
                    
                    # Hourly distribution comparison
                    hourly_comp_fig = visualizer.plot_hourly_comparison(
                        metrics['period1']['hourly_distribution'],
                        metrics['period2']['hourly_distribution']
                    )
                    st.plotly_chart(hourly_comp_fig, use_container_width=True)
                    
                    # Display daily patterns
                    st.write("### Daily Patterns")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### Period 1")
                        daily_pattern1 = pd.DataFrame(
                            metrics['period1']['daily_pattern'].items(),
                            columns=['Day', 'Crawls']
                        ).set_index('Day')
                        st.dataframe(daily_pattern1)
                    
                    with col2:
                        st.write("#### Period 2")
                        daily_pattern2 = pd.DataFrame(
                            metrics['period2']['daily_pattern'].items(),
                            columns=['Day', 'Crawls']
                        ).set_index('Day')
                        st.dataframe(daily_pattern2)

            with tab5:
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

            with tab6:
                st.subheader("📊 Visualization Presets")
                st.markdown("""
                Choose from predefined visualization combinations to quickly analyze specific aspects of your crawler data.
                """)
                
                # Preset selection
                preset_name = st.selectbox(
                    "Select a Visualization Preset",
                    options=list(visualizer.presets.keys()),
                    format_func=lambda x: visualizer.presets[x].name
                )
                
                # Display preset description
                st.info(visualizer.get_preset_description(preset_name))
                
                # Get charts for the selected preset
                charts = visualizer.get_preset_charts(preset_name)
                
                # Display charts based on the preset
                for chart_type in charts:
                    if chart_type == 'daily_crawls':
                        st.subheader("Daily Crawl Frequency")
                        fig = visualizer.plot_daily_crawls(combined_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'monthly_crawls':
                        st.subheader("Monthly Crawl Distribution")
                        fig = visualizer.plot_monthly_crawls(combined_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'heatmap':
                        st.subheader("Crawl Frequency Heat Map")
                        fig = visualizer.create_heatmap(combined_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'time_series_decomposition':
                        st.subheader("Time Series Decomposition")
                        stats_results = data_processor.perform_statistical_analysis(combined_df)
                        fig = visualizer.plot_time_series_decomposition(
                            stats_results['trend'],
                            stats_results['seasonal'],
                            stats_results['residual']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'url_distribution':
                        st.subheader("URL Distribution")
                        stats_results = data_processor.perform_statistical_analysis(combined_df)
                        fig = visualizer.plot_url_distribution(stats_results['url_diversity']['top_urls'])
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file(s): {str(e)}")

if __name__ == "__main__":
    main()