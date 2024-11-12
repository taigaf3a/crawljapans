import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import io
import gzip
import re
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

class DataProcessor:
    @st.cache_data
    def parse_log_line(_self, line):
        """Parse a single log line with caching for repeated patterns."""
        patterns = [
            # Apache/Nginx format with domain prefix
            r'(?:\S+\s+)?(?P<ip>[\d.]+)\s+[-\w]+\s+[-\w]+\s+\[(?P<datetime>[^\]]+)\]\s+"(?P<method>\w+)\s+(?P<url>[^\s"]+)[^"]*"\s+(?P<status>\d+)\s+(?P<bytes>[-\d]+)\s+"([^"]*)"\s+"(?P<useragent>[^"]*)"',
            # Alternative format with domain prefix
            r'(?:\S+\s+)?(?P<ip>[\d.]+)\s+\[(?P<datetime>[^\]]+)\]\s+"(?P<method>\w+)\s+(?P<url>[^\s"]+)[^"]*"\s+(?P<status>\d+)\s+(?P<bytes>[-\d]+)\s+"([^"]*)"\s+"(?P<useragent>[^"]*)"',
            # Basic format with domain prefix
            r'(?:\S+\s+)?(?P<ip>[\d.]+)\s+\S+\s+\S+\s+\[(?P<datetime>[^\]]+)\]\s+"[^"]+"\s+(?P<status>\d+)\s+(?P<bytes>[-\d]+)\s+"[^"]*"\s+"(?P<useragent>[^"]*)"'
        ]

        try:
            # Try each pattern
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    data = match.groupdict()
                    if re.search(r'googlebot', data['useragent'], re.IGNORECASE):
                        # Parse datetime
                        try:
                            dt = datetime.strptime(data['datetime'], '%d/%b/%Y:%H:%M:%S %z')
                        except ValueError:
                            try:
                                dt = datetime.strptime(data['datetime'], '%d/%b/%Y:%H:%M:%S')
                            except ValueError:
                                return None

                        return {
                            'url': data['url'],
                            'date': dt.date(),
                            'time': dt.strftime('%H:%M:%S'),
                            'status': data['status'],
                            'user_agent': data['useragent']
                        }
        except Exception:
            return None
        return None

    @st.cache_data
    def load_data(_self, file):
        """Load and process crawler data from access logs (.log, .txt, or .gz files)."""
        try:
            # Reset file position
            file.seek(0)
            content = file.read()
            
            # Handle text content
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            
            # Split into lines and process
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            total_lines = len(lines)
            
            if total_lines == 0:
                raise ValueError(f"File {file.name} appears to be empty")
            
            parsed_data = []
            invalid_lines = 0
            googlebot_entries = 0
            
            for line in lines:
                data = _self.parse_log_line(line)
                if data:
                    parsed_data.append(data)
                    googlebot_entries += 1
                else:
                    invalid_lines += 1
            
            if not parsed_data and total_lines > 0:
                sample_lines = lines[:5]  # First 5 lines
                error_msg = f"No Googlebot entries found in {file.name}. \n"
                error_msg += f"Total lines: {total_lines}, Invalid lines: {invalid_lines}\n"
                error_msg += "Sample of first few lines:\n"
                error_msg += "\n".join(sample_lines)
                raise ValueError(error_msg)
            
            st.info(f"Processed {total_lines} lines, found {googlebot_entries} Googlebot entries")
            df = pd.DataFrame(parsed_data)
            
            # Convert date column to datetime and add derived columns
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.strftime('%Y-%m')
            df['day_of_week'] = df['date'].dt.day_name()
            df['hour'] = df['date'].dt.hour
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing file {file.name}: {str(e)}")

    @st.cache_data
    def calculate_crawl_frequency(_self, df):
        """Calculate crawl frequency with proper data type handling and caching."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Group by date and URL, ensuring float64 type for counts
        daily_counts = (df.groupby(['date', 'url'])
                       .size()
                       .reset_index(name='crawl_count')
                       .astype({'crawl_count': 'float64'}))
        
        # Sort by date for consistency in caching
        daily_counts = daily_counts.sort_values('date')
        
        return daily_counts

    @st.cache_data
    def calculate_monthly_stats(_self, df):
        """Calculate monthly crawl statistics."""
        monthly_stats = df.groupby('month').agg({
            'url': 'count',
            'date': 'nunique'
        }).reset_index()
        monthly_stats.columns = ['month', 'total_crawls', 'unique_days']
        return monthly_stats

    @st.cache_data
    def get_url_patterns(_self, df, start_date=None, end_date=None, sort_by='total_crawls', ascending=False):
        """
        Analyze URL patterns and frequency with filtering and sorting options.
        
        Parameters:
        - df: DataFrame containing the crawl data
        - start_date: Optional start date for filtering
        - end_date: Optional end date for filtering
        - sort_by: Column to sort by ('total_crawls', 'months_active', 'latest_crawl')
        - ascending: Sort order
        """
        # Apply date filtering if provided
        if start_date is not None:
            df = df[df['date'] >= start_date]
        if end_date is not None:
            df = df[df['date'] <= end_date]

        # Group by URL and calculate metrics
        url_patterns = df.groupby('url').agg({
            'date': ['count', 'min', 'max', 'nunique'],
            'month': 'nunique',
            'status': lambda x: (x == 200).mean() * 100  # Success rate in percentage
        }).reset_index()

        # Rename columns
        url_patterns.columns = [
            'url', 'total_crawls', 'first_crawl', 'latest_crawl',
            'unique_days', 'months_active', 'success_rate'
        ]

        # Add average daily crawls
        url_patterns['avg_daily_crawls'] = url_patterns['total_crawls'] / url_patterns['unique_days']

        # Sort based on specified column
        if sort_by in url_patterns.columns:
            url_patterns = url_patterns.sort_values(sort_by, ascending=ascending)

        return url_patterns

    @st.cache_data
    def perform_statistical_analysis(_self, df):
        """Perform advanced statistical analysis on crawl data."""
        # Convert daily_series to float64 explicitly
        daily_series = df.groupby('date').size().astype('float64')
        
        basic_stats = {
            'mean_daily_crawls': float(daily_series.mean()),
            'median_daily_crawls': float(daily_series.median()),
            'std_daily_crawls': float(daily_series.std()),
            'skewness': float(daily_series.skew()),
            'kurtosis': float(daily_series.kurtosis())
        }

        trend = pd.Series(dtype='float64')
        seasonal = pd.Series(dtype='float64')
        residual = pd.Series(dtype='float64')

        if len(daily_series) >= 14:  # Minimum length for decomposition
            try:
                daily_series = daily_series.asfreq('D', fill_value=0.0)  # Use float
                decomposition = seasonal_decompose(daily_series, period=7)
                trend = decomposition.trend.astype('float64')
                seasonal = decomposition.seasonal.astype('float64')
                residual = decomposition.resid.astype('float64')
            except Exception:
                pass

        # Convert hourly stats to float64
        hourly_stats = df.groupby('hour').size().astype('float64')
        peak_hours = hourly_stats.nlargest(3).index.tolist()
        
        # Ensure URL counts are float64
        url_counts = df.groupby('url').size().astype('float64')
        url_diversity = {
            'unique_urls': float(len(url_counts)),
            'gini_coefficient': float(_self._calculate_gini(url_counts.values)),
            'top_urls': {k: float(v) for k, v in url_counts.nlargest(5).to_dict().items()}
        }

        return {
            'basic_stats': basic_stats,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'peak_hours': peak_hours,
            'url_diversity': url_diversity
        }

    @st.cache_data
    def _calculate_gini(_self, array):
        """Calculate the Gini coefficient of inequality."""
        array = np.array(array)
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    @st.cache_data
    def export_data(_self, df, export_type='csv'):
        """Export data to different formats."""
        if export_type == 'csv':
            output = io.BytesIO()
            df.to_csv(output, index=False)
            return output.getvalue()
        elif export_type == 'excel':
            output = io.BytesIO()
            df.to_excel(output, index=False)
            return output.getvalue()
        elif export_type == 'gz':
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            return gzip.compress(csv_str.encode('utf-8'))
        else:
            raise ValueError("Unsupported export format")

    @st.cache_data
    def compare_time_periods(_self, df, start_date1, end_date1, start_date2, end_date2):
        """Compare Googlebot crawl data between two time periods."""
        period1 = df[(df['date'] >= start_date1) & (df['date'] <= end_date1)]
        period2 = df[(df['date'] >= start_date2) & (df['date'] <= end_date2)]
        
        metrics = {
            'period1': {
                'total_crawls': len(period1),
                'unique_urls': len(period1['url'].unique()),
                'avg_daily_crawls': period1.groupby('date').size().mean(),
                'peak_hours': period1.groupby('hour').size().nlargest(3).index.tolist(),
                'top_urls': period1['url'].value_counts().nlargest(5).to_dict(),
                'hourly_distribution': period1.groupby('hour').size().to_dict(),
                'daily_pattern': period1.groupby('day_of_week').size().to_dict(),
                'status_codes': period1['status'].value_counts().to_dict(),
                'googlebot_variants': period1['user_agent'].value_counts().to_dict()
            },
            'period2': {
                'total_crawls': len(period2),
                'unique_urls': len(period2['url'].unique()),
                'avg_daily_crawls': period2.groupby('date').size().mean(),
                'peak_hours': period2.groupby('hour').size().nlargest(3).index.tolist(),
                'top_urls': period2['url'].value_counts().nlargest(5).to_dict(),
                'hourly_distribution': period2.groupby('hour').size().to_dict(),
                'daily_pattern': period2.groupby('day_of_week').size().to_dict(),
                'status_codes': period2['status'].value_counts().to_dict(),
                'googlebot_variants': period2['user_agent'].value_counts().to_dict()
            }
        }
        
        return metrics, period1, period2

class Visualizer:
    def visualize_url_distribution(_self, url_patterns):
        """Visualize URL distribution."""
        st.subheader("URL Distribution Analysis")
        st.write("**Note:** Click on the column headers to sort the table.")
        st.dataframe(url_patterns)
        
        # Create a chart showing top 10 URLs
        top_urls = url_patterns.sort_values('total_crawls', ascending=False).head(10)
        st.subheader("Top 10 URLs")
        fig = px.bar(
            top_urls,
            x='url',
            y='total_crawls',
            title="Top 10 URLs by Crawl Count",
            labels={"url": "URL", "total_crawls": "Total Crawls"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a chart showing URL success rate
        st.subheader("URL Success Rate")
        fig = px.histogram(
            url_patterns,
            x='success_rate',
            title="Distribution of URL Success Rates",
            labels={"success_rate": "Success Rate (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create a chart showing URL crawl frequency over time
        st.subheader("URL Crawl Frequency over Time")
        fig = px.line(
            url_patterns,
            x='latest_crawl',
            y='total_crawls',
            title="Crawl Frequency of URLs over Time",
            labels={"latest_crawl": "Latest Crawl Date", "total_crawls": "Total Crawls"}
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Googlebot Crawl Data Analysis")
    st.sidebar.title("Data Options")
    uploaded_file = st.sidebar.file_uploader("Upload a log file (.log, .txt, .gz)", type=['log', 'txt', 'gz'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".gz"):
                file = gzip.open(uploaded_file, 'rt', encoding='utf-8')
            else:
                file = uploaded_file
            
            # Load data from the file
            data_processor = DataProcessor()
            df = data_processor.load_data(file)

            # Display tabs for different analysis types
            tab1, tab2, tab3 = st.tabs(['Statistical Insights', 'URL Distribution', 'Time Period Comparison'])

            with tab1:
                st.header("Statistical Insights")
                
                # Filter Data by Date
                start_date = st.date_input("Start Date", value=df['date'].min().date())
                end_date = st.date_input("End Date", value=df['date'].max().date())
                
                # Perform statistical analysis
                filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                stats = data_processor.perform_statistical_analysis(filtered_df)
                
                # Display basic statistics
                st.subheader("Basic Crawl Statistics")
                st.write(f"**Mean Daily Crawls:** {stats['basic_stats']['mean_daily_crawls']:.2f}")
                st.write(f"**Median Daily Crawls:** {stats['basic_stats']['median_daily_crawls']:.2f}")
                st.write(f"**Standard Deviation:** {stats['basic_stats']['std_daily_crawls']:.2f}")
                st.write(f"**Skewness:** {stats['basic_stats']['skewness']:.2f}")
                st.write(f"**Kurtosis:** {stats['basic_stats']['kurtosis']:.2f}")
                
                # Display trend, seasonal, and residual
                st.subheader("Time Series Decomposition")
                st.line_chart(stats['trend'], use_container_width=True)
                st.line_chart(stats['seasonal'], use_container_width=True)
                st.line_chart(stats['residual'], use_container_width=True)
                
                # Display peak hours
                st.subheader("Peak Crawl Hours")
                st.write(f"Peak hours: {stats['peak_hours']}")
                
                # Display URL diversity
                st.subheader("URL Diversity")
                st.write(f"**Unique URLs:** {stats['url_diversity']['unique_urls']}")
                st.write(f"**Gini Coefficient:** {stats['url_diversity']['gini_coefficient']:.2f}")
                st.write("**Top 5 URLs:**")
                for url, count in stats['url_diversity']['top_urls'].items():
                    st.write(f"- {url}: {count:.2f}")

            with tab2:
                st.header("URL Distribution Analysis")

                # Filter Data by Date
                start_date = st.date_input("Start Date (optional)", value=None, key="start_date_tab2")
                end_date = st.date_input("End Date (optional)", value=None, key="end_date_tab2")

                # Sort Options
                sort_by = st.selectbox("Sort by:", ['total_crawls', 'latest_crawl', 'months_active'], key="sort_by_tab2")
                ascending = st.radio("Sort Order:", ['ascending', 'descending'], index=1, key="ascending_tab2")
                if ascending == 'ascending':
                    ascending = True
                else:
                    ascending = False

                # Get and visualize URL patterns
                url_patterns = data_processor.get_url_patterns(df, start_date, end_date, sort_by, ascending)
                visualizer = Visualizer()
                visualizer.visualize_url_distribution(url_patterns)

            with tab3:
                st.header("Time Period Comparison")
                
                # Input for time periods
                start_date1 = st.date_input("Start Date for Period 1", value=df['date'].min().date())
                end_date1 = st.date_input("End Date for Period 1", value=df['date'].max().date())
                start_date2 = st.date_input("Start Date for Period 2", value=df['date'].min().date())
                end_date2 = st.date_input("End Date for Period 2", value=df['date'].max().date())
                
                # Compare time periods
                metrics, period1, period2 = data_processor.compare_time_periods(df, start_date1, end_date1, start_date2, end_date2)
                
                # Display comparison results
                st.subheader("Comparison Metrics")
                st.write(f"**Period 1:** {start_date1.strftime('%Y-%m-%d')} to {end_date1.strftime('%Y-%m-%d')}")
                st.write(f"**Period 2:** {start_date2.strftime('%Y-%m-%d')} to {end_date2.strftime('%Y-%m-%d')}")
                
                for metric in metrics.keys():
                    st.write(f"**{metric.upper()}**")
                    for key, value in metrics[metric].items():
                        if isinstance(value, dict):
                            st.write(f"   - {key}:")
                            for k, v in value.items():
                                st.write(f"       - {k}: {v}")
                        else:
                            st.write(f"   - {key}: {value}")

                # Option to export data
                export_type = st.selectbox("Export Data as:", ['csv', 'excel', 'gz'])
                if st.button("Export Data"):
                    if export_type == 'gz':
                        data = data_processor.export_data(df, export_type)
                        st.download_button("Download Data (gzip)", data, file_name="crawl_data.gz", mime="application/gzip")
                    else:
                        data = data_processor.export_data(df, export_type)
                        st.download_button("Download Data", data, file_name=f"crawl_data.{export_type}", mime=f"application/{export_type}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()