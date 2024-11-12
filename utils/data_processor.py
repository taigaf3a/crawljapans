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
    def parse_log_line(_self, line):
        # Apache combined log format pattern with Googlebot filtering
        pattern = r'(?P<ip>[\d.]+)\s+[-\w]+\s+[-\w]+\s+\[(?P<datetime>[^\]]+)\]\s+"(?P<method>\w+)\s+(?P<url>[^\s"]+)[^"]*"\s+(?P<status>\d+)\s+(?P<bytes>[-\d]+)\s+"[^"]*"\s+"(?P<useragent>[^"]*)"'
        
        try:
            match = re.match(pattern, line)
            if match:
                data = match.groupdict()
                
                # Check if the user agent contains Googlebot (case insensitive)
                if not re.search(r'googlebot', data['useragent'], re.IGNORECASE):
                    return None
                    
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
        """Load and process crawler data from CSV, GZ, or log file."""
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
            
            if not parsed_data:
                raise ValueError(f"No Googlebot entries found in {file.name}. Total lines: {total_lines}, Invalid lines: {invalid_lines}")
            
            st.info(f"Processed {total_lines} lines, found {googlebot_entries} Googlebot entries")
            df = pd.DataFrame(parsed_data)
            
            # Convert date column to datetime and add derived columns
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.strftime('%Y-%m')
            df['day_of_week'] = df['date'].dt.day_name()
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing file {file.name}: {str(e)}")

    @st.cache_data
    def calculate_crawl_frequency(self, df):
        """Calculate crawl frequency metrics."""
        daily_counts = df.groupby(['date', 'url']).size().reset_index(name='crawl_count')
        return daily_counts

    @st.cache_data
    def calculate_monthly_stats(self, df):
        """Calculate monthly crawl statistics."""
        monthly_stats = df.groupby('month').agg({
            'url': 'count',
            'date': 'nunique'
        }).reset_index()
        monthly_stats.columns = ['month', 'total_crawls', 'unique_days']
        return monthly_stats

    @st.cache_data
    def get_url_patterns(self, df):
        """Analyze URL patterns and frequency."""
        url_patterns = df.groupby('url').agg({
            'date': 'count',
            'month': 'nunique'
        }).reset_index()
        url_patterns.columns = ['url', 'total_crawls', 'months_active']
        return url_patterns

    @st.cache_data
    def perform_statistical_analysis(self, df):
        """Perform advanced statistical analysis on crawl data."""
        # Daily crawl count series
        daily_series = df.groupby('date').size()
        
        # Basic statistics
        basic_stats = {
            'mean_daily_crawls': daily_series.mean(),
            'median_daily_crawls': daily_series.median(),
            'std_daily_crawls': daily_series.std(),
            'skewness': daily_series.skew(),
            'kurtosis': daily_series.kurtosis()
        }

        # Time series decomposition
        if len(daily_series) >= 14:  # Minimum length for decomposition
            try:
                # Convert to regular time series with missing values filled
                daily_series = daily_series.asfreq('D', fill_value=0)
                decomposition = seasonal_decompose(daily_series, period=7)
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
            except Exception:
                trend = seasonal = residual = pd.Series()
        else:
            trend = seasonal = residual = pd.Series()

        # Hourly distribution analysis
        hourly_stats = df.groupby('hour').size()
        peak_hours = hourly_stats.nlargest(3).index.tolist()
        
        # URL diversity analysis
        url_counts = df.groupby('url').size()
        url_diversity = {
            'unique_urls': len(url_counts),
            'gini_coefficient': _self._calculate_gini(url_counts.values),
            'top_urls': url_counts.nlargest(5).to_dict()
        }

        return {
            'basic_stats': basic_stats,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'peak_hours': peak_hours,
            'url_diversity': url_diversity
        }

    def _calculate_gini(self, array):
        """Calculate the Gini coefficient of inequality."""
        array = np.array(array)
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def export_data(self, df, export_type='csv'):
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
            # Export as CSV and compress with gzip
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            return gzip.compress(csv_str.encode('utf-8'))
        else:
            raise ValueError("Unsupported export format")

    @st.cache_data
    def compare_time_periods(self, df, start_date1, end_date1, start_date2, end_date2):
        """Compare Googlebot crawl data between two time periods."""
        period1 = df[(df['date'] >= start_date1) & (df['date'] <= end_date1)]
        period2 = df[(df['date'] >= start_date2) & (df['date'] <= end_date2)]
        
        # Calculate metrics for both periods
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