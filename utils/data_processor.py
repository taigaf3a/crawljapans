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
    def get_url_patterns(_self, df):
        """Analyze URL patterns and frequency."""
        url_patterns = df.groupby('url').agg({
            'date': 'count',
            'month': 'nunique'
        }).reset_index()
        url_patterns.columns = ['url', 'total_crawls', 'months_active']
        return url_patterns

    @st.cache_data
    def perform_statistical_analysis(_self, df):
        """Perform advanced statistical analysis on crawl data."""
        daily_series = df.groupby('date').size()
        
        basic_stats = {
            'mean_daily_crawls': daily_series.mean(),
            'median_daily_crawls': daily_series.median(),
            'std_daily_crawls': daily_series.std(),
            'skewness': daily_series.skew(),
            'kurtosis': daily_series.kurtosis()
        }

        trend = pd.Series()
        seasonal = pd.Series()
        residual = pd.Series()

        if len(daily_series) >= 14:  # Minimum length for decomposition
            try:
                daily_series = daily_series.asfreq('D', fill_value=0)
                decomposition = seasonal_decompose(daily_series, period=7)
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
            except Exception:
                pass

        hourly_stats = df.groupby('hour').size()
        peak_hours = hourly_stats.nlargest(3).index.tolist()
        
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