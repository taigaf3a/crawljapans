import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import io
import gzip

class DataProcessor:
    @st.cache_data
    def load_data(self, file):
        """Load and process crawler data from CSV or GZ file."""
        try:
            if file.name.endswith('.gz'):
                # Read gzipped file
                content = gzip.decompress(file.read())
                df = pd.read_csv(io.BytesIO(content))
            else:
                # Read CSV file
                df = pd.read_csv(file)
            
            # Ensure required columns exist
            required_columns = ['url', 'date', 'time']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("File must contain 'url', 'date', and 'time' columns")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.strftime('%Y-%m')
            df['day_of_week'] = df['date'].dt.day_name()
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            
            return df
        except Exception as e:
            raise ValueError(f"Error processing file: {str(e)}")

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