import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np

class VisualizationPreset:
    def __init__(self, name, description, charts):
        self.name = name
        self.description = description
        self.charts = charts

class Visualizer:
    def __init__(self):
        self.presets = {
            'overview': VisualizationPreset(
                'Overview',
                'General crawling behavior overview',
                ['daily_crawls', 'monthly_crawls', 'heatmap']
            ),
            'temporal': VisualizationPreset(
                'Temporal Analysis',
                'Detailed time-based analysis',
                ['daily_crawls', 'time_series_decomposition', 'hourly_comparison']
            ),
            'url_focused': VisualizationPreset(
                'URL Analysis',
                'URL crawling patterns and distribution',
                ['url_distribution', 'monthly_crawls']
            ),
            'comparison': VisualizationPreset(
                'Comparative Analysis',
                'Compare different time periods',
                ['period_comparison', 'hourly_comparison']
            )
        }

    @st.cache_data
    def plot_daily_crawls(_self, df):
        daily_crawls = df.groupby('date')['url'].count().astype('float64').reset_index()
        fig = px.line(
            daily_crawls,
            x='date',
            y='url',
            title='Daily Crawl Frequency',
            labels={'url': 'Number of Crawls', 'date': 'Date'}
        )
        fig.update_layout(
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='white'
        )
        return fig

    @st.cache_data
    def plot_monthly_crawls(_self, df):
        monthly_crawls = (
            df.groupby('month')['url']
            .agg('count')
            .astype('float64')
            .reset_index(name='count')
        )
        fig = px.bar(
            monthly_crawls,
            x='month',
            y='count',
            title='Monthly Crawl Distribution',
            labels={'count': 'Number of Crawls', 'month': 'Month'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white'
        )
        return fig

    @st.cache_data
    def create_heatmap(_self, df):
        # Convert counts to float64 before pivot
        counts = df.groupby(['day_of_week', 'hour']).size().astype('float64')
        
        # Create pivot table with float64 values
        heatmap_data = counts.reset_index().pivot(
            index='day_of_week',
            columns='hour',
            values=0
        ).fillna(0.0)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Crawl Frequency Heat Map (Hour vs Day)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        return fig

    @st.cache_data
    def plot_time_series_decomposition(_self, trend, seasonal, residual):
        fig = go.Figure()
        
        if not trend.empty:
            fig.add_trace(go.Scatter(
                x=trend.index,
                y=trend.values.astype('float64'),
                name='Trend',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=seasonal.index,
                y=seasonal.values.astype('float64'),
                name='Seasonal',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=residual.index,
                y=residual.values.astype('float64'),
                name='Residual',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Time Series Decomposition',
                xaxis_title='Date',
                yaxis_title='Component Value',
                height=600,
                showlegend=True,
                plot_bgcolor='white'
            )
        return fig

    @st.cache_data
    def plot_url_distribution(_self, url_counts):
        df = pd.DataFrame(url_counts.items(), columns=['URL', 'Count'])
        df['Count'] = df['Count'].astype('float64')
        df = df.sort_values('Count', ascending=True).tail(10)
        
        fig = px.bar(
            df,
            x='Count',
            y='URL',
            orientation='h',
            title='Top 10 Most Crawled URLs'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='white',
            height=400
        )
        return fig

    @st.cache_data
    def plot_period_comparison(_self, period1_data, period2_data, metric, title):
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Period 1',
            x=['Period 1'],
            y=[float(period1_data)],
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.add_trace(go.Bar(
            name='Period 2',
            x=['Period 2'],
            y=[float(period2_data)],
            marker_color='rgb(26, 118, 255)'
        ))

        fig.update_layout(
            title=title,
            showlegend=True,
            plot_bgcolor='white',
            barmode='group'
        )
        return fig

    @st.cache_data
    def plot_hourly_comparison(_self, period1_dist, period2_dist):
        hours = list(range(24))
        
        # Convert to float64
        p1_values = [float(period1_dist.get(hour, 0)) for hour in hours]
        p2_values = [float(period2_dist.get(hour, 0)) for hour in hours]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=p1_values,
            name='Period 1',
            line=dict(color='rgb(55, 83, 109)')
        ))
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=p2_values,
            name='Period 2',
            line=dict(color='rgb(26, 118, 255)')
        ))
        
        fig.update_layout(
            title='Hourly Distribution Comparison',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Crawls',
            showlegend=True,
            plot_bgcolor='white'
        )
        return fig

    @st.cache_data
    def get_preset_charts(_self, preset_name):
        if preset_name in _self.presets:
            return _self.presets[preset_name].charts
        return []

    @st.cache_data
    def get_preset_description(_self, preset_name):
        if preset_name in _self.presets:
            return _self.presets[preset_name].description
        return ""
