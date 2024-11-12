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
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_crawls['date'],
                y=daily_crawls['url'],
                name='Daily Crawls',
                uid='daily_crawls_line'
            )
        )
        fig.update_layout(
            title='Daily Crawl Frequency',
            xaxis_title='Date',
            yaxis_title='Number of Crawls',
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='white',
            uirevision='daily_crawls'
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
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly_crawls['month'],
                y=monthly_crawls['count'],
                name='Monthly Crawls',
                uid='monthly_crawls_bar'
            )
        )
        fig.update_layout(
            title='Monthly Crawl Distribution',
            xaxis_title='Month',
            yaxis_title='Number of Crawls',
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            uirevision='monthly_crawls'
        )
        return fig

    @st.cache_data
    def create_heatmap(_self, df):
        grouped = (df.groupby(['day_of_week', 'hour'])
                  .size()
                  .reset_index(name='count')
                  .astype({'count': 'float64'}))
        
        heatmap_data = grouped.pivot(
            index='day_of_week',
            columns='hour',
            values='count'
        ).fillna(0.0).astype('float64')
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values.astype('float64'),
            x=heatmap_data.columns.astype('float64'),
            y=heatmap_data.index,
            colorscale='Viridis',
            hoverongaps=False,
            uid='heatmap_main'
        ))
        
        fig.update_layout(
            title='Crawl Frequency Heat Map (Hour vs Day)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400,
            uirevision='heatmap'
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
                line=dict(color='blue'),
                uid='trend_line'
            ))
            
            fig.add_trace(go.Scatter(
                x=seasonal.index,
                y=seasonal.values.astype('float64'),
                name='Seasonal',
                line=dict(color='green'),
                uid='seasonal_line'
            ))
            
            fig.add_trace(go.Scatter(
                x=residual.index,
                y=residual.values.astype('float64'),
                name='Residual',
                line=dict(color='red'),
                uid='residual_line'
            ))
            
            fig.update_layout(
                title='Time Series Decomposition',
                xaxis_title='Date',
                yaxis_title='Component Value',
                height=600,
                showlegend=True,
                plot_bgcolor='white',
                uirevision='decomposition'
            )
        return fig

    @st.cache_data
    def plot_url_distribution(_self, url_data, metric='total_crawls', top_n=10):
        """Enhanced URL distribution visualization with multiple metric options."""
        df = url_data.nlargest(top_n, metric)
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df[metric],
                y=df['url'],
                orientation='h',
                name=metric.replace('_', ' ').title(),
                text=df[metric].round(2),
                textposition='auto',
                uid=f'url_dist_bars_{metric}'
            )
        )
        
        fig.update_layout(
            title=f'Top {top_n} URLs by {metric.replace("_", " ").title()}',
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='white',
            height=max(400, 50 * top_n),  # Dynamic height based on number of URLs
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title='URL',
            uirevision=f'url_distribution_{metric}'
        )
        return fig

    @st.cache_data
    def plot_period_comparison(_self, period1_data, period2_data, metric, title):
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Period 1',
            x=['Period 1'],
            y=[float(period1_data)],
            marker_color='rgb(55, 83, 109)',
            uid='period1_bar'
        ))
        
        fig.add_trace(go.Bar(
            name='Period 2',
            x=['Period 2'],
            y=[float(period2_data)],
            marker_color='rgb(26, 118, 255)',
            uid='period2_bar'
        ))

        fig.update_layout(
            title=title,
            showlegend=True,
            plot_bgcolor='white',
            barmode='group',
            uirevision='period_comparison'
        )
        return fig

    @st.cache_data
    def plot_hourly_comparison(_self, period1_dist, period2_dist):
        hours = list(range(24))
        
        p1_values = [float(period1_dist.get(hour, 0)) for hour in hours]
        p2_values = [float(period2_dist.get(hour, 0)) for hour in hours]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=p1_values,
            name='Period 1',
            line=dict(color='rgb(55, 83, 109)'),
            uid='hourly_period1'
        ))
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=p2_values,
            name='Period 2',
            line=dict(color='rgb(26, 118, 255)'),
            uid='hourly_period2'
        ))
        
        fig.update_layout(
            title='Hourly Distribution Comparison',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Crawls',
            showlegend=True,
            plot_bgcolor='white',
            uirevision='hourly_comparison'
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