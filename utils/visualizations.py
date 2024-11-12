import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

class Visualizer:
    @st.cache_data
    def plot_daily_crawls(self, df):
        """Create daily crawl frequency plot."""
        daily_crawls = df.groupby('date')['url'].count().reset_index()
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
    def plot_monthly_crawls(self, df):
        """Create monthly crawl distribution plot."""
        monthly_crawls = df.groupby('month')['url'].count().reset_index()
        fig = px.bar(
            monthly_crawls,
            x='month',
            y='url',
            title='Monthly Crawl Distribution',
            labels={'url': 'Number of Crawls', 'month': 'Month'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white'
        )
        return fig

    @st.cache_data
    def create_heatmap(self, df):
        """Create crawl frequency heatmap."""
        heatmap_data = df.pivot_table(
            index='day_of_week',
            columns='hour',
            values='url',
            aggfunc='count',
            fill_value=0
        )
        
        # Reorder days of week
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
