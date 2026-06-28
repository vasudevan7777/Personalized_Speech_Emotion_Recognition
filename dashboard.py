"""
Dashboard Visualization Module.
Generates Plotly interactive graphs for the Speech Emotion Recognition dashboard.
"""

import plotly.graph_objects as go
import pandas as pd

# Design Color Palette for emotions
EMOTION_COLORS = {
    'neutral': '#6C757D',     # Cool Gray
    'calm': '#4EA8DE',        # Soft Serene Blue
    'happy': '#FFC01E',       # Sunny Gold
    'sad': '#560BAD',         # Deep Blue-Purple
    'angry': '#E63946',        # Crimson Red
    'fearful': '#7209B7',      # Moody Violet
    'disgust': '#40916C',      # Forest Green
    'surprised': '#FF7096'     # Coral Pink
}

def plot_emotion_distribution_donut(df):
    """
    Generate a donut chart showing the percentage distribution of predicted emotions.
    """
    if df.empty:
        return None
        
    counts = df['emotion'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [EMOTION_COLORS.get(label.lower(), '#9E9E9E') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color='#1E1E2F', width=2)),
        textinfo='percent+label',
        hoverinfo='label+value',
        textfont=dict(size=12, color='white')
    )])
    
    fig.update_layout(
        title={
            'text': "<b>Emotion Distribution</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='white')
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        height=280
    )
    
    return fig

def plot_risk_gauge_meter(risk_score, category=""):
    """
    Generate a gauge chart (speedometer) to display the Emotional Risk Score.
    """
    # Define gauge colors
    # Low (0-30): green, Mod (30-60): yellow, High (60-85): orange, Extreme (85-100): red
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>Risk Level: {category}</b>", 'font': {'size': 16, 'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#6A0DAD", 'thickness': 0.25},  # purple indicator bar
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(40, 167, 69, 0.4)'},      # Green
                {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.4)'},     # Yellow
                {'range': [60, 85], 'color': 'rgba(253, 126, 20, 0.4)'},    # Orange
                {'range': [85, 100], 'color': 'rgba(220, 53, 69, 0.4)'}     # Red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter, sans-serif"},
        height=240,
        margin=dict(l=30, r=30, t=50, b=20)
    )
    
    return fig

def plot_emotion_timeline(df):
    """
    Generate a scatter-line plot showing emotions detected chronologically.
    """
    if df.empty:
        return None
        
    # Sort chronologically
    df_sorted = df.sort_values('timestamp').copy()
    df_sorted['time_str'] = df_sorted['timestamp'].dt.strftime('%H:%M:%S')
    
    fig = go.Figure()
    
    # Trace for risk score over time
    fig.add_trace(go.Scatter(
        x=df_sorted['time_str'],
        y=df_sorted['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#764ba2', width=3),
        marker=dict(
            size=10, 
            color=[EMOTION_COLORS.get(e.lower(), '#9E9E9E') for e in df_sorted['emotion']],
            line=dict(width=2, color='white')
        ),
        text=[f"Emotion: {e}<br>Confidence: {c*100:.1f}%" for e, c in zip(df_sorted['emotion'], df_sorted['confidence'])],
        hoverinfo='text+x+y'
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Emotional Risk & Prediction Timeline</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='white')
        },
        xaxis=dict(
            title=dict(text="Time of Day", font=dict(color='white')),
            tickfont=dict(color='gray'),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Emotional Risk (0-100)", font=dict(color='white')),
            tickfont=dict(color='gray'),
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.05)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=50, b=40),
        height=280
    )
    
    return fig

def plot_confidence_radar_or_bar(all_probs, predicted_emotion):
    """
    Plot predicted emotion probabilities using a styled horizontal bar chart.
    """
    if not all_probs:
        return None
        
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1])
    emotions = [e.capitalize() for e, _ in sorted_probs]
    probs = [p * 100 for _, p in sorted_probs]
    
    # Highlight current prediction color
    colors = [
        '#764ba2' if e.lower() != predicted_emotion.lower() else EMOTION_COLORS.get(predicted_emotion.lower(), '#667eea')
        for e in emotions
    ]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#1E1E2F', width=1.5)),
        text=[f"{p:.1f}%" for p in probs],
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Probability Distribution Breakdown</b>",
            'font': dict(size=14, color='white')
        },
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(color='white')),
            tickfont=dict(color='gray'),
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            tickfont=dict(color='white'),
            showgrid=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=30),
        height=280
    )
    
    return fig
