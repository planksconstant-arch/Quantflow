"""
Enhanced Plotly Chart Configuration
Professional-grade interactivity and readability settings
"""

# Universal chart configuration for all QuantFlow charts
CHART_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath', 
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape'
    ],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'quantflow_chart',
        'height': 1080,
        'width': 1920,
        'scale': 3  # Higher resolution
    }
}

# Professional layout template
PROFESSIONAL_LAYOUT = {
    # Typography - Larger, clearer fonts
    'font': {
        'family': 'Inter, Arial, sans-serif',
        'size': 14,
        'color': '#E5E7EB'  # Light gray for readability
    },
    
    # Title styling
    'title': {
        'font': {
            'size': 22,
            'color': '#00FF88',
            'family': 'Inter, Arial, sans-serif'
        },
        'x': 0.5,
        'xanchor': 'center',
        'y': 0.98,
        'yanchor': 'top'
    },
    
    # Axis improvements
    'xaxis': {
        'showgrid': True,
        'gridwidth': 1,
        'gridcolor': 'rgba(255,255,255,0.1)',
        'showline': True,
        'linewidth': 2,
        'linecolor': 'rgba(255,255,255,0.3)',
        'mirror': True,
        'ticks': 'outside',
        'tickfont': {'size': 13, 'color': '#D1D5DB'},
        'title': {'font': {'size': 15, 'color': '#00C8FF'}},
        'zeroline': True,
        'zerolinewidth': 2,
        'zerolinecolor': 'rgba(255,255,255,0.2)'
    },
    
    'yaxis': {
        'showgrid': True,
        'gridwidth': 1,
        'gridcolor': 'rgba(255,255,255,0.1)',
        'showline': True,
        'linewidth': 2,
        'linecolor': 'rgba(255,255,255,0.3)',
        'mirror': True,
        'ticks': 'outside',
        'tickfont': {'size': 13, 'color': '#D1D5DB'},
        'title': {'font': {'size': 15, 'color': '#00C8FF'}},
        'zeroline': True,
        'zerolinewidth': 2,
        'zerolinecolor': 'rgba(255,255,255,0.2)'
    },
    
    # Dark theme
    'template': 'plotly_dark',
    'paper_bgcolor': '#0F172A',
    'plot_bgcolor': '#1E293B',
    
    # Hover improvements
    'hovermode': 'x unified',
    'hoverlabel': {
        'bgcolor': '#1E293B',
        'font': {
            'size': 14,
            'family': 'Inter, Arial, sans-serif',
            'color': '#E5E7EB'
        },
        'bordercolor': '#00FF88',
        'namelength': -1  # Show full names
    },
    
    # Legend improvements
    'showlegend': True,
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1,
        'bgcolor': 'rgba(30,41,59,0.8)',
        'bordercolor': 'rgba(255,255,255,0.2)',
        'borderwidth': 1,
        'font': {'size': 13, 'color': '#E5E7EB'}
    },
    
    # Margins
    'margin': dict(l=80, r=80, t=100, b=80),
    
    # Animation
    'transition': {
        'duration': 300,
        'easing': 'cubic-in-out'
    }
}

# Enhanced hover template
def create_hover_template(metric_name, unit="", description=""):
    """Create professional hover template with context"""
    return (
        f"<b>{metric_name}</b><br>"
        f"Value: %{{y:.4f}}{unit}<br>"
        f"Price: $%{{x:.2f}}<br>"
        f"<i>{description}</i><br>"
        "<extra></extra>"
    )

# Color scheme - High contrast, accessible
COLORS = {
    'primary': '#00FF88',      # Neon green (positive)
    'secondary': '#00C8FF',    # Cyan (info)
    'accent': '#FFB020',       # Amber (warning)
    'danger': '#FF6B6B',       # Coral (negative, not harsh)
    'critical': '#FF004D',     # Crimson (critical)
    'neutral': '#9CA3AF',      # Gray
    'background': '#0F172A',   # Dark navy
    'surface': '#1E293B',      # Slate
    'grid': 'rgba(255,255,255,0.1)',
    'text': '#E5E7EB'          # Light gray
}

# Line styles for clarity
LINE_STYLES = {
    'solid': dict(width=3),
    'dash': dict(width=3, dash='dash'),
    'dot': dict(width=2, dash='dot'),
    'dashdot': dict(width=3, dash='dashdot')
}

# Marker styles
MARKER_STYLES = {
    'current': dict(
        size=12,
        color='#FF004D',
        symbol='diamond',
        line=dict(width=2, color='white')
    ),
    'strike': dict(
        size=10,
        color='#00C8FF',
        symbol='circle',
        line=dict(width=2, color='white')
    ),
    'highlight': dict(
        size=10,
        color='#FFB020',
        symbol='star',
        line=dict(width=2, color='white')
    )
}

def add_crosshair(fig):
    """Add interactive crosshair to chart"""
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='#FFB020',
        spikethickness=1,
        spikedash='dot'
    )
    
    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='#FFB020',
        spikethickness=1,
        spikedash='dot'
    )
    
    return fig

def add_range_selector(fig):
    """Add range selector for time-based zoom"""
    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.05,
            bgcolor='#1E293B',
            bordercolor='#00C8FF',
            borderwidth=1
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=10, label="10 pts", step="all"),
                dict(count=25, label="25 pts", step="all"),
                dict(count=50, label="50 pts", step="all"),
                dict(step="all", label="All")
            ]),
            bgcolor='#1E293B',
            activecolor='#00FF88',
            font=dict(color='#E5E7EB')
        )
    )
    
    return fig
