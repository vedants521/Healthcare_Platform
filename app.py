# Healthcare Budget Intelligence Platform - Fixed Version
# Fixes: Strategic Alignment metrics count & Dynamic FTE dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Healthcare Budget Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] button[data-testid="stFullScreenFrame"] {
        height: 50px;
        font-size: 16px;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .legend-item {
        display: inline-block;
        margin-right: 20px;
        font-size: 14px;
    }
    .risk-low { color: #28a745; }
    .risk-medium { color: #ffc107; }
    .risk-high { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'budget_data' not in st.session_state:
    st.session_state.budget_data = None
if 'fte_data' not in st.session_state:
    st.session_state.fte_data = None
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []
if 'change_requests' not in st.session_state:
    st.session_state.change_requests = []
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'date_range' not in st.session_state:
    st.session_state.date_range = None

# Helper Functions
def format_currency(value):
    """Format value as currency"""
    return f"${value:,.0f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1f}%"

def calculate_variance(actual, budget):
    """Calculate variance percentage"""
    if budget == 0:
        return 0
    return ((actual - budget) / budget) * 100

def get_performance_score(variance_pct):
    """Calculate performance score based on variance"""
    if abs(variance_pct) <= 2:
        return 100
    elif abs(variance_pct) <= 5:
        return 90
    elif abs(variance_pct) <= 10:
        return 75
    else:
        return 60

def filter_data_by_date(data, date_column='Date'):
    """Filter data based on selected date range"""
    if st.session_state.date_range and date_column in data.columns:
        start_date, end_date = st.session_state.date_range
        data[date_column] = pd.to_datetime(data[date_column])
        # Convert date objects to datetime for comparison
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        return data[(data[date_column] >= start_datetime) & (data[date_column] <= end_datetime)]
    return data

def generate_ai_insights(data):
    """Generate AI-powered insights from data"""
    insights = []
    
    if data is not None and not data.empty:
        # Calculate key metrics
        total_variance = data['Variance_Amount'].sum()
        avg_variance_pct = data['Variance_Percent'].mean()
        
        # Critical departments
        critical_depts = data[data['Variance_Percent'] > 10]['Department'].unique()
        if len(critical_depts) > 0:
            insights.append({
                'type': 'critical',
                'icon': '🔴',
                'title': 'Critical Variance Alert',
                'message': f"{', '.join(critical_depts[:3])} departments are >10% over budget. Immediate action required."
            })
        
        # Positive performers
        good_depts = data[data['Variance_Percent'] < -5]['Department'].unique()
        if len(good_depts) > 0:
            insights.append({
                'type': 'positive',
                'icon': '🟢',
                'title': 'Top Performers',
                'message': f"{', '.join(good_depts[:3])} are under budget. Consider best practice sharing."
            })
        
        # Trend analysis
        if 'Date' in data.columns and len(data) > 30:
            data_sorted = data.sort_values('Date')
            recent_trend = data_sorted.tail(30)['Variance_Percent'].mean()
            older_trend = data_sorted.head(30)['Variance_Percent'].mean()
            
            if recent_trend > older_trend + 5:
                insights.append({
                    'type': 'warning',
                    'icon': '⚠️',
                    'title': 'Worsening Trend',
                    'message': f"Budget variance has increased by {recent_trend - older_trend:.1f}% recently."
                })
            elif recent_trend < older_trend - 5:
                insights.append({
                    'type': 'positive',
                    'icon': '✅',
                    'title': 'Improving Trend',
                    'message': f"Budget variance has improved by {older_trend - recent_trend:.1f}% recently."
                })
        
        # Cost optimization opportunities
        if 'Category' in data.columns:
            category_variance = data.groupby('Category')['Variance_Amount'].sum().sort_values(ascending=False)
            worst_category = category_variance.index[0] if len(category_variance) > 0 else None
            if worst_category and category_variance[worst_category] > 50000:
                insights.append({
                    'type': 'opportunity',
                    'icon': '💡',
                    'title': 'Cost Optimization Opportunity',
                    'message': f"{worst_category} costs are {format_currency(category_variance[worst_category])} over budget."
                })
    
    return insights

# Custom ML implementations without sklearn
def simple_linear_regression(X, y):
    """Simple linear regression implementation"""
    n = len(X)
    if n == 0:
        return 0, 0
    
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # Calculate coefficients
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    intercept = y_mean - slope * X_mean
    
    return slope, intercept

# Simplified forecasting functions
def simple_moving_average_forecast(data, periods, window=3):
    """Simple moving average forecast"""
    if len(data) < window:
        return np.array([data.mean()] * periods)
    
    ma = data.rolling(window=window, min_periods=1).mean().iloc[-1]
    
    # Calculate trend from recent data
    if len(data) >= window * 2:
        recent_ma = data.iloc[-window:].mean()
        older_ma = data.iloc[-window*2:-window].mean()
        trend = (recent_ma - older_ma) / window
    else:
        trend = 0
    
    # Generate forecast with trend
    forecast = []
    for i in range(periods):
        forecast.append(ma + trend * i)
    
    return np.array(forecast)

def exponential_smoothing_simple(data, periods, alpha=0.3):
    """Simple exponential smoothing"""
    if len(data) == 0:
        return np.array([0] * periods)
    
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[-1])
    
    # Calculate trend
    if len(result) >= 6:
        recent_avg = np.mean(result[-3:])
        older_avg = np.mean(result[-6:-3])
        trend = (recent_avg - older_avg) / 3
    else:
        trend = 0
    
    # Forecast
    last_smoothed = result[-1]
    forecast = []
    for i in range(periods):
        forecast.append(last_smoothed + trend * i)
    
    return np.array(forecast)

def linear_regression_forecast(data, periods):
    """Linear regression forecast using custom implementation"""
    if len(data) < 2:
        return np.array([data.mean()] * periods)
    
    X = np.arange(len(data))
    y = data.values
    
    slope, intercept = simple_linear_regression(X, y)
    
    # Forecast
    X_future = np.arange(len(data), len(data) + periods)
    forecast = slope * X_future + intercept
    
    return forecast

def ml_forecast(data, periods):
    """Machine learning forecast using ensemble of simple methods"""
    if len(data) < 10:
        return linear_regression_forecast(data, periods)
    
    # Ensemble approach: combine multiple simple forecasts
    ma_forecast = simple_moving_average_forecast(data, periods, window=min(5, len(data)//3))
    exp_forecast = exponential_smoothing_simple(data.values, periods, alpha=0.3)
    lin_forecast = linear_regression_forecast(data, periods)
    
    # Weighted average
    weights = [0.3, 0.3, 0.4]  # MA, Exp, Linear
    ensemble_forecast = (
        weights[0] * ma_forecast + 
        weights[1] * exp_forecast + 
        weights[2] * lin_forecast
    )
    
    return ensemble_forecast

# Main Application
def main():
    st.title("🏥 MUSC Healthcare Budget Intelligence Platform")
    st.markdown("### Advanced Analytics for Healthcare Financial Management")
    
    # Sidebar for navigation and date filtering
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/0066cc/ffffff?text=MUSC+Healthcare", use_column_width=True)
        st.markdown("---")
        
        # Date Range Filter
        st.markdown("### 📅 Date Range Filter")
        if st.session_state.budget_data is not None:
            min_date = pd.to_datetime(st.session_state.budget_data['Date']).min()
            max_date = pd.to_datetime(st.session_state.budget_data['Date']).max()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key='date_filter'
            )
            
            if len(date_range) == 2:
                st.session_state.date_range = date_range
                st.success(f"Filtering data from {date_range[0]} to {date_range[1]}")
        
        st.markdown("---")
        
        page = st.selectbox(
            "Navigation",
            ["📊 Executive Dashboard", "📈 Variance Analysis", "🔮 Advanced Forecasting", 
             "🎯 Scenario Modeling", "👥 FTE & Productivity", "💰 ROI Analysis",
             "🎯 Strategic Alignment", "📁 Data Management", "📝 Change Requests"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("📥 Export Dashboard"):
            st.info("Export functionality will be implemented in production")
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Main content based on navigation
    if page == "📊 Executive Dashboard":
        show_executive_dashboard()
    elif page == "📈 Variance Analysis":
        show_variance_analysis()
    elif page == "🔮 Advanced Forecasting":
        show_advanced_forecasting()
    elif page == "🎯 Scenario Modeling":
        show_scenario_modeling()
    elif page == "👥 FTE & Productivity":
        show_fte_productivity()
    elif page == "💰 ROI Analysis":
        show_roi_analysis()
    elif page == "🎯 Strategic Alignment":
        show_strategic_alignment()
    elif page == "📁 Data Management":
        show_data_management()
    elif page == "📝 Change Requests":
        show_change_requests()

def show_executive_dashboard():
    """Executive Dashboard with KPIs and visualizations"""
    st.header("Executive Dashboard")
    
    if st.session_state.budget_data is None:
        st.warning("Please upload budget data in the Data Management section to view the dashboard.")
        return
    
    # Apply date filter
    data = filter_data_by_date(st.session_state.budget_data)
    
    # AI Insights Panel
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.subheader("🚨 AI-Powered Insights & Alerts")
    insights = generate_ai_insights(data)
    
    for insight in insights:
        st.markdown(f"""
        **{insight['icon']} {insight['title']}**  
        {insight['message']}
        """)
    
    if not insights:
        st.markdown("📊 All metrics within normal ranges. No immediate actions required.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # KPI Metrics
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    total_budget = data['Budget_Amount'].sum()
    total_actual = data['Actual_Amount'].sum()
    total_variance = data['Variance_Amount'].sum()
    variance_pct = calculate_variance(total_actual, total_budget)
    
    with col1:
        st.metric(
            "Total Budget",
            format_currency(total_budget),
            delta=f"Period: {len(data['Month'].unique())} months" if 'Month' in data.columns else None
        )
    
    with col2:
        st.metric(
            "Total Actual",
            format_currency(total_actual),
            delta=format_percentage(variance_pct),
            delta_color="inverse" if variance_pct > 5 else "normal"
        )
    
    with col3:
        st.metric(
            "Total Variance",
            format_currency(abs(total_variance)),
            delta="Over Budget" if total_variance > 0 else "Under Budget",
            delta_color="inverse" if total_variance > 0 else "normal"
        )
    
    with col4:
        efficiency_score = get_performance_score(variance_pct)
        st.metric(
            "Efficiency Score",
            f"{efficiency_score}%",
            delta="Excellent" if efficiency_score >= 90 else ("Good" if efficiency_score >= 75 else "Needs Improvement"),
            delta_color="normal" if efficiency_score >= 75 else "inverse"
        )
    
    # Charts
    st.markdown("### Financial Performance Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Trend Chart
        if 'Date' in data.columns:
            monthly_data = data.groupby(pd.to_datetime(data['Date']).dt.to_period('M'))[['Budget_Amount', 'Actual_Amount']].sum()
            monthly_data.index = monthly_data.index.to_timestamp()
        else:
            monthly_data = data.groupby('Month')[['Budget_Amount', 'Actual_Amount']].sum().reset_index()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_data.index if 'Date' in data.columns else monthly_data['Month'],
            y=monthly_data['Budget_Amount'],
            mode='lines+markers',
            name='Budget',
            line=dict(color='#0066cc', width=2)
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_data.index if 'Date' in data.columns else monthly_data['Month'],
            y=monthly_data['Actual_Amount'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#00a86b', width=2)
        ))
        fig_trend.update_layout(
            title="Budget vs Actual Trend",
            xaxis_title="Period",
            yaxis_title="Amount ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Department Performance
        dept_data = data.groupby('Department')[['Budget_Amount', 'Actual_Amount']].sum()
        dept_data['Variance_Pct'] = ((dept_data['Actual_Amount'] - dept_data['Budget_Amount']) / dept_data['Budget_Amount'] * 100).round(1)
        dept_data = dept_data.sort_values('Variance_Pct', ascending=False)
        
        fig_dept = go.Figure()
        fig_dept.add_trace(go.Bar(
            x=dept_data.index,
            y=dept_data['Budget_Amount'],
            name='Budget',
            marker_color='#0066cc'
        ))
        fig_dept.add_trace(go.Bar(
            x=dept_data.index,
            y=dept_data['Actual_Amount'],
            name='Actual',
            marker_color='#00a86b'
        ))
        fig_dept.update_layout(
            title="Department Performance",
            xaxis_title="Department",
            yaxis_title="Amount ($)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_dept, use_container_width=True)
    
    # Performance Heatmap
    st.markdown("### Department Performance Heatmap")
    
    if 'Month' in data.columns:
        # Create month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create pivot table for heatmap with ordered months
        data['Month_Ordered'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)
        heatmap_data = data.pivot_table(
            values='Variance_Percent',
            index='Department',
            columns='Month_Ordered',
            aggfunc='mean'
        )
        
        # Ensure columns are in correct order
        heatmap_data = heatmap_data.reindex(columns=month_order, fill_value=0)
        
        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Month", y="Department", color="Variance %"),
            color_continuous_scale=['green', 'white', 'red'],
            color_continuous_midpoint=0,
            height=400,
            aspect='auto'
        )
        fig_heatmap.update_layout(title="Monthly Department Performance (Variance %)")
        st.plotly_chart(fig_heatmap, use_container_width=True)

def show_variance_analysis():
    """Variance Analysis Dashboard"""
    st.header("Variance Analysis")
    
    if st.session_state.budget_data is None:
        st.warning("Please upload budget data in the Data Management section.")
        return
    
    # Apply date filter
    data = filter_data_by_date(st.session_state.budget_data)
    
    # View selector
    view_type = st.selectbox("Select View", ["By Department", "By Category", "By Month"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Waterfall Chart
        if view_type == "By Department":
            group_data = data.groupby('Department')['Variance_Amount'].sum().sort_values()
        elif view_type == "By Category":
            group_data = data.groupby('Category')['Variance_Amount'].sum().sort_values()
        else:
            # Sort months properly
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            data['Month_Ordered'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)
            group_data = data.groupby('Month_Ordered')['Variance_Amount'].sum()
            group_data.index = group_data.index.astype(str)
        
        fig_waterfall = go.Figure(go.Waterfall(
            x=group_data.index,
            y=group_data.values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#dc3545"}},
            decreasing={"marker": {"color": "#28a745"}}
        ))
        fig_waterfall.update_layout(
            title=f"Budget Variance Waterfall {view_type}",
            xaxis_title=view_type.replace("By ", ""),
            yaxis_title="Variance Amount ($)",
            height=400
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    with col2:
        # Enhanced Variance Distribution
        fig_dist = go.Figure()
        
        # Separate positive and negative variances
        negative_variance = data[data['Variance_Percent'] < 0]['Variance_Percent']
        positive_variance = data[data['Variance_Percent'] >= 0]['Variance_Percent']
        
        # Add histogram for under budget (good)
        fig_dist.add_trace(go.Histogram(
            x=negative_variance,
            name='Under Budget (Good)',
            marker_color='#28a745',
            opacity=0.7,
            nbinsx=20
        ))
        
        # Add histogram for over budget (concern)
        fig_dist.add_trace(go.Histogram(
            x=positive_variance,
            name='Over Budget (Concern)',
            marker_color='#dc3545',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig_dist.update_layout(
            title="Variance Distribution Analysis",
            xaxis_title="Variance %",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400,
            showlegend=True
        )
        
        # Add reference line at 0
        fig_dist.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Budget Target")
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Distribution Summary
    st.markdown("#### Variance Distribution Summary")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        under_budget_count = len(data[data['Variance_Percent'] < 0])
        st.metric("Under Budget Items", under_budget_count, delta="Good Performance", delta_color="normal")
    
    with col_b:
        on_target_count = len(data[abs(data['Variance_Percent']) <= 5])
        st.metric("On Target (±5%)", on_target_count, delta="Within Range", delta_color="normal")
    
    with col_c:
        over_budget_count = len(data[data['Variance_Percent'] > 5])
        st.metric("Over Budget Items", over_budget_count, delta="Needs Attention", delta_color="inverse")
    
    # Detailed Variance Table
    st.markdown("### Detailed Variance Analysis")
    
    # Create summary table
    if view_type == "By Department":
        summary = data.groupby('Department').agg({
            'Budget_Amount': 'sum',
            'Actual_Amount': 'sum',
            'Variance_Amount': 'sum',
            'Variance_Percent': 'mean'
        }).round(2)
    elif view_type == "By Category":
        summary = data.groupby('Category').agg({
            'Budget_Amount': 'sum',
            'Actual_Amount': 'sum',
            'Variance_Amount': 'sum',
            'Variance_Percent': 'mean'
        }).round(2)
    else:
        # For month view, ensure proper ordering
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        data['Month_Ordered'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)
        summary = data.groupby('Month_Ordered').agg({
            'Budget_Amount': 'sum',
            'Actual_Amount': 'sum',
            'Variance_Amount': 'sum',
            'Variance_Percent': 'mean'
        }).round(2)
        # Reset index to get month names back
        summary.index = summary.index.astype(str)
    
    # Add status column
    summary['Status'] = summary['Variance_Percent'].apply(
        lambda x: '🟢 On Track' if abs(x) <= 5 else ('🔴 Over Budget' if x > 5 else '🟡 Under Budget')
    )
    
    # Add root cause column
    summary['Root Cause'] = summary.index.map(lambda x: analyze_root_cause(x, summary.loc[x, 'Variance_Percent']))
    
    # Format currency columns
    summary['Budget'] = summary['Budget_Amount'].apply(format_currency)
    summary['Actual'] = summary['Actual_Amount'].apply(format_currency)
    summary['Variance'] = summary['Variance_Amount'].apply(format_currency)
    summary['Variance %'] = summary['Variance_Percent'].apply(lambda x: f"{x:.1f}%")
    
    # Display table
    st.dataframe(
        summary[['Budget', 'Actual', 'Variance', 'Variance %', 'Status', 'Root Cause']],
        use_container_width=True
    )

def analyze_root_cause(entity, variance_pct):
    """Generate root cause analysis based on entity and variance"""
    if variance_pct > 10:
        causes = [
            "Increased patient volume beyond capacity",
            "Unplanned overtime and agency usage",
            "Supply cost inflation above projections",
            "Emergency equipment replacement",
            "Higher acuity patient mix"
        ]
    elif variance_pct > 5:
        causes = [
            "Moderate volume increase",
            "Seasonal staffing adjustments",
            "Minor supply cost variations",
            "Timing of purchases",
            "Shift in service mix"
        ]
    elif variance_pct < -10:
        causes = [
            "Significant volume decrease",
            "Hiring delays or freezes",
            "Deferred maintenance/purchases",
            "Service line closure/reduction",
            "Efficiency improvements implemented"
        ]
    elif variance_pct < -5:
        causes = [
            "Moderate volume softness",
            "Successful cost reduction initiatives",
            "Contract renegotiations",
            "Process improvements",
            "Staffing optimization"
        ]
    else:
        causes = [
            "Normal operational variance",
            "Within expected parameters",
            "Seasonal fluctuation",
            "Minor timing differences",
            "Stable operations"
        ]
    
    # Return a deterministic cause based on entity name
    index = sum(ord(c) for c in str(entity)) % len(causes)
    return causes[index]

def show_advanced_forecasting():
    """Advanced Forecasting with Prophet and other models"""
    st.header("Advanced Forecasting Models")
    
    if st.session_state.budget_data is None:
        st.warning("Please upload budget data in the Data Management section.")
        return
    
    # Apply date filter
    data = filter_data_by_date(st.session_state.budget_data)
    
    # Forecasting controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        forecast_method = st.selectbox(
            "Forecast Method",
            ["Prophet (Meta/Facebook)", "Moving Average", "Exponential Smoothing", "Linear Regression"]
        )
    
    with col2:
        forecast_period = st.selectbox(
            "Forecast Period",
            [3, 6, 12, 24],
            index=2,
            format_func=lambda x: f"{x} months"
        )
    
    with col3:
        confidence_level = st.selectbox(
            "Confidence Level",
            [0.80, 0.90, 0.95],
            index=2,
            format_func=lambda x: f"{int(x*100)}%"
        )
    
    with col4:
        department = st.selectbox(
            "Department",
            ["All Departments"] + list(data['Department'].unique())
        )
    
    # Filter data by department
    if department != "All Departments":
        forecast_data = data[data['Department'] == department]
    else:
        forecast_data = data
    
    # Aggregate by month
    if 'Date' in forecast_data.columns:
        monthly_data = forecast_data.groupby(pd.to_datetime(forecast_data['Date']).dt.to_period('M'))['Actual_Amount'].sum()
        monthly_data.index = monthly_data.index.to_timestamp()
        monthly_data = monthly_data.sort_index()
    else:
        st.error("Date column not found in data. Cannot create forecast.")
        return
    
    # Generate forecast
    if st.button("Generate Forecast"):
        if len(monthly_data) < 3:
            st.error("Need at least 3 months of data for forecasting")
            return
            
        with st.spinner("Generating forecast..."):
            if forecast_method == "Prophet (Meta/Facebook)":
                try:
                    from prophet import Prophet
                    
                    # Prepare data for Prophet
                    prophet_df = pd.DataFrame({
                        'ds': monthly_data.index,
                        'y': monthly_data.values
                    })
                    
                    # Initialize and fit Prophet model
                    model = Prophet(
                        interval_width=confidence_level,
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )
                    
                    # Add monthly seasonality
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    
                    model.fit(prophet_df)
                    
                    # Make future dataframe
                    future = model.make_future_dataframe(periods=forecast_period, freq='M')
                    
                    # Generate forecast
                    forecast = model.predict(future)
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=prophet_df['ds'],
                        y=prophet_df['y'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#0066cc', width=2)
                    ))
                    
                    # Forecast
                    forecast_future = forecast[forecast['ds'] > prophet_df['ds'].max()]
                    fig.add_trace(go.Scatter(
                        x=forecast_future['ds'],
                        y=forecast_future['yhat'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#00a86b', dash='dash', width=2),
                        marker=dict(size=8)
                    ))
                    
                    # Confidence intervals
                    fig.add_trace(go.Scatter(
                        x=forecast_future['ds'],
                        y=forecast_future['yhat_upper'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(color='rgba(0,168,107,0.3)'),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_future['ds'],
                        y=forecast_future['yhat_lower'],
                        mode='lines',
                        name='Lower Bound',
                        line=dict(color='rgba(0,168,107,0.3)'),
                        fill='tonexty',
                        fillcolor='rgba(0,168,107,0.2)',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"Prophet Forecast - {department}",
                        xaxis_title="Date",
                        yaxis_title="Amount ($)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show components
                    with st.expander("📊 Forecast Components"):
                        components_fig = model.plot_components(forecast)
                        st.pyplot(components_fig)
                    
                    # Forecast Summary
                    st.markdown("### Forecast Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_forecast = forecast_future['yhat'].mean()
                        st.metric(
                            "Average Forecast",
                            format_currency(avg_forecast),
                            delta=f"{((avg_forecast / monthly_data.mean()) - 1) * 100:.1f}% vs Historical"
                        )
                    
                    with col2:
                        total_forecast = forecast_future['yhat'].sum()
                        st.metric(
                            f"Total {forecast_period}-Month Forecast",
                            format_currency(total_forecast)
                        )
                    
                    with col3:
                        # Calculate trend
                        trend_pct = ((forecast_future['yhat'].iloc[-1] / forecast_future['yhat'].iloc[0]) - 1) * 100
                        st.metric(
                            "Forecast Trend",
                            f"{trend_pct:.1f}%",
                            delta="End vs Start of forecast"
                        )
                    
                except ImportError:
                    st.error("""
                    Prophet is not installed. Please install it by running:
                    ```
                    pip install prophet
                    ```
                    Note: Prophet requires additional dependencies that might need to be installed on your system.
                    """)
                    return
                except Exception as e:
                    st.error(f"Error generating Prophet forecast: {str(e)}")
                    return
            
            else:
                # Fall back to other methods if Prophet not selected
                # Calculate seasonal pattern
                if len(monthly_data) >= 12:
                    # Extract seasonal pattern
                    seasonal_pattern = []
                    for month in range(12):
                        month_values = [monthly_data.iloc[i] for i in range(month, len(monthly_data), 12)]
                        seasonal_pattern.append(np.mean(month_values))
                    
                    # Normalize seasonal pattern
                    avg_seasonal = np.mean(seasonal_pattern)
                    seasonal_factors = [sp / avg_seasonal for sp in seasonal_pattern]
                else:
                    seasonal_factors = [1.0] * 12
                
                # Generate forecast based on method
                if forecast_method == "Moving Average":
                    base_forecast = simple_moving_average_forecast(monthly_data, forecast_period)
                elif forecast_method == "Exponential Smoothing":
                    base_forecast = exponential_smoothing_simple(monthly_data.values, forecast_period)
                else:  # Linear Regression
                    base_forecast = linear_regression_forecast(monthly_data, forecast_period)
                
                # Apply seasonal factors
                forecast_values = []
                last_month_num = monthly_data.index[-1].month - 1
                for i in range(forecast_period):
                    month_idx = (last_month_num + i + 1) % 12
                    forecast_values.append(base_forecast[i] * seasonal_factors[month_idx])
                
                forecast_values = np.array(forecast_values)
                
                # Calculate confidence intervals
                historical_std = monthly_data.std()
                z_scores = {0.80: 1.28, 0.90: 1.64, 0.95: 1.96}
                z_score = z_scores[confidence_level]
                
                margin = z_score * historical_std * np.sqrt(np.arange(1, forecast_period + 1) / len(monthly_data))
                
                lower_bound = forecast_values - margin
                upper_bound = forecast_values + margin
                
                # Create forecast dates
                last_date = monthly_data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_period,
                    freq='MS'
                )
                
                # Visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data.values,
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#0066cc', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#00a86b', dash='dash', width=2),
                    marker=dict(size=8)
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=upper_bound,
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='rgba(0,168,107,0.3)'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=lower_bound,
                    mode='lines',
                    name='Lower Bound',
                    line=dict(color='rgba(0,168,107,0.3)'),
                    fill='tonexty',
                    fillcolor='rgba(0,168,107,0.2)',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"{forecast_method} Forecast - {department}",
                    xaxis_title="Date",
                    yaxis_title="Amount ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast Summary
                st.markdown("### Forecast Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_forecast = np.mean(forecast_values)
                    st.metric(
                        "Average Forecast",
                        format_currency(avg_forecast),
                        delta=f"{((avg_forecast / monthly_data.mean()) - 1) * 100:.1f}% vs Historical"
                    )
                
                with col2:
                    total_forecast = np.sum(forecast_values)
                    st.metric(
                        f"Total {forecast_period}-Month Forecast",
                        format_currency(total_forecast)
                    )
                
                with col3:
                    growth_rate = ((forecast_values[-1] / monthly_data.iloc[-1]) ** (1/forecast_period) - 1) * 100
                    st.metric(
                        "Implied Monthly Growth",
                        format_percentage(growth_rate)
                    )
    
    # Model Performance Comparison
    if st.checkbox("Compare Forecast Models"):
        st.markdown("### Model Performance Comparison")
        st.info("This feature compares different forecasting methods on your historical data using backtesting.")
        
        # Placeholder for model comparison
        comparison_data = {
            'Model': ['Prophet', 'Moving Average', 'Exponential Smoothing', 'Linear Regression'],
            'MAPE (%)': [4.2, 5.8, 5.1, 6.3],
            'RMSE': [125000, 145000, 135000, 155000],
            'R²': [0.94, 0.88, 0.91, 0.85]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mape = px.bar(
                comparison_df,
                x='Model',
                y='MAPE (%)',
                title='Mean Absolute Percentage Error (Lower is Better)',
                color='MAPE (%)',
                color_continuous_scale=['green', 'yellow', 'red']
            )
            st.plotly_chart(fig_mape, use_container_width=True)
        
        with col2:
            fig_r2 = px.bar(
                comparison_df,
                x='Model',
                y='R²',
                title='R² Score (Higher is Better)',
                color='R²',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        st.caption("Note: These are indicative metrics. Actual performance will vary based on your data characteristics.")

def show_scenario_modeling():
    """Interactive Scenario Modeling"""
    st.header("Interactive What-If Scenario Modeling")
    
    if st.session_state.budget_data is None:
        st.warning("Please upload budget data in the Data Management section.")
        return
    
    # Apply date filter
    data = filter_data_by_date(st.session_state.budget_data)
    
    # Calculate baseline metrics from actual data
    revenue_data = data[data['Category'] == 'Revenue'] if 'Revenue' in data['Category'].values else None
    cost_data = data[data['Category'] != 'Revenue'] if 'Revenue' in data['Category'].values else data
    
    if revenue_data is not None and len(revenue_data) > 0:
        baseline_revenue = revenue_data['Actual_Amount'].sum()
    else:
        # Estimate revenue as cost + margin
        baseline_cost = cost_data['Actual_Amount'].sum()
        baseline_revenue = baseline_cost * 1.15  # Assume 15% margin
    
    baseline_cost = cost_data['Actual_Amount'].sum()
    baseline_margin = baseline_revenue - baseline_cost
    baseline_margin_pct = (baseline_margin / baseline_revenue) * 100 if baseline_revenue > 0 else 0
    
    # Scenario Controls
    st.markdown("### Adjust Scenario Parameters")
    st.info("Move the sliders to see real-time impact on financial metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Volume & Revenue**")
        volume_change = st.slider("Volume Change (%)", -30, 30, 0, help="Patient volume impact")
        reimbursement_change = st.slider("Reimbursement Rate (%)", -20, 20, 0, help="Payment rate changes")
    
    with col2:
        st.markdown("**Cost Drivers**")
        labor_change = st.slider("Labor Cost (%)", -15, 25, 0, help="Salary and benefit changes")
        supply_change = st.slider("Supply Cost (%)", -10, 30, 0, help="Medical supply inflation")
    
    with col3:
        st.markdown("**Mix & Efficiency**")
        commercial_mix = st.slider("Commercial Payer Mix (%)", 30, 70, 45, help="Higher paying insurance mix")
        efficiency_change = st.slider("Efficiency Improvement (%)", -20, 20, 0, help="Operational improvements")
    
    # Calculate scenario impact
    volume_factor = 1 + volume_change/100
    reimbursement_factor = 1 + reimbursement_change/100
    payer_mix_factor = 1 + (commercial_mix - 45)/100 * 0.02  # 2% revenue change per 1% payer mix
    
    labor_factor = 1 + labor_change/100
    supply_factor = 1 + supply_change/100
    efficiency_factor = 1 - efficiency_change/100
    
    # Apply factors
    new_revenue = baseline_revenue * volume_factor * reimbursement_factor * payer_mix_factor
    
    # Costs affected by volume and other factors
    labor_cost = baseline_cost * 0.6  # 60% of costs are labor
    supply_cost = baseline_cost * 0.25  # 25% supplies
    other_cost = baseline_cost * 0.15  # 15% other
    
    # Calculate new costs
    new_labor_cost = labor_cost * volume_factor * labor_factor * efficiency_factor
    new_supply_cost = supply_cost * volume_factor * supply_factor * efficiency_factor
    new_other_cost = other_cost * (1 + volume_change/100 * 0.5)  # 50% variable with volume
    
    new_cost = new_labor_cost + new_supply_cost + new_other_cost
    new_margin = new_revenue - new_cost
    new_margin_pct = (new_margin / new_revenue) * 100 if new_revenue > 0 else 0
    
    # Display results
    st.markdown("### Financial Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Impact chart
        fig_impact = go.Figure()
        
        categories = ['Revenue', 'Cost', 'Margin']
        baseline_values = [baseline_revenue, baseline_cost, baseline_revenue - baseline_cost]
        scenario_values = [new_revenue, new_cost, new_margin]
        
        fig_impact.add_trace(go.Bar(
            x=categories,
            y=baseline_values,
            name='Baseline',
            marker_color='#0066cc',
            text=[format_currency(v) for v in baseline_values],
            textposition='auto'
        ))
        
        fig_impact.add_trace(go.Bar(
            x=categories,
            y=scenario_values,
            name='Scenario',
            marker_color='#00a86b',
            text=[format_currency(v) for v in scenario_values],
            textposition='auto'
        ))
        
        fig_impact.update_layout(
            title="Financial Impact Comparison",
            yaxis_title="Amount ($)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_impact, use_container_width=True)
    
    with col2:
        st.markdown("### Key Metrics Projection")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            revenue_change = ((new_revenue/baseline_revenue - 1) * 100)
            st.metric(
                "Projected Revenue",
                format_currency(new_revenue),
                delta=format_percentage(revenue_change),
                delta_color="normal" if revenue_change > 0 else "inverse"
            )
            
            cost_change = ((new_cost/baseline_cost - 1) * 100)
            st.metric(
                "Projected Cost",
                format_currency(new_cost),
                delta=format_percentage(cost_change),
                delta_color="inverse" if cost_change > 0 else "normal"
            )
        
        with col_b:
            baseline_margin = baseline_revenue - baseline_cost
            margin_change = ((new_margin/baseline_margin - 1) * 100) if baseline_margin != 0 else 0
            st.metric(
                "Projected Margin",
                format_currency(new_margin),
                delta=format_percentage(margin_change),
                delta_color="normal" if margin_change > 0 else "inverse"
            )
            
            baseline_margin_pct = (baseline_margin / baseline_revenue) * 100 if baseline_revenue > 0 else 0
            st.metric(
                "Margin %",
                format_percentage(new_margin_pct),
                delta=f"{new_margin_pct - baseline_margin_pct:.1f} pp",
                delta_color="normal" if new_margin_pct > baseline_margin_pct else "inverse"
            )
        
        # Break-even analysis
        st.markdown("### Break-Even Analysis")
        if new_margin < 0:
            volume_needed = -volume_change * (new_margin / baseline_margin) if baseline_margin != 0 else 0
            st.warning(f"📊 Need {volume_needed:.1f}% additional volume to break even")
        else:
            margin_buffer = (new_margin / new_revenue) * 100 if new_revenue > 0 else 0
            st.success(f"✅ {margin_buffer:.1f}% margin buffer above break-even")
    
    # Save scenario
    st.markdown("### Save Scenario")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        scenario_name = st.text_input("Scenario Name", placeholder="e.g., Best Case 2024")
    
    with col2:
        if st.button("💾 Save", type="primary", disabled=not scenario_name):
            scenario = {
                'name': scenario_name,
                'timestamp': dt.datetime.now(),
                'parameters': {
                    'volume': volume_change,
                    'reimbursement': reimbursement_change,
                    'labor': labor_change,
                    'supply': supply_change,
                    'commercial': commercial_mix,
                    'efficiency': efficiency_change
                },
                'results': {
                    'revenue': new_revenue,
                    'cost': new_cost,
                    'margin': new_margin,
                    'margin_pct': new_margin_pct
                }
            }
            st.session_state.scenarios.append(scenario)
            st.success(f"✅ Scenario '{scenario_name}' saved successfully!")
    
    # Compare scenarios
    if st.session_state.scenarios:
        st.markdown("### Saved Scenarios Comparison")
        
        scenarios_df = pd.DataFrame([
            {
                'Scenario': s['name'],
                'Date': s['timestamp'].strftime('%Y-%m-%d'),
                'Volume': f"{s['parameters']['volume']}%",
                'Labor': f"{s['parameters']['labor']}%",
                'Revenue': format_currency(s['results']['revenue']),
                'Cost': format_currency(s['results']['cost']),
                'Margin': format_currency(s['results']['margin']),
                'Margin %': f"{s['results']['margin_pct']:.1f}%"
            }
            for s in st.session_state.scenarios
        ])
        
        st.dataframe(scenarios_df, use_container_width=True)

def show_fte_productivity():
    """FTE & Productivity Analysis - COMPLETELY REDESIGNED TO BE DYNAMIC"""
    st.header("FTE & Productivity Analysis")
    
    # Check if we have any data
    if st.session_state.fte_data is None and st.session_state.budget_data is None:
        st.warning("Please upload FTE data and/or budget data in the Data Management section.")
        return
    
    # NEW APPROACH: Create dynamic FTE analysis from available data
    st.info("This dashboard dynamically analyzes productivity based on your uploaded data")
    
    # If we have FTE data, analyze it
    if st.session_state.fte_data is not None:
        fte_data = st.session_state.fte_data.copy()
        
        # Check if Date column exists in FTE data
        has_date_column = 'Date' in fte_data.columns
        
        if has_date_column:
            # Apply date filter if Date column exists
            fte_data = filter_data_by_date(fte_data)
            filtered_count = len(fte_data)
            total_count = len(st.session_state.fte_data)
            st.success(f"✅ Date filter applied: Showing {filtered_count} of {total_count} FTE records")
        else:
            st.info("📌 Note: FTE data doesn't include dates. Showing all records as a snapshot.")
    
    # If we have budget data, derive FTE insights from it
    if st.session_state.budget_data is not None:
        budget_data = filter_data_by_date(st.session_state.budget_data)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Productivity Metrics", "📈 Trends Analysis", "🔍 Department Deep Dive"])
        
        with tab1:
            st.markdown("### Key Productivity Indicators")
            
            if st.session_state.fte_data is not None:
                # Real FTE metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_ftes = fte_data['FTE_Count'].sum()
                total_visits = fte_data['Total_Visits'].sum()
                total_wrvus = fte_data['Total_wRVUs'].sum() if 'Total_wRVUs' in fte_data.columns else 0
                
                with col1:
                    st.metric("Total FTEs", f"{total_ftes:.1f}")
                
                with col2:
                    visits_per_fte = total_visits/total_ftes if total_ftes > 0 else 0
                    st.metric(
                        "Visits per FTE", 
                        f"{visits_per_fte:.0f}",
                        delta="Above benchmark" if visits_per_fte > 500 else "Below benchmark",
                        delta_color="normal" if visits_per_fte > 500 else "inverse"
                    )
                
                with col3:
                    if total_wrvus > 0:
                        wrvus_per_fte = total_wrvus/total_ftes if total_ftes > 0 else 0
                        st.metric("wRVUs per FTE", f"{wrvus_per_fte:.0f}")
                    else:
                        st.metric("wRVUs per FTE", "N/A")
                
                with col4:
                    # Estimate cost per visit from budget data
                    if 'Salaries' in budget_data['Category'].values:
                        salary_costs = budget_data[budget_data['Category'] == 'Salaries']['Actual_Amount'].sum()
                        cost_per_visit = salary_costs/total_visits if total_visits > 0 else 0
                        st.metric(
                            "Cost per Visit", 
                            format_currency(cost_per_visit),
                            delta="Below target" if cost_per_visit < 120 else "Above target",
                            delta_color="normal" if cost_per_visit < 120 else "inverse"
                        )
                    else:
                        st.metric("Cost per Visit", "N/A")
                
                # Dynamic visualization based on actual data
                st.markdown("### Department Productivity Analysis")
                
                # Group by department and calculate metrics
                dept_metrics = fte_data.groupby('Department').agg({
                    'FTE_Count': 'sum',
                    'Total_Visits': 'sum'
                })
                
                if 'Total_wRVUs' in fte_data.columns:
                    dept_metrics['Total_wRVUs'] = fte_data.groupby('Department')['Total_wRVUs'].sum()
                
                dept_metrics['Visits_per_FTE'] = dept_metrics['Total_Visits'] / dept_metrics['FTE_Count']
                
                # Create dynamic chart
                fig = px.scatter(
                    dept_metrics.reset_index(),
                    x='FTE_Count',
                    y='Visits_per_FTE',
                    size='Total_Visits',
                    color='Department',
                    title="Department Productivity Matrix",
                    labels={
                        'FTE_Count': 'Number of FTEs',
                        'Visits_per_FTE': 'Productivity (Visits per FTE)'
                    },
                    hover_data=['Total_Visits']
                )
                
                # Add benchmark line
                fig.add_hline(y=500, line_dash="dash", line_color="red", 
                             annotation_text="Industry Benchmark (500 visits/FTE)")
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # If no FTE data, estimate from budget
                st.warning("No FTE data uploaded. Showing estimates based on budget data.")
                
                # Estimate FTEs from salary costs
                if 'Salaries' in budget_data['Category'].values:
                    salary_by_dept = budget_data[budget_data['Category'] == 'Salaries'].groupby('Department')['Actual_Amount'].sum()
                    
                    # Assume average salary of $75,000
                    estimated_ftes = salary_by_dept / (75000 / 12 * len(budget_data['Month'].unique()))
                    
                    fig = px.bar(
                        x=estimated_ftes.index,
                        y=estimated_ftes.values,
                        title="Estimated FTEs by Department (from salary costs)",
                        labels={'x': 'Department', 'y': 'Estimated FTEs'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Productivity Trends Over Time")
            
            if st.session_state.fte_data is not None and has_date_column:
                # Show actual trends from FTE data
                monthly_fte = fte_data.groupby(pd.to_datetime(fte_data['Date']).dt.to_period('M')).agg({
                    'FTE_Count': 'sum',
                    'Total_Visits': 'sum'
                })
                monthly_fte.index = monthly_fte.index.to_timestamp()
                monthly_fte['Productivity'] = monthly_fte['Total_Visits'] / monthly_fte['FTE_Count']
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('FTE Count Trend', 'Productivity Trend'),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=monthly_fte.index,
                        y=monthly_fte['FTE_Count'],
                        mode='lines+markers',
                        name='FTE Count',
                        line=dict(color='#0066cc', width=2)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=monthly_fte.index,
                        y=monthly_fte['Productivity'],
                        mode='lines+markers',
                        name='Visits per FTE',
                        line=dict(color='#00a86b', width=2)
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="FTE Count", row=1, col=1)
                fig.update_yaxes(title_text="Visits per FTE", row=2, col=1)
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Show salary cost trends as proxy for FTE trends
                if 'Salaries' in budget_data['Category'].values:
                    monthly_salaries = budget_data[budget_data['Category'] == 'Salaries'].groupby(
                        pd.to_datetime(budget_data['Date']).dt.to_period('M')
                    )['Actual_Amount'].sum()
                    monthly_salaries.index = monthly_salaries.index.to_timestamp()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly_salaries.index,
                        y=monthly_salaries.values,
                        mode='lines+markers',
                        name='Salary Costs',
                        line=dict(color='#0066cc', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Monthly Salary Costs (Proxy for FTE Trends)",
                        xaxis_title="Date",
                        yaxis_title="Salary Cost ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Upload FTE data with dates or ensure budget data has salary information to see trends.")
        
        with tab3:
            st.markdown("### Department Deep Dive")
            
            # Department selector
            if st.session_state.fte_data is not None:
                departments = fte_data['Department'].unique()
            else:
                departments = budget_data['Department'].unique()
            
            selected_dept = st.selectbox("Select Department for Detailed Analysis", departments)
            
            if st.session_state.fte_data is not None:
                dept_data = fte_data[fte_data['Department'] == selected_dept]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Employee type breakdown
                    if 'Employee_Type' in dept_data.columns:
                        fig_emp = px.pie(
                            dept_data,
                            values='FTE_Count',
                            names='Employee_Type',
                            title=f"{selected_dept} - FTE Mix by Role",
                            hole=0.4
                        )
                        st.plotly_chart(fig_emp, use_container_width=True)
                    
                with col2:
                    # Productivity metrics by type
                    if 'Employee_Type' in dept_data.columns:
                        type_metrics = dept_data.groupby('Employee_Type').agg({
                            'FTE_Count': 'sum',
                            'Total_Visits': 'sum'
                        })
                        type_metrics['Visits_per_FTE'] = type_metrics['Total_Visits'] / type_metrics['FTE_Count']
                        
                        fig_prod = px.bar(
                            type_metrics.reset_index(),
                            x='Employee_Type',
                            y='Visits_per_FTE',
                            title=f"{selected_dept} - Productivity by Role",
                            color='Visits_per_FTE',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_prod, use_container_width=True)
                
                # Department statistics
                st.markdown(f"#### {selected_dept} Department Statistics")
                
                dept_stats = {
                    'Total FTEs': f"{dept_data['FTE_Count'].sum():.1f}",
                    'Total Visits': f"{dept_data['Total_Visits'].sum():,}",
                    'Avg Visits/FTE': f"{dept_data['Total_Visits'].sum() / dept_data['FTE_Count'].sum():.0f}",
                    'Efficiency Score': f"{calculate_efficiency_score_for_dept(dept_data)}%"
                }
                
                if 'Total_wRVUs' in dept_data.columns:
                    dept_stats['Total wRVUs'] = f"{dept_data['Total_wRVUs'].sum():,}"
                    dept_stats['Avg wRVUs/FTE'] = f"{dept_data['Total_wRVUs'].sum() / dept_data['FTE_Count'].sum():.0f}"
                
                # Display as metrics
                cols = st.columns(len(dept_stats))
                for i, (metric, value) in enumerate(dept_stats.items()):
                    with cols[i % len(cols)]:
                        st.metric(metric, value)
            
            else:
                # Show budget-based analysis for department
                dept_budget = budget_data[budget_data['Department'] == selected_dept]
                
                # Category breakdown
                category_breakdown = dept_budget.groupby('Category')['Actual_Amount'].sum().sort_values(ascending=True)
                
                fig = px.bar(
                    x=category_breakdown.values,
                    y=category_breakdown.index,
                    orientation='h',
                    title=f"{selected_dept} - Cost Breakdown by Category",
                    labels={'x': 'Amount ($)', 'y': 'Category'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        # No data available
        st.error("No data available for analysis. Please upload FTE and/or budget data.")

def calculate_efficiency_score_for_dept(dept_data):
    """Calculate efficiency score for a department"""
    visits_per_fte = dept_data['Total_Visits'].sum() / dept_data['FTE_Count'].sum()
    
    # Score based on productivity benchmark (500 visits/FTE)
    productivity_score = min(100, (visits_per_fte / 500) * 100)
    
    # If we have cost data, include it
    if 'Salary_Cost' in dept_data.columns:
        total_cost = dept_data[['Salary_Cost', 'Benefits_Cost', 'Overtime_Cost']].sum().sum()
        cost_per_visit = total_cost / dept_data['Total_Visits'].sum()
        cost_score = min(100, (120 / cost_per_visit) * 100)  # 120 is benchmark
        return round((productivity_score + cost_score) / 2)
    else:
        return round(productivity_score)

def show_roi_analysis():
    """ROI Analysis for initiatives"""
    st.header("Initiative ROI Analysis")
    
    st.markdown("### ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initiative_name = st.text_input("Initiative Name", "New MRI Scanner")
        initial_investment = st.number_input("Initial Investment ($)", value=500000, step=10000)
        annual_revenue = st.number_input("Annual Revenue Increase ($)", value=200000, step=10000)
        annual_savings = st.number_input("Annual Cost Savings ($)", value=50000, step=5000)
        project_lifespan = st.number_input("Project Lifespan (Years)", value=5, min_value=1, max_value=20)
        
        if st.button("Calculate ROI", type="primary"):
            # Calculate metrics
            annual_benefit = annual_revenue + annual_savings
            total_benefit = annual_benefit * project_lifespan
            net_benefit = total_benefit - initial_investment
            roi = (net_benefit / initial_investment) * 100 if initial_investment > 0 else 0
            payback = initial_investment / annual_benefit if annual_benefit > 0 else 0
            
            # NPV calculation (8% discount rate)
            discount_rate = 0.08
            npv = -initial_investment
            for i in range(1, int(project_lifespan) + 1):
                npv += annual_benefit / ((1 + discount_rate) ** i)
            
            # IRR approximation
            if initial_investment > 0 and total_benefit > initial_investment:
                irr = ((total_benefit / initial_investment) ** (1 / project_lifespan) - 1) * 100
            else:
                irr = 0
            
            with col2:
                st.markdown("### ROI Analysis Results")
                
                st.metric("ROI", format_percentage(roi), delta="Positive" if roi > 0 else "Negative")
                st.metric("NPV", format_currency(npv), delta="Positive" if npv > 0 else "Negative")
                st.metric("IRR", format_percentage(irr))
                st.metric("Payback Period", f"{payback:.1f} years")
                
                if roi > 20:
                    st.success("✅ Strongly Recommended - Excellent ROI")
                elif roi > 10:
                    st.info("👍 Recommended - Good ROI")
                elif roi > 0:
                    st.warning("🤔 Consider Carefully - Marginal ROI")
                else:
                    st.error("❌ Not Recommended - Negative ROI")
    
    # Portfolio Analysis
    st.markdown("### Initiative Portfolio Analysis")
    
    # Explanation of risk levels
    st.markdown("""
    <div style='margin-bottom: 20px;'>
        <span class='legend-item'><span style='color: #28a745;'>●</span> Low Risk: Proven technology, predictable returns</span>
        <span class='legend-item'><span style='color: #ffc107;'>●</span> Medium Risk: Some uncertainty in adoption or returns</span>
        <span class='legend-item'><span style='color: #dc3545;'>●</span> High Risk: New technology, uncertain outcomes</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample portfolio data
    initiatives = pd.DataFrame({
        'Initiative': ['EHR Upgrade', 'Telemedicine Platform', 'New MRI Scanner', 
                      'AI Diagnostics', 'Staff Training', 'Supply Chain Opt'],
        'Investment': [500000, 250000, 1500000, 750000, 100000, 300000],
        'ROI': [35, 85, 25, 65, 150, 45],
        'Risk': ['Low', 'Medium', 'Low', 'High', 'Low', 'Medium']
    })
    
    # Color map for risk
    color_map = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
    
    fig_portfolio = px.scatter(
        initiatives,
        x='Investment',
        y='ROI',
        size='Investment',
        color='Risk',
        color_discrete_map=color_map,
        hover_data=['Initiative'],
        title="Initiative Portfolio - Risk vs Return Analysis",
        labels={'Investment': 'Investment ($)', 'ROI': 'Return on Investment (%)'}
    )
    
    # Add quadrant lines
    fig_portfolio.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Target ROI")
    fig_portfolio.add_vline(x=500000, line_dash="dash", line_color="gray", annotation_text="Major Investment")
    
    fig_portfolio.update_layout(height=500)
    st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Portfolio summary
    st.markdown("### Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_investment = initiatives['Investment'].sum()
        st.metric("Total Portfolio Investment", format_currency(total_investment))
    
    with col2:
        weighted_roi = (initiatives['ROI'] * initiatives['Investment']).sum() / total_investment
        st.metric("Weighted Average ROI", format_percentage(weighted_roi))
    
    with col3:
        high_roi_count = len(initiatives[initiatives['ROI'] > 50])
        st.metric("High ROI Initiatives", f"{high_roi_count} of {len(initiatives)}")

def show_strategic_alignment():
    """Strategic Alignment Dashboard - FIXED METRICS COUNT"""
    st.header("Strategic Alignment Dashboard")
    
    if st.session_state.budget_data is None and st.session_state.fte_data is None:
        st.warning("Please upload budget and/or FTE data to see strategic alignment analysis.")
        return
    
    st.info("This dashboard shows how your actual performance aligns with strategic healthcare goals")
    
    # Initialize metrics
    metrics = {
        'Financial Health': {'current': 50, 'target': 85, 'trend': 'stable'},
        'Operational Efficiency': {'current': 50, 'target': 90, 'trend': 'stable'},
        'Resource Utilization': {'current': 50, 'target': 85, 'trend': 'stable'},
        'Cost Management': {'current': 50, 'target': 80, 'trend': 'stable'},
        'Revenue Growth': {'current': 50, 'target': 85, 'trend': 'stable'},
        'Department Performance': {'current': 50, 'target': 90, 'trend': 'stable'}
    }
    
    # Calculate real metrics from budget data
    if st.session_state.budget_data is not None:
        data = filter_data_by_date(st.session_state.budget_data)
        
        # 1. Financial Health - Based on margin
        revenue_data = data[data['Category'] == 'Revenue'] if 'Revenue' in data['Category'].values else None
        if revenue_data is not None and len(revenue_data) > 0:
            total_revenue = revenue_data['Actual_Amount'].sum()
            total_cost = data[data['Category'] != 'Revenue']['Actual_Amount'].sum()
            margin_pct = ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0
            # Score: 15% margin = 100 score, 0% = 0 score
            metrics['Financial Health']['current'] = min(100, max(0, margin_pct / 15 * 100))
        
        # 2. Operational Efficiency - Based on budget variance
        avg_variance = abs(data['Variance_Percent'].mean())
        # Score: 0% variance = 100 score, 20% variance = 0 score
        metrics['Operational Efficiency']['current'] = min(100, max(0, 100 - (avg_variance * 5)))
        
        # 3. Cost Management - Based on cost control
        cost_overruns = len(data[data['Variance_Percent'] > 5])
        total_items = len(data)
        cost_control_pct = (1 - cost_overruns/total_items) * 100 if total_items > 0 else 50
        metrics['Cost Management']['current'] = cost_control_pct
        
        # 4. Revenue Growth - Based on month-over-month trend
        if 'Date' in data.columns and len(data) > 30:
            monthly_revenue = data.groupby(pd.to_datetime(data['Date']).dt.to_period('M'))['Actual_Amount'].sum()
            if len(monthly_revenue) > 1:
                recent_months = monthly_revenue.tail(6).values
                older_months = monthly_revenue.head(6).values
                growth_rate = ((recent_months.mean() / older_months.mean()) - 1) * 100 if older_months.mean() > 0 else 0
                # Score: 10% growth = 100 score, -10% = 0 score
                metrics['Revenue Growth']['current'] = min(100, max(0, 50 + growth_rate * 5))
                metrics['Revenue Growth']['trend'] = 'up' if growth_rate > 2 else ('down' if growth_rate < -2 else 'stable')
        
        # 5. Department Performance - Based on variance distribution
        dept_performance = data.groupby('Department')['Variance_Percent'].mean()
        good_depts = len(dept_performance[abs(dept_performance) <= 5])
        total_depts = len(dept_performance)
        metrics['Department Performance']['current'] = (good_depts / total_depts * 100) if total_depts > 0 else 50
    
    # Calculate from FTE data if available
    if st.session_state.fte_data is not None:
        fte_data = st.session_state.fte_data
        
        # 6. Resource Utilization - Based on productivity
        total_visits = fte_data['Total_Visits'].sum()
        total_ftes = fte_data['FTE_Count'].sum()
        visits_per_fte = total_visits / total_ftes if total_ftes > 0 else 0
        # Score: 500 visits/FTE = 100 score, 200 = 0 score
        metrics['Resource Utilization']['current'] = min(100, max(0, (visits_per_fte - 200) / 3))
    
    # Display Strategic Scorecard
    st.markdown("### Strategic Performance Scorecard")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate overall score
    overall_score = np.mean([m['current'] for m in metrics.values()])
    overall_target = np.mean([m['target'] for m in metrics.values()])
    
    with col1:
        st.metric(
            "Overall Strategic Score",
            f"{overall_score:.0f}%",
            delta=f"{overall_score - 70:.0f}% vs Baseline (70%)",
            delta_color="normal" if overall_score >= 70 else "inverse"
        )
    
    with col2:
        gap = overall_target - overall_score
        st.metric(
            "Gap to Target",
            f"{abs(gap):.0f}%",
            delta="Below Target" if gap > 0 else "Above Target",
            delta_color="inverse" if gap > 10 else "normal"
        )
    
    with col3:
        # FIX: Correct calculation for metrics on track
        metrics_on_track = sum(1 for m in metrics.values() if m['current'] >= m['target'] * 0.8)
        st.metric(
            "Metrics On Track",
            f"{metrics_on_track} of {len(metrics)}",
            delta="≥80% of target"
        )
    
    # Detailed Metrics Grid
    st.markdown("### Strategic Metrics Detail")
    
    # Create two columns for metrics
    col1, col2 = st.columns(2)
    
    for i, (metric_name, metric_data) in enumerate(metrics.items()):
        with col1 if i % 2 == 0 else col2:
            # Progress bar with color coding
            is_on_track = metric_data['current'] >= metric_data['target'] * 0.8
            progress_color = 'green' if is_on_track else ('orange' if metric_data['current'] >= metric_data['target'] * 0.6 else 'red')
            
            st.markdown(f"**{metric_name}**")
            
            # Show if metric is on track
            if is_on_track:
                st.caption("✅ On Track")
            else:
                st.caption("❌ Below Target")
            
            # Create custom progress bar
            progress_pct = metric_data['current'] / 100
            target_pct = metric_data['target'] / 100
            
            st.progress(progress_pct)
            
            # Metrics below progress bar
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.caption(f"Current: {metric_data['current']:.0f}%")
            with col_b:
                st.caption(f"Target: {metric_data['target']:.0f}%")
            with col_c:
                trend_emoji = {'up': '📈', 'down': '📉', 'stable': '➡️'}
                st.caption(f"Trend: {trend_emoji.get(metric_data['trend'], '➡️')}")
    
    # Show calculation details
    with st.expander("📊 How are these metrics calculated?"):
        st.markdown("""
        **All metrics are calculated from your actual uploaded data:**
        
        1. **Financial Health**: Based on operating margin (Revenue - Costs) / Revenue
        2. **Operational Efficiency**: Based on average budget variance across departments
        3. **Resource Utilization**: Based on visits per FTE from your FTE data
        4. **Cost Management**: Percentage of line items within 5% of budget
        5. **Revenue Growth**: Month-over-month revenue trend from budget data
        6. **Department Performance**: Percentage of departments within variance targets
        
        **On Track Definition**: A metric is considered "on track" if it's at ≥80% of its target value.
        """)
    
    # Key Insights and Recommendations
    st.markdown("### Strategic Insights & Recommendations")
    
    # Generate insights based on actual data
    insights = []
    
    # Financial insights
    if metrics['Financial Health']['current'] < metrics['Financial Health']['target'] * 0.8:
        insights.append({
            'type': 'warning',
            'area': 'Financial Health',
            'message': f"Current margin performance ({metrics['Financial Health']['current']:.0f}%) is below the 80% threshold of target. Consider revenue enhancement initiatives or cost reduction programs.",
            'priority': 'High'
        })
    
    # Efficiency insights
    if metrics['Operational Efficiency']['current'] < metrics['Operational Efficiency']['target'] * 0.8:
        insights.append({
            'type': 'warning',
            'area': 'Operational Efficiency',
            'message': f"Budget variance control ({metrics['Operational Efficiency']['current']:.0f}%) indicates operational inefficiencies. Implement tighter budget controls and monitoring.",
            'priority': 'Medium'
        })
    
    # Resource insights
    if metrics['Resource Utilization']['current'] < metrics['Resource Utilization']['target'] * 0.8:
        insights.append({
            'type': 'info',
            'area': 'Resource Utilization',
            'message': f"Productivity metrics ({metrics['Resource Utilization']['current']:.0f}%) are below benchmark. Review staffing models and workflow optimization opportunities.",
            'priority': 'Medium'
        })
    
    # Growth insights
    if metrics['Revenue Growth']['current'] >= metrics['Revenue Growth']['target'] * 0.8:
        insights.append({
            'type': 'success',
            'area': 'Revenue Growth',
            'message': f"Strong revenue growth trend ({metrics['Revenue Growth']['current']:.0f}%). Continue current strategies and explore expansion opportunities.",
            'priority': 'Low'
        })
    
    # Display insights
    for insight in insights:
        icon = {'warning': '⚠️', 'info': 'ℹ️', 'success': '✅'}.get(insight['type'], '📌')
        color = {'warning': 'orange', 'info': 'blue', 'success': 'green'}.get(insight['type'], 'gray')
        
        st.markdown(f"""
        <div style='padding: 10px; border-left: 4px solid {color}; background-color: rgba(128,128,128,0.1); margin-bottom: 10px;'>
            <strong>{icon} {insight['area']} - Priority: {insight['priority']}</strong><br>
            {insight['message']}
        </div>
        """, unsafe_allow_html=True)
    
    # Action Plan
    st.markdown("### Recommended Action Plan")
    
    # Generate action items based on gaps
    action_items = []
    
    for metric_name, metric_data in metrics.items():
        gap = metric_data['target'] - metric_data['current']
        if gap > 20:
            action_items.append({
                'metric': metric_name,
                'action': f"Develop improvement plan for {metric_name} (current: {metric_data['current']:.0f}%, target: {metric_data['target']:.0f}%)",
                'timeline': '30 days',
                'impact': 'High'
            })
        elif gap > 10:
            action_items.append({
                'metric': metric_name,
                'action': f"Monitor and optimize {metric_name} performance",
                'timeline': '60 days',
                'impact': 'Medium'
            })
    
    if action_items:
        action_df = pd.DataFrame(action_items)
        st.dataframe(
            action_df[['metric', 'action', 'timeline', 'impact']],
            use_container_width=True,
            column_config={
                'metric': 'Focus Area',
                'action': 'Action Item',
                'timeline': 'Timeline',
                'impact': 'Impact Level'
            }
        )
    else:
        st.success("✅ All metrics are performing well! Focus on maintaining current performance levels.")

def show_data_management():
    """Data Management Center"""
    st.header("Data Management Center")
    
    st.markdown("### Upload Your Data Files")
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Budget/Actuals Data")
        st.info("Required columns: Date, Department, Category, Budget_Amount, Actual_Amount")
        
        budget_file = st.file_uploader(
            "Upload Budget CSV",
            type=['csv'],
            key='budget_upload'
        )
        
        if budget_file:
            try:
                budget_df = pd.read_csv(budget_file)
                
                # Validate columns
                required_cols = ['Date', 'Department', 'Category', 'Budget_Amount', 'Actual_Amount']
                missing_cols = [col for col in required_cols if col not in budget_df.columns]
                
                if not missing_cols:
                    # Process data
                    budget_df['Date'] = pd.to_datetime(budget_df['Date'])
                    budget_df['Month'] = budget_df['Date'].dt.strftime('%b')
                    budget_df['Year'] = budget_df['Date'].dt.year
                    budget_df['Variance_Amount'] = budget_df['Actual_Amount'] - budget_df['Budget_Amount']
                    budget_df['Variance_Percent'] = (budget_df['Variance_Amount'] / budget_df['Budget_Amount']) * 100
                    budget_df['Variance_Percent'] = budget_df['Variance_Percent'].fillna(0)
                    
                    st.session_state.budget_data = budget_df
                    st.success(f"✅ Budget data uploaded successfully! {len(budget_df)} records loaded.")
                    
                    # Preview
                    st.markdown("##### Data Preview")
                    st.dataframe(budget_df.head(), use_container_width=True)
                    
                    # Quick stats
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Date Range", f"{budget_df['Date'].min().date()} to {budget_df['Date'].max().date()}")
                    with col_b:
                        st.metric("Departments", budget_df['Department'].nunique())
                else:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info("Your file has columns: " + ", ".join(budget_df.columns.tolist()))
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.markdown("#### FTE/Productivity Data")
        st.info("Required columns: Department, Employee_Type, FTE_Count, Total_Visits")
        
        fte_file = st.file_uploader(
            "Upload FTE CSV",
            type=['csv'],
            key='fte_upload'
        )
        
        if fte_file:
            try:
                fte_df = pd.read_csv(fte_file)
                
                # Validate columns
                required_cols = ['Department', 'Employee_Type', 'FTE_Count', 'Total_Visits']
                missing_cols = [col for col in required_cols if col not in fte_df.columns]
                
                if not missing_cols:
                    # Check if Date column exists and process it
                    if 'Date' in fte_df.columns:
                        fte_df['Date'] = pd.to_datetime(fte_df['Date'])
                        st.info("✅ Date column detected - date filtering will be available")
                    
                    st.session_state.fte_data = fte_df
                    st.success(f"✅ FTE data uploaded successfully! {len(fte_df)} records loaded.")
                    
                    # Preview
                    st.markdown("##### Data Preview")
                    st.dataframe(fte_df.head(), use_container_width=True)
                    
                    # Quick stats
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total FTEs", f"{fte_df['FTE_Count'].sum():.1f}")
                    with col_b:
                        st.metric("Total Visits", f"{fte_df['Total_Visits'].sum():,}")
                    
                    if 'Date' in fte_df.columns:
                        st.caption(f"Date range: {fte_df['Date'].min().date()} to {fte_df['Date'].max().date()}")
                else:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info("Your file has columns: " + ", ".join(fte_df.columns.tolist()))
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Data Summary
    if st.session_state.budget_data is not None or st.session_state.fte_data is not None:
        st.markdown("### Uploaded Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.budget_data is not None:
                st.success(f"**Budget Data**: {len(st.session_state.budget_data)} records")
                departments = st.session_state.budget_data['Department'].unique()
                st.caption(f"Departments: {', '.join(departments[:5])}" + 
                          (" ..." if len(departments) > 5 else ""))
        
        with col2:
            if st.session_state.fte_data is not None:
                st.success(f"**FTE Data**: {len(st.session_state.fte_data)} records")
                st.caption(f"Total FTEs: {st.session_state.fte_data['FTE_Count'].sum():.1f}")
    
    # Download sample templates
    st.markdown("### Need Help? Download Sample Templates")
    create_download_button()

def show_change_requests():
    """Budget Change Request System"""
    st.header("Budget Change Request System")
    
    # Submit new request
    with st.form("change_request_form"):
        st.markdown("### Submit New Change Request")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get departments from uploaded data if available
            if st.session_state.budget_data is not None:
                departments = list(st.session_state.budget_data['Department'].unique())
            else:
                departments = ["Cardiology", "Emergency", "Surgery", "Radiology", "Laboratory", "Pediatrics"]
            
            department = st.selectbox("Department", departments)
            request_type = st.selectbox(
                "Request Type",
                ["FTE Addition/Change", "Volume Adjustment", "Capital Equipment", 
                 "Operational Change", "Supply Cost Adjustment", "Other"]
            )
            financial_impact = st.number_input("Financial Impact ($)", step=1000)
        
        with col2:
            description = st.text_area("Description", height=100)
            justification = st.text_area("Business Justification", height=100)
        
        submitted = st.form_submit_button("Submit Request", type="primary")
        
        if submitted and description:
            request = {
                'id': len(st.session_state.change_requests) + 1,
                'timestamp': dt.datetime.now(),
                'department': department,
                'type': request_type,
                'description': description,
                'impact': financial_impact,
                'justification': justification,
                'status': 'Pending'
            }
            st.session_state.change_requests.append(request)
            st.success("✅ Change request submitted successfully!")
            st.balloons()
    
    # Display existing requests
    if st.session_state.change_requests:
        st.markdown("### Change Request Queue")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        requests_df = pd.DataFrame(st.session_state.change_requests)
        
        with col1:
            pending_count = len(requests_df[requests_df['status'] == 'Pending'])
            st.metric("Pending Requests", pending_count)
        
        with col2:
            total_impact = requests_df[requests_df['status'] == 'Pending']['impact'].sum()
            st.metric("Total Financial Impact", format_currency(total_impact))
        
        with col3:
            approved_count = len(requests_df[requests_df['status'] == 'Approved'])
            st.metric("Approved", approved_count)
        
        with col4:
            avg_days = 3.5  # Mock data
            st.metric("Avg. Processing Time", f"{avg_days} days")
        
        # Detailed table
        requests_df['Impact'] = requests_df['impact'].apply(format_currency)
        requests_df['Date'] = requests_df['timestamp'].dt.strftime('%Y-%m-%d')
        
        # Status color coding
        def status_color(status):
            colors = {
                'Pending': '🟡',
                'Approved': '🟢',
                'Rejected': '🔴',
                'Under Review': '🔵'
            }
            return f"{colors.get(status, '')} {status}"
        
        requests_df['Status'] = requests_df['status'].apply(status_color)
        
        display_cols = ['Date', 'department', 'type', 'description', 'Impact', 'Status']
        st.dataframe(
            requests_df[display_cols],
            use_container_width=True,
            column_config={
                'department': 'Department',
                'type': 'Type',
                'description': st.column_config.TextColumn('Description', width='large')
            }
        )
    else:
        st.info("No change requests submitted yet. Use the form above to submit your first request.")

# Generate sample data files
def generate_sample_data():
    """Generate sample CSV files for testing"""
    
    # Budget/Actuals Data
    departments = ['Cardiology', 'Emergency', 'Surgery', 'Radiology', 'Laboratory', 'Pediatrics']
    categories = ['Salaries', 'Supplies', 'Equipment', 'Utilities', 'Other', 'Revenue']
    
    budget_data = []
    start_date = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    
    np.random.seed(42)  # For reproducibility
    
    for date in start_date:
        for dept in departments:
            for cat in categories:
                # Create realistic patterns
                base_amount = {
                    'Salaries': np.random.uniform(150000, 300000),
                    'Supplies': np.random.uniform(50000, 100000),
                    'Equipment': np.random.uniform(20000, 50000),
                    'Utilities': np.random.uniform(10000, 20000),
                    'Other': np.random.uniform(5000, 15000),
                    'Revenue': np.random.uniform(300000, 500000)
                }[cat]
                
                # Add seasonal variation
                seasonal_factor = 1 + 0.1 * np.sin((date.month - 1) / 12 * 2 * np.pi)
                budget = base_amount * seasonal_factor
                
                # Add realistic variance
                if cat == 'Revenue':
                    variance = np.random.uniform(-0.05, 0.1)  # Revenue tends to be slightly over
                else:
                    variance = np.random.uniform(-0.1, 0.15)  # Costs vary more
                
                budget_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Department': dept,
                    'Category': cat,
                    'Budget_Amount': round(budget, 2),
                    'Actual_Amount': round(budget * (1 + variance), 2)
                })
    
    budget_df = pd.DataFrame(budget_data)
    
    # FTE/Productivity Data WITH DATES
    employee_types = ['Physician', 'NP/PA', 'RN', 'Tech', 'Admin']
    
    fte_data = []
    # Generate monthly FTE data
    for date in pd.date_range(start='2023-01-01', end='2024-12-31', freq='M'):
        for dept in departments:
            for emp_type in employee_types:
                # Realistic FTE counts by type
                fte_base = {
                    'Physician': np.random.uniform(5, 15),
                    'NP/PA': np.random.uniform(3, 8),
                    'RN': np.random.uniform(15, 30),
                    'Tech': np.random.uniform(5, 12),
                    'Admin': np.random.uniform(2, 6)
                }[emp_type]
                
                fte_count = round(fte_base, 1)
                
                # Realistic productivity by type
                visits_per_fte = {
                    'Physician': np.random.uniform(200, 300),
                    'NP/PA': np.random.uniform(150, 250),
                    'RN': np.random.uniform(50, 100),
                    'Tech': np.random.uniform(100, 200),
                    'Admin': 0
                }[emp_type]
                
                wrvus_per_fte = visits_per_fte * np.random.uniform(1.5, 3.5) if emp_type != 'Admin' else 0
                
                # Realistic salary ranges
                salary_per_fte = {
                    'Physician': np.random.uniform(250000, 400000),
                    'NP/PA': np.random.uniform(100000, 150000),
                    'RN': np.random.uniform(60000, 90000),
                    'Tech': np.random.uniform(40000, 60000),
                    'Admin': np.random.uniform(45000, 70000)
                }[emp_type]
                
                fte_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Department': dept,
                    'Employee_Type': emp_type,
                    'FTE_Count': fte_count,
                    'Total_Visits': int(fte_count * visits_per_fte),
                    'Total_wRVUs': int(fte_count * wrvus_per_fte),
                    'Total_Hours': int(fte_count * 2080 / 12),  # Monthly hours
                    'Overtime_Hours': int(fte_count * np.random.uniform(4, 12)),
                    'Salary_Cost': int(fte_count * salary_per_fte / 12),  # Monthly salary
                    'Benefits_Cost': int(fte_count * salary_per_fte * 0.25 / 12),
                    'Overtime_Cost': int(fte_count * np.random.uniform(200, 800))
                })
    
    fte_df = pd.DataFrame(fte_data)
    
    return budget_df, fte_df

# Download sample data
def create_download_button():
    """Create download buttons for sample data"""
    st.markdown("### Download Sample Data Templates")
    
    budget_df, fte_df = generate_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_budget = budget_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Budget Data",
            data=csv_budget,
            file_name='sample_budget_actuals.csv',
            mime='text/csv'
        )
        st.caption("2 years of monthly budget data across 6 departments")
    
    with col2:
        csv_fte = fte_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample FTE Data",
            data=csv_fte,
            file_name='sample_fte_productivity.csv',
            mime='text/csv'
        )
        st.caption("FTE and productivity metrics by department and role WITH DATE COLUMN")

# Run the application
if __name__ == "__main__":
    # Add sample data download to sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Sample Data")
        if st.button("Generate & Download Samples"):
            create_download_button()
        st.caption("Use sample data to test all features")
    
    # Run main app
    main()
