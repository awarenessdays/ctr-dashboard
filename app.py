import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Journey Further - CTR Study",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Journey Further branding
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #e0e7ff;
        margin: 0;
        font-size: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-delta {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and process the CTR data"""
    # Try different possible filenames
    possible_filenames = [
        'CTR_Study_data.csv',
        'CTR Study - data.csv',
        'ctr-study-data.csv',
        'data.csv',
        'CTR_data.csv'
    ]
    
    for filename in possible_filenames:
        try:
            # Read CSV with special handling for quoted numbers and commas
            df = pd.read_csv(
                filename,
                thousands=',',
                quotechar='"',
                skipinitialspace=True,
                na_values=['', 'NULL', 'null', 'NaN'],
                keep_default_na=True
            )
            
            # Check if dataframe is empty
            if df.empty:
                st.warning(f"⚠️ File {filename} is empty.")
                continue
                
            # Check if required columns exist
            required_columns = ['Date', 'Primary Industry', 'Clicks', 'Impressions', 'CTR']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing columns in {filename}: {missing_columns}")
                st.write(f"Available columns: {list(df.columns)}")
                continue
            
            st.success(f"✅ Successfully loaded {len(df)} rows from: {filename}")
            break
            
        except pd.errors.EmptyDataError:
            st.error(f"❌ File {filename} is empty or corrupted.")
            continue
        except pd.errors.ParserError as e:
            st.error(f"❌ Parse error in {filename}: {str(e)}")
            continue
        except FileNotFoundError:
            continue
        except Exception as e:
            st.error(f"❌ Error reading {filename}: {str(e)}")
            continue
    else:
        # If no file found or all failed, show error and use sample data
        st.error("❌ Could not load any CSV file. Please check:")
        st.write("• File exists in the repository")
        st.write("• File is not empty") 
        st.write("• File has the correct format")
        st.warning("Using sample data for demonstration.")
        return create_sample_data()
    
    # Process the data
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Clean numeric columns - handle quoted numbers with commas
        if 'Clicks' in df.columns:
            df['Clicks'] = df['Clicks'].astype(str).str.replace(',', '').str.replace('"', '').astype(float).fillna(0).astype(int)
        
        if 'Impressions' in df.columns:
            df['Impressions'] = df['Impressions'].astype(str).str.replace(',', '').str.replace('"', '').astype(float).fillna(0).astype(int)
        
        if 'CTR' in df.columns:
            df['CTR'] = df['CTR'].astype(str).str.replace('%', '').str.replace('"', '').astype(float).fillna(0)
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        if df.empty:
            st.error("❌ No valid data rows after processing.")
            return create_sample_data()
            
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    dates = pd.date_range('2024-04-21', '2024-05-15', freq='D')
    industries = ['Retail & E-commerce', 'Home & Living', 'Business Services', 'Health & Wellness', 'Technology & Software']
    business_models = ['E-commerce', 'Services', 'B2B Services', 'Marketplace']
    markets = ['UK Primary', 'EU Primary', 'Global', 'US Primary']
    intents = ['Commercial', 'Transactional', 'Informational']
    
    data = []
    site_id = 1
    
    for date in dates:
        for industry in industries:
            for _ in range(np.random.randint(2, 6)):  # 2-5 sites per industry per day
                impressions = np.random.randint(1000, 100000)
                ctr = np.random.uniform(0.5, 8.0)
                clicks = int(impressions * ctr / 100)
                
                data.append({
                    'Date': date,
                    'Site ID': site_id,
                    'Primary Industry': industry,
                    'Business Model': np.random.choice(business_models),
                    'Target Audience': np.random.choice(['B2C Consumer', 'B2B Professional']),
                    'Geographic Market': np.random.choice(markets),
                    'Search Intent Profile': np.random.choice(intents),
                    'Clicks': clicks,
                    'Impressions': impressions,
                    'CTR': round(ctr, 2),
                    'Avg. Position': round(np.random.uniform(10, 40), 1)
                })
                site_id += 1
    
    return pd.DataFrame(data)

def apply_smoothing(df, column, window_size=7):
    """Apply moving average smoothing to a column"""
    return df[column].rolling(window=window_size, center=True, min_periods=1).mean()

def calculate_period_comparison(df):
    """Calculate comparison between first and last period"""
    df_sorted = df.sort_values('Date')
    total_days = (df_sorted['Date'].max() - df_sorted['Date'].min()).days
    split_date = df_sorted['Date'].min() + pd.Timedelta(days=total_days//2)
    
    first_period = df_sorted[df_sorted['Date'] <= split_date]
    last_period = df_sorted[df_sorted['Date'] > split_date]
    
    # Calculate metrics for each period
    first_impressions = first_period['Impressions'].sum()
    last_impressions = last_period['Impressions'].sum()
    first_clicks = first_period['Clicks'].sum()
    last_clicks = last_period['Clicks'].sum()
    
    first_ctr = (first_clicks / first_impressions * 100) if first_impressions > 0 else 0
    last_ctr = (last_clicks / last_impressions * 100) if last_impressions > 0 else 0
    
    return {
        'impressions_change': ((last_impressions - first_impressions) / first_impressions * 100) if first_impressions > 0 else 0,
        'clicks_change': ((last_clicks - first_clicks) / first_clicks * 100) if first_clicks > 0 else 0,
        'ctr_change': last_ctr - first_ctr
    }

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Journey Further - CTR Study</h1>
        <p>Sector Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading CTR data...'):
        df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Filter controls
    industries = ['All'] + sorted(df['Primary Industry'].unique().tolist())
    selected_industry = st.sidebar.selectbox('Primary Industry', industries)
    
    business_models = ['All'] + sorted(df['Business Model'].unique().tolist())
    selected_business_model = st.sidebar.selectbox('Business Model', business_models)
    
    markets = ['All'] + sorted(df['Geographic Market'].unique().tolist())
    selected_market = st.sidebar.selectbox('Geographic Market', markets)
    
    intents = ['All'] + sorted(df['Search Intent Profile'].unique().tolist())
    selected_intent = st.sidebar.selectbox('Search Intent Profile', intents)
    
    # Reset filters button
    if st.sidebar.button('🔄 Reset Filters'):
        st.rerun()
    
    # Data smoothing options
    st.sidebar.header("📈 Data Smoothing")
    enable_smoothing = st.sidebar.checkbox('Enable Data Smoothing')
    if enable_smoothing:
        smoothing_window = st.sidebar.slider('Smoothing Window (days)', 3, 14, 7)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_industry != 'All':
        filtered_df = filtered_df[filtered_df['Primary Industry'] == selected_industry]
    if selected_business_model != 'All':
        filtered_df = filtered_df[filtered_df['Business Model'] == selected_business_model]
    if selected_market != 'All':
        filtered_df = filtered_df[filtered_df['Geographic Market'] == selected_market]
    if selected_intent != 'All':
        filtered_df = filtered_df[filtered_df['Search Intent Profile'] == selected_intent]
    
    if filtered_df.empty:
        st.error("No data available for the selected filters. Please adjust your selection.")
        return
    
    # Calculate KPIs
    total_impressions = filtered_df['Impressions'].sum()
    total_clicks = filtered_df['Clicks'].sum()
    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    
    # Period comparison
    comparison = calculate_period_comparison(filtered_df)
    
    # KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Impressions</h3>
            <div class="metric-value">{total_impressions:,}</div>
            <div class="metric-delta">
                {'🔺' if comparison['impressions_change'] > 0 else '🔻' if comparison['impressions_change'] < 0 else '➖'} 
                {comparison['impressions_change']:+.1f}% vs period start
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Clicks</h3>
            <div class="metric-value">{total_clicks:,}</div>
            <div class="metric-delta">
                {'🔺' if comparison['clicks_change'] > 0 else '🔻' if comparison['clicks_change'] < 0 else '➖'} 
                {comparison['clicks_change']:+.1f}% vs period start
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>CTR</h3>
            <div class="metric-value">{avg_ctr:.2f}%</div>
            <div class="metric-delta">
                {'🔺' if comparison['ctr_change'] > 0 else '🔻' if comparison['ctr_change'] < 0 else '➖'} 
                {comparison['ctr_change']:+.2f}pp vs period start
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prepare daily aggregated data for charts
    daily_data = filtered_df.groupby('Date').agg({
        'Clicks': 'sum',
        'Impressions': 'sum',
        'CTR': 'mean'
    }).reset_index()
    
    # Calculate actual CTR from clicks and impressions
    daily_data['Actual_CTR'] = (daily_data['Clicks'] / daily_data['Impressions'] * 100).round(2)
    
    # Apply smoothing if enabled
    if enable_smoothing:
        daily_data['Clicks_Smooth'] = apply_smoothing(daily_data, 'Clicks', smoothing_window)
        daily_data['Impressions_Smooth'] = apply_smoothing(daily_data, 'Impressions', smoothing_window)
        daily_data['CTR_Smooth'] = apply_smoothing(daily_data, 'Actual_CTR', smoothing_window)
    
    # CTR Trend Chart
    st.subheader("📊 CTR Trend Over Time")
    
    fig_ctr = px.line(
        daily_data, 
        x='Date', 
        y='CTR_Smooth' if enable_smoothing else 'Actual_CTR',
        title='Click-Through Rate Over Time',
        labels={'CTR_Smooth' if enable_smoothing else 'Actual_CTR': 'CTR (%)'},
        template='plotly_white'
    )
    fig_ctr.update_traces(line=dict(color='#06b6d4', width=3))
    fig_ctr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        title_font_size=16
    )
    st.plotly_chart(fig_ctr, use_container_width=True)
    
    # Impressions and Clicks Chart
    st.subheader("📈 Impressions and Clicks Over Time")
    
    # Create subplot with secondary y-axis
    fig_traffic = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Impressions
    fig_traffic.add_trace(
        go.Scatter(
            x=daily_data['Date'], 
            y=daily_data['Impressions_Smooth'] if enable_smoothing else daily_data['Impressions'],
            name='Impressions',
            line=dict(color='#ec4899', width=3)
        ),
        secondary_y=False,
    )
    
    # Add Clicks
    fig_traffic.add_trace(
        go.Scatter(
            x=daily_data['Date'], 
            y=daily_data['Clicks_Smooth'] if enable_smoothing else daily_data['Clicks'],
            name='Clicks',
            line=dict(color='#06b6d4', width=3)
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig_traffic.update_xaxes(title_text="Date")
    fig_traffic.update_yaxes(title_text="Impressions", secondary_y=False, title_font_color="#ec4899")
    fig_traffic.update_yaxes(title_text="Clicks", secondary_y=True, title_font_color="#06b6d4")
    
    fig_traffic.update_layout(
        title="Impressions and Clicks Over Time",
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        title_font_size=16
    )
    
    st.plotly_chart(fig_traffic, use_container_width=True)
    
    # Performance Summary Table
    st.subheader("📋 Performance Summary by Industry")
    
    # Group by industry for summary
    industry_summary = filtered_df.groupby('Primary Industry').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'CTR': 'mean'
    }).reset_index()
    
    # Calculate period comparison for each industry
    industry_summary['Avg_CTR'] = (industry_summary['Clicks'] / industry_summary['Impressions'] * 100).round(2)
    
    # Add period comparison
    industry_changes = []
    for industry in industry_summary['Primary Industry']:
        industry_data = filtered_df[filtered_df['Primary Industry'] == industry]
        if len(industry_data) > 1:
            industry_comparison = calculate_period_comparison(industry_data)
            industry_changes.append(industry_comparison['ctr_change'])
        else:
            industry_changes.append(0)
    
    industry_summary['CTR_Change'] = industry_changes
    industry_summary = industry_summary.sort_values('Impressions', ascending=False)
    
    # Format the summary table
    summary_display = industry_summary.copy()
    summary_display['Total Impressions'] = summary_display['Impressions'].apply(lambda x: f"{x:,}")
    summary_display['Total Clicks'] = summary_display['Clicks'].apply(lambda x: f"{x:,}")
    summary_display['Avg CTR'] = summary_display['Avg_CTR'].apply(lambda x: f"{x:.2f}%")
    summary_display['CTR Change'] = summary_display['CTR_Change'].apply(
        lambda x: f"{'🔺' if x > 0 else '🔻' if x < 0 else '➖'} {x:+.2f}pp"
    )
    
    st.dataframe(
        summary_display[['Primary Industry', 'Total Impressions', 'Total Clicks', 'Avg CTR', 'CTR Change']],
        use_container_width=True,
        hide_index=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown("**Journey Further** - CTR Sector Analysis Dashboard")
    st.markdown(f"*Data points analyzed: {len(filtered_df):,}*")

if __name__ == "__main__":
    main()
