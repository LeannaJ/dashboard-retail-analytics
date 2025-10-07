import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# ---------- Utility Functions ----------
@st.cache_data(show_spinner=False)
def load_data():
    """Load and preprocess data"""
    try:
        # Check if files exist first
        required_files = [
            'transaction_data.csv',
            'hh_demographic.csv', 
            'product.csv',
            'campaign_table.csv',
            'campaign_desc.csv',
            'coupon_redempt.csv',
            'coupon.csv'
        ]
        
        st.info("Checking if all required files exist...")
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"Missing files: {', '.join(missing_files)}")
            st.info("Please ensure all CSV files are in the same directory as the Streamlit app.")
            return None, None, None, None, None
        
        st.success("All required files found!")
        
        # Load data with error handling for each file
        st.info("Loading transaction data...")
        
        # Debug: Check file size and first few bytes
        file_size = os.path.getsize('transaction_data.csv')
        st.info(f"File size: {file_size} bytes")
        
        # Try to read first few lines to debug
        try:
            with open('transaction_data.csv', 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                st.info(f"First line: {first_line[:100]}...")
        except Exception as e:
            st.warning(f"Could not read first line: {e}")
        
        # Try different approaches to load the CSV
        try:
            transaction_df = pd.read_csv('transaction_data.csv', encoding='utf-8')
            st.info(f"Transaction data loaded successfully: {transaction_df.shape}")
            st.info(f"Columns: {list(transaction_df.columns)}")
        except pd.errors.EmptyDataError:
            st.error("EmptyDataError: File appears to be empty")
            # Try with different parameters
            try:
                st.info("Trying with different parameters...")
                transaction_df = pd.read_csv('transaction_data.csv', encoding='utf-8', sep=',', skipinitialspace=True)
                st.success(f"Success with different params: {transaction_df.shape}")
            except Exception as e2:
                st.error(f"Still failed: {e2}")
                raise
        
        st.info("Loading demographic data...")
        demographic_df = pd.read_csv('hh_demographic.csv', encoding='utf-8')
        st.info(f"Demographic data loaded: {demographic_df.shape}")
        
        st.info("Loading product data...")
        product_df = pd.read_csv('product.csv', encoding='utf-8')
        st.info(f"Product data loaded: {product_df.shape}")
        
        st.info("Loading campaign table data...")
        campaign_table_df = pd.read_csv('campaign_table.csv', encoding='utf-8')
        st.info(f"Campaign table data loaded: {campaign_table_df.shape}")
        
        st.info("Loading campaign description data...")
        campaign_desc_df = pd.read_csv('campaign_desc.csv', encoding='utf-8')
        st.info(f"Campaign description data loaded: {campaign_desc_df.shape}")
        
        st.info("Loading coupon redemption data...")
        coupon_redempt_df = pd.read_csv('coupon_redempt.csv', encoding='utf-8')
        st.info(f"Coupon redemption data loaded: {coupon_redempt_df.shape}")
        
        st.info("Loading coupon data...")
        coupon_df = pd.read_csv('coupon.csv', encoding='utf-8')
        st.info(f"Coupon data loaded: {coupon_df.shape}")
        
        # Date conversion (relative dates based on 2022-01-01)
        transaction_df['DATE'] = pd.to_datetime('2022-01-01') + pd.to_timedelta(transaction_df['DAY'] - 1, unit='D')
        campaign_desc_df['START_DATE'] = pd.to_datetime('2022-01-01') + pd.to_timedelta(campaign_desc_df['START_DAY'] - 1, unit='D')
        campaign_desc_df['END_DATE'] = pd.to_datetime('2022-01-01') + pd.to_timedelta(campaign_desc_df['END_DAY'] - 1, unit='D')
        
        # Data merging
        # Transaction data + Product information
        df_merged = transaction_df.merge(product_df, on='PRODUCT_ID', how='left')
        
        # Transaction data + Customer information
        df_merged = df_merged.merge(demographic_df, on='household_key', how='left')
        
        # Add campaign information
        campaign_merged = campaign_table_df.merge(campaign_desc_df, on=['DESCRIPTION', 'CAMPAIGN'], how='left')
        df_merged = df_merged.merge(campaign_merged[['household_key', 'CAMPAIGN', 'DESCRIPTION', 'START_DATE', 'END_DATE']], 
                                   on='household_key', how='left')
        
        # Add coupon usage information
        coupon_merged = coupon_redempt_df.merge(coupon_df, on=['COUPON_UPC', 'CAMPAIGN'], how='left')
        
        return df_merged, demographic_df, product_df, campaign_desc_df, coupon_merged
        
    except FileNotFoundError as e:
        st.error(f"Data file not found: {str(e)}")
        st.info("Please ensure all CSV files are in the same directory as the Streamlit app.")
        return None, None, None, None, None
    except pd.errors.EmptyDataError as e:
        st.error(f"Empty data file detected: {str(e)}")
        st.info("The CSV file appears to be empty or corrupted.")
        return None, None, None, None, None
    except pd.errors.ParserError as e:
        st.error(f"CSV parsing error: {str(e)}")
        st.info("The CSV file format may be incorrect or corrupted.")
        return None, None, None, None, None
    except UnicodeDecodeError as e:
        st.error(f"File encoding error: {str(e)}")
        st.info("The file encoding is not supported. Trying with different encoding...")
        try:
            # Try with different encodings
            transaction_df = pd.read_csv('transaction_data.csv', encoding='latin-1')
            st.success("Successfully loaded with latin-1 encoding!")
            # Continue with other files...
        except Exception as e2:
            st.error(f"Still unable to load file: {str(e2)}")
            return None, None, None, None, None
    except Exception as e:
        st.error(f"Error occurred while loading data: {str(e)}")
        st.info("Please check the data files and try again.")
        st.code(f"Error details: {type(e).__name__}: {str(e)}")
        return None, None, None, None, None

def kpi_card(label: str, value, help_txt: str = ""):
    """Create a KPI metric card"""
    st.metric(label, value if value is not None else "—", help=help_txt)

def try_mean(df, col):
    """Safely calculate mean of a column"""
    if hasattr(df, 'columns') and col in df.columns and df[col].notna().any():
        return float(df[col].mean())
    elif hasattr(df, 'name') and df.name == col and df.notna().any():
        return float(df.mean())
    else:
        return None

def try_sum(df, col):
    """Safely calculate sum of a column"""
    if hasattr(df, 'columns') and col in df.columns and df[col].notna().any():
        return float(df[col].sum())
    elif hasattr(df, 'name') and df.name == col and df.notna().any():
        return float(df.sum())
    else:
        return None

def try_unique(df, col):
    """Safely count unique values in a column"""
    if hasattr(df, 'columns') and col in df.columns:
        return int(df[col].nunique())
    elif hasattr(df, 'name') and df.name == col:
        return int(df.nunique())
    else:
        return None

def format_currency(value, decimals=1):
    """Format large numbers with appropriate units (K, M, B)"""
    if value is None:
        return "—"
    
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    else:
        return f"${value:.{decimals}f}"

def optional_chart(title, chart):
    """Display a plotly chart with title if chart is not None"""
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

# ---------- Chart Creation Functions ----------
def create_line_chart(df, x_col, y_col, title, color="#1f77b4"):
    """Create a line chart using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
        markers=True
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    fig.update_traces(line=dict(width=3))
    return fig

def create_bar_chart(df, x_col, y_col, title, color="#ff7f0e", orientation='v'):
    """Create a bar chart using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    if orientation == 'v':
        fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=[color])
        x_title = x_col.replace("_", " ").title()
        y_title = y_col.replace("_", " ").title()
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=[color], orientation='h')
        x_title = y_col.replace("_", " ").title()
        y_title = x_col.replace("_", " ").title()
    
    fig.update_layout(
        height=400,
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_pie_chart(df, names_col, values_col, title, color_palette=None):
    """Create a pie chart using plotly express"""
    if df.empty or names_col not in df.columns or values_col not in df.columns:
        return None
    
    if color_palette is None:
        color_palette = px.colors.qualitative.Set3
    
    fig = px.pie(
        df, 
        names=names_col, 
        values=values_col,
        title=title,
        color_discrete_sequence=color_palette
    )
    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_histogram(df, column, title, color="#1f77b4", nbins=30):
    """Create a histogram using plotly express"""
    if df.empty or column not in df.columns:
        return None
    
    fig = px.histogram(
        df, 
        x=column,
        title=title,
        color_discrete_sequence=[color],
        nbins=nbins
    )
    fig.update_layout(
        height=400,
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title="Count",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_scatter_plot(df, x_col, y_col, title, color="#ff7f0e"):
    """Create a scatter plot using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
        opacity=0.6
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_box_plot(df, x_col, y_col, title, color="#1f77b4"):
    """Create a box plot using plotly express"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color]
    )
    fig.update_layout(
        height=400,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig

def create_correlation_heatmap(df, title):
    """Create a correlation heatmap using plotly"""
    if df.empty:
        return None
    
    # Select numeric columns for correlation
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter for specific columns we want to include
    desired_columns = [
        'SALES_VALUE', 'QUANTITY', 'RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC'
    ]
    
    # Only include columns that exist in the data
    correlation_columns = [col for col in desired_columns if col in numeric_columns]
    
    if len(correlation_columns) < 2:
        return None
    
    # Calculate correlation matrix
    correlation_data = df[correlation_columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

# ---------- Analysis Functions ----------
def create_overview_metrics(df):
    """Create overview metrics"""
    total_revenue = df['SALES_VALUE'].sum()
    total_transactions = df['BASKET_ID'].nunique()
    total_customers = df['household_key'].nunique()
    total_products = df['PRODUCT_ID'].nunique()
    avg_basket_size = df.groupby('BASKET_ID')['QUANTITY'].sum().mean()
    
    return {
        'total_revenue': total_revenue,
        'total_transactions': total_transactions,
        'total_customers': total_customers,
        'total_products': total_products,
        'avg_basket_size': avg_basket_size
    }

def create_time_series_analysis(df):
    """Time series analysis"""
    # Daily revenue
    daily_revenue = df.groupby('DATE')['SALES_VALUE'].sum().reset_index()
    
    # Weekly revenue
    df['WEEK'] = df['DATE'].dt.isocalendar().week
    df['YEAR'] = df['DATE'].dt.year
    weekly_revenue = df.groupby(['YEAR', 'WEEK'])['SALES_VALUE'].sum().reset_index()
    weekly_revenue['DATE'] = pd.to_datetime(weekly_revenue['YEAR'].astype(str) + '-W' + 
                                          weekly_revenue['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
    
    # Monthly revenue
    df['MONTH'] = df['DATE'].dt.to_period('M')
    monthly_revenue = df.groupby('MONTH')['SALES_VALUE'].sum().reset_index()
    monthly_revenue['DATE'] = monthly_revenue['MONTH'].dt.to_timestamp()
    
    return daily_revenue, weekly_revenue, monthly_revenue

def create_customer_analysis(df, demographic_df):
    """Customer Analysis"""
    # Customer revenue
    customer_revenue = df.groupby('household_key')['SALES_VALUE'].sum().reset_index()
    customer_revenue = customer_revenue.merge(demographic_df, on='household_key', how='left')
    
    # RFM Analysis
    customer_metrics = df.groupby('household_key').agg({
        'DATE': 'max',  # Last purchase date
        'SALES_VALUE': ['sum', 'count'],  # Total purchase amount, purchase frequency
        'BASKET_ID': 'nunique'  # Unique transaction count
    }).reset_index()
    
    customer_metrics.columns = ['household_key', 'last_purchase', 'monetary', 'frequency', 'transactions']
    
    # Recency calculation (days since last purchase)
    max_date = df['DATE'].max()
    customer_metrics['recency'] = (max_date - customer_metrics['last_purchase']).dt.days
    
    # RFM score calculation (1-5 scale)
    customer_metrics['R_score'] = pd.qcut(customer_metrics['recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
    customer_metrics['F_score'] = pd.qcut(customer_metrics['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    customer_metrics['M_score'] = pd.qcut(customer_metrics['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    return customer_revenue, customer_metrics

def create_product_analysis(df):
    """Product Analysis"""
    # Product revenue
    product_revenue = df.groupby(['PRODUCT_ID', 'COMMODITY_DESC', 'BRAND', 'DEPARTMENT']).agg({
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
        'BASKET_ID': 'nunique'
    }).reset_index()
    
    # Category analysis
    category_revenue = df.groupby('COMMODITY_DESC')['SALES_VALUE'].sum().sort_values(ascending=False)
    brand_revenue = df.groupby('BRAND')['SALES_VALUE'].sum().sort_values(ascending=False)
    department_revenue = df.groupby('DEPARTMENT')['SALES_VALUE'].sum().sort_values(ascending=False)
    
    return product_revenue, category_revenue, brand_revenue, department_revenue

def create_campaign_analysis(df, campaign_desc_df):
    """Campaign Analysis"""
    # Campaign effectiveness analysis
    campaign_effectiveness = df.groupby('CAMPAIGN').agg({
        'SALES_VALUE': 'sum',
        'BASKET_ID': 'nunique',
        'household_key': 'nunique'
    }).reset_index()
    
    campaign_effectiveness = campaign_effectiveness.merge(
        campaign_desc_df[['CAMPAIGN', 'DESCRIPTION']], on='CAMPAIGN', how='left'
    )
    
    # Campaign participants vs non-participants analysis
    campaign_customers = df[df['CAMPAIGN'].notna()]['household_key'].unique()
    non_campaign_customers = df[~df['household_key'].isin(campaign_customers)]['household_key'].unique()
    
    campaign_analysis = pd.DataFrame({
        'customer_type': ['Campaign Participants', 'Non-Participants'],
        'avg_spending': [
            df[df['household_key'].isin(campaign_customers)]['SALES_VALUE'].sum() / len(campaign_customers),
            df[df['household_key'].isin(non_campaign_customers)]['SALES_VALUE'].sum() / len(non_campaign_customers)
        ],
        'customer_count': [len(campaign_customers), len(non_campaign_customers)]
    })
    
    return campaign_effectiveness, campaign_analysis

def create_cohort_analysis(df):
    """Cohort Analysis"""
    # Calculate customer's first purchase month
    first_purchase = df.groupby('household_key')['DATE'].min().reset_index()
    first_purchase['cohort_month'] = first_purchase['DATE'].dt.to_period('M')
    
    # Calculate month for each transaction
    df['order_month'] = df['DATE'].dt.to_period('M')
    
    # Create cohort table
    cohort_data = df.merge(first_purchase, on='household_key')
    cohort_table = cohort_data.groupby(['cohort_month', 'order_month']).agg({
        'household_key': 'nunique'
    }).reset_index()
    
    # Calculate cohort index
    cohort_sizes = cohort_table.groupby('cohort_month')['household_key'].first().reset_index()
    cohort_table = cohort_table.merge(cohort_sizes, on='cohort_month', suffixes=('', '_cohort_size'))
    cohort_table['cohort_index'] = cohort_table['household_key'] / cohort_table['household_key_cohort_size']
    
    # Create pivot table
    cohort_pivot = cohort_table.pivot(index='cohort_month', columns='order_month', values='cohort_index')
    
    return cohort_pivot

def create_clv_analysis(df):
    """Customer Lifetime Value (CLV) Analysis"""
    # Calculate CLV by customer
    customer_clv = df.groupby('household_key').agg({
        'SALES_VALUE': 'sum',
        'DATE': ['min', 'max'],
        'BASKET_ID': 'nunique'
    }).reset_index()
    
    customer_clv.columns = ['household_key', 'total_revenue', 'first_purchase', 'last_purchase', 'total_transactions']
    customer_clv['lifespan_days'] = (customer_clv['last_purchase'] - customer_clv['first_purchase']).dt.days
    customer_clv['lifespan_days'] = customer_clv['lifespan_days'].replace(0, 1)  # Change 0 days to 1 day
    customer_clv['clv'] = customer_clv['total_revenue']
    customer_clv['avg_order_value'] = customer_clv['total_revenue'] / customer_clv['total_transactions']
    customer_clv['purchase_frequency'] = customer_clv['total_transactions'] / (customer_clv['lifespan_days'] / 30)  # Monthly purchase frequency
    
    return customer_clv

# ---------- App Configuration ----------
st.set_page_config(
    page_title="Retail Analytics Dashboard", 
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS Styles ----------
st.markdown("""
<style>
/* More specific CSS selectors for expander hover */
[data-testid="stExpander"] details:hover summary,
[data-testid="stExpander"] details[open] summary:hover,
.stExpander details:hover summary,
.stExpander details[open] summary:hover {
    color: #1f77b4 !important;
}

/* Alternative approach - target all possible expander elements */
div[data-testid="stExpander"] > div > details > summary:hover,
div[data-testid="stExpander"] > div > details[open] > summary:hover {
    color: #1f77b4 !important;
}

/* Tab styling - font color and highlight */
.stTabs [data-baseweb="tab-list"] button {
    color: #666 !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #1f77b4 !important;
}

.stTabs [data-baseweb="tab-list"] button:hover {
    color: #1f77b4 !important;
}

/* Tab highlight (underline) color */
[data-baseweb="tab-highlight"] {
    background-color: #1f77b4 !important;
}

/* Remove tab border completely */
[data-baseweb="tab-border"] {
    display: none !important;
}

/* Expander toggle button (arrow) color */
[data-testid="stExpander"] details summary svg,
[data-testid="stExpander"] details[open] summary svg,
.stExpander details summary svg,
.stExpander details[open] summary svg {
    color: #1f77b4 !important;
    fill: #1f77b4 !important;
}

/* Alternative approach for expander arrow */
[data-testid="stExpander"] svg,
.stExpander svg {
    color: #1f77b4 !important;
    fill: #1f77b4 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Main App ----------
def main():
    # Portfolio Header
    st.markdown("""
    <div style="padding: 2rem; margin-bottom: 2rem;">
        <h1 style="text-align: center; font-size: 3rem; margin: 0;">
            <span style="font-size: 3rem;">🛒</span>
            <span style="color: #1f77b4; display: inline-block; background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: #1f77b4;"> Retail Analytics Dashboard</span>
        </h1>
        <p style="color: #1f77b4; background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: #1f77b4; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Comprehensive Analysis of Consumer Goods Retail Data
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project Info
    st.markdown("### 📋 Project Overview")
    with st.expander("**Project Goal:** Analyze retail consumer behavior and market trends using industrial dataset", expanded=False):
        st.markdown("""
        **Data Sources:** 
        - Consumer Goods Dataset from retail industry
        - Transaction data, customer demographics, product information
        - Campaign and coupon redemption data
        
        **Technologies Used:**
        - Python (Pandas, Streamlit, Plotly)
        - Data Visualization and Interactive Dashboards
        - Statistical Analysis and Customer Segmentation
        
        **Key Insights:**
        - Customer behavior patterns
        - Product performance analysis
        - Campaign effectiveness
        - Market trends and seasonality
        """)
    
    st.markdown("")

    # Data loading
    with st.spinner("Loading data..."):
        df, demographic_df, product_df, campaign_desc_df, coupon_merged = load_data()
    
    if df is None:
        st.error("Unable to load data. Please check the file paths and ensure all CSV files are present.")
        st.stop()
    
    # Check if data is empty
    if df.empty:
        st.error("Loaded data is empty. Please check your data files.")
        st.stop()

    # ---------- Sidebar Configuration ----------
    st.sidebar.markdown("### 🎛️ Dashboard Controls")
    
    # Sidebar Info Section
    st.sidebar.metric("Total Transactions", f"{df['BASKET_ID'].nunique():,}", "Unique Transactions")
    st.sidebar.metric("Total Customers", f"{df['household_key'].nunique():,}", "Unique Households")
    st.sidebar.metric("Total Products", f"{df['PRODUCT_ID'].nunique():,}", "Unique Products")
    
    st.sidebar.markdown("---")
    
    # Filters Section
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Date filter
    min_date = df['DATE'].min().date()
    max_date = df['DATE'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Analysis Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['DATE'].dt.date >= start_date) & (df['DATE'].dt.date <= end_date)]
    else:
        df_filtered = df
    
    # Customer segment filter
    age_groups = ['All'] + list(demographic_df['classification_1'].unique())
    selected_age = st.sidebar.selectbox("Customer Segment Group(Age)", age_groups)
    
    if selected_age != 'All':
        df_filtered = df_filtered[df_filtered['classification_1'] == selected_age]
    
    # Product category filter
    departments = ['All'] + list(df['DEPARTMENT'].unique())
    selected_dept = st.sidebar.selectbox("Product Category(Department)", departments)
    
    if selected_dept != 'All':
        df_filtered = df_filtered[df_filtered['DEPARTMENT'] == selected_dept]

    # ---------- KPIs ----------
    st.markdown("### 📊 Key Performance Indicators")
    
    # Main KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = try_sum(df_filtered, 'SALES_VALUE')
        kpi_card("Total Revenue", format_currency(total_revenue), "Total sales value in current filter")
    
    with col2:
        total_transactions = try_unique(df_filtered, 'BASKET_ID')
        kpi_card("Total Transactions", f"{total_transactions:,}", "Unique transactions in current filter")
    
    with col3:
        total_customers = try_unique(df_filtered, 'household_key')
        kpi_card("Total Customers", f"{total_customers:,}", "Unique customers in current filter")
    
    with col4:
        basket_sizes = df_filtered.groupby('BASKET_ID')['QUANTITY'].sum()
        avg_basket_size = float(basket_sizes.mean()) if not basket_sizes.empty else None
        kpi_card("Avg Basket Size", f"{avg_basket_size:.1f}", "Average items per basket")
    
    # Secondary metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        total_products = try_unique(df_filtered, 'PRODUCT_ID')
        kpi_card("Total Products", f"{total_products:,}", "Unique products sold")
    
    with col6:
        transaction_values = df_filtered.groupby('BASKET_ID')['SALES_VALUE'].sum()
        avg_transaction_value = float(transaction_values.mean()) if not transaction_values.empty else None
        kpi_card("Avg Transaction Value", format_currency(avg_transaction_value), "Average value per transaction")
    
    with col7:
        campaign_participation = df_filtered['CAMPAIGN'].notna().sum() / len(df_filtered) if len(df_filtered) > 0 else 0
        kpi_card("Campaign Participation", f"{campaign_participation:.1%}", "Percentage of transactions with campaigns")
    
    with col8:
        coupon_usage = df_filtered['COUPON_DISC'].fillna(0).gt(0).sum() / len(df_filtered) if len(df_filtered) > 0 else 0
        kpi_card("Coupon Usage", f"{coupon_usage:.1%}", "Percentage of transactions with coupons")
    
    st.markdown("")

    # ---------- Charts ----------
    st.markdown("### 📈 Data Visualizations")
    
    # Create tabs for different chart categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["◽ Overview", "◽ Customer Analysis", "◽ Product Analysis", "◽ Campaign Analysis", "◽ Advanced Analytics"])
    
    with tab1:
        st.markdown("##### (1) Business Overview")
        
        # Time series analysis
        daily_revenue, weekly_revenue, monthly_revenue = create_time_series_analysis(df_filtered)
        
        chart_type = st.selectbox("Chart Type", ["Daily", "Weekly", "Monthly"])
        
        if chart_type == "Daily":
            revenue_chart = create_line_chart(daily_revenue, 'DATE', 'SALES_VALUE', 'Daily Revenue Trend')
        elif chart_type == "Weekly":
            revenue_chart = create_line_chart(weekly_revenue, 'DATE', 'SALES_VALUE', 'Weekly Revenue Trend')
        else:
            revenue_chart = create_line_chart(monthly_revenue, 'DATE', 'SALES_VALUE', 'Monthly Revenue Trend')
        
        optional_chart("", revenue_chart)
        
        # Day of week / Hour analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week revenue
            df_filtered['WEEKDAY'] = df_filtered['DATE'].dt.day_name()
            weekday_revenue = df_filtered.groupby('WEEKDAY')['SALES_VALUE'].sum().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            ).reset_index()
            
            weekday_chart = create_bar_chart(weekday_revenue, 'WEEKDAY', 'SALES_VALUE', 'Revenue by Day of Week')
            optional_chart("", weekday_chart)
        
        with col2:
            # Hourly revenue
            df_filtered['HOUR'] = df_filtered['TRANS_TIME'] // 100
            hourly_revenue = df_filtered.groupby('HOUR')['SALES_VALUE'].sum().reset_index()
            
            hourly_chart = create_bar_chart(hourly_revenue, 'HOUR', 'SALES_VALUE', 'Revenue by Hour')
            optional_chart("", hourly_chart)
    
    with tab2:
        st.markdown("##### (2) Customer Analysis")
        
        customer_revenue, customer_metrics = create_customer_analysis(df_filtered, demographic_df)
        
        # Customer segment analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by age group
            age_revenue = customer_revenue.groupby('classification_1')['SALES_VALUE'].sum().reset_index()
            age_pie = create_pie_chart(age_revenue, 'classification_1', 'SALES_VALUE', 'Revenue Share by Age Group')
            optional_chart("", age_pie)
        
        with col2:
            # Homeowner type analysis
            homeowner_revenue = customer_revenue.groupby('HOMEOWNER_DESC')['SALES_VALUE'].sum().reset_index()
            homeowner_bar = create_bar_chart(homeowner_revenue, 'HOMEOWNER_DESC', 'SALES_VALUE', 'Revenue by Homeowner Type')
            optional_chart("", homeowner_bar)
        
        # RFM Analysis
        st.markdown("##### RFM Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recency_hist = create_histogram(customer_metrics, 'recency', 'Recency Distribution')
            optional_chart("", recency_hist)
        
        with col2:
            frequency_hist = create_histogram(customer_metrics, 'frequency', 'Frequency Distribution')
            optional_chart("", frequency_hist)
        
        with col3:
            monetary_hist = create_histogram(customer_metrics, 'monetary', 'Monetary Distribution')
            optional_chart("", monetary_hist)
    
    with tab3:
        st.markdown("##### (3) Product Analysis")
        
        product_revenue, category_revenue, brand_revenue, department_revenue = create_product_analysis(df_filtered)
        
        # Top products analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Product Categories**")
            top_categories = category_revenue.head(10).reset_index()
            top_categories.columns = ['Category', 'Revenue']
            category_bar = create_bar_chart(top_categories, 'Revenue', 'Category', 'Top Categories by Revenue', orientation='h')
            optional_chart("", category_bar)
        
        with col2:
            st.markdown("**Department Revenue Distribution**")
            dept_pie = create_pie_chart(department_revenue.reset_index(), 'DEPARTMENT', 'SALES_VALUE', 'Department Revenue Share')
            optional_chart("", dept_pie)
        
        # Brand analysis
        st.markdown("**Top 15 Brands by Revenue**")
        top_brands = brand_revenue.head(15).reset_index()
        top_brands.columns = ['Brand', 'Revenue']
        brand_bar = create_bar_chart(top_brands, 'Revenue', 'Brand', 'Top Brands by Revenue', orientation='h')
        optional_chart("", brand_bar)
        
        # Product performance analysis
        st.markdown("**Product Performance Matrix**")
        product_performance = df_filtered.groupby(['PRODUCT_ID', 'COMMODITY_DESC']).agg({
            'BASKET_ID': 'nunique',
            'SALES_VALUE': 'mean'
        }).reset_index()
        product_performance.columns = ['Product_ID', 'Category', 'Transaction_Frequency', 'Avg_Revenue']
        
        performance_scatter = create_scatter_plot(product_performance, 'Transaction_Frequency', 'Avg_Revenue', 'Product Performance Matrix')
        optional_chart("", performance_scatter)
    
    with tab4:
        st.markdown("##### (4) Campaign Analysis")
        
        campaign_effectiveness, campaign_analysis = create_campaign_analysis(df_filtered, campaign_desc_df)
        
        # Campaign effectiveness analysis
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_bar = create_bar_chart(campaign_effectiveness, 'DESCRIPTION', 'SALES_VALUE', 'Revenue by Campaign Type')
            optional_chart("", campaign_bar)
        
        with col2:
            participant_bar = create_bar_chart(campaign_analysis, 'customer_type', 'avg_spending', 'Average Spending: Campaign vs Non-Campaign')
            optional_chart("", participant_bar)
        
        # Campaign participant count
        campaign_participants = create_bar_chart(campaign_effectiveness, 'DESCRIPTION', 'household_key', 'Campaign Participants by Type')
        optional_chart("", campaign_participants)
    
    with tab5:
        st.markdown("##### (5) Advanced Analytics")
        
        # Cohort analysis
        st.markdown("**Customer Retention Cohort Analysis**")
        cohort_pivot = create_cohort_analysis(df_filtered)
        
        if not cohort_pivot.empty:
            # Create heatmap
            fig = px.imshow(cohort_pivot.values, 
                           x=[str(col) for col in cohort_pivot.columns],
                           y=[str(idx) for idx in cohort_pivot.index],
                           color_continuous_scale='Blues',
                           title='Customer Retention Cohort Analysis')
            optional_chart("", fig)
        
        # Correlation analysis
        st.markdown("**Correlation Analysis**")
        correlation_heatmap = create_correlation_heatmap(df_filtered, 'Variable Correlation Matrix')
        optional_chart("", correlation_heatmap)
        
        # CLV analysis
        st.markdown("**Customer Lifetime Value Analysis**")
        customer_clv = create_clv_analysis(df_filtered)
        
        col1, col2 = st.columns(2)
        
        with col1:
            clv_hist = create_histogram(customer_clv, 'clv', 'Customer Lifetime Value Distribution')
            optional_chart("", clv_hist)
        
        with col2:
            clv_scatter = create_scatter_plot(customer_clv, 'purchase_frequency', 'avg_order_value', 'Purchase Frequency vs Average Order Value')
            optional_chart("", clv_scatter)
    
    st.markdown("---")
    
    # ---------- Key Insights Section ----------
    st.markdown("### 💡 Key Insights & Analysis")
    
    # Generate insights based on filtered data
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("##### (1) Market Insights")
        
        # Revenue insights
        total_rev = try_sum(df_filtered, 'SALES_VALUE')
        transaction_values = df_filtered.groupby('BASKET_ID')['SALES_VALUE'].sum()
        avg_transaction = float(transaction_values.mean()) if not transaction_values.empty else None
        
        st.markdown(f"""
        <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <strong>💰 Revenue Performance</strong><br>
        Total revenue: {format_currency(total_rev)} across all transactions<br>
        Average transaction value: {format_currency(avg_transaction)} per basket
        </div>
        """, unsafe_allow_html=True)
        
        # Customer insights
        total_customers = try_unique(df_filtered, 'household_key')
        basket_sizes = df_filtered.groupby('BASKET_ID')['QUANTITY'].sum()
        avg_basket = float(basket_sizes.mean()) if not basket_sizes.empty else None
        
        st.markdown(f"""
        <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <strong>👥 Customer Behavior</strong><br>
        {total_customers:,} unique customers with average basket size of {avg_basket:.1f} items<br>
        Customer engagement shows strong purchasing patterns and loyalty
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("##### (2) Strategic Recommendations")
        
        # Campaign insights
        campaign_participation = df_filtered['CAMPAIGN'].notna().sum() / len(df_filtered) if len(df_filtered) > 0 else 0
        
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <strong>🎯 Campaign Optimization</strong><br>
        {campaign_participation:.1%} of transactions involve campaigns - consider expanding targeted marketing efforts and personalized campaigns
        </div>
        """, unsafe_allow_html=True)
        
        # Product insights
        top_category = category_revenue.index[0] if not category_revenue.empty else "N/A"
        
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem;">
        <strong>🛍️ Product Strategy</strong><br>
        Top performing category: {top_category} - focus inventory and marketing on high-performing categories for maximum profitability
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong> Retail Analytics Dashboard</strong> | Built with Python, Streamlit & Plotly</p>
        <p>Data-driven insights for retail analytics and customer behavior analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()