# 🛒 Dunnhumby Retail Analytics Dashboard

**Data Analyst Portfolio Project** - Consumer Goods Retail Analytics

## 📋 Project Overview

This project is a comprehensive retail analytics dashboard built using Dunnhumby's synthetic consumer goods company data as a Data Analyst portfolio project.

## 🎯 Key Features

### 📈 Business Overview
- Real-time sales and transaction metrics monitoring
- Time series sales analysis (daily/weekly/monthly)
- Sales pattern analysis by day of week and time period

### 👥 Customer Analysis
- **RFM Analysis**: Customer segmentation based on Recency, Frequency, and Monetary scores
- Customer demographic analysis (age groups, residence types)
- Customer Lifetime Value (CLV) calculation and analysis
- Customer retention analysis through cohort analysis

### 🛍️ Product Analysis
- Sales analysis by product category
- Brand performance analysis
- Product performance matrix (frequency vs. average sales)
- Sales distribution by department

### 🎯 Campaign Analysis
- Campaign effectiveness analysis by type
- Campaign participants vs. non-participants comparison
- Campaign ROI analysis

### 📊 Advanced Analytics
- Cohort analysis
- Correlation analysis
- Simple predictive modeling (sales trend forecasting)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Language**: Python 3.8+

## 📊 Dataset Structure

### Main Tables
1. **transaction_data.csv** - Customer purchase transaction data
2. **hh_demographic.csv** - Customer demographic information
3. **product.csv** - Product information (brand, category, department, etc.)
4. **campaign_table.csv** - Campaign participant customer information
5. **campaign_desc.csv** - Campaign descriptions and periods
6. **coupon_redempt.csv** - Coupon redemption history
7. **coupon.csv** - Coupon information

### Key Columns
- **Transaction Data**: household_key, BASKET_ID, PRODUCT_ID, SALES_VALUE, QUANTITY
- **Customer Data**: Age groups, residence types, children presence, etc.
- **Product Data**: Brand, category, department, product description
- **Campaign Data**: Campaign type, participating customers, duration

## 🚀 How to Run

### 1. Environment Setup
```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run streamlit_retail.py
```

### 3. Access in Browser
- Default URL: http://localhost:8501

## 📈 Key Analysis Insights

### 1. Sales Trends
- Sales pattern identification through time series analysis
- Seasonality and trend analysis

### 2. Customer Segmentation
- Customer characteristics by tier through RFM analysis
- High-value customer identification and targeting strategy development

### 3. Product Performance
- Sales contribution analysis by category
- Brand performance comparison

### 4. Campaign Effectiveness
- Actual ROI measurement of marketing campaigns
- Effective campaign type identification

## 🎨 Dashboard Features

### Interactive Filtering
- Date range selection
- Customer segment filtering
- Product category filtering

### Various Visualizations
- Dynamic charts (Plotly)
- Heatmaps and correlation analysis
- Cohort analysis heatmaps

### Responsive Design
- Mobile-friendly layout
- Intuitive navigation

## 📊 Business Value

### 1. Data-Driven Decision Making
- Real-time business metrics monitoring
- Strategy development through quantitative analysis

### 2. Enhanced Customer Understanding
- Customer behavior pattern analysis
- Personalized marketing strategy development

### 3. Operational Efficiency
- Product performance optimization
- Inventory management improvement

### 4. Marketing ROI Improvement
- Campaign effectiveness measurement and optimization
- Targeted marketing strategy development

## 🔧 Customization

### Adding New Analysis
1. Add new tab in `main()` function
2. Implement corresponding analysis function
3. Add visualizations and insights

### Changing Data Sources
1. Modify `load_data()` function
2. Adjust data preprocessing logic
3. Update column mapping

## 📝 License

This project is created for portfolio purposes.

## 🤝 Contributing

Feedback and improvement suggestions for this project are always welcome!

---

**Built with ❤️ for Data Analytics Portfolio**