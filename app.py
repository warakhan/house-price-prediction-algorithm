import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Responsive title */
    h1 {
        font-size: clamp(1.5rem, 5vw, 2.5rem) !important;
        margin-bottom: 1rem !important;
    }
    
    /* Card-like containers */
    .stAlert, .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: clamp(1.2rem, 3vw, 2rem) !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Responsive columns */
    @media (max-width: 768px) {
        .row-widget.stRadio > div {
            flex-direction: column;
        }
    }
    
    /* Chart container */
    .plot-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
    }
    
    /* Price display */
    .price-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .price-amount {
        font-size: clamp(2rem, 6vw, 3.5rem);
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header with icon
col1, col2 = st.columns([1, 10])
with col1:
    st.write("🏠")
with col2:
    st.title("House Price Prediction System")

st.markdown("""
    <p style='font-size: clamp(0.9rem, 2vw, 1.1rem); color: #666;'>
    Predict house prices based on property features using Machine Learning
    </p>
""", unsafe_allow_html=True)

st.divider()

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("housing.csv")
        return data, None
    except FileNotFoundError:
        return None, "housing.csv file not found. Please ensure the file exists."
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

data, error = load_data()

if error:
    st.error(error)
    st.stop()

# Train model
@st.cache_resource
def train_model(data):
    X = data[['area', 'bedrooms', 'bathrooms']]
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return model, X_test, y_test, y_pred, metrics

model, X_test, y_test, y_pred, metrics = train_model(data)

# Sidebar - Input Section
with st.sidebar:
    st.header("🏗️ Property Details")
    st.markdown("---")
    
    # Input fields with better defaults and validation
    area = st.number_input(
        "🏡 Area (sq ft)",
        min_value=500,
        max_value=10000,
        value=1500,
        step=100,
        help="Enter the total area of the property"
    )
    
    bedrooms = st.number_input(
        "🛏️ Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of bedrooms"
    )
    
    bathrooms = st.number_input(
        "🚿 Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Number of bathrooms"
    )
    
    st.markdown("---")
    
    predict_button = st.button("🔮 Predict House Price", use_container_width=True)
    
    st.markdown("---")
    st.caption("💡 Adjust the values above to get price predictions")

# Main content area with responsive columns
if predict_button:
    # Prediction
    input_data = np.array([[area, bedrooms, bathrooms]])
    price = model.predict(input_data)
    
    # Display prediction in a prominent card
    st.markdown(f"""
        <div class="price-display">
            <h3 style='margin:0; font-size: clamp(1.2rem, 3vw, 1.5rem);'>Estimated House Price</h3>
            <div class="price-amount">₹ {int(price[0]):,}</div>
            <p style='margin:0; font-size: clamp(0.9rem, 2vw, 1.1rem); opacity: 0.9;'>
                {area:,} sq ft • {bedrooms} bed • {bathrooms} bath
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Additional insights in responsive columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_per_sqft = price[0] / area
        st.metric("Price per sq ft", f"₹ {int(price_per_sqft):,}")
    
    with col2:
        avg_price = data['price'].mean()
        diff = ((price[0] - avg_price) / avg_price) * 100
        st.metric("vs Average", f"{diff:+.1f}%")
    
    with col3:
        bedroom_price = price[0] / bedrooms
        st.metric("Price per bedroom", f"₹ {int(bedroom_price):,}")

st.markdown("---")

# Model Performance Section - Responsive layout
st.subheader("📊 Model Performance Metrics")

# Metrics in responsive columns
met_col1, met_col2, met_col3 = st.columns(3)

with met_col1:
    st.metric(
        label="R² Score",
        value=f"{metrics['r2']:.3f}",
        help="Coefficient of determination (0-1, higher is better)"
    )

with met_col2:
    st.metric(
        label="Mean Absolute Error",
        value=f"₹ {int(metrics['mae']):,}",
        help="Average prediction error"
    )

with met_col3:
    st.metric(
        label="Root Mean Squared Error",
        value=f"₹ {int(metrics['rmse']):,}",
        help="Standard deviation of prediction errors"
    )

st.markdown("---")

# Visualization Section - Responsive charts
st.subheader("📈 Model Visualization")

# Use tabs for better mobile experience
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📉 Residuals", "🔍 Feature Analysis"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel("Actual Price (₹)", fontsize=12)
    ax.set_ylabel("Predicted Price (₹)", fontsize=12)
    ax.set_title("Actual vs Predicted Prices", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel("Predicted Price (₹)", fontsize=12)
    ax.set_ylabel("Residuals (₹)", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    features = ['area', 'bedrooms', 'bathrooms']
    for idx, feature in enumerate(features):
        axes[idx].scatter(data[feature], data['price'], alpha=0.5)
        axes[idx].set_xlabel(feature.capitalize(), fontsize=11)
        axes[idx].set_ylabel("Price (₹)", fontsize=11)
        axes[idx].set_title(f"Price vs {feature.capitalize()}", fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Dataset Section - Expandable
with st.expander("📄 View Dataset Sample", expanded=False):
    # Responsive dataframe display
    st.dataframe(
        data.head(10),
        use_container_width=True,
        height=300
    )
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Shape:**", data.shape)
        st.write("**Total Records:**", len(data))
    
    with col2:
        st.write("**Features:**", len(data.columns) - 1)
        st.write("**Missing Values:**", data.isnull().sum().sum())

# Footer
st.markdown("---")
st.warning(
    "⚠️ **Note:** This model predicts prices based only on area, bedrooms, and bathrooms. "
    "Location, age, amenities, and other factors are not considered in these predictions."
)

st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #666;'>
        <p style='font-size: clamp(0.8rem, 2vw, 0.9rem);'>
            Built with Streamlit • Machine Learning Powered
        </p>
    </div>
""", unsafe_allow_html=True)
