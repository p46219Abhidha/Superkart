import streamlit as st
import pandas as pd
import joblib

# Set the page title and layout
st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart Sales Prediction Tool")
st.write("Enter the product and store details below to predict the total sales.")

# Load the trained model, scaler, and column list
model = joblib.load('superkart_model.pkl')
scaler = joblib.load('superkart_scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# Note: The mappings below correspond to the LabelEncoder alphabetical sorting in Colab.
# Product_Sugar_Content: Low Sugar=0, No Sugar=1, Regular=2
# Store_Size: High=0, Medium=1, Small=2
# Store_Location_City_Type: Tier 1=0, Tier 2=1, Tier 3=2
# Store_Type: Departmental Store=0, Food Mart=1, Supermarket Type1=2, Supermarket Type2=3

st.header("Product & Store Details")
col1, col2, col3 = st.columns(3)

with col1:
    weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.5, step=0.1)
    sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'No Sugar', 'Regular'])
    allocated_area = st.number_input("Product Allocated Area (Ratio)", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
    product_type = st.selectbox("Product Type", ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'])

with col2:
    mrp = st.number_input("Maximum Retail Price (MRP)", min_value=0.0, max_value=500.0, value=150.0, step=0.1)
    store_size = st.selectbox("Store Size", ['High', 'Medium', 'Small'])
    city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    store_type = st.selectbox("Store Type", ['Departmental Store', 'Food Mart', 'Supermarket Type1', 'Supermarket Type2'])

with col3:
    store_age = st.number_input("Store Age (Years)", min_value=0, max_value=50, value=20, step=1)

# Map the text inputs back to the numbers used during training
sugar_map = {'Low Sugar': 0, 'No Sugar': 1, 'Regular': 2}
size_map = {'High': 0, 'Medium': 1, 'Small': 2}
city_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
store_type_map = {'Departmental Store': 0, 'Food Mart': 1, 'Supermarket Type1': 2, 'Supermarket Type2': 3}

# Create a mapping for Product Type based on the alphabetical order in Colab
product_types_list = ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods']
product_type_map = {name: i for i, name in enumerate(product_types_list)}

if st.button("Predict Sales", type="primary"):
    # Create a dictionary with the encoded values
    input_data = {
        'Product_Weight': weight,
        'Product_Sugar_Content': sugar_map[sugar_content],
        'Product_Allocated_Area': allocated_area,
        'Product_Type': product_type_map[product_type],
        'Product_MRP': mrp,
        'Store_Size': size_map[store_size],
        'Store_Location_City_Type': city_map[city_type],
        'Store_Type': store_type_map[store_type],
        'Store_Age': store_age
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the column order matches exactly
    input_df = input_df[model_columns]

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(scaled_input)[0]

    # Display the result
    st.success(f"### Predicted Product Store Sales Total: ₹ {prediction:,.2f}")
