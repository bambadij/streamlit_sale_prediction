import pandas as pd
import streamlit as st
import numpy as np
import sklearn
import joblib


num_imputer = joblib.load('toolkit/numerical_imputer.joblib')
cat_imputer = joblib.load('toolkit/categorical_imputer.joblib')
encoder = joblib.load('toolkit/encoder.joblib')
scaler = joblib.load('toolkit/scaler.joblib')
dt_model = joblib.load('toolkit/Final_model.joblib')

# Add a title and subtitle
st.write("<center><h1>Sales Prediction App</h1></center>", unsafe_allow_html=True)

# Load the image
# image = Image.open("grocery_shopping_woman.png")

# Set up the layout
col1, col2, col3 = st.columns([1, 3, 3])
# col2.image(image, width=600)

st.write("This app uses machine learning to predict sales based on certain input parameters. Simply enter the required information and click 'Predict' to get a sales prediction!")

st.subheader("Enter the details to predict sales")

# Create the input fields
input_data = {}
col1,col2 = st.columns(2)
with col1:
    input_data['store_nbr'] = st.slider("enter store number",0,54)
    input_data['family'] = st.selectbox("choose product family", ['AUTOMOTIVE', 'BEAUTY AND FASHION', 'BEVERAGES AND LIQUOR',
       'SCHOOL AND OFFICE SUPPLIES', 'GROCERY', 'HOME CARE AND GARDEN',
       'FROZEN FOODS', 'HOME AND KITCHEN', 'PET SUPPLIES'])
    input_data['onpromotion'] =st.number_input("number of items on promotion",step=1)
    input_data['oil_price'] = st.slider("oil price",min_value=0.0, max_value=100.0, step=0.1)
    input_data['store_type'] = st.selectbox("store_type",['D', 'C', 'B', 'E', 'A'])
    input_data['store_cluster'] = st.slider("store cluster",0,17)

with col2:
    input_data['events'] = st.slider("Is it a holiday? 0=no, 1=yes",0,1)
    input_data['Year'] = st.number_input("year",step=1)
    input_data['Month'] = st.slider("month",1,12)
    input_data['Day'] = st.slider("day",1,31)
    input_data['Quarter'] = st.slider("quarter",1,4)
    input_data['Week_of_Year'] = st.slider("week of the year",1,53)
    input_data['Day_of_Week'] = st.number_input("dayofweek,0=Sun and 6=Sat",step=1)
    input_data['Is_Weekend'] = st.number_input("Is it a weekend? 0=No, 1=Yes",0,1)

# Define the custom CSS
predict_button_css = """
    <style>
    .predict-button {
        background-color: #C4C4C4;
        color: gray;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
"""

# Display the custom CSS
st.markdown(predict_button_css, unsafe_allow_html=True)


  # Create a button to make a prediction

if st.button("Predict", key="predict_button", help="Click to make a prediction."):
    # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([input_data])


# Selecting categorical and numerical columns separately
        cat_columns = [col for col in input_df.columns if input_df[col].dtype == 'object']
        num_columns = [col for col in input_df.columns if input_df[col].dtype != 'object']


# Apply the imputers
        input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns])
        input_df_imputed_num = num_imputer.transform(input_df[num_columns])


 # Encode the categorical columns
        input_encoded_df = pd.DataFrame(encoder.transform(input_df_imputed_cat).toarray(),
                                   columns=encoder.get_feature_names_out(cat_columns))

# Scale the numerical columns
        input_df_scaled = scaler.transform(input_df_imputed_num)
        input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_columns)

#joining the cat encoded and num scaled
        final_df = pd.concat([input_scaled_df, input_encoded_df], axis=1)

# Make a prediction
        prediction = dt_model.predict(final_df)[0]


# Display the prediction
        st.write(f"The predicted sales are: {prediction}.")
        input_df.to_csv("data/input_data.csv", index=False)
        st.table(input_df)
