import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# -------------------------------
# Define PyTorch Model
# -------------------------------
class PurchaseModel(nn.Module):
    def __init__(self, input_dim):
        super(PurchaseModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.output(x))
        return x


# -------------------------------
# Streamlit App
# -------------------------------
st.title("🛒 Customer Purchase Prediction App")

st.sidebar.header("Upload Your Files")
raw_file = st.sidebar.file_uploader("Upload raw_customer_data.csv", type=["csv"])
model_file = st.sidebar.file_uploader("Upload model_data.csv", type=["csv"])
validation_file = st.sidebar.file_uploader("Upload validation_features.csv", type=["csv"])

if raw_file and model_file:
    # -------------------------------
    # Step 1: Data Cleaning
    # -------------------------------
    st.subheader("🔹 Step 1: Data Cleaning")
    df = pd.read_csv(raw_file)

    # Handle missing values
    df['time_spent'] = df['time_spent'].fillna(df['time_spent'].median())
    df['pages_viewed'] = df['pages_viewed'].fillna(df['pages_viewed'].mean())
    df['basket_value'] = df['basket_value'].fillna(0)
    df['device_type'] = df['device_type'].fillna("Unknown")
    df['customer_type'] = df['customer_type'].fillna("New")

    # Convert datatypes
    df['customer_id'] = df['customer_id'].astype(int)
    df['pages_viewed'] = df['pages_viewed'].astype(int)

    clean_data = df.copy()
    st.write("✅ Cleaned Data (first 5 rows):")
    st.dataframe(clean_data.head())

    # -------------------------------
    # Step 2: Preprocessing for Model
    # -------------------------------
    st.subheader("🔹 Step 2: Feature Engineering")

    df_model = pd.read_csv(model_file)
    numerical_cols = ['time_spent', 'pages_viewed', 'basket_value']
    categorical_cols = ['device_type', 'customer_type']
    target_col = 'purchase'

    # Scaling
    scaler = MinMaxScaler()
    scaled_numerical = pd.DataFrame(
        scaler.fit_transform(df_model[numerical_cols]), columns=numerical_cols
    )

    # One-hot encoding
    encoded_categorical = pd.get_dummies(df_model[categorical_cols], prefix=categorical_cols)

    # Final dataset
    model_feature_set = pd.concat([
        df_model[['customer_id']],
        scaled_numerical,
        encoded_categorical,
        df_model[target_col]
    ], axis=1)

    st.write("✅ Feature Engineered Data (first 5 rows):")
    st.dataframe(model_feature_set.head())

    # -------------------------------
    # Step 3: Model Training
    # -------------------------------
    st.subheader("🔹 Step 3: Train Model")

    X = model_feature_set.drop(columns=['customer_id', 'purchase']).values.astype('float32')
    y = model_feature_set['purchase'].values.astype('float32')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model
    input_dim = X_train.shape[1]
    purchase_model = PurchaseModel(input_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(purchase_model.parameters(), lr=0.001)

    progress = st.progress(0)
    epochs = 50
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = purchase_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        progress.progress((epoch + 1) / epochs)

    # Validation Accuracy
    with torch.no_grad():
        val_tensor = torch.tensor(X_val)
        predictions = purchase_model(val_tensor).numpy().flatten()
        predicted_labels = (predictions >= 0.5).astype(int)
        accuracy = accuracy_score(y_val, predicted_labels)

    st.success(f"🎯 Model Trained! Validation Accuracy: {accuracy:.4f}")

    # -------------------------------
    # Step 4: Predictions on Validation Set
    # -------------------------------
    if validation_file:
        st.subheader("🔹 Step 4: Make Predictions")
        val_df = pd.read_csv(validation_file)
        X_validation = val_df.drop(columns=['customer_id']).values.astype('float32')
        customer_ids = val_df['customer_id'].values

        with torch.no_grad():
            val_tensor = torch.tensor(X_validation)
            predictions = purchase_model(val_tensor).numpy().flatten()
            predicted_labels = (predictions >= 0.5).astype(int)

        validation_predictions = pd.DataFrame({
            'customer_id': customer_ids,
            'purchase': predicted_labels
        })

        st.write("✅ Predictions (first 10 rows):")
        st.dataframe(validation_predictions.head())

        # Download button
        csv = validation_predictions.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
