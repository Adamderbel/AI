import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, low_memory=False)

def preprocess_data(df):
    """Preprocess data: filter dates, sort, encode labels, and remove specific codes."""
    df['date_saisie'] = pd.to_datetime(df['date_saisie'])
    df = df[(df['date_saisie'] >= '2006-01-01') & (df['date_saisie'] <= '2023-12-31')]
    df = df.sort_values(by='date_saisie', ascending=True)

    label_encoder = LabelEncoder()
    df['Fk_prestation_code'] = label_encoder.fit_transform(df['Fk_famille_prestation'])

    codes_to_remove = [18, 0, 7, 8, 11, 1, 4, 16, 13, 10, 19]
    df = df[~df['Fk_prestation_code'].isin(codes_to_remove)]
    
    return df, df[['Fk_prestation_code', 'Fk_famille_prestation']].drop_duplicates()

def train_test_split(df):
    """Split the data into training and testing sets."""
    X_train = df[(df['date_saisie'].dt.year >= 2006) & (df['date_saisie'].dt.year <= 2018)][['Fk_prestation_code']]
    X_test = df[df['date_saisie'].dt.year == 2019][['Fk_prestation_code']]
    y_train = df[(df['date_saisie'].dt.year >= 2006) & (df['date_saisie'].dt.year <= 2018)]['montant_HT']
    y_test = df[df['date_saisie'].dt.year == 2019]['montant_HT']
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def make_predictions(model, X_test, y_test, df):
    """Make predictions and prepare the results dataframe."""
    y_pred = model.predict(X_test)
    
    predictions_df = X_test.copy()
    predictions_df['predicted_montant_HT'] = y_pred
    predictions_df['actual_montant_HT'] = y_test.values
    predictions_df['date_saisie'] = df[df['date_saisie'].dt.year == 2019]['date_saisie'].values
    
    return predictions_df.reset_index(drop=True)

def main():
    file_path = "D:\\PFE\\Fact_Table_Rev.csv"
    
    # Load and preprocess data
    data = load_data(file_path)
    df, mapping_df = preprocess_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions_df = make_predictions(model, X_test, y_test, df)
    
    # Merge with mapping and save results
    final_df = predictions_df.merge(mapping_df, on='Fk_prestation_code')
    final_df.to_csv('Predlinear2019.csv', index=False)
    
    print(final_df)
if __name__ == "__main__":
    main()
