import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

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
    
    mapping_df = df[['Fk_prestation_code', 'Fk_famille_prestation']].drop_duplicates()
    
    df = df.groupby(['date_saisie', 'Fk_prestation_code'])['montant_HT'].sum().reset_index()
    df.set_index('date_saisie', inplace=True)
    
    return df, mapping_df

def check_index_type(df):
    """Check the type of the index and if it is a DateTimeIndex."""
    index_type = type(df.index)
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    print("Index type:", index_type)
    print("Is DateTimeIndex:", is_datetime_index)

def fit_sarimax(train_data, test_data):
    """Fit the SARIMAX model and make predictions."""
    model = SARIMAX(train_data['montant_HT'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit()
    start = len(train_data)
    end = len(train_data) + len(test_data) - 1
    predictions = list(result.predict(start, end))
    
    return predictions

def forecast_for_years(df, years):
    """Forecast for given years and return the results."""
    results = list()
    all_prestation = df.Fk_prestation_code.unique()
    
    for prestation in tqdm(all_prestation):
        sub_df = df[df.Fk_prestation_code == prestation]
        train_data = sub_df.loc['2010':'2018']
        test_data = sub_df[sub_df.index.year == years[0]]
        
        predictions = fit_sarimax(train_data, test_data)
        
        prestation_result = pd.DataFrame({
            "montant_predit": predictions,
            "date": list(test_data.index),
            "Fk_prestation_code": prestation
        })
        
        results.append(prestation_result)
    
    return pd.concat(results)

def forecast_future(df, start_year, end_year):
    """Forecast for future periods and return the results."""
    results = list()
    all_prestation = df.Fk_prestation_code.unique()
    
    for prestation in tqdm(all_prestation):
        sub_df = df[df.Fk_prestation_code == prestation]
        train_data = pd.concat([sub_df.loc['2010':'2019'], sub_df.loc['2022':'2023']])
        future_periods = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31')
        
        model = SARIMAX(train_data['montant_HT'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        result = model.fit()
        
        predictions = result.predict(start=len(train_data), end=len(train_data) + len(future_periods) - 1)
        
        prestation_result = pd.DataFrame({
            "montant_predit": predictions,
            "date": future_periods,
            "Fk_prestation_code": prestation
        })
        
        results.append(prestation_result)
    
    return pd.concat(results)

def main():
    file_path = "D:\\PFE\\Fact_Table_Rev.csv"
    
    # Load and preprocess data
    data = load_data(file_path)
    df, mapping_df = preprocess_data(data)
    
    # Check index type
    check_index_type(df)
    
    # Forecast for 2019
    final_pred = forecast_for_years(df, [2019])
    test_data = df[df.index.year == 2019].reset_index()
    test_data.rename(columns={"date_saisie": 'date'}, inplace=True)
    merged_data = pd.merge(test_data, final_pred, how='left', on=['date', 'Fk_prestation_code'])
    merged_data = merged_data.merge(mapping_df, on='Fk_prestation_code')
    merged_data.to_csv('Predsarimax2019.csv', index=False)
    
    # Forecast for 2024 and beyond
    final_results = forecast_future(df, 2024, 2026)
    merged_data1 = final_results.merge(mapping_df, on='Fk_prestation_code')
    merged_data1.to_csv('Pred2024andmore.csv', index=False)
    
if __name__ == "__main__":
    main()
