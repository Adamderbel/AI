# med LSTM

This repository contains a Long Short-Term Memory (LSTM) model for predicting medical costs based on historical data. The model is implemented in Python using TensorFlow and scikit-learn.

## Files

* **lstmModel.py**: Contains the implementation of the LSTM model class, including data preprocessing, model training, prediction, and evaluation.
* **med_lstm.ipynb**: A Jupyter Notebook demonstrating the usage of the LSTM model to train and evaluate models for different medical services.

## Usage

The `med_lstm.ipynb` notebook provides a step-by-step guide on how to use the LSTM model. The following steps are involved:

1. **Data Loading and Preprocessing:**
   - Load the medical cost data from a CSV file (`Fact_Table_Rev.csv`).
   - Filter the data to include records between 2006 and 2019.
   - Extract unique medical services from the data.

2. **Model Training and Evaluation:**
   - Iterate through each medical service.
   - Split the data for the current service into training and testing sets.
   - Create an instance of the `lstm_model` class.
   - Train the LSTM model using the training data.
   - Predict medical costs for the testing data.
   - Evaluate the model's performance using metrics such as Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE).

3. **Model Storage:**
   - Store trained models for each medical service in a dictionary (`models_dict`).

## LSTM Model

The `lstm_model` class in `lstmModel.py` provides the following functionality:

* **Data Preprocessing:**
   - Splits the data into training and testing sets.
   - Creates input and output sequences for the LSTM model.
* **Model Training:**
   - Defines the LSTM model architecture with an LSTM layer and a Dense layer.
   - Compiles the model using the Adam optimizer and mean squared error loss.
   - Trains the model using the training data and specified epochs and batch size.
* **Prediction:**
   - Predicts medical costs for the testing data.
* **Evaluation:**
   - Computes the MSE and MAPE to evaluate the model's performance.

## Dependencies

* Python 3
* pandas
* tqdm
* TensorFlow
* scikit-learn

## Data

The model is trained on a CSV file named `Fact_Table_Rev.csv` containing historical medical cost data. The file should have columns for date, medical service, company type, and amount.

## Notes

* The notebook provides an example with a sequence length of 10 and one epoch for demonstration purposes. You can adjust these parameters for optimal performance based on your data and requirements.
* The model can be extended to include additional features, such as patient demographics and medical history, to improve prediction accuracy.
* This repository is intended for educational purposes and may require modifications for real-world applications.

## License

This project is licensed under the Clinisys License.