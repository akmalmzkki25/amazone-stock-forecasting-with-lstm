# amazone-stock-forecasting-with-lstm

Time series forecasting of Amazon (AMZN) daily closing prices with an LSTM network built in PyTorch. The work is a single Jupyter notebook that goes from raw price data to a forecast plot.

## What the notebook does

1. Loads a daily price CSV (`AMZN.csv`), checks for missing values and keeps only the `Date` and `Close` columns.
2. Plots the closing price history interactively with Plotly.
3. Builds a supervised dataset with a sliding window: each sample uses the previous 7 closing prices (lookback = 7) to predict the next one.
4. Scales the data to the range -1 to 1 with `MinMaxScaler` and splits it chronologically, 95% train and 5% test.
5. Wraps the tensors in a PyTorch `Dataset` and `DataLoader` (batch size 16).
6. Defines an LSTM model (`nn.LSTM` followed by a linear output layer, hidden size 4, one layer) and trains it with MSE loss and the Adam optimizer for 10 epochs, using a GPU when one is available.
7. Inverts the scaling and plots actual versus predicted closing prices for the train and test sets.

## Tech stack

- Python, Jupyter Notebook
- PyTorch
- pandas, NumPy, scikit-learn (scaling, metrics)
- Matplotlib, Plotly

## Files

- `amazone-stock-lstm.ipynb`: the full analysis and model
- `.gitignore`: standard Python ignores

## Running it

The notebook reads `AMZN.csv`, which is not stored in this repository. Place a price file with `Date` and `Close` columns next to the notebook, install the libraries listed above, and run the cells in order with Jupyter.
