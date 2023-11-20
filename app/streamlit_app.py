import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import torch
import torch.nn as nn
import numpy as np


class Receipt_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Receipt_Predictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x,_ = self.lstm3(x)
        x = self.dense(x[:, -1, :])  # Select the last time step output
        return x
model = Receipt_Predictor(1,64,1)
model.load_state_dict(torch.load('data/lstm_weights.pt'))

file = open('data/loss_history.json')
loss_history = json.load(file)
file.close()

df = pd.DataFrame.from_dict(loss_history)
fig = px.line(df, title = 'Loss curve').update_layout(
  xaxis_title = 'Epoch',
  yaxis_title = 'Loss'
)
st.plotly_chart(fig)



user_input = st.text_input(label = 'Enter a month')
user_input = user_input.lower()
days = {
  'january' : (1,31),
  'february' : (32, 58),
  'march' : (59,89),
  'april' : (90,119),
  'may' : (120,150),
  'june' : (151, 180),
  'july' : (181, 211),
  'august' : (212, 242),
  'september' : (243, 272),
  'october' : (273, 303),
  'november' : (304, 333),
  'december' : (334, 364)
}
if user_input:
  window_size = 30
  start, num_future_steps = days[user_input]
  y = np.loadtxt('data/percent.txt')
  y = torch.tensor(y, dtype = torch.float32)
  df = pd.read_csv('data/data_daily.csv')
  num_future_steps = 365
  predictions = []
  current_input = y[-window_size:]

  model.eval()

  for i in range(num_future_steps):
      # Convert current_input to PyTorch tensor
      current_input_tensor = torch.Tensor(current_input).view(1, window_size, 1)

      # Use the model to make a prediction
      with torch.no_grad():
          current_prediction = model(current_input_tensor)

      # Convert the prediction to a NumPy array
      current_prediction = current_prediction.view(-1).numpy()
      
      predictions.append(current_prediction)

      # Update current_input for the next iteration
      current_input = np.append(current_input, predictions[-1])
      current_input = current_input[-window_size:]

  preds = [] 
  firstPreds = df['Receipt_Count'].values.tolist()
  for i,p in enumerate(predictions):
    if i == 0: preds.append((1+p)*firstPreds[-1])
    else: preds.append((1+p)*preds[-1])
  
  res = preds[start:]

  fig2 = px.line(res).update_layout(
    title = 'Total receipt count for ' + user_input[0].upper() + user_input[1:] + ' = ' + str(int(sum(res))),
    xaxis_title = 'Days',
    yaxis_title = 'Receipt Count'
  )
  st.plotly_chart(fig2)