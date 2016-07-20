require 'nn'; lstm = require 'LSTM_LN';

a = lstm.lstm(10,10)

output = a:forward({torch.rand(100,10), torch.rand(100,10), torch.rand(100,10)})

print(output[1])
