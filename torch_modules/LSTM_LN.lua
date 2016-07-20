require 'nngraph'
require 'LayerNormalization'

local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm(inputSize, hiddenSize)
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum_bias(bias)
        -- transforms input
        local i2h            = nn.Linear(inputSize, hiddenSize)(x)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(hiddenSize, hiddenSize)(prev_h)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum_bias(0))
    local forget_gate      = nn.Sigmoid()(new_input_sum_bias(-4.))
    local out_gate         = nn.Sigmoid()(new_input_sum_bias(0))
    local in_transform     = nn.Tanh()(new_input_sum_bias(0))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    -- layer normalizes the cell output
    local ln = nn.LayerNormalization(hiddenSize)
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(ln(next_c))})

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end

return LSTM

