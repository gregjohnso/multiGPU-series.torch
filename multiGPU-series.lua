require 'cutorch' -- Use CUDA if available
require 'cunn'
require 'nn'
require 'optim'

count = cutorch.getDeviceCount()
for i = 1, count do
    print('Device ' .. i .. ':')
    freeMemory, totalMemory = cutorch.getMemoryUsage(i)
    print('\t Free memory ' .. freeMemory)
    print('\t Total memory ' .. totalMemory)
end

-- assumes you have 3 gpus
gpu1 = 1
gpu2 = 2
gpu3 = 3

cutorch.setDevice(gpu1)

torch.manualSeed(1) 
cutorch.manualSeed(torch.random())

net = nn.Sequential()

inSize = 10
batchSize = 1000
niters = 250

-- simple autoencoder
net:add(nn.GPU(nn.Linear(inSize, 25), gpu1))
net:add(nn.GPU(nn.BatchNormalization(25), gpu1))

net:add(nn.GPU(nn.ReLU(true), gpu2))
net:add(nn.GPU(nn.Linear(25, 2), gpu2))
net:add(nn.GPU(nn.BatchNormalization(2), gpu2))

net:add(nn.GPU(nn.ReLU(true), gpu3))
net:add(nn.GPU(nn.Linear(2, 25), gpu3))
net:add(nn.GPU(nn.BatchNormalization(25), gpu3))

net:add(nn.GPU(nn.ReLU(true), gpu1))
net:add(nn.GPU(nn.Linear(25, inSize), gpu1))
net:add(nn.GPU(nn.BatchNormalization(inSize), gpu1))

criterion = nn.MSECriterion()

-- convert to cuda
net:cuda()
criterion:cuda()

-- optim parameters
optParam = {}
optParam.optimizer = 'rmsprop'
optParam.learningRate = 0.01
optParam.momentum = 0
optParam.numUpdates = 0

-- this variable records the state
optStates = {}

-- general function to evaluate
local function feval()
    net:zeroGradParameters()
    
    -- generate some data
    local x = torch.CudaTensor(batchSize, inSize):rand(batchSize, inSize)
    local y = x:clone()
    
    local yHat = net:forward(x)
    
    local loss = criterion:forward(yHat, y)

    local lossGrad = criterion:backward(yHat, y)
    
    net:backward(x, lossGrad)
    
    return loss, gradTheta
end

-- the function to perform the optimization 
local function optim_step(net, loss, optParam, optStates)
    -- this function assumes that all modules are nn.GPU-decorated
    local function feval_dummy(param)
        return loss, thisgrad
    end
    
    -- since each module has a block of parameters that may be on a different device, we optimaze each module one at a time
    local c = 1
    for i = 1, #net.modules do
        cutorch.setDevice(net.modules[i].device)

        local theta, gradTheta = net.modules[i]:parameters()
        
        for j = 1,#theta do
            local thisparam = theta[j]
            thisgrad = gradTheta[j]
            
            local optState = optStates[c] or {}
            optim[optParam.optimizer](feval_dummy, thisparam, optParam, optState)
            optStates[c] = optState
            c = c+1
        end
    end
end

-- main loop
for i = 1,niters do 
    -- do the forward/backward pass in feval()
    local loss = feval()
    
    -- update the parameter values in optim_step()
    optim_step(net, loss, optParam, optStates)

    print(loss)
    cutorch.setDevice(gpu1)
end



