
load net

analyzeNetwork(net)

layer = 2;
name = net.Layers(layer).Name

channels = 1:36;
I = deepDreamImage(net,name,channels, ...
    'PyramidLevels',1)