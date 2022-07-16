function [loss,gradientsSubnet,gradientsParams] = modelLoss(net,fcParams,X1,X2,X3,batchsize)

% Pass the similar image pair through the network.
Y1 = abs(forwardSiamese(net,fcParams,X1,X2)).^2;
Y1=Cleaneps(Y1);

% Pass the dissimilar image pair through the network.
Y2 = abs(forwardSiamese(net,fcParams,X1,X3)).^2;
Y2=Cleaneps(Y2);

% Calculate distance between the two values.
loss=sum(Y1 - Y2)/batchsize

loss = max([loss+0.9,0]); %maximise distance between them

% Calculate gradients of the loss with respect to the network learnable
% parameters.
[gradientsSubnet,gradientsParams] = dlgradient(loss,net.Learnables,fcParams);

end

