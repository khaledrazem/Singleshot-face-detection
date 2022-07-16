filepath = fileparts(mfilename('fullpath'));
trainPath=append(filepath,'\FaceDatabase\Train\'); % These training/testing folders need to be in the same root folder of this code. 
testPath=append(filepath,'\FaceDatabase\Test\');   % Or you can use the full folder path here


%load datastores
loaded=0;
if exist ('trainImgSet','var') == 0
    disp("training set not found... Loading training set")
    [trainImgSet, trainPersonID]=loadTrainingSet(trainPath);
end
disp(exist ('net.mat','file'))
if exist ('net','var') ~= 0 && exist ('fcParams','var') ~= 0 
    disp("TRRAIN ALREADY LOADED")
    loaded=1;


%load pretrained model
elseif exist ('net.mat','file') ~= 0 && exist ('fcParams.mat','file') ~= 0
disp("saved model found... loading model")
load fcParams; 
load net;
loaded=1;
end


if loaded==0

imgsize=200;

layers = [  %define network layers
    imageInputLayer([imgsize imgsize 3],Normalization="none",name="inputlayer")
    convolution2dLayer(10,64,name='conv1',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer(name='relu1')
    maxPooling2dLayer(2,name='pool1',Stride=2)

    convolution2dLayer(7,128,name='conv2',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer(name='relu2')
    maxPooling2dLayer(2,name='pool2',Stride=2)

    convolution2dLayer(4,128,name='conv3',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer(name='relu3')
    maxPooling2dLayer(2,name='pool3',Stride=2)

    convolution2dLayer(5,256,name='conv4',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer(name='relu4')

    fullyConnectedLayer(4096,name='fc',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")];%3072

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

fcWeights = dlarray(0.01*randn(1,4096));
fcBias = dlarray(0.01*randn(1,1));

fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);


%define hyperparameters
numIterations = 10000; 
miniBatchSize = 64;

learningRate = 1e-5;
gradDecay = 0.9;
gradDecaySq = 0.99;

executionEnvironment = "Auto";


%initialise loss graph
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];

start = tic;

print("training")
% Loop over mini-batches.
for iteration = 1:numIterations

    % Extract mini-batch of image pairs and pair labels
    [X1,X2,X3] = getSiameseBatch(trainImgSet,trainPersonID,miniBatchSize,imgsize);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "SSCB" (spatial, spatial, channel, batch) for image data
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");
    X3 = dlarray(X3,"SSCB");
    
    % If training on a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
        X3 = gpuArray(X3);
    end

    % Evaluate the model loss and gradients using dlfeval and the modelLoss
    % function listed at the end of the example.
    [loss,gradientsSubnet,gradientsParams] = dlfeval(@modelLoss,net,fcParams,X1,X2,X3,miniBatchSize);

    % Update the Siamese subnetwork parameters.
    [net,trailingAvgSubnet,trailingAvgSqSubnet] = adamupdate(net,gradientsSubnet, ...
        trailingAvgSubnet,trailingAvgSqSubnet,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the fullyconnect parameters.
    [fcParams,trailingAvgParams,trailingAvgSqParams] = adamupdate(fcParams,gradientsParams, ...
        trailingAvgParams,trailingAvgSqParams,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    lossValue = double(extractdata(loss));
    lossValue=lossValue(1)
    addpoints(lineLossTrain,iteration,lossValue);
    title("Elapsed: " + string(D))
    drawnow
end

disp("saving model")
save(filepath,'net')
save(filepath,'fcParams')

end

disp("TESTING")
if exist ('imdsTest','var') == 0
    disp("testing set not found... Loading testing set")
    imdsTest=testdata(testPath);
end


TP=0;
TN=0;
FP=0;
FN=0;


classlen=size(imdsTest.Files,1);
trainlen=size(trainImgSet,4);


trainingtrain=zeros([imgsize,imgsize,3 ,100]);
for k = 1:size(trainingtrain,4)
    trainingtrain(:,:,:,k)=im2single(imresize(trainImgSet(:,:,:,k),[imgsize imgsize]));
   
end
tic;
classes=zeros(classlen,3);

for k=1:classlen
    testin=im2single(imresize(((getSimilarPair(imread(imdsTest.Files{k})))),[200 200]));
    bbatch = dlarray(testin,"SSCB");
    Y= predictSiamese(net,fcParams,dlarray(trainingtrain,"SSCB"),bbatch);
    Y=extractdata(Y);
    dvals=sort(Y);
    best=dvals(1:3);
    best=[find(best(1) == Y),find(best(2) == Y),find(best(3) == Y)];
    classes(k,:)=best;
    disp(k)
    disp(best)
end

runTime=toc

correctP=0;
for i=1:size(classes,1)
    memcheck=ismember(imdsTest.Labels(i,:),categorical(classes(i,:)));
    if (memcheck==1)
        correctP=correctP+1;
    end
end
recAccuracy=correctP/classlen*100  %Recognition accuracy
disp(recAccuracy)
disp("END")


