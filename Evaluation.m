 clear all;
 close all;
 trainPath=append(fileparts(mfilename('fullpath')),'\FaceDatabase\Train\'); % These training/testing folders need to be in the same root folder of this code. 
 testPath=append(fileparts(mfilename('fullpath')),'\FaceDatabase\Test\');   % Or you can use the full folder path here
%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images

size(trainImgSet)  % if successfully loaded this should be with dimension of 600,600,3,100

%% Now we need to do facial recognition: Baseline Method
tic;
   outputID=FaceRecognition(trainImgSet, trainPersonID, testPath);
runTime=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputID(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy

% Method developed by you


imgsize=200;

%load pretrained model
disp("LOADING")
load net;
load fcParams;
disp("LOADED")



%load datasets if not loaded
if exist ('imdsTest','var') == 0
    disp("testing set not found... Loading testing set")
    imdsTest=testdata(testPath);
end
if exist ('trainImgSet','var') == 0
    disp("training set not found... Loading training set")
    [trainImgSet, trainPersonID]=loadTrainingSet(trainPath);
end

%create a list of class images to be compared with
trainingtrain=zeros([imgsize,imgsize,3 ,100]);
for k = 1:size(trainingtrain,4)
    trainingtrain(:,:,:,k)=im2single(imresize(trainImgSet(:,:,:,k),[imgsize imgsize]));
   
end

%initialise variables
TP=0;
TN=0;
FP=0;
FN=0;
classlen=size(imdsTest.Files,1);
trainlen=size(trainImgSet,4);
correctP=0;
classes=zeros(classlen,3);
trainingtrain=dlarray(trainingtrain,"SSCB");
tic;


for k=1:classlen

    %formate test image and predict distance
    testimg1=im2single(imresize(((getSimilarPair(imread(imdsTest.Files{k})))),[imgsize imgsize]));  %cycle through tess dataset
    testimg1 = dlarray(testimg1,"SSCB");

    Y= predictSiamese(net,fcParams,trainingtrain,testimg1);
    
    %calculate distance prediciton accuracy
    predround=round(extractdata(Y));
    if predround(:,int64(imdsTest.Labels(k,:)))==0
        TP=TP+1;
        FP=FP+100-sum(predround)-1;
        TN=TN+sum(predround);
    else
        FP=FP+100-sum(predround);
        FN=FN+1;
        TN=TN+sum(predround)-1;

    end

    %find 3 lowest distance classes
    Y=extractdata(Y);
    dvals=sort(Y);
    best=dvals(1:3);
    best=[find(best(1) == Y),find(best(2) == Y),find(best(3) == Y)];
    classes(k,:)=best;
    disp(best)
    disp(imdsTest.Labels(k,:))

    %check if true class has been predicted correctly
    memcheck=ismember(imdsTest.Labels(k,:),categorical(best));
    if (memcheck==1)
        correctP=correctP+1;
    end

end

classacc=(TP+TN)/(TP+TN+FP+FN)
methodNewTime=toc



recAccuracy=correctP/classlen*100  %Recognition accuracy
disp(recAccuracy)
disp("END")

