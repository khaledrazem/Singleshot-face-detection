load('rcnnStopSigns.mat','cifar10Net')  ;
testPath='C:\Users\khale\Documents\MATLAB\Lab5Materials\FaceDatabase\Test\';   % Or you can use the full folder path here
trainPath='C:\Users\khale\Documents\MATLAB\Lab5Materials\FaceDatabase\Train\';
testImgNames=dir([testPath,'*.jpg']);
testimgSet=zeros(32,32,3,200);
[trainImgSet, trainPersonID]=loadTrainingSet(trainPath);

imshow(trainImgSet(:,:,:,1))
% k=1;
% for i=1:size(testImgNames,1)
%     
%     if ~isempty(testImgNames(i,:).name) && k<200
%         testimgSet(:,:,:,k)=imresize(imread([testPath, testImgNames(i,:).name]),[32,32]);
%         k=k+1;
%     end
% end
% testimgSet=uint8(testimgSet(:,:,:,1:k-1));
% test=testimgSet(:,:,:,5)
% size(test)
% ans=classify(cifar10Net,test)
