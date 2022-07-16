% 
idx=1;
training=im2single(imresize((getSimilarPair( trainImgSet(:,:,:,idx))),[200 200]));
testin=im2single(imresize(((getSimilarPair(imread(imdsTest.Files{15})))),[200 200]));


% trainimage=trainImgSet(:,:,:,idx)
%[abatch,bbatch,lbatch]=getSiameseBatch(trainImgSet,trainPersonID,10,200);
% aug=getSimilarPair(trainImgSet(:,:,:,idx));

%montage({(abatch(:,:,:)),bbatch(:,:,:)})
montage({training,testin})
testingtrain=im2single(imresize(imread(imdsTest.Files{624}),[100 100]));
   


% trainimage=dlarray(im2single(abatch(:,:,:,idx)),"SSCB");
% testimage=dlarray((im2single(bbatch(:,:,:,idx))),"SSCB");
abatch = dlarray(training,"SSCB");
bbatch = dlarray(testin,"SSCB");
lbatch = dlarray(lbatch,"SSCB");

% x1=dlarray(im2single(imresize(abatch(:,:,:,idx),[140 140])),"SSCB");
% x2=dlarray(im2single(imresize(bbatch(:,:,:,idx),[140 140])),"SSCB");
% 
% trainaug=dlarray(im2single(imresize(aug{1},[140 140])),"SSCB");
% 
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
    disp(best)
end

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
% Y1= abs(predictSiamese(net,fcParams,abatch,lbatch)).^2
% 
% precision = underlyingType(Y);
% % Convert values less than floating point precision to eps.
% Y(Y < eps(precision)) = eps(precision);
% % Convert values between 1-eps and 1 to 1-eps.
% Y(Y > 1 - eps(precision)) = 1 - eps(precision);
% 
% precision = underlyingType(Y1);
% % Convert values less than floating point precision to eps.
% Y1(Y1 < eps(precision)) = eps(precision);
% % Convert values between 1-eps and 1 to 1-eps.
% Y1(Y1 > 1 - eps(precision)) = 1 - eps(precision);
% sum(Y-Y1)+0.5
% sum(abs(pdist(extractdata(prediction),extractdata(prediction1))));
% % [x, best] = max(prediction)
% % disp(best)


% 
% correctP=0;
% for i=1:size(classes,1)
%     if (categorical(classes(i,:))==imdsTest.Labels(i,:))
%         correctP=correctP+1;
%     end
% end
% recAccuracy=correctP/classlen*100  %Recognition accuracy
% disp(recAccuracy)
% disp("END")
