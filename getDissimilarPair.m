function  pair1 = getDissimilarPair(imgs,idx)

listSize = size(imgs,4);
% Find all unique classes.

% Choose two different classes randomly which will be used to get a
% dissimilar pair.


pairIdx1=idx;


while idx ==pairIdx1
    classesChoice = randperm(listSize,1);
    pairIdx1 = classesChoice(1);

    pair1=imgs(:,:,:,pairIdx1);
end


end

