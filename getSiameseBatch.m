function [X1,X2,X3] = getSiameseBatch(imds,labels,miniBatchSize,imgsize)




X1 = zeros([imgsize,imgsize,3 ,miniBatchSize]);
X2 = zeros([imgsize,imgsize,3 ,miniBatchSize]);
X3 = zeros([imgsize,imgsize,3 ,miniBatchSize]);
for i = 1:miniBatchSize
   

    choice = rand(1);
    idx=round(unifrnd(1,100));
    
    
    simimg1=imds(:,:,:,idx);
    simimg2=imds(:,:,:,idx);

    if choice>0.03
        simimg1 = getSimilarPair(simimg1);
        simimg2 = getSimilarPair(simimg2);
    end
    X1(:,:,:,i) = im2single(imresize(simimg1, [imgsize imgsize]));
    X2(:,:,:,i) = im2single(imresize(simimg2, [imgsize imgsize]));
 
%         figure
%         disp("SIM")
%         subplot(1,2,1), imshow(X1(:,:,:,i))
%         subplot(1,2,2), imshow(X2(:,:,:,i))



    pair1 = getDissimilarPair(imds,idx);
    if choice<0.97
        pair1=getSimilarPair(pair1);

    end
    X3(:,:,:,i) = im2single(imresize(pair1, [imgsize imgsize]));

%         disp("disSIM")

%         subplot(1,2,1), imshow(X1(:,:,:,i))
%         subplot(1,2,2), imshow(X2(:,:,:,i))




    
end

end
