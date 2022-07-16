

function imdsTest=loadTrainingSet(testPath)

imdsTest = imageDatastore(testPath, ...
    IncludeSubfolders=true, ...
    LabelSource="none");



load testLabel;
testLabel=num2cell(testLabel,2);
testLabel=arrayfun( @(x) str2double(x(:)), testLabel);

imdsTest.Labels = categorical(testLabel);

end
