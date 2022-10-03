clear all
%%
imageFolder='.\train_';
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
%%
[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
%%
net = vgg16;

imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
%%
layers = net.Layers;
layers(23) = fullyConnectedLayer(7);
layers(25) = classificationLayer;

opts = trainingOptions('sgdm',...
    'InitialLearnRate', 0.001,...
    'MaxEpochs', 30,...
    'MiniBatchSize', 64,...
    'Plots','training-progress');
%%
%augmentedTrainingSet.ReadFcn = @readFunctionTrain;
net = trainNetwork(augmentedTrainingSet, layers, opts);
%%
predictedLabels = classify(net, augmentedTestSet);

testLabels=testSet.Labels;

accuracy = mean((predictedLabels == testSet.Labels))*100

confMat1 = confusionchart(testLabels, predictedLabels);

%performance metrics calculation
confMat = confusionmat(testLabels, predictedLabels);
[Precision,Recall,f1_score] = Perfomance(confMat)
