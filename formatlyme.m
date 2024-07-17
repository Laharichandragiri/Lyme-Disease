% Load necessary libraries and dependencies
clc;
clear;
close all;

% Load your dataset and perform data preprocessing
dataDir = 'DATA'; % Specify the path to your dataset
imds = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the dataset into training and testing sets (80-20 split)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Define the input image size
imageSize = [256 256 3];

% Define the layers of the CNN model with input size 256x256
layers = [
    imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'zscore')
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];


% Define label smoothing factor
smoothingFactor = 0.1;

% Convert labels to one-hot encoding
Y_train = double((imdsTrain.Labels));

% Apply label smoothing regularization
Y_train_smoothed = (1 - smoothingFactor) * Y_train + smoothingFactor / numClasses;

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 12, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10);

% Train the CNN model with label smoothing
net = trainNetwork(imdsTrain, layers, options);

% Evaluate the model on the test set
Y_test = imdsTest.Labels;

% Calculate ROC curve
[Y_score, ~] = classify(net, imdsTest);
% Evaluate the model on the test set
Y_pred = classify(net, imdsTest);
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
% Convert categorical labels to numeric form
Y_test_numeric = double(Y_test) - 1; % Convert to 0 and 1
% Obtain scores for positive class
Y_scores = net.predict(imdsTest);
% Convert scores to numeric form
Y_scores_numeric = Y_scores(:,2); % Scores for the positive class
% Calculate ROC curve
[X,Y,~,AUC] = perfcurve(Y_test_numeric,Y_scores_numeric,1);
% Display results
disp(['Test accuracy: ', num2str(accuracy * 100), '%']);
disp(['Area under ROC curve: ', num2str(AUC)]);
% Plot ROC curve
figure;
plot(X,Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
% Convert categorical labels to numeric form
Y_test_numeric = double(Y_test) - 1; % Convert to 0 and 1
% Obtain scores for positive class
Y_scores = net.predict(imdsTest);
% Convert scores to numeric form
Y_scores_numeric = Y_scores(:,2); % Scores for the positive class
% Calculate ROC curve and ROC AUC
[X_roc,Y_roc,~,AUC_roc] = perfcurve(Y_test_numeric,Y_scores_numeric,1);
% Calculate precision-recall curve and PR AUC
[precision, recall, ~, AUC_pr] = perfcurve(Y_test_numeric, Y_scores_numeric, 1, 'XCrit', 'reca', 'YCrit', 'prec');
% Display results
% disp(['Test accuracy: ', num2str(accuracy * 100), '%']);
disp(['ROC AUC: ', num2str(AUC_roc)]);

disp(['PR AUC: ', num2str(AUC_pr)]);
confMat = confusionmat(Y_test, Y_pred);
% Calculate precision, recall, and F1 score
precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);
disp(['ROC AUC: ', num2str(accuracy * 100), '%']);
disp(['PR AUC: ', num2str(accuracy * 100), '%']);
% Display results
disp(['Test accuracy: ', num2str(accuracy * 100), '%']);
disp('Confusion Matrix:');
disp(confMat);
disp('Precision:');
disp(precision');
disp('Recall:');
disp(recall);
disp('F1 Score:');
disp(f1Score');
% Plot confusion matrix
figure;
confusionchart(Y_test, Y_pred);
