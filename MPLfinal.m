% ************************************************************************
%                   HYPER-PARAMETER BAYESIAN OPTIMIZATION
% ************************************************************************
% This script performs both Bayesian Optimization and Gradient Descent to 
%search for the best combination of values for hyper-parameters for Feed 
%Forward Nueral Net Network. It studies MLP with upto two hidden
%layers and upto 40 nuerons in each layer.

%% Read the dataset
Daten = readtable('C:\Users\Prerna Prakash Gupta\Desktop\nndata5000.xlsx');
Daten=table2array(Daten)
[m,n] = size(Daten) ;
q=Daten(:,end-1)
%% Partition the data into stratified train and straified test set (class imbalance), with 30% test hold-out dataset
train_test_stratified=cvpartition(q,'HoldOut',0.3);
traindata=Daten(train_test_stratified.training,:);
testdata=Daten(train_test_stratified.test,:);
xtest=testdata(:,1:end-2);
ytest=testdata(:,end-1:end);
ytest=vertcat(ytest')
ytest=vec2ind(ytest)
xtrain=traindata(:,1:end-2);
ytrain=traindata(:,end-1:end);
[rows_train,cols_train]=size(xtrain);
q1=ytrain(:,end-1)
%% Create stratified dataset 10-fold cross-validation from the 70% training data
K=10;
cv_partitions=cvpartition(q1,'KFold',K);
%% Create stratified dataset 10-fold cross-validation from the 70% training data
K=10;
cv_partitions=cvpartition(q1,'KFold',K);
validation_performance=[];
validation_f_score=[];
prectrain=[]
recalltrain=[]
mcv=[]
%% MLP with single Hidden Layer
%In order to evaluate the dispersion of Test F1 score, every hyper-parameter combination is evaluated on the 30% Hold-Out test set

%N=No. of nuerons in hidden layer
N = [ 5, 10, 15, 20, 25, 30, 35, 40 ];
%learn-rate combinations
learnRate = [0.1 0.25 0.5 1];
%two optimisation algorithms
%"trainbr"=Bayesian Regularization
%"traingd"=Gradient Descent
TF = ["traingd", "trainbr"];
%Two activation Functions: Log-sig and Poslin
activationfun = ["logsig","poslin"];
%weightreg corresponds to weight momentum
weightreg=[0 0.0001 0.001 0.1 ]
Scores_EB=[]
tElapsed=[]

tstart=tic;
for ii = 1:length(N)
      % Create a Fitting Network & set number of neurons
      hiddenLayerSize = N(ii); 
      %K-fold cross validation for hyperparameter combinations
      for k = 1: length(TF)
          trainFcn = TF(k);
          net = patternnet(hiddenLayerSize,trainFcn)
          for q = 1: length(learnRate)
              net.trainParam.lr = learnRate(q) ;
              %set cross-entrophy loss function to minimise back
              %propogation cross entrophy error in each iteration
              net.performFcn = 'crossentropy'
              %Early-stopping criterion at 10 epochs
              net.trainParam.max_fail = 10;
              %Set maximum no. of epochs to 400
              net.trainParam.epochs = 400;
              for w =1:length(weightreg)
                  weighreg1=weightreg(w)
                  net.performParam.regularization=weighreg1
                  %activation f(n) for input layers
                  for a=1:length(activationfun)
                      actfun=activationfun(a)
                      net.layers{1}.transferFcn = actfun;
                      %setting output layer as soft-max for classification
                      net.layers{2}.transferFcn = 'softmax';
                      %K-fold Cross validation for different
                      %hyperparameters
                      for i=1:K
                          cv_train_index=cv_partitions.training(i);
                          cv_validation_index=cv_partitions.test(i);
                          cv_train_x=xtrain(cv_train_index,:);
                          cv_train_y=ytrain(cv_train_index,:);
                          cv_validation_y=ytrain(cv_validation_index,:);
                          cv_validation_x=xtrain(cv_validation_index,:);
                          % Model fit on training data and evaluated on validation data for every fold
                          net = train(net,cv_train_x',cv_train_y');
                          predicted_validation=net(cv_validation_x');
                          predicted_validation=round(predicted_validation);
                          predicted_validation=vec2ind(predicted_validation);
                          cv_validation_y=vec2ind(cv_validation_y');
                          perfvalidation = perform(net,predicted_validation,cv_validation_y);
                          %Confusion Matrix, F1 -score, misscallification erorr;Precision and Recall for every out of sample validation fold
                          misclassval=sum(predicted_validation-cv_validation_y)/length(predicted_validation)
                          confMat = confusionmat(cv_validation_y, predicted_validation);
                          for j =1:size(confMat,1)
                          recall(j)=confMat(j,j)/sum(confMat(j,:));
                          end
                          recall(isnan(recall))=[];
                          Recall=sum(recall)/size(confMat,1)
                          for j =1:size(confMat,1)
                              precision(j)=confMat(j,j)/sum(confMat(:,j));
                          end
                          precision(isnan(precision))=[];
                          Precision=sum(precision)/size(confMat,1)
                          Fscorevalidation=2*Recall*Precision/(Precision+Recall);
                          prectrain=[prectrain,Precision ]
                          recalltrain=[recalltrain,Recall ]
                          validation_performance=[validation_performance; perfvalidation];
                          validation_f_score=[validation_f_score; Fscorevalidation];
                          mcv=[mcv,misclassval]
                      end
                      % Calculating the mean F1 -score,misscallification erorr;Precision and Recall
                      %for all out-of sample validation scores to get cross-validation metrics
                      validation_perf=mean(validation_performance)
                      validation_f_score=mean(validation_f_score)
                      prectrainval=mean(prectrain)
                      reaclltrainval=mean(recalltrain)
                      MCV= mean(mcv)
                      %Time calculation post K-fold cross validation for the given hyperparameter combination
                      tElapsed = toc(tStart);
                      
                      %Evaluate every-hyper parameter-combination on hold-out test data to study dispersion of test scores
                      ytestpredicted=net(xtest');
                      ytestpredicted =round(ytestpredicted);
                      ytestpredicted =vec2ind(ytestpredicted)
                      perftest = perform(net,ytestpredicted,ytest);
                      MCT= sum(ytestpredicted-ytest
                      confMat1 = confusionmat(ytest, ytestpredicted);
                      for j1 =1:size(confMat1,1)
                          recall(j1)=confMat1(j1,j1)/sum(confMat1(j1,:));
                      end
                      recall(isnan(recall))=[];
                      Recall=sum(recall)/size(confMat1,1)
                      for j1 =1:size(confMat1,1)
                          precision(j1)=confMat1(j1,j1)/sum(confMat1(:,j1));
                      end
                      precision(isnan(precision))=[];
                      Precision=sum(precision)/size(confMat1,1)
                      Fscoretest=2*Recall*Precision/(Precision+Recall);
                      Scores_EB = [Scores_EB; N(ii), TF(k),learnRate(q),weightreg(w),MCV, MCT,activationfun(a),tElapsed,prectrainval, reaclltrainval, validation_perf, validation_f_score,Fscoretest, perftest,Recall,Precision]
                  end
              end
          end
      end
end 
%save the grid search
writematrix(Scores_EB,'C:\Users\Prerna Prakash Gupta\Desktop\narthur1layerppg.csv')

%% Grid search for two hidden layers
%In order to evaluate the dispersion of Test F1 score, every hyper-parameter combination is evaluated on the 30% Hold-Out test set
Scores_EB1=[]
%N=number of nuerons in layer 1
N = [ 5, 10, 15, 20, 25, 30, 35, 40];
%N1=number of nuerons in layer 2
N1 = [ 5, 10, 15, 20, 25, 30, 35, 40];
%two optimisation algorithms
%"trainbr"=Bayesian Regularization
%"traingd"=Gradient Descent
TF = ["traingd", "trainbr"];
%Learning Rate
learnRate = [0.1 0.25 0.5 1];

%Two choices Log-sig and Poslin activation functions for each layer
activationfun = ["logsig","poslin"];
%Weight Momentum Parameters
weightreg=[0 0.0001 0.001 0.1 ]
Scores_EB1=[]

%start time
tstart=tic;
% Fitting Network & set number of neurons for each layer (N and N1)
for  I = 1:length(N)
    hiddenLayerSize1 = N(I);
    for ii= 1:length(N1)
        hiddenLayerSize2 = N1(ii);
        % %K-fold cross validation for hyperparameter combinations
        for k = 1: length(TF)
              trainFcn = TF(k);
              net = patternnet([hiddenLayerSize1, hiddenLayerSize2], trainFcn);
              for q = 1: length(learnRate)
                  net.trainParam.lr = learnRate(q) ;
                  net.performFcn = 'crossentropy';
                  %Earlt stopping criterion after 10 epochs
                  net.trainParam.max_fail = 10;
                  %Set maximum no. of epochs to 400
                  net.trainParam.epochs = 400;
                  for w =1:length(weightreg)
                      %diff values of weight regularization
                      weighreg1=weightreg(w);
                      net.performParam.regularization=weighreg1;
                      %two -hidden layers
                      for a=1:length(activationfun)
                          actfun=activationfun(a)
                          net.layers{1}.transferFcn = actfun;
                          net.layers{2}.transferFcn = actfun;
                          %setting hyper-parameter as soft-max in poutput
                          %layer
                          net.layers{3}.transferFcn = 'softmax';
                          %10-fold cross validation
                          for i=1:K
                          cv_train_index=cv_partitions.training(i);
                          cv_validation_index=cv_partitions.test(i);
                          cv_train_x=xtrain(cv_train_index,:);
                          cv_train_y=ytrain(cv_train_index,:);
                          cv_validation_y=ytrain(cv_validation_index,:);
                          cv_validation_x=xtrain(cv_validation_index,:);
                          net = train(net,cv_train_x',cv_train_y');
                          %Confusion Matrix, F1 -score, misscallification erorr;Precision and Recall for every out of sample validation fold
                          predicted_validation=net(cv_validation_x');
                          predicted_validation=round(predicted_validation);
                          predicted_validation=vec2ind(predicted_validation);
                          cv_validation_y=vec2ind(cv_validation_y');
                          perfvalidation = perform(net,predicted_validation,cv_validation_y);
                          misclassval=sum(predicted_validation-cv_validation_y)/length(predicted_validation)
                          confMat = confusionmat(cv_validation_y, predicted_validation);
                          for j =1:size(confMat,1)
                          recall(j)=confMat(j,j)/sum(confMat(j,:));
                          end
                          recall(isnan(recall))=[];
                          Recall=sum(recall)/size(confMat,1);
                          for j =1:size(confMat,1)
                              precision(j)=confMat(j,j)/sum(confMat(:,j));
                          end
                           % Calculating the mean F1 -score,misscallification erorr;Precision and Recall
                           %for all out-of sample validation scores to get cross-validation metrics
                          precision(isnan(precision))=[];
                          Precision=sum(precision)/size(confMat,1)
                          Fscorevalidation=2*Recall*Precision/(Precision+Recall);
                          prectrain=[prectrain,Precision ]
                          recalltrain=[recalltrain,Recall ]
                          validation_performance=[validation_performance; perfvalidation];
                          validation_f_score=[validation_f_score; Fscorevalidation];
                          mcv=[mcv,misclassval]
                         
                          
                          %Time calculation post K-fold cross validation for the given hyperparameter combination
                          tElapsed = toc(tStart);
                          end
                      
                      validation_perf=mean(validation_performance)
                      validation_f_score=mean(validation_f_score)
                      prectrainval=mean(prectrain)
                      reaclltrainval=mean(recalltrain)
                      MCV= mean(mcv)
                      
                      %Evaluate every-hyper parameter-combination on hold-out test data to study dispersion of test scores
                      ytestpredicted=net(xtest');
                      ytestpredicted =round(ytestpredicted);
                      ytestpredicted =vec2ind(ytestpredicted)
                      perftest = perform(net,ytestpredicted,ytest);
                      MCT= sum(ytestpredicted-ytest)
                      confMat1 = confusionmat(ytest, ytestpredicted);
                      for j1 =1:size(confMat1,1)
                          recall(j1)=confMat1(j1,j1)/sum(confMat1(j1,:));
                      end
                      recall(isnan(recall))=[];
                      Recall=sum(recall)/size(confMat1,1);
                      for j1 =1:size(confMat1,1)
                          precision(j1)=confMat1(j1,j1)/sum(confMat1(:,j1));
                      end
                      precision(isnan(precision))=[];
                      Precision=sum(precision)/size(confMat1,1)
                      Fscoretest=2*Recall*Precision/(Precision+Recall);
                      Scores_EB1 = [Scores_EB1; N(I),N1(ii), TF(k),learnRate(q),weightreg(w),MCV, MCT,activationfun(a),tElapsed,prectrainval, reaclltrainval, validation_perf, validation_f_score,Fscoretest, perftest,Recall,Precision]
                  end
              end
         end
        end   
    end 
end
%save the grod-search results for two-hidden layers
writematrix(Scores_EB,'C:\Users\Prerna Prakash Gupta\Desktop\narthur2layerppg.csv')



