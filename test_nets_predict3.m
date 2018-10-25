% x = randn(100,1); 
% y = x * rand + rand; 

y = randn(100,1); 
x = y * rand + rand; 

my = mean(y); mx = mean(x);
b = x' * y / (y' * y);
v = y' * x / (x' * x);
yhat1 = (x - mx) / b; 
yhat2 = x * v + my;
figure(1) 
subplot(1,3,1)
scatter(y,yhat1); ylim([-3 3]); xlim([-3 3])
subplot(1,3,2)
scatter(y,yhat2); ylim([-3 3]); xlim([-3 3])
subplot(1,3,3)
scatter(yhat1,yhat2); ylim([-3 3]); xlim([-3 3])
[sum((yhat1-y).^2) sum((yhat2-y).^2)]



%%

addpath('../shared/')
addpath('../../Regression/glmnet_matlab/')
load arrhythmia

cs = [];
Permutations = [];
confounds = [];

parameters = struct();
parameters.relaxed = 0; 
parameters.biascorrect = 0; 
[predictedY00,stats00] = nets_predict3(Y,X,family,parameters,cs,Permutations,confounds);

parameters.relaxed = 1; 
parameters.biascorrect = 0;
[predictedY10,stats10] = nets_predict3(Y,X,family,parameters,cs,Permutations,confounds);

parameters.relaxed = 0; 
parameters.biascorrect = 1;
[predictedY01,stats01] = nets_predict3(Y,X,family,parameters,cs,Permutations,confounds);

parameters.relaxed = 1; 
parameters.biascorrect = 1;
[predictedY11,stats11] = nets_predict3(Y,X,family,parameters,cs,Permutations,confounds);


%%

%%%%%%%%%%%%%%%%%%%%

% real age is NPYoutC 
% predicted age is NPpredictedYC 

%%%%%%%%% adjust for the bias in the regularisation in elastic net
%%%%  NPpredictedYC = [NPYoutC ones] grotbeta + error;
grot=[NPYoutC ones(length(NPYoutC),1)];     grotbeta = pinv(grot) * NPpredictedYC;
% pred = age * m + c
% ( pred - c ) / m = ageNewlyPredicted
NPpredictedYC2 = ( NPpredictedYC - grotbeta(2) ) / grotbeta(1);

%%%%%%%%% or - more principled - LOO-based correction
grotI=randi(10,length(NPYoutC),1); NPpredictedYC2=0*NPYoutC/0;
for i=1:10
  grotOUT=grotI==i;   grotIN=grotI~=i;
  grot=[NPYoutC(grotIN) ones(length(NPYoutC(grotIN)),1)]; grotbeta = pinv(grot) * NPpredictedYC(grotIN);
  NPpredictedYC2(grotOUT) = ( NPpredictedYC(grotOUT) - grotbeta(2) ) / grotbeta(1);
end

subplot(2,2,1); dscatter(NPYoutC,NPpredictedYC); hold on; plot([40 85],[40 85],'k','LineWidth',3);
subplot(2,2,3); dscatter(NPYoutC,NPpredictedYC-NPYoutC)
subplot(2,2,2); dscatter(NPYoutC,NPpredictedYC2); hold on; plot([40 85],[40 85],'k','LineWidth',3);
subplot(2,2,4); dscatter(NPYoutC,NPpredictedYC2-NPYoutC)