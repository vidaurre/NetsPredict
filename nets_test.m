function [predictedY,predictedYD] = nets_test(Xin,estimation,dec_estimation,confounds)
%
% Elastic-net training of betas, with two-stage feature selection,
% using (stratified) LOO and permutation testing
%
% Diego Vidaurre
% Aarhus Uni / OHBA Oxford, 2020
%
% Ypredicted = nets_test(X,estimation);
% Ypredicted = nets_test(X,estimation,dec_estimation,confounds);

% INPUTS
% X - predictor matrix (samples X features)
% estimation - Estimated model from nets_train
% confounds (optional) - features that potentially influence the inputs, 
%       and the outputs for family="gaussian'
%
% OUTPUTS
% Ypredicted - Predicted responses from Xin
% Ypredicted - Predicted responses from Xin in deconfounded space 

Method = estimation.Method;
family = estimation.family;
enet = strcmpi(Method,'lasso') || strcmpi(Method,'glmnet');

% Standardizing Xin
mx = estimation.mx; sx = estimation.sx; N = size(Xin,1); 
Xin = Xin - repmat(mx,N,1);
Xin(:,sx>0) = Xin(:,sx>0) ./ repmat(sx(sx>0),N,1);

% Deconfounding X
if nargin > 2
try
    [~,Xin] = nets_deconfound(Xin,[],confounds,'gaussian',dec_estimation.betaX);
catch
   error('Both deconfounding parameters and confounds have to be supplied in the right format') 
end

beta = estimation.beta; 

if strcmpi(family,'gaussian')
    if enet
        a0 = estimation.a0; 
        predictedY = Xin * beta + a0;
    else % ridge or unregularized
        Xin = [ones(size(Xin,1),1) Xin];
        predictedY = Xin * estimation.beta;
    end
elseif strcmpi(family,'multinomial') % deconfounded space, both predictedYp and predictedYp0
    if strcmpi(Method,'glmnet')
        predictedY = nets_glmnetpredict(estimation,Xin,estimation.lambda(end),'response');
    elseif strcmpi(Method,'lasso')
        a0 = estimation.a0;
        beta_final = [a0; beta];
        predictedY = glmval(beta_final,Xin,'logit');
    elseif strcmpi(Method,'unregularized')
        predictedY = mnrval(estimation.beta,Xin);
        if any(isnan(predictedY(:))) % unregularized logistic often goes out of precision
            predictedY(isnan(predictedY)) = 1;
            predictedY = predictedY ./ repmat(sum(predictedY,2),1,q);
        end
    end
elseif strcmpi(family,'poisson')
    if strcmpi(Method,'glmnet')
        predictedY = ...
            max(nets_glmnetpredict(estimation,Xin,estimation.lambda(end),'response'),eps);
    elseif strcmpi(Method,'lasso')
        a0 = estimation.a0;
        beta_final = [a0; beta];
        predictedY = glmval(beta_final,Xin,'log');
    end
    %predictedYp(J) = exp(XJ * glmfit.beta(:,end) + glmfit.a0(end) );
else % cox
    predictedY = exp(Xin * estimation.beta);
end

predictedYD = predictedY;
if nargin > 2 && isfield(dec_estimation,'betaY')
    predictedY = nets_confound(predictedY,confounds,estimation.family,dec_estimation.betaY);
end

end


