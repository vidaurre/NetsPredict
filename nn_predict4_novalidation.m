% nn_predict - nearest-neighbour estimation using a distance matrix
% using (stratified) LOO and permutation testing
% Diego Vidaurre 
%
% [predictedY,stats] = nn_predict4(Y,X,family,parameters);
% [predictedY,stats] = nn_predict4(Y,X,family,parameters,correlation_structure);
% [predictedY,stats] = nn_predict4(Y,X,family,parameters,correlation_structure,Permutations);
% [predictedY,stats] = nn_predict4(Y,X,family,parameters,correlation_structure,Permutations,confounds);
%
% INPUTS
% Y - response vector (samples X 1)
% X - predictor matrix (samples X features)
% family - probability distribution of the response, one of the following:
%   + 'gaussian': standard linear regression on a continuous response
%   + 'poisson': non-negative counts
%   + 'multinomial': a binary-valued matrix with as columns as classes
%   + 'cox': a two-column matrix with the 1st column for time and the 2d for status: 1 for death and 0 right censored.
%   If a list with two elements, the first is the family for the feature selection stage (typically gaussian)
% parameters is a structure with:
%   + Method - One of 'glmnet', 'lasso' or 'ridge'. Glmnet and lasso are
%           the elastic net, but with different code: glmnet uses the glmnet
%           package, essentially a matlab wrap of Fortran code - it is very quick
%           but ocassionally crashes taking down the whole Matlab; 'lasso' uses the
%           matlab function for the elastic net, it is considerable slower and only works for
%           the 'gaussian' family, but never crashes. 
%           Finally, 'ridge' is just ridge regression with built-in code.
%   + Nfeatures - Proportion of features to initially filter  
%             if Nfeatures has an optional second component, for early stopping on Elastic net
%   + alpha - a vector of weights on the L2 penalty on the regression
%           coefficients, if 'Method' is 'lasso' or 'glmnet'; otherwise, it is the
%           values of the ridge penalty. 
%   + riemann - run in tangent space, closer to riemannian geometry? 
%   + deconfounding - a vector of two elements, first referring to Xin and
%           the second to Yin, telling which deconfounding strategy to follow if confounds 
%           are specified: if 0, no deconfounding is applied; if 1, confounds
%           are regressed out; if 2, confounds are regressed out using cross-validation
%   + CVscheme - vector of two elements: first is number of folds for model evaluation;
%             second is number of folds for the model selection phase (0 in both for LOO)
%   + CVfolds - prespecified CV folds for the outer loop
%   + Nperm - number of permutations (set to 0 to skip permutation testing)
%   + show_scatter - set to 1 to show a scatter plot of predicted_Y vs Y (only for family='gaussian' or 'poisson')
%   + verbose -  display progress?
% correlation_structure (optional) - A (Nsamples X Nsamples) matrix with
%                                   integer dependency labels (e.g., family structure), or 
%                                    A (Nsamples X 1) vector defining some
%                                    grouping: (1...no.groups) or 0 for no group
% Permutations (optional but must also have correlation_structure) - pre-created set of permutations
% confounds (optional) - features that potentially influence the inputs, and the outputs for family="gaussian'
%
% OUTPUTS
% predictedY - predicted response,in the original (non-decounfounded) space
% stats structure, with fields
%   + pval - permutation-based p-value, if permutation is run;
%            otherwise, correlation-based (family=gaussian) or multinomial-based p-value (family='multinomial')
%   + dev - cross-validated deviance (for family='gaussian', this is the sum of squared errors)
%   + cod - coeficient of determination
%   + dev_deconf - cross-validated deviance in the deconfounded space (family='gaussian')
%   + cod_deconf - coeficient of determination in the deconfounded space (family='gaussian')
%   + accuracy - cross-validated classification accuracy (family='multinomial')
% predictedYC - the predicted response, in the deconfounded space
% YoutC - the actual response, in the deconfounded space
% predictedYmean - the estimated mean of the response, in original space
% beta - the regression coefficients, which correspond to standarized predictors
% Cin - the mean covariance matrix, used for the riemannian transformation
% grotperms - the deviance values obtained for each permutation


function [predictedY,stats,predictedYC,Ymean,YCmean] ...
    = nn_predict4_novalidation(Yin,Din,family,parameters,varargin)

N = size(Yin,1);
if nargin<3, family = 'gaussian'; end
if nargin<4, parameters = {}; end
if ~isfield(parameters,'K')
    K = 1:min(50,round(0.5*N));
else
    K = parameters.K; 
end
nK = length(K);
if ~isfield(parameters,'deconfounding'), deconfounding=1;
else, deconfounding = parameters.deconfounding; end
if ~isfield(parameters,'verbose'), verbose=0;
else, verbose = parameters.verbose; end
    
if strcmp(family,'multinomial') 
    error('Family multinomial not yet implemented')
end
q = size(Yin,2);

tmpnm = tempname; mkdir(tmpnm); mkdir(strcat(tmpnm,'/out')); mkdir(strcat(tmpnm,'/params')); 

if (nargin>4) && ~isempty(varargin{1})
    cs=varargin{1};
    if ~isempty(cs)
        if size(cs,2)>1 % matrix format
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
        else
            allcs = [];
            nz = cs>0; 
            gr = unique(cs(nz));  
            for g=gr'
               ss = find(cs==g);
               for s1=ss
                   for s2=ss
                       allcs = [allcs; [s1 s2]];
                   end
               end
            end
            % grotMZi and grotDZi need to be computer here
        end
    end
else
    cs = [];
end

% get confounds, and deconfound Xin
if (nargin>5) && ~isempty(varargin{2})
    if strcmp(family,'multinomial')
        error('Deconfounding not yet allowed with multinomial family')
    end
    confounds = varargin{2};
    confounds = confounds - repmat(mean(confounds),N,1);
else
    confounds = []; deconfounding = 0;
end

% confounded
predictedY = zeros(N,q,nK);
Ymean = zeros(N,q);
% deconfounded
predictedYC = zeros(N,q,nK);
YCmean = zeros(N,q);

stats = struct();
stats.K = zeros(1,nK );

done = false(N,1);

while any(~done)
    
    J = find(~done,1); % test
    if (~isempty(allcs))  % leave out all samples related to the one in question
        if size(find(allcs(:,1)==J),1)>0
            J = [J allcs(allcs(:,1)==J,2)'];
        end
    end
    ji = setdiff(1:N,J); % train
    
    D = Din(ji,J); Y = Yin(ji,:);
    
    % deconfounding business
    YC = Y; 
    if deconfounding==1
        betaY = cell(q,1); interceptY = cell(q,1);
        for i = 1:q
            [~,~,~,betaY{i},interceptY{i},YC(:,i)] = nets_deconfound([],Y(:,i),confounds(ji,:),family,...
                [],[],[],[],tmpnm);
        end
    end
    
    % getting mean
    if strcmp(family,'gaussian')
        my = mean(YC);
        YCmean(J,:) = repmat(my,length(J),1);
    end
    
    for j = 1:length(J)
        [~,order] = sort(D(:,j));
        Yordered = YC(order,:);
        for ik = 1:nK
            predictedYC(J(j),:,ik) = mean(Yordered(1:K(ik),:));
        end
    end
    
    if deconfounding % in order to later estimate prediction accuracy in deconfounded space
        if ~isempty(betaY)
            for ik = 1:nK
                for i = 1:q
                    predictedY(J,:,ik) = nets_confound(predictedYC(J,:,ik),confounds(J,:),family,betaY{i},interceptY{i}); % original space
                end
                Ymean(J,:) = nets_confound(YCmean(J,:),confounds(J,:),family,betaY{i},interceptY{i});
            end
        end
    end
    
    if verbose
        disp(['Done ' num2str( sum(done)/length(done) ) '%'])
    end
    done(J) = true;
end

if ~deconfounding
    predictedY = predictedYC;
    Ymean = YCmean;
end

stats.nulldev = sum((Yin-Ymean).^2);
stats.dev = zeros(nK,q);
stats.cod = zeros(nK,q);
for ik = 1:nK
    stats.dev(ik,:) = sum((Yin-predictedY(:,:,ik)).^2);
    stats.cod(ik,:) = 1 - stats.dev(ik,:) ./ stats.nulldev; 
end

       
end

% function S = getOrder(D)
% N = size(D,1);
% S = zero(N,N-1);
% for j = 1:N
%     ind = setdiff(1:N,j);
%     [~,S(j,:)] = tiedrank(D(j,ind));
% end
% end
% 
% 
% function y = getPrediction(s,Y,k)
% y = mean(Y(s
% 
% end
% 
% 





