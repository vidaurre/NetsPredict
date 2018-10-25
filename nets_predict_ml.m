% nets_predict - elastic-net estimation, with two-stage feature selection,
% using (stratified) LOO and permutation testing
% Diego Vidaurre and Steve Smith 
% FMRIB Oxford, 2013-2014
%
% [predictedY,stats] = nets_predict3(Y,X,family,parameters);
% [predictedY,stats] = nets_predict3(Y,X,family,parameters,correlation_structure);
% [predictedY,stats] = nets_predict3(Y,X,family,parameters,correlation_structure,Permutations);
% [predictedY,stats] = nets_predict3(Y,X,family,parameters,correlation_structure,Permutations,confounds);
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
%   coefficients, if 'Method' is 'lasso' or 'glmnet'; otherwise, it is the
%   values of the ridge penalty. 
%   + riemann - run in tangent space, closer to riemannian geometry? 
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


function [predictedY,stats,predictedYC,YoutC,predictedYmean,beta] ...
    = nets_predict_ml(Yin,Xin,parameters,varargin)

if nargin<3, parameters = {}; end
if ~isfield(parameters,'CVscheme'), CVscheme = [10 10];
else, CVscheme = parameters.CVscheme; end
if ~isfield(parameters,'CVfolds'), CVfolds = [];
else, CVfolds = parameters.CVfolds; end
if ~isfield(parameters,'Nperm'), Nperm=1;
else, Nperm = parameters.Nperm; end
if ~isfield(parameters,'Nfeatures'), Nfeatures=0;
else, Nfeatures = parameters.Nfeatures; end
if ~isfield(parameters,'standardize'), standardize = 1;
else, standardize = parameters.standardize; end
if ~isfield(parameters,'verbose'), verbose=0;
else, verbose = parameters.verbose; end

% Standardizing Xin
p = size(Xin,2); N = size(Xin,1); 
mx = mean(Xin);  sx = std(Xin);
if standardize
    Xin = Xin - repmat(mx,N,1);
    Xin(:,sx>0) = Xin(:,sx>0) ./ repmat(sx(sx>0),N,1);
end
if (Nperm<2),  Nperm=1;  end
cs=[];
if (nargin>3) && ~isempty(varargin{1})
    cs=varargin{1};
    if ~isempty(cs)
        if size(cs,2)>1 % matrix format
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
            [grotMZi(:,2),grotMZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==1));
            [grotDZi(:,2),grotDZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==2));
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
if ~exist('allcs','var'), allcs = []; end
PrePerms=0;
if (nargin>4) && ~isempty(varargin{2})
    Permutations=varargin{2};
    if ~isempty(Permutations)
        PrePerms=1;
        Nperm=size(Permutations,2);
    end
else
    Permutations = [];
end
% get confounds, and deconfound Xin
if (nargin>5) && ~isempty(varargin{3})
    confounds = varargin{3};
    confounds = confounds - repmat(mean(confounds),N,1);
    [~,Xin] = nets_deconfound(Xin,[],confounds,family{1},[],[],tmpnm);
else
    confounds = [];
end

alpha = 0.00001;


YinORIG = Yin; 
YinORIGmean = zeros(size(Yin));
%if exist('Yin2','var'), YinORIG2=Yin2; YinORIGmean2 = zeros(size(Yin2)); end 
grotperms = zeros(Nperm,1);
if ~isempty(confounds)
    YC = zeros(size(Yin));
    YCmean = zeros(size(Yin));
end

if isempty(CVfolds)
    beta = zeros(p,CVscheme(1));
else
    beta = zeros(p,length(CVfolds));
end

for perm=1:Nperm
    if (perm>1)
        if isempty(cs)           % simple full permutation with no correlation structure
            rperm = randperm(N);
            Yin=YinORIG(rperm,:);
        elseif (PrePerms==0)          % complex permutation, taking into account correlation structure
            PERM=zeros(1,N);
            perm1=randperm(size(grotMZi,1));
            for ipe=1:length(perm1)
                if rand<0.5, wt=[1 2]; else wt=[2 1]; end
                PERM(grotMZi(ipe,1))=grotMZi(perm1(ipe),wt(1));
                PERM(grotMZi(ipe,2))=grotMZi(perm1(ipe),wt(2));
            end
            perm1=randperm(size(grotDZi,1));
            for ipe=1:length(perm1)
                if rand<0.5, wt=[1 2]; else wt=[2 1]; end
                PERM(grotDZi(ipe,1))=grotDZi(perm1(ipe),wt(1));
                PERM(grotDZi(ipe,2))=grotDZi(perm1(ipe),wt(2));
            end
            from=find(PERM==0);  pto=randperm(length(from));  to=from(pto);  PERM(from)=to;
            Yin=YinORIG(PERM,:);
        else                   % pre-supplied permutation
            Yin=YinORIG(Permutations(:,perm),:);  % or maybe it should be the other way round.....?
        end
    end
    
    predictedYp = zeros(N,1);
    if ~isempty(confounds), predictedYpC = zeros(N,1); end
    
    % create the inner CV structure - stratified for family=multinomial
    if isempty(CVfolds)
        if CVscheme(1)==1
            folds = {1:N};
        else            
            folds = cvfolds(Yin,'gaussian',CVscheme(1),allcs);
        end
    else
        folds = CVfolds;
    end
    
    for ifold = 1:length(folds) 
        
        if verbose, fprintf('CV iteration %d \n',ifold); end
        
        J = folds{ifold}; % test
        if isempty(J), continue; end
        if length(folds)==1
            ji = J;
        else
            ji = setdiff(1:N,J); % train
        end
        QN = length(ji);
        X = Xin(ji,:); Y = Yin(ji,:); 
        
        % family structure for this fold
        Qallcs=[]; 
        if (~isempty(cs))
            [Qallcs(:,2),Qallcs(:,1)]=ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));  
        end
        
        % deconfounding business
        if ~isempty(confounds)
            [~,~,betaY,Y] = nets_deconfound([],Y,confounds(ji,:),family{1},[],[],tmpnm);
        end
        
        % centering response
        my = 0;%mean(Y); 
        Y = Y - my;
        YCmean(J) = my;

        % pre-kill features
        if Nfeatures(1)<p && Nfeatures(1)>0
            dev = nets_glmnet(X, Y, family{1}, 1);
            [~,groti0]=sort(dev);
            groti0=groti0(end-Nfeatures(1)+1:end);
        else
            groti0 = find(sx>0);
        end
        
        X = X(:,groti0);
        
        b = (X' * X + alpha*eye(p)) \ X' * Y;  
        if perm==1 
            beta(groti0,ifold) = b; 
        end
        
        % predict the test fold
        XJ = Xin(J,groti0);
        predictedYp(J) = XJ * b + repmat(my,length(J),1);
                        
        % predictedYpC and YC in deconfounded space; Yin and predictedYp are confounded
        predictedYpC(J,:) = predictedYp(J,:); 
        YC(J,:) = Yin(J,:);
        YinORIGmean(J) = YCmean(J);
        if ~isempty(confounds) % in order to later estimate prediction accuracy in deconfounded space
            [~,~,~,YC(J,:)] = nets_deconfound([],YC(J,:),confounds(J,:),family{2},[],betaY,tmpnm);
            if ~isempty(betaY)
                predictedYp(J,:) = nets_confound(predictedYp(J,:),confounds(J,:),family{2},betaY); % original space
                YinORIGmean(J) = nets_confound(YCmean(J,:),confounds(J,:),family{2},betaY); 
            end
        end
        
    end
  
    grotperms(perm) = sum((YC-predictedYpC).^2);
    
    if perm==1
        predictedY = predictedYp;
        predictedYmean = YinORIGmean;
        predictedYC = predictedYpC;
        YoutC = YC;
        
        stats.dev = sum((YinORIG-predictedYp).^2);
        stats.nulldev = sum((YinORIG-YinORIGmean).^2);
        stats.corr = corr(YinORIG,predictedYp);
        if Nperm==1
            [~,pv] = corrcoef(YinORIG,predictedYp); stats.pval=pv(1,2);
            if corr(YinORIG,predictedYp)<0, pv = 1; end
        end
        if ~isempty(confounds)
            stats.dev_deconf = sum((YC-predictedYpC).^2);
            stats.nulldev_deconf = sum((YC-YCmean).^2);
            if Nperm==1
                [~,pvd] = corrcoef(YC,predictedYpC); stats.pval_deconf=pvd(1,2);
            end
        end

        stats.cod = 1 - stats.dev / stats.nulldev;
        if ~isempty(confounds), stats.cod_deconf = 1 - stats.dev_deconf / stats.nulldev_deconf; end
        
    else
        fprintf('Permutation %d \n',perm)
    end
end

if Nperm>1 
    stats.pval = sum(grotperms<=grotperms(1)) / (Nperm+1);
end
beta = squeeze(beta);

end

