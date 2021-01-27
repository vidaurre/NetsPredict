function [stats,predictedY,predictedY0,predictedYD,predictedYD0,beta] ...
    = nets_predict5(Yin,Xin,family,parameters,varargin)
% nets_predict - elastic-net estimation, with two-stage feature selection,
% using (stratified) LOO and permutation testing
%
% Diego Vidaurre and Steve Smith 
% Aarhus Uni / FMRIB Oxford, 2020
%
% [stats,predictedY] = nets_predict5(Y,X,family,parameters);
% [stats,predictedY] = nets_predict5(Y,X,family,parameters,correlation_structure);
% [stats,predictedY] = nets_predict5(Y,X,family,parameters,correlation_structure,Permutations);
% [stats,predictedY] = nets_predict5(Y,X,family,parameters,correlation_structure,Permutations,confounds);
%
% INPUTS
% Y - response vector (samples X 1), with integers representing classes if family is 'multinomial'
% X - predictor matrix (samples X features)
% family - probability distribution of the response, one of the following:
%   + 'gaussian': standard linear regression on a continuous response
%   + 'poisson': non-negative counts
%   + 'multinomial': a binary-valued matrix with as columns as classes
%   + 'cox': a two-column matrix with the 1st column for time and the 2d for status: 
%       1 for death and 0 right censored.
% parameters is a structure with:
%   + Method - One of 'glmnet', 'lasso', 'ridge' or 'unregularized'. Glmnet and lasso are
%           the elastic net, but with different code: glmnet uses the glmnet
%           package, essentially a matlab wrap of Fortran code - it is very quick
%           but ocassionally crashes taking down the whole Matlab; 'lasso' uses the
%           matlab function for the elastic net, it is considerable slower and only works for
%           the 'multinomial' family when there only 2 classes, but never crashes. 
%           'ridge' is ridge regression, and 'unregularized' does not impose a penalty.
%           Note that for 'multinomial' family, only 'glmnet' and
%           'unregularized' are allowed, but they way they treat the
%           estimation is different: 'glmnet' is symmetric, whereas when
%           'unregularized' is used, the last category is used as a reference
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
%   + CVfolds - prespecified CV folds for the outer loop: it must be a (nfolds x 1) cell
%   with one vector per fold, indicating which samples are going to
%   be used for testing in that fold; therefore the intersection of all the vectors
%   should amount to 1:N, where N is the number of samples
%   + Nperm - number of permutations (set to 0 to skip permutation testing)
%   + verbose -  display progress?
% correlation_structure (optional) - A (Nsamples X Nsamples) matrix with
%                                   integer dependency labels (e.g., family structure), or 
%                                    A (Nsamples X 1) vector defining some
%                                    grouping: (1...no.groups) or 0 for no group
% Permutations (optional but must also have correlation_structure) - 
%       pre-created set of permutations
% confounds (optional) - features that potentially influence the inputs, 
%       and the outputs for family="gaussian'
%
% OUTPUTS
% stats structure, with fields
%   + pval - permutation-based p-value, if permutation is run;
%            otherwise, correlation-based (family=gaussian) or multinomial-based p-value 
%           (family='multinomial')
%   + cod - coeficient of determination (family='gaussian')
%   + corr - correlation between predicted and observed Y (family='gaussian')
%   + baseline_corr - baseline correlation between predicted and observed Y  for null model 
%           (family='gaussian')
%   + dev - deviance (e.g. sum of squares if family='gaussian')
%   + baseline_dev - baseline deviance for the null model 
%           (e.g. sum of squares if family='gaussian')
%   + accuracy - cross-validated classification accuracy (family='multinomial')
%   + baseline_accuracy - cross-validated classification accuracy (family='multinomial')
%   PLUS: All of the above +'_deconf' in the deconfounded space, if counfounds were specified
% predictedY - predicted response,in the original (non-decounfounded) space
% predictedYD - the predicted response, in the deconfounded space
% predictedY0 - predicted baseline response, in the original (non-decounfounded) space
% predictedYD0 - the predicted baseline response, in the deconfounded space  
% beta - A (no. of outer-loop CV folds x no. of predictors) matrix
%           with the estimated regression coefficients for each CV fold
%           (which correspond to standarized predictors); only implemented
%           for Gaussian family so far. 

if nargin<3, family = 'gaussian'; end
if nargin<4, parameters = {}; end
if ~isfield(parameters,'Method')
    ch = which('glmnet');
    if ~isempty(ch),  Method = 'glmnet';
    else, Method = 'lasso';
    end
    try
        glmnet(randn(10,2),rand(10,1),'gaussian')
    catch
        warning('glmnet is not working correctly (recompile?), using lasso instead')
        Method = 'lasso';
    end
else
    Method = parameters.Method; 
    if strcmpi(Method,'glmnet') 
        ch = which('glmnet');
        if isempty(ch)
            error('Package glmnet not found, use Method=''lasso'' instead')
        end
    end
end

if strcmpi(Method,'unregularized') || strcmpi(Method,'ridge')
    if strcmpi(family,'poisson')
        error('Poisson family is only implemented for the glmnet and lasso methods')
    end
    if strcmpi(family,'cox')
        error('Cox family is only implemented for the glmnet and lasso methods')
    end
end

if strcmpi(family,'cox')
    warning('Family cox is yet not well tested')
end
if strcmpi(family,'poisson')
    warning('Family poisson is yet not well tested')
end

if ~isfield(parameters,'alpha')
    if strcmpi(Method,'ridge')
        alpha = [0.00001 0.0001 0.001 0.01 0.1 0.4 0.7 0.9 1.0 10 100];
    else
        alpha = [0.01 0.1 0.4 0.7 0.9 0.99];
    end
else
    alpha = parameters.alpha; 
end
if ~isfield(parameters,'CVscheme'), CVscheme = [10 10];
else, CVscheme = parameters.CVscheme; end
if ~isfield(parameters,'CVfolds'), CVfolds = [];
else, CVfolds = parameters.CVfolds; end
if ~isfield(parameters,'Nfeatures'), Nfeatures=0;
else, Nfeatures = parameters.Nfeatures; end
if ~isfield(parameters,'deconfounding'), deconfounding=[1 0];
else, deconfounding = parameters.deconfounding; end
if ~isfield(parameters,'biascorrect'), biascorrect = 0;
else, biascorrect = parameters.biascorrect; end
if ~isfield(parameters,'Nperm'), Nperm=1;
else, Nperm = parameters.Nperm; end
if ~isfield(parameters,'nlambda'), nlambda=2000;
else, nlambda = parameters.nlambda; end
if ~isfield(parameters,'riemann'), riemann = length(size(Xin))==3;
else, riemann = parameters.riemann; end
if ~isfield(parameters,'keepvar'), keepvar = 1;
else, keepvar = parameters.keepvar; end
if ~isfield(parameters,'verbose'), verbose=0;
else, verbose = parameters.verbose; end

if biascorrect == 1 
    if ~strcmpi(family,'gaussian')
        error('biascorrect can only be used for gaussian family')
    end
    parameters.biascorrect = 0;
    parameters.riemann = 0; 
    parameters.verbose = 0; 
end

if any(deconfounding==2)
    parameters_dec = struct();
    parameters_dec.alpha = [0.0001 0.001 0.01 0.1 1 10 100 1000];
    parameters_dec.CVscheme = [10 10];
    parameters_dec.Method = 'ridge';
end
if deconfounding(2)==2
   error('This deconfounding strategy is not available for Y - deconfounding(2) must be 0 or 1') 
end
    
enet = strcmpi(Method,'lasso') || strcmpi(Method,'glmnet');
if ~enet, nlambda = 1; end
if Nperm > 1 && enet
   warning('Using permutation testing with Method lasso or glmnet can be computationally very costly') 
end

% put Xin in the right format, which depends on riemann=1
Xin = reshape_pred(Xin,riemann,keepvar); N = size(Xin,1); 

tmpnm = tempname; 
if strcmpi(Method,'glmnet')
    mkdir(tmpnm); mkdir(strcat(tmpnm,'/out')); mkdir(strcat(tmpnm,'/params'));
end

% Format Yin appropriately
if strcmpi(family,'multinomial') 
    if (strcmpi(Method,'glmnet') || strcmpi(Method,'unregularized'))
        if size(Yin,2)==1, Yin = nets_class_vectomat(Yin); end
        q = size(Yin,2);
        if q>9, warning('That is a lot of classes (>9), you sure this is correct?'); end
    elseif strcmpi(Method,'lasso')
        if size(Yin,2)>1, error('Correct format for Yin is (N x 1)'); end
        q = length(unique(Yin));
        if q>2, error('Method ''lasso'' does not support more than 2 classes'); end
        val = unique(Yin); 
        if any(val~=0 & val~=1), error('Values for Yin have to be either 0 or 1'); end
    elseif  strcmpi(Method,'ridge')
        error('Multinomial family for more than 2 classes is not implemented for ''ridge'' method')
    else 
        error('Method not supported')
    end
end    

% Putting Xin it in tangent space if riemann=1
if riemann
    Xin = permute(Xin,[2 3 1]);
    for j=1:size(Xin,3)
        ev = eig(Xin(:,:,j));
        if any(ev<0)
            error(sprintf('The matrix for subject %d is not positive definite',j))
        end
    end
    Cin = mean_covariances(Xin,'riemann');
    Xin = Tangent_space(Xin,Cin)'; 
else
    Cin = [];
end

% Standardizing Xin
p = size(Xin,2);
mx = mean(Xin);  sx = std(Xin);
Xin = Xin - repmat(mx,N,1);
Xin(:,sx>0) = Xin(:,sx>0) ./ repmat(sx(sx>0),N,1);

% check correlation structure
allcs = []; 
if (nargin>4) && ~isempty(varargin{1})
    cs=varargin{1};
    if ~isempty(cs)
        is_cs_matrix = (size(cs,2) == size(cs,1));
        if is_cs_matrix 
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
            [grotMZi(:,2),grotMZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==1));
            [grotDZi(:,2),grotDZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==2));
        else
            allcs = find(cs > 0);
            grotMZi = find(cs == 1); 
            grotDZi = find(cs == 2); 
        end
    end
else, cs = []; 
end

if (Nperm<2),  Nperm=1;  end
PrePerms=0;
if (nargin>5) && ~isempty(varargin{2})
    Permutations=varargin{2};
    if ~isempty(Permutations)
        PrePerms=1;
        Nperm=size(Permutations,2);
    end
else
    Permutations = [];
end

% get confounds, and deconfound Xin
if (nargin>6) && ~isempty(varargin{3})
    confounds = varargin{3};
    confounds = confounds - repmat(mean(confounds),N,1);
    if deconfounding(1)==1
        [~,Xin] = nets_deconfound(Xin,[],confounds,'gaussian',[],[],tmpnm);
    elseif deconfounding(1)==2
        for j = 1:p
           Xin(:,j) = Xin(:,j) - nets_predict5(Xin(:,j),confounds,'gaussian',parameters_dec); 
        end
    end
else
    confounds = []; deconfounding = [0 0];
end
if ~strcmpi(family,'gaussian') && deconfounding(2) > 0
    error('Deconfounding still only implemented for family gaussian')
end
    
if strcmpi(Method,'ridge') 
   ridg_pen_scale = mean(diag(Xin' * Xin));
end

YinORIG = Yin; 
YD = zeros(size(Yin)); % deconfounded signal
grotperms = zeros(Nperm,1);

if strcmpi(family,'multinomial') && (strcmpi(Method,'glmnet') || strcmpi(Method,'unregularized'))
    predictedYp = zeros(N,q); predictedYp0 = zeros(N,q);
    predictedYpD = zeros(N,q); predictedYpD0 = zeros(N,q);
else
    predictedYp = zeros(N,1); predictedYp0 = zeros(N,1);
    predictedYpD = zeros(N,1); predictedYpD0 = zeros(N,1);
end

if strcmpi(family,'gaussian')
    if isempty(CVfolds)
        beta = zeros(p,CVscheme(1));
    else
        beta = zeros(p,length(CVfolds));
    end
else
    beta = [];
end

for perm=1:Nperm
    
    if (perm>1)
        if isempty(cs)  % simple full permutation with no correlation structure
            rperm = randperm(N);
            Yin = YinORIG(rperm,:);
        elseif (PrePerms==0)  % complex permutation, taking into account correlation structure
            PERM = zeros(1,N);
            if is_cs_matrix
                perm1 = randperm(size(grotMZi,1));
                for ipe=1:length(perm1)
                    if rand<0.5, wt=[1 2]; else wt=[2 1]; end
                    PERM(grotMZi(ipe,1))=grotMZi(perm1(ipe),wt(1));
                    PERM(grotMZi(ipe,2))=grotMZi(perm1(ipe),wt(2));
                end
                perm1 = randperm(size(grotDZi,1));
                for ipe = 1:length(perm1)
                    if rand<0.5, wt=[1 2]; else wt=[2 1]; end
                    PERM(grotDZi(ipe,1))=grotDZi(perm1(ipe),wt(1));
                    PERM(grotDZi(ipe,2))=grotDZi(perm1(ipe),wt(2));
                end
                from = find(PERM==0);  pto=randperm(length(from));  to=from(pto);  PERM(from)=to;
                Yin = YinORIG(PERM,:);
            else
                families = unique(cs);
                for j = 1:length(families)
                    ind = find(cs == families(j));
                    rp = randperm(length(ind));
                    PERM(ind) = ind(rp);
                end
            end

        else   % pre-supplied permutation
            Yin = YinORIG(Permutations(:,perm),:);  % or maybe it should be the other way round.....?
        end
    end
    
    % create the inner CV structure - stratified for family=multinomial
    if isempty(CVfolds)
        if CVscheme(1)==1
            folds = {1:N};
        else            
            folds = cvfolds(Yin,family,CVscheme(1),allcs);
        end
    else
        folds = CVfolds;
    end
    
    if perm==1 && ~strcmpi(Method,'unregularized')
        stats = struct();
        stats.alpha = zeros(1,length(folds) );
        stats.alpha_ridge = zeros(1,length(folds) );
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
        X0 = randn(size(X,1),1);
        
        % family structure for this fold
        Qallcs=[]; 
        if (~isempty(cs))
            if is_cs_matrix
                [Qallcs(:,2),Qallcs(:,1)] = ...
                    ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));
            else
                Qallcs = find(cs(ji) > 0);
                %QgrotMZi = find(cs(ji) == 1);
                %QgrotDZi = find(cs(ji) == 2);
            end
        end
                
        % deconfounding business
        if deconfounding(2)==1
            [~,~,betaY,Y] = ...
                nets_deconfound([],Y,confounds(ji,:),family,[],[],tmpnm);
        end
        
        % pre-kill features
        if Nfeatures(1)<p && Nfeatures(1)>0
            dev = nets_screen(X, Y, family);
            [~,groti] = sort(dev);
            groti = groti(end-Nfeatures(1)+1:end);
        else
            groti = find(sx>0);
        end
        
        QXin = Xin(ji,groti);
        % uncomment this if you want to deconfound in inner loop
        %QYin = Yin(ji,:); 
        QYin = Y;
        
        %if ~isempty(confounds), Qconfounds=confounds(ji,:); end
        X = X(:,groti);
        
        % create the inner CV structure - stratified for family=multinomial
        Qfolds = cvfolds(Y,family,CVscheme(2),Qallcs);

        Dev = Inf(nlambda,length(alpha));
        Lambda = {};
        
        if ~strcmpi(Method,'unregularized')
            
            for ialph = 1:length(alpha)
                
                if strcmpi(family,'multinomial') && strcmpi(Method,'glmnet')
                    QpredictedYp = Inf(QN,q,nlambda);
                else
                    QpredictedYp = Inf(QN,nlambda);
                end
                options = {}; options.standardize = false;
                if strcmpi(family,'gaussian'), options.intr = false; end
                options.alpha = alpha(ialph); options.nlambda = nlambda;
                
                QYinCOMPARE = QYin;
                
                % Inner CV loop
                for Qifold = 1:length(Qfolds)
                    QJ = Qfolds{Qifold};
                    Qji = setdiff(1:QN,QJ);
                    QX = QXin(Qji,:); QY = QYin(Qji,:);
                    if enet
                        if Qifold>1, options.lambda = Lambda{ialph};
                        elseif isfield(options,'lambda'), options = rmfield(options,'lambda');
                        end
                    end
                    
                    switch Method
                        case {'glmnet','Glmnet'}
                            estimation = glmnet(QX, QY, family, options); 
                        case {'lasso','Lasso'}
                            estimation = struct();
                            if strcmpi(family,'gaussian')
                                if Qifold==1
                                    [estimation.beta,lassostats] = ...
                                        lasso(QX,QY,'Alpha',alpha(ialph),'NumLambda',nlambda);
                                else
                                    [estimation.beta,lassostats] = ...
                                        lasso(QX,QY,'Alpha',alpha(ialph),'Lambda',Lambda{ialph});
                                end
                            else
                                if strcmpi(family,'multinomial')
                                    strfam = 'binomial'; % q > 2 not implemented here
                                elseif strcmpi(family,'cox')
                                    error('Cox family not implemented for lasso method');
                                else
                                    strfam = family;
                                end
                                if Qifold==1
                                    [estimation.beta,lassostats] = ...
                                        lassoglm(QX,QY,strfam,'Alpha',alpha(ialph),'NumLambda',nlambda);
                                else
                                    [estimation.beta,lassostats] = ...
                                        lassoglm(QX,QY,strfam,'Alpha',alpha(ialph),'Lambda',Lambda{ialph});
                                end
                            end
                            estimation.lambda = lassostats.Lambda;
                            estimation.a0 = lassostats.Intercept;
                            
                        case {'ridge','Ridge'}
                            if strcmpi(family,'gaussian')
                                estimation = struct();
                                QX = [ones(size(QX,1),1) QX];
                                R = alpha(ialph) * ridg_pen_scale * eye(size(QX,2)); R(1,1) = 0;
                                estimation.beta = (QX' * QX + R) \ (QX' * QY);
                            else
                                error('Only Gaussian family is implemented for ridge method');
                            end
                    end
                    
                    if Qifold == 1 && enet
                        Lambda{ialph} = estimation.lambda;
                        options = rmfield(options,'nlambda');
                    end
                    
                    QXJ = QXin(QJ,:);
                    
                    if strcmpi(family,'gaussian')
                        if enet % glmnet, lasso
                            QpredictedYp(QJ,1:length(estimation.lambda)) = ...
                                QXJ * estimation.beta + ...
                                repmat(estimation.a0(end),length(QJ),length(estimation.lambda));
                        else % ridge
                            QXJ = [ones(size(QXJ,1),1) QXJ];
                            QpredictedYp(QJ) = QXJ * estimation.beta;
                        end
                    elseif strcmpi(family,'multinomial')
                        if strcmpi(Method,'glmnet')
                            QpredictedYp(QJ,:,1:length(estimation.lambda)) = ...
                                nets_glmnetpredict(estimation,QXJ,estimation.lambda,'response');
                        elseif strcmpi(Method,'lasso')
                            QpredictedYp(QJ,1:length(estimation.lambda)) = ...
                                glmval([estimation.a0; estimation.beta] ,QXJ,'logit');
                        elseif strcmpi(Method,'ridge')
                            error('Only Gaussian family is implemented for ridge method');
                        end
                    elseif strcmpi(family,'poisson')
                        if strcmpi(Method,'glmnet')
                            QpredictedYp(QJ,1:length(estimation.lambda)) = ...
                                max(nets_glmnetpredict(estimation,QXJ,estimation.lambda,'response'),eps);
                        elseif strcmpi(Method,'lasso')
                            QpredictedYp(QJ,1:length(estimation.lambda)) = ...
                                glmval([estimation.a0; estimation.beta],QXJ,'log');
                        end
                        %exp(QXJ * glmfit.beta + repmat(glmfit.a0',length(QJ),1) );
                    else % cox
                        QpredictedYp(QJ,1:length(estimation.lambda)) = exp(QXJ * estimation.beta);
                    end
                end
                % Pick the one with the lowest deviance (=quadratic error for family="gaussian")
                if strcmpi(family,'gaussian') % it's actually QN*log(sum.... but it doesn't matter
                    if enet
                        Qdev = sum(( QpredictedYp(:,1:length(Lambda{ialph})) - ...
                            repmat(QYinCOMPARE,1,length(Lambda{ialph}))).^2) / QN; 
                        Qdev = Qdev';
                    else
                        Qdev = sum(( QpredictedYp - QYinCOMPARE ).^2) / QN;
                    end
                elseif strcmpi(family,'multinomial')
                    if strcmpi(Method,'glmnet')  
                        Qdev = Inf(length(Lambda{ialph}),1);
                        for i=1:length(Lambda{ialph})
                            Qdev(i) = - sum(log(sum(QYinCOMPARE .* QpredictedYp(:,:,i) ,2))); 
                        end
                    else % lasso
                        for i=1:length(Lambda{ialph})
                            Qdev(i) = - sum(log( ((1-QYinCOMPARE) .* (1-QpredictedYp(:,i))))) ...
                                - sum(log( (QYinCOMPARE .* QpredictedYp(:,i))));
                        end
                    end
                elseif strcmpi(family,'poisson')
                    Ye = repmat(QYinCOMPARE,1,length(Lambda{ialph}));
                    Qdev = sum(Ye .* log( (Ye+(Ye==0)) ./ QpredictedYp(:,1:length(Lambda{ialph}))) - ...
                        (Ye - QpredictedYp(:,1:length(Lambda{ialph}))));
                    Qdev = Qdev';
                else % cox
                    failures = find(QYinCOMPARE(:,2) == 1)';
                    Qdev = zeros(length(Lambda{ialph}),1);
                    for i=1:length(Lambda{ialph})
                        for n=failures
                            Qdev(i) = Qdev(i) + ...
                                log( QpredictedYp(n,i) / ...
                                sum(QpredictedYp(QYinCOMPARE(:,1) >= QYinCOMPARE(n,1),i)) );
                        end
                    end
                    Qdev = -2 * Qdev;
                end
                
                Dev(1:length(Qdev),ialph) = Qdev;
            end
            
            [~,opt] = min(Dev(:));
            
            if enet
                ialph = ceil(opt / nlambda);
                if ialph==0, ialph = length(Lambda); end
                if ialph>length(Lambda), ialph = length(Lambda); end
                ilamb = mod(opt,nlambda);
                if ilamb==0, ilamb = nlambda; end
                if ilamb>length(Lambda{ialph}), ilamb = length(Lambda{ialph}); end
                options.alpha = alpha(ialph);
                if verbose, fprintf('Alpha chosen to be  %f \n',options.alpha); end
                options.lambda = (2:-.2:1)' * Lambda{ialph}(ilamb); % it doesn't like just 1 lambda
                switch Method
                    case {'glmnet','Glmnet'}
                        estimation = glmnet(X,Y,family,options);
                        estimation0 = glmnet(X0,Y,family,options);
                    case {'lasso','Lasso'}
                        estimation = struct(); estimation0 = struct();
                        if strcmpi(family,'gaussian')
                            [estimation.beta,lassostats] = ...
                                lasso(X,Y,'Alpha',alpha(ialph),'Lambda',options.lambda);
                            [estimation0.beta,lassostats0] = ...
                                lasso(X0,Y,'Alpha',alpha(ialph),'Lambda',options.lambda);
                        else
                            [estimation.beta,lassostats] = ...
                                lassoglm(X,Y,strfam,'Alpha',alpha(ialph),'Lambda',options.lambda);
                            [estimation0.beta,lassostats0] = ...
                                lassoglm(X0,Y,strfam,'Alpha',alpha(ialph),'Lambda',options.lambda);
                            
                        end
                        estimation.a0 = lassostats.Intercept; estimation0.a0 = lassostats0.Intercept;
                end 
                if perm==1 && nargout>=6 && strcmpi(family,'gaussian')
                    beta(groti,ifold) = estimation.beta(:,end); 
                end
                
            else % ridge
                ialph = opt;
                options.alpha = alpha(ialph);
                if verbose, fprintf('Alpha chosen to be  %f \n',options.alpha); end
                estimation = struct(); estimation0 = struct();
                X = [ones(size(X,1),1) X]; 
                R =  options.alpha * ridg_pen_scale * eye(size(X,2)); R(1,1) = 0;
                estimation.beta = (X' * X + R) \ (X' * Y);
                X0 = [ones(size(X,1),1) X0]; 
                R = zeros(2); R(2,2) = options.alpha * ridg_pen_scale;
                estimation0.beta = (X0' * X0 + R) \ (X0' * Y);
                if perm==1 && nargout>=6 && strcmpi(family,'gaussian') 
                    beta(groti,ifold) = estimation.beta(2:end); 
                end
            end
            
        else
            estimation = struct(); estimation0 = struct();
            if strcmpi(family,'gaussian')
                X = [ones(size(X,1),1) X]; 
                estimation.beta = (X' * X) \ (X' * Y);
                X0 = [ones(size(X0,1),1) X0]; 
                estimation0.beta = (X0' * X0) \ (X0' * Y);
                if perm==1 && nargout>=6, beta(groti,ifold) = estimation.beta(2:end); end
            else
                estimation.beta = mnrfit(X,Y); 
                estimation0.beta = mnrfit(X0,Y); 
            end
            
        end

        % predict the test fold
        XJ = Xin(J,groti);
        XJ0 = randn(size(XJ,1),1);
                
        if strcmpi(family,'gaussian')
            beta_final = estimation.beta(:,end); beta_final0 = estimation0.beta(:,end);
            if enet 
                predictedYp(J) = XJ * beta_final + estimation.a0(end);
                predictedYp0(J) = XJ0 * beta_final0 + estimation0.a0(end);
            else % ridge or unregularized
                XJ = [ones(size(XJ,1),1) XJ];
                predictedYp(J) = XJ * estimation.beta;
                XJ0 = [ones(size(XJ0,1),1) XJ0];
                predictedYp0(J) = XJ0 * estimation0.beta;
            end
        elseif strcmpi(family,'multinomial') % deconfounded space, both predictedYp and predictedYp0
            if strcmpi(Method,'glmnet')
                predictedYp(J,:) = nets_glmnetpredict(estimation,XJ,estimation.lambda(end),'response');
                predictedYp0(J,:) = nets_glmnetpredict(estimation0,XJ0,estimation.lambda(end),'response');
            elseif strcmpi(Method,'lasso')
                beta_final = [estimation.a0(end); estimation.beta(:,end)]; 
                beta_final0 = [estimation0.a0(end); estimation0.beta(:,end)];
                predictedYp(J) = glmval(beta_final,XJ,'logit');
                predictedYp0(J) = glmval(beta_final0,XJ0,'logit');
            elseif strcmpi(Method,'unregularized')
                predictedYp(J,:) = mnrval(estimation.beta,XJ);
                predictedYp0(J,:) = mnrval(estimation0.beta,XJ0);
                if any(isnan(predictedYp(:))) % unregularized logistic often goes out of precision
                    predictedYp(isnan(predictedYp)) = 1;
                    predictedYp = predictedYp ./ repmat(sum(predictedYp,2),1,q);
                end
            end
            elseif strcmpi(family,'poisson')
            if strcmpi(Method,'glmnet')
                predictedYp(J) = ...
                    max(nets_glmnetpredict(estimation,XJ,estimation.lambda(end),'response'),eps);
                predictedYp0(J) = ...
                    max(nets_glmnetpredict(estimation0,XJ0,estimation0.lambda(end),'response'),eps);
            elseif strcmpi(Method,'lasso')
                beta_final = [estimation.a0(end); estimation.beta(:,end)]; 
                beta_final0 = [estimation0.a0(end); estimation0.beta(:,end)];
                predictedYp(J) = glmval(beta_final,XJ,'log');
                predictedYp0(J) = glmval(beta_final0,XJ0,'log');
            end
            %predictedYp(J) = exp(XJ * glmfit.beta(:,end) + glmfit.a0(end) );
        else % cox
            predictedYp(J) = exp(XJ * estimation.beta(:,end));
            predictedYp0(J) = exp(XJ0 * estimation0.beta(:,end));
        end
        
        % predictedYpD and YD in deconfounded space; Yin and predictedYp are confounded
        predictedYpD(J,:) = predictedYp(J,:); 
        predictedYpD0(J,:) = predictedYp0(J,:); 
        YD(J,:) = Yin(J,:);
        
        if deconfounding(2) % in order to later estimate prediction accuracy in deconfounded space
            [~,~,~,YD(J,:)] = ...
                nets_deconfound([],YD(J,:),confounds(J,:),family,[],betaY,tmpnm);
            if ~isempty(betaY) % into original space
                predictedYp(J,:) = ...
                    nets_confound(predictedYp(J,:),confounds(J,:),family,betaY); 
                predictedYp0(J,:) = ...
                    nets_confound(predictedYp0(J,:),confounds(J,:),family,betaY); 
            end
        end
        
        if biascorrect % we do this in the original space
            if enet            
                Yhattrain = Xin(ji,groti) * beta_final + repmat(estimation.a0(end),length(ji),1);
            else
                Xin1 = [ones(length(ji),1) Xin(ji,groti)];
                Yhattrain = Xin1 * beta_final;
            end
            if deconfounding(2)
                Yhattrain = nets_confound(Yhattrain,confounds(ji,:),family,betaY);
            end
            Ytrain = [QYin ones(size(QYin,1),1)]; 
            b = pinv(Ytrain) * Yhattrain;
            predictedYp(J,:) = (predictedYp(J,:) - b(2)) / b(1);
        end
        
        if perm==1 && ~strcmpi(Method,'unregularized')
            stats.alpha(ifold) = options.alpha;
        end
        
    end
     
    % grotperms computed in deconfounded space
    if strcmpi(family,'gaussian')  
        grotperms(perm) = sum((YD-predictedYpD).^2);
    elseif strcmpi(family,'multinomial')
        if (strcmpi(Method,'glmnet') || strcmpi(Method,'unregularized')) %>1 column
            grotperms(perm) = - 2 * sum(log(sum(YD .* predictedYpD,2)));
        else % lasso
            grotperms(perm) = - 2 * sum(log( ((1-YD) .* (1-predictedYpD)))) ...
                - 2 * sum(log( (YD .* predictedYpD)));
        end
    elseif strcmpi(family,'poisson')
        grotperms(perm) = 2 * sum(YD.*log((YD+(YD==0)) ./ predictedYpD) - (YD - predictedYpD) );
    else % cox - in the current version there is no response deconfounding for family="cox"
        grotperms(perm) = 0;
        failures = find(YD(:,2) == 1)';
        for n=failures, grotperms(perm) = grotperms(perm) + ...
                log( predictedYpD(n) / sum(predictedYpD(YD(:,1) >= YD(n,1))) ); end
        grotperms(perm) = -2 * grotperms(perm);
    end
    
    if perm==1
        predictedY = predictedYp;
        predictedY0 = predictedYp0; 
        predictedYD = predictedYpD;
        predictedYD0 = predictedYpD0;
        if strcmpi(family,'gaussian')
            stats.dev = sum((YinORIG-predictedY).^2);
            stats.baseline_dev = sum((YinORIG-predictedY0).^2);
            stats.corr = corr(YinORIG,predictedY);
            stats.baseline_corr = corr(YinORIG,predictedY0);
            stats.cod = 1 - stats.dev / stats.baseline_dev;
            if Nperm==1
                [~,pv] = corrcoef(YinORIG,predictedY); stats.pval=pv(1,2);
                if corr(YinORIG,predictedYp)<0, pv = 1; end
            end
            if deconfounding(2)
                stats.dev_deconf = sum((YD-predictedYD).^2);
                stats.baseline_dev_deconf = sum((YD-predictedYD0).^2);
                stats.corr_deconf = corr(YD,predictedYD);
                stats.baseline_corr_deconf = corr(YD,predictedYD0);
                stats.cod_deconf = 1 - stats.dev_deconf / stats.baseline_dev_deconf;
                if Nperm==1
                    [~,pvd] = corrcoef(YD,predictedYD); stats.pval_deconf=pvd(1,2);
                end
            end
            
        elseif strcmpi(family,'multinomial')
            if (strcmpi(Method,'glmnet') || strcmpi(Method,'unregularized'))
                stats.accuracy = mean(sum(YinORIG .* predictedY,2));
                stats.baseline_accuracy = mean(sum(YinORIG .* predictedY0,2));
            else
                stats.accuracy = (sum(YinORIG .* predictedY) + ...
                    sum((1-YinORIG) .* (1-predictedY))) / size(YinORIG,1);
                stats.baseline_accuracy = (sum(YinORIG .* predictedY0) + ...
                    sum((1-YinORIG) .* (1-predictedY0))) / size(YinORIG,1);
            end
            if deconfounding(2)
                if (strcmpi(Method,'glmnet') || strcmpi(Method,'unregularized'))
                    stats.accuracy_deconf = mean(sum(YD .* predictedYD,2));
                    stats.baseline_accuracy_deconf = mean(sum(YD .* predictedYD0,2));
                else
                    stats.accuracy_deconf = (sum(YD .* predictedYD) + ...
                        sum((1-YD) .* (1-predictedYD))) / size(YD,1);
                    stats.baseline_accuracy_deconf = (sum(YD .* predictedYD0) + ...
                        sum((1-YD) .* (1-predictedYD0))) / size(YD,1);
                end
            end                

        elseif strcmpi(family,'poisson')
            stats.dev = 2 * sum(YinORIG.*log((YinORIG+(YinORIG==0)) ./ ...
                predictedY) - (YinORIG - predictedY) );
            stats.baseline_dev = 2 * sum(YinORIG.*log((YinORIG+(YinORIG==0)) ./ ...
                predictedY0) - (YinORIG - predictedY0) );
            if deconfounding(2)
                stats.dev_deconf = 2 * sum(YD.*log((YD+(YD==0)) ./ ...
                    predictedYD) - (YD - predictedYD) );
                stats.baseline_dev_deconf = 2 * sum(YD.*log((YD+(YD==0)) ./ ...
                    predictedYD0) - (YD - predictedYD0) );
            end
        else % cox
            failures = find(YinORIG(:,2) == 1)';
            stats.dev = 0; stats.dev_baseline = 0;
            for n = failures 
                stats.dev = stats.dev + log( predictedY(n) / ...
                    sum(predictedY(YinORIG(:,1) >= YinORIG(n,1))) ); 
                stats.dev_baseline = stats.dev_baseline + ...
                     log( predictedY0(n) / sum(predictedY0(YinORIG(:,1) >= YinORIG(n,1))) );             
            end
            
        end
        
    else
        fprintf('Permutation %d \n',perm)
    end
end

if Nperm>1 
    stats.pval = sum(grotperms<=grotperms(1)) / (Nperm+1);
end

system(['rm -fr ',tmpnm]);

end

