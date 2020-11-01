function [estimation,dec_estimation] = nets_train(Yin,Xin,family,parameters,varargin)
%
% Elastic-net training of betas, with two-stage feature selection,
% using (stratified) LOO and permutation testing
%
% Diego Vidaurre
% Aarhus Uni / OHBA Oxford, 2020
%
% beta = nets_train(Y,X,family,parameters);
% beta = nets_train(Y,X,family,parameters,correlation_structure);
% beta = nets_train(Y,X,family,parameters,correlation_structure,confounds);
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
%   + CVscheme - the number of folds for the model selection phase (0 for LOO)
%   + CVfolds - prespecified CV folds: it must be a (nfolds x 1) cell
%   with one vector per fold, indicating which samples are going to
%   be used for testing in that fold; therefore the intersection of all the vectors
%   should amount to 1:N, where N is the number of samples
% correlation_structure (optional) - A (Nsamples X Nsamples) matrix with
%                                   integer dependency labels (e.g., family structure), or
%                                    A (Nsamples X 1) vector defining some
%                                    grouping: (1...no.groups) or 0 for no group
% confounds (optional) - features that potentially influence the inputs, 
%       and the outputs for family="gaussian'
%
% OUTPUTS
% estimation - estimated regression parameters in the original (non-decounfounded) space
%           including the intercept
% dec_estimation - parameters used to deconfound X and Y
% betaY - decounfounding regression parameters for Y
% betaX - decounfounding regression parameters for X


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
if ~isfield(parameters,'CVscheme'), CVscheme = 10;
else, CVscheme = parameters.CVscheme; end
if ~isfield(parameters,'CVfolds'), CVfolds = [];
else, CVfolds = parameters.CVfolds; end
if ~isfield(parameters,'Nfeatures'), Nfeatures=0;
else, Nfeatures = parameters.Nfeatures; end
if ~isfield(parameters,'deconfounding'), deconfounding=[1 0];
else, deconfounding = parameters.deconfounding; end
if ~isfield(parameters,'nlambda'), nlambda=2000;
else, nlambda = parameters.nlambda; end
if ~isfield(parameters,'riemann'), riemann = length(size(Xin))==3;
else, riemann = parameters.riemann; end
if ~isfield(parameters,'keepvar'), keepvar = 1;
else, keepvar = parameters.keepvar; end

if riemann 
   error('Riemann not yet implemented here') 
end

if deconfounding(2)==2
    error('This deconfounding strategy is not available for Y - deconfounding(2) must be 0 or 1')
end

enet = strcmpi(Method,'lasso') || strcmpi(Method,'glmnet');
if ~enet, nlambda = 1; end

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

% get confounds, and deconfound Xin
dec_estimation = [];
if (nargin>5) && ~isempty(varargin{2})
    confounds = varargin{2};
    confounds = confounds - repmat(mean(confounds),N,1);
    if deconfounding(1)==1
        [dec_estimation.betaX,Xin] = ...
            nets_deconfound(Xin,[],confounds,'gaussian',[],[],tmpnm);
    elseif deconfounding(1)==2
        error('deconfounding==2 not available in nets_train')
    end
    if deconfounding(2)==1 && strcmpi(family,'gaussian')
        [~,~,dec_estimation.betaY,Yin] = ...
            nets_deconfound([],Yin,confounds,'gaussian',[],[],tmpnm);
    elseif deconfounding(2)==2
        error('deconfounding==2 not available in nets_train')
    elseif deconfounding(2)==1 && ~strcmpi(family,'gaussian')
        error('deconfounding only available for family gaussian')
    end
end
if ~strcmpi(family,'gaussian') && deconfounding(2) > 0
    error('Deconfounding still only implemented for family gaussian')
end

if strcmpi(Method,'ridge')
    ridg_pen_scale = mean(diag(Xin' * Xin));
end

% create the inner CV structure - stratified for family=multinomial
if isempty(CVfolds)
    if CVscheme==1
        folds = {1:N};
    else
        folds = cvfolds(Yin,family,CVscheme,allcs);
    end
else
    folds = CVfolds;
end

% pre-kill features
feature_selection = Nfeatures(1)<p && Nfeatures(1)>0;
if feature_selection
    dev = nets_screen(Xin, Yin, family);
    [~,groti] = sort(dev);
    groti = groti(end-Nfeatures(1)+1:end);
else
    groti = find(sx>0);
end
Xin = Xin(:,groti);

if ~strcmpi(Method,'unregularized')
    
    Dev = Inf(nlambda,length(alpha));
    Lambda = {};
    
    for ialph = 1:length(alpha)
        
        options = {}; options.standardize = false;
        if strcmpi(family,'gaussian'), options.intr = false; end
        options.alpha = alpha(ialph); options.nlambda = nlambda;  
        
        if strcmp(family,'multinomial'), predictedYp = Inf(N,q,nlambda);
        else, predictedYp = Inf(N,nlambda);
        end
        
        % Inner CV loop
        for ifold = 1:length(folds)
            
            if enet
                if ifold>1, options.lambda = Lambda{ialph};
                elseif isfield(options,'lambda'), options = rmfield(options,'lambda');
                end
            end
            
            J = folds{ifold};
            ji = setdiff(1:N,J);
            X = Xin(ji,:); Y = Yin(ji,:);

            switch Method
                
                case {'glmnet','Glmnet'}
                    estimation = glmnet(X,Y,family,options);
                    
                case {'lasso','Lasso'}
                    estimation = struct();
                    if strcmpi(family,'gaussian')
                        if ifold==1
                            [estimation.beta,lassostats] = lasso(X,Y,'Alpha',alpha(ialph),'NumLambda',nlambda);
                        else
                            [estimation.beta,lassostats] = lasso(X,Y,'Alpha',alpha(ialph),'Lambda',Lambda{ialph});
                        end
                    else
                        if strcmpi(family,'multinomial')
                            strfam = 'binomial'; % q > 2 not implemented here
                        elseif strcmpi(family,'cox')
                            error('Cox family not implemented for lasso method');
                        else
                            strfam = family;
                        end
                        if ifold==1
                            [estimation.beta,lassostats] = ...
                                lassoglm(X,Y,strfam,'Alpha',alpha(ialph),'NumLambda',nlambda);
                        else
                            [estimation.beta,lassostats] = ...
                                lassoglm(X,Y,strfam,'Alpha',alpha(ialph),'Lambda',Lambda{ialph});
                        end
                    end
                    estimation.lambda = lassostats.Lambda;
                    estimation.a0 = lassostats.Intercept;
                    
                case {'ridge','Ridge'}
                    if strcmpi(family,'gaussian')
                        estimation = struct();
                        X = [ones(size(X,1),1) X];
                        R = alpha(ialph) * ridg_pen_scale * eye(size(X,2)); R(1,1) = 0;
                        estimation.beta = (X' * X + R) \ (X' * Y);
                    else
                        error('Only Gaussian family is implemented for ridge method');
                    end
            end
            
            if ifold == 1 && enet
                Lambda{ialph} = estimation.lambda;
                options = rmfield(options,'nlambda');
            end
            
            XJ = Xin(J,groti);
            
            if strcmpi(family,'gaussian')
                if enet
                    predictedYp(J,1:length(estimation.lambda)) = ...
                        XJ * estimation.beta + ...
                        repmat(estimation.a0(end),length(J),length(estimation.lambda));
                else % ridge
                    XJ = [ones(size(XJ,1),1) XJ];
                    predictedYp(J) = XJ * estimation.beta;
                end
            elseif strcmpi(family,'multinomial')
                if strcmpi(Method,'glmnet')
                    predictedYp(J,:,1:length(estimation.lambda)) = ...
                        nets_glmnetpredict(estimation,XJ,estimation.lambda,'response');
                elseif strcmpi(Method,'lasso')
                    predictedYp(J,1:length(estimation.lambda)) = ...
                        glmval([estimation.a0; estimation.beta] ,XJ,'logit');
                elseif strcmpi(Method,'ridge')
                    error('Only Gaussian family is implemented for ridge method');
                else % 'unregularized'
                    
                end
            elseif strcmpi(family,'poisson')
                if strcmpi(Method,'glmnet')
                    predictedYp(J,1:length(estimation.lambda)) = ...
                        max(nets_glmnetpredict(estimation,XJ,estimation.lambda,'response'),eps);
                elseif strcmpi(Method,'lasso')
                    predictedYp(J,1:length(estimation.lambda)) = ...
                        glmval([estimation.a0; estimation.beta],XJ,'log');
                end
            else % cox
                predictedYp(J,1:length(estimation.lambda)) = exp(XJ * estimation.beta);
            end
            
        end
        
        % Pick the one with the lowest deviance (=quadratic error for family="gaussian")
        if strcmpi(family,'gaussian') % it's actually N*log(sum.... but it doesn't matter
            if enet
                dev = sum(( predictedYp(:,1:length(Lambda{ialph})) - ...
                    repmat(Yin,1,length(Lambda{ialph}))).^2) / N; 
                dev = dev';
            else
                dev = sum(( predictedYp - Yin ).^2) / N;
            end
        elseif strcmpi(family,'multinomial')
            if strcmpi(Method,'glmnet')
                dev = Inf(length(Lambda{ialph}),1);
                for i=1:length(Lambda{ialph})
                    dev(i) = - sum(log(sum(Yin .* predictedYp(:,:,i) ,2))); 
                end
            else % lasso
                for i=1:length(Lambda{ialph})
                    dev(i) = - sum(log( ((1-Yin) .* (1-predictedYp(:,i))))) ...
                        - sum(log( (Yin .* predictedYp(:,i))));
                end
            end
        elseif strcmpi(family,'poisson')
            Ye = repmat(Yin,1,length(Lambda{ialph}));
            dev = sum(Ye .* log( (Ye+(Ye==0)) ./ predictedYp(:,1:length(Lambda{ialph}))) - ...
                (Ye - predictedYp(:,1:length(Lambda{ialph}))));
            dev = dev';
        else % cox
            failures = find(Yin(:,2) == 1)';
            dev = zeros(length(Lambda{ialph}),1);
            for i=1:length(Lambda{ialph})
                for n=failures
                    dev(i) = dev(i) + ...
                        log( predictedYp(n,i) / sum(predictedYp(Yin(:,1) >= Yin(n,1),i)) );
                end
            end
            dev = -2 * dev;
        end
        
        Dev(1:length(dev),ialph) = dev;
    end
    
    [~,opt] = min(Dev(:));
    
    if enet
        ialph = ceil(opt / nlambda);
        if ialph==0, ialph = length(Lambda); end; if ialph>length(Lambda), ialph = length(Lambda); end
        ilamb = mod(opt,nlambda);
        if ilamb==0, ilamb = nlambda; end
        if ilamb>length(Lambda{ialph}), ilamb = length(Lambda{ialph}); end
        options.alpha = alpha(ialph);
        options.lambda = (2:-.2:1)' * Lambda{ialph}(ilamb); % it doesn't like just 1 lambda
        switch Method
            case {'glmnet','Glmnet'}
                estimation = glmnet(Xin,Yin,family,options);
                if strcmpi(family,'multinomial')
                    for j = 1:length(estimation.beta)
                        estimation.beta{j} = estimation.beta{j}(:,end); 
                    end
                    estimation.a0 = estimation.a0(:,end);
                else
                    estimation.beta = estimation.beta(:,end);
                    estimation.a0 = estimation.a0(end);
                end
                estimation.lambda = estimation.lambda(end);
            case {'lasso','Lasso'}
                estimation = struct();
                if strcmpi(family,'gaussian')
                    [estimation.beta,lassostats] = ...
                        lasso(Xin,Yin,'Alpha',alpha(ialph),'Lambda',options.lambda);
                else
                    [estimation.beta,lassostats] = ...
                        lassoglm(Xin,Yin,strfam,'Alpha',alpha(ialph),'Lambda',options.lambda);
                end
                estimation.beta = estimation.beta(:,end);
                estimation.a0 = lassostats.Intercept(end);
        end
    else % ridge
        ialph = opt;
        options.alpha = alpha(ialph);
        Xin = [ones(size(Xin,1),1) Xin]; 
        R =  options.alpha * ridg_pen_scale * eye(size(Xin,2)); R(1,1) = 0;
        estimation = struct();
        estimation.beta = (Xin' * Xin + R) \ (Xin' * Yin);
    end
else
    estimation = struct(); estimation0 = struct();
    if strcmpi(family,'gaussian')
        Xin = [ones(size(Xin,1),1) Xin];
        estimation.beta = (Xin' * Xin) \ (Xin' * Yin);
    else
        estimation.beta = mnrfit(Xin,Yin);
    end
end

estimation.Method = Method;
estimation.family = family;
estimation.mx = mx;
estimation.sx = sx;

if feature_selection
    if strcmpi(Method,'glmnet') && strcmpi(family,'multinomial')
        beta_star = cell(length(estimation.beta));
        for j = 1:length(estimation.beta)
            beta_star{j} = zeros(p,1);
            beta_star{j}(groti) = estimation.beta{j};
        end
    else
        beta_star = zeros(p,size(estimation.beta,2));
        beta_star(groti,:) = estimation.beta;
    end
    estimation.beta = beta_star;
end


end


