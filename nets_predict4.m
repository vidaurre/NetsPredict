% nets_predict - elastic-net estimation, with two-stage feature selection,
% using (stratified) LOO and permutation testing
% Diego Vidaurre and Steve Smith 
% FMRIB Oxford, 2013-2014
%
% [predictedY,stats] = nets_predict4(Y,X,family,parameters);
% [predictedY,stats] = nets_predict4(Y,X,family,parameters,correlation_structure);
% [predictedY,stats] = nets_predict4(Y,X,family,parameters,correlation_structure,Permutations);
% [predictedY,stats] = nets_predict4(Y,X,family,parameters,correlation_structure,Permutations,confounds);
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


function [predictedY,stats,predictedYC,YoutC,predictedYmean,beta,Cin,grotperms,folds] ...
    = nets_predict4(Yin,Xin,family,parameters,varargin)

if nargin<3, family = 'gaussian'; end
if nargin<4, parameters = {}; end
if ~isfield(parameters,'Method')
    ch = which('glmnet');
    if ~isempty(ch),  Method = 'glmnet';
    else, Method = 'lasso';
    end
else
    Method = parameters.Method; 
    if strcmp(Method,'glmnet') 
        ch = which('glmnet');
        if isempty(ch)
            error('Package glmnet not found ? use Method=''lasso'' instead')
        end
    end
end

if ~isfield(parameters,'alpha')
    if strcmp(Method,'ridge')
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
if ~isfield(parameters,'relaxed'), relaxed = 0; %strcmp(family,'gaussian');
else, relaxed = parameters.relaxed; end
if ~isfield(parameters,'Nperm'), Nperm=1;
else, Nperm = parameters.Nperm; end
if ~isfield(parameters,'nlambda'), nlambda=2000;
else, nlambda = parameters.nlambda; end
if ~isfield(parameters,'riemann'), riemann = length(size(Xin))==3;
else, riemann = parameters.riemann; end
if ~isfield(parameters,'standardize'), standardize = 1;
else, standardize = parameters.standardize; end
if ~isfield(parameters,'keepvar'), keepvar = 1;
else, keepvar = parameters.keepvar; end
if ~isfield(parameters,'show_scatter'), show_scatter=0;
else, show_scatter = parameters.show_scatter; end
if ~isfield(parameters,'verbose'), verbose=0;
else, verbose = parameters.verbose; end

if biascorrect == 1 
    if ~strcmp(family,'gaussian')
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
    
if strcmp(Method,'lasso') || strcmp(Method,'ridge')
   if ~strcmp(family,'gaussian')
       error('Families other than gaussian are only implemented for the glmnet method')
   end
end
enet = strcmp(Method,'lasso') || strcmp(Method,'glmnet');
if ~enet
    nlambda = 1; 
end

% put Xin in the right format, which depends on riemann=1
Xin = reshape_pred(Xin,riemann,keepvar); N = size(Xin,1); 

if ischar(family)==1
    f = family; family = {}; family{1} = f; family{2} = f; clear f;
elseif ~relaxed
    error('If not relaxed, then family must be a string')
end

tmpnm = tempname; mkdir(tmpnm); mkdir(strcat(tmpnm,'/out')); mkdir(strcat(tmpnm,'/params')); 

Yin2 = Yin; % used for the variable selection stage
if strcmp(family{1},'multinomial') || strcmp(family{2},'multinomial')
    if strcmp(family{1},'multinomial') 
        if size(Yin,2)==1, Yin = nets_class_vectomat(Yin); end
        q = size(Yin,2);
    end
    if strcmp(family{2},'multinomial') 
        if size(Yin,2)==1, Yin2 = nets_class_vectomat(Yin);  
        else Yin2 = Yin; end 
        q = size(Yin2,2);
    end
    if q>9, error('Too many classes!'); end
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
if standardize
    Xin = Xin - repmat(mx,N,1);
    Xin(:,sx>0) = Xin(:,sx>0) ./ repmat(sx(sx>0),N,1);
end
if (Nperm<2),  Nperm=1;  end
if (nargin>4) && ~isempty(varargin{1})
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
        [~,~,Xin] = nets_deconfound(Xin,[],confounds,'gaussian',[],[],[],[],tmpnm);
    elseif deconfounding(1)==2
        for j = 1:p
           Xin(:,j) = Xin(:,j) - nets_predict4(Xin(:,j),confounds,'gaussian',parameters_dec); 
        end
    end
else
    confounds = []; deconfounding = [0 0];
end

if strcmp(Method,'ridge') || relaxed
   ridg_pen_scale = mean(diag(Xin' * Xin));
end

if relaxed
   alpha_ridge = [0.00001 0.0001 0.001 0.01 0.1 0.4 0.7 0.9 1.0 10 100]; 
end

YinORIG = Yin; 
YinORIGmean = zeros(size(Yin));
YinORIG2 = Yin2;
%if exist('Yin2','var'), YinORIG2=Yin2; YinORIGmean2 = zeros(size(Yin2)); end 
grotperms = zeros(Nperm,1);
YC = zeros(size(Yin)); % deconfounded signal
YCmean = zeros(size(Yin)); % mean in deconfounded space

if isempty(CVfolds)
    if enet && relaxed
        beta = zeros(p,CVscheme(1),2);
    else 
        beta = zeros(p,CVscheme(1));
    end
else
    if enet && relaxed
        beta = zeros(p,length(CVfolds),2);
    else
        beta = zeros(p,length(CVfolds));
    end
end

for perm=1:Nperm
    if (perm>1)
        if isempty(cs)           % simple full permutation with no correlation structure
            rperm = randperm(N);
            Yin=YinORIG(rperm,:);
            Yin2=YinORIG2(rperm,:);
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
            Yin2=YinORIG2(PERM,:); 
        else                   % pre-supplied permutation
            Yin=YinORIG(Permutations(:,perm),:);  % or maybe it should be the other way round.....?
            Yin2=YinORIG2(Permutations(:,perm),:); 
        end
    end
    
    if strcmp(family{1},'multinomial')
        predictedYp = zeros(N,q); 
        if deconfounding(2), predictedYpC = zeros(N,q); end  
    else
        predictedYp = zeros(N,1);
        if deconfounding(2), predictedYpC = zeros(N,1); end 
    end
    
    % create the inner CV structure - stratified for family=multinomial
    if isempty(CVfolds)
        if CVscheme(1)==1
            folds = {1:N};
        else            
            folds = cvfolds(Yin,family{1},CVscheme(1),allcs);
        end
    else
        folds = CVfolds;
    end
    
    if perm==1
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
        
        % family structure for this fold
        Qallcs=[]; 
        if (~isempty(cs))
            [Qallcs(:,2),Qallcs(:,1)]=ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));  
        end
                
        % deconfounding business
        if deconfounding(2)==1
            [~,~,~,betaY,interceptY,Y] = nets_deconfound([],Y,confounds(ji,:),family{1},[],[],[],[],tmpnm);
        end
        
        % centering response
        if strcmp(family{1},'gaussian')
            my = mean(Y); Y = Y - my; 
            YCmean(J) = my;
        elseif deconfounding(2) % YCmean's only used when ~isempty(confounds)
            glmfit0 = nets_glmnet(randn(size(Y,1),2),Y,family{1},0,tmpnm,options0); %  deconfounded space
            pred0 = nets_glmnetpredict(glmfit0,randn(length(J),2),[],'response'); 
            if strcmp(family{1},'multinomial'), YCmean(J,:) = pred0(:,:,1); 
            elseif strcmp(family{1},'poisson'), YCmean(J,:) = max(pred0(:,1),eps);
            else YCmean(J,:) = pred0(:,1); 
            end
            %if any(YCmean<0), keyboard; end
        end

        % pre-kill features
        if Nfeatures(1)<p && Nfeatures(1)>0
            options = {}; options.lambda = [1 0]';
            if strcmp(family{1},'cox'), options.intr = false; end
            dev = nets_glmnet(X, Y, family{1}, 1, tmpnm, options);
            [~,groti0]=sort(dev);
            groti0=groti0(end-Nfeatures(1)+1:end);
        else
            groti0 = find(sx>0);
        end
        
        QXin = Xin(ji,groti0);
        % uncomment this if you want to deconfound in inner loop
        %QYin = Yin(ji,:); 
        QYin = Y;
        
        %if ~isempty(confounds), Qconfounds=confounds(ji,:); end
        X = X(:,groti0);
        options = {}; options.standardize = false;
        if strcmp(family{1},'gaussian'), options.intr = false; end
        
        % create the inner CV structure - stratified for family=multinomial
        Qfolds = cvfolds(Y,family{1},CVscheme(2),Qallcs);
        
        % Variable selection with the elastic net
        if enet && length(Nfeatures)==2
            options.pmax = Nfeatures(2);
        end
        Dev = Inf(nlambda,length(alpha));
        Lambda = {};
        for ialph = 1:length(alpha)
            if strcmp(family{1},'multinomial'), QpredictedYp = Inf(QN,q,nlambda);
            else QpredictedYp = Inf(QN,nlambda);
            end
            options.alpha = alpha(ialph); options.nlambda = nlambda;
            QYinCOMPARE=QYin;
            % Inner CV loop
            for Qifold = 1:length(Qfolds)
                QJ = Qfolds{Qifold};
                Qji=setdiff(1:QN,QJ); 
                QX = QXin(Qji,:); QY = QYin(Qji,:);
                if strcmp(family{1},'gaussian'), Qmy=mean(QY);  QY=QY-Qmy; end
                if enet
                    if Qifold>1, options.lambda = Lambda{ialph};
                    elseif isfield(options,'lambda'), options = rmfield(options,'lambda');
                    end
                end
                switch Method
                    case {'glmnet','Glmnet'}
                        glmfit = nets_glmnet(QX,QY,family{1},0,tmpnm,options);
                    case {'lasso','Lasso'}
                        glmfit = struct();
                        if Qifold==1
                            [glmfit.beta,lassostats] = lasso(QX,QY,'Alpha',alpha(ialph),'NumLambda',nlambda);
                        else
                            [glmfit.beta,lassostats] = lasso(QX,QY,'Alpha',alpha(ialph),'Lambda',Lambda{ialph});
                        end
                        glmfit.lambda = lassostats.Lambda; 
                    case {'ridge','Ridge'}
                        glmfit = struct();
                        glmfit.beta = (QX' * QX + alpha(ialph) * ridg_pen_scale * eye(size(QX,2))) \ (QX' * QY);
                end
                if Qifold == 1 && enet
                    Lambda{ialph} = glmfit.lambda; 
                    options = rmfield(options,'nlambda');
                end
                QXJ=QXin(QJ,:);
                if strcmp(family{1},'gaussian')
                    if enet
                        QpredictedYp(QJ,1:length(glmfit.lambda)) = ...
                            QXJ * glmfit.beta + repmat(Qmy,length(QJ),length(glmfit.lambda));
                    else
                        QpredictedYp(QJ) = QXJ * glmfit.beta + repmat(Qmy,length(QJ),1);
                    end
                elseif strcmp(family{1},'multinomial')
                    QpredictedYp(QJ,:,1:length(glmfit.lambda)) = ...
                        nets_glmnetpredict(glmfit,QXJ,glmfit.lambda,'response');
                elseif strcmp(family{1},'poisson')
                    QpredictedYp(QJ,1:length(glmfit.lambda)) = ...
                        max(nets_glmnetpredict(glmfit,QXJ,glmfit.lambda,'response'),eps); 
                        %exp(QXJ * glmfit.beta + repmat(glmfit.a0',length(QJ),1) );
                else % cox
                    QpredictedYp(QJ,1:length(glmfit.lambda)) = exp(QXJ * glmfit.beta);
                end
            end
            % Pick the one with the lowest deviance (=quadratic error for family="gaussian")
            if strcmp(family{1},'gaussian') % it's actually QN*log(sum.... but it doesn't matter
                if enet
                    Qdev = sum(( QpredictedYp(:,1:length(Lambda{ialph})) - ...
                        repmat(QYinCOMPARE,1,length(Lambda{ialph}))).^2) / QN; Qdev = Qdev';
                else
                    Qdev = sum(( QpredictedYp - QYinCOMPARE ).^2) / QN;  
                end
            elseif strcmp(family{1},'multinomial')
                Qdev = Inf(length(Lambda{ialph}),1);
                for i=1:length(Lambda{ialph}), Qdev(i) = - sum(log(sum(QYinCOMPARE .* QpredictedYp(:,:,i) ,2))); end
            elseif strcmp(family{1},'poisson')
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
                            log( QpredictedYp(n,i) / sum(QpredictedYp(QYinCOMPARE(:,1) >= QYinCOMPARE(n,1),i)) ); 
                    end
                end; Qdev = -2 * Qdev;
            end
            Dev(1:length(Qdev),ialph) = Qdev;
        end
        [~,opt] = min(Dev(:));
        if enet
            ialph = ceil(opt / nlambda);
            if ialph==0, ialph = length(Lambda); end; if ialph>length(Lambda), ialph = length(Lambda); end
            ilamb = mod(opt,nlambda);
            if ilamb==0, ilamb = nlambda; end; if ilamb>length(Lambda{ialph}), ilamb = length(Lambda{ialph}); end
            options.alpha = alpha(ialph);
            if verbose, fprintf('Alpha chosen to be  %f \n',options.alpha); end
            options.lambda = (2:-.1:1)' * Lambda{ialph}(ilamb); % it doesn't like just 1 lambda
            % we set lambda instead of pmax/dfmax because of a bug in glmnet that arises when pmax is specified
            %options.pmax = max(2,ceil(mean(Qdf{ialph}(ilamb,:))));
            %options.dfmax = options.pmax;
            %options = rmfield(options,'lambda');
            % and run again on the whole fold
            if strcmp(Method,'glmnet')
                glmfit = nets_glmnet(X,Y,family{1},0,tmpnm,options);
            else
                glmfit = struct();
                glmfit.beta = lasso(X,Y,'Alpha',options.alpha,'Lambda',options.lambda);
            end
        else
            ialph = opt; 
            options.alpha = alpha(ialph);
            if verbose, fprintf('Alpha chosen to be  %f \n',options.alpha); end
            glmfit = struct();
            glmfit.beta = (X' * X + options.alpha * ridg_pen_scale * eye(size(X,2))) \ (X' * Y);
        end
        if perm==1 
            if strcmp(family{1},'multinomial')
                beta(groti0,ifold,1) = glmfit.beta{1}(:,end);
            else
                beta(groti0,ifold,1) = glmfit.beta(:,end);
            end
        end

        if strcmp(family{2},'multinomial') % because there are a different set of coefficients per class
            groti = false(size(glmfit.beta{1},1),1);
            for nb=1:length(glmfit.beta), groti(glmfit.beta{nb}(:,end)~=0) = true; end
        else
            groti = glmfit.beta(:,end)~=0;
        end
        if sum(groti)<2
            if verbose, fprintf('The empty model turns out to be the best...\n '); end
            if sum(groti)==1 && groti(1)==0, groti(1) = 1;
            elseif sum(groti)==1, groti(2) = 1;
            else groti(1:2) = 1; 
            end
        end
        
        if enet && relaxed
            QXin=QXin(:,groti);
            X = X(:,groti);
            groti = groti0(groti);
        else
            groti = groti0;
        end
        
        % Prediction with the elastic net, keeping the (CV) best no. of variables for each value of alpha
        % It uses the best value of alpha from the first stage
        % Inner CV loop (2) - only for enet when relaxed is applied
        if enet && relaxed
            QYin = Yin2(ji,:);
            QpredictedYp = Inf(QN,length(alpha_ridge));
            for Qifold = 1:length(Qfolds)
                QJ = Qfolds{Qifold};
                Qji = setdiff(1:QN,QJ); 
                QX = QXin(Qji,:);  QY=QYin(Qji,:);
                if strcmp(family{2},'gaussian'), Qmy=mean(QY);  QY=QY-Qmy; 
                else, error('Family other than Gaussian not currenty implemented for relaxed estimation');
                end
                QXJ=QXin(QJ,:);
                for ialph = 1:length(alpha_ridge)
                    beta_relaxed = (QX' * QX + alpha_ridge(ialph) * ridg_pen_scale * eye(size(QX,2))) \ (QX' * QY);
                    QpredictedYp(QJ,ialph) = QXJ * beta_relaxed + repmat(Qmy,length(QJ),1);
                end
            end
            Qdev = sum(( QpredictedYp(:,1:length(glmfit.lambda)) - ...
                repmat(QYinCOMPARE,1,length(alpha_ridge))).^2) / QN; Qdev = Qdev';
            [~, ialph] = min(Qdev);
            beta_final = (X' * X + alpha_ridge(ialph) * ridg_pen_scale * eye(size(X,2))) \ (X' * Y);
            beta(groti,ifold,2) = beta_final;
        else
            beta_final = glmfit.beta(:,end);
        end
        % predict the test fold
        XJ=Xin(J,groti);
                
        if strcmp(family{2},'gaussian')
            predictedYp(J) = XJ * beta_final + repmat(my,length(J),1);
        elseif strcmp(family{2},'multinomial') % deconfounded space, both predictedYp and predictedYp0
            predictedYp(J,:) = nets_glmnetpredict(glmfit,XJ,glmfit.lambda(end),'response'); 
        elseif strcmp(family{2},'poisson')
            predictedYp(J,:) = max(nets_glmnetpredict(glmfit,XJ,glmfit.lambda(end),'response'),eps);  
            %predictedYp(J) = exp(XJ * glmfit.beta(:,end) + glmfit.a0(end) );
        else % cox
            predictedYp(J) = exp(XJ * glmfit.beta(:,end));
        end
        
        % predictedYpC and YC in deconfounded space; Yin and predictedYp are confounded
        predictedYpC(J,:) = predictedYp(J,:); 
        YC(J,:) = Yin(J,:);
        YinORIGmean(J) = YCmean(J,:);
        if deconfounding(2) % in order to later estimate prediction accuracy in deconfounded space
            [~,~,~,~,~,YC(J,:)] = nets_deconfound([],YC(J,:),confounds(J,:),family{2},[],[],betaY,interceptY,tmpnm);
            if ~isempty(betaY)
                predictedYp(J,:) = nets_confound(predictedYp(J,:),confounds(J,:),family{2},betaY,interceptY); % original space
                YinORIGmean(J) = nets_confound(YCmean(J,:),confounds(J,:),family{2},betaY,interceptY); 
            end
        end
        
        if biascorrect % we do this in the original space
            Yhattrain = Xin(ji,groti) * beta_final + repmat(my,length(ji),1);
            if deconfounding(2)
                Yhattrain = nets_confound(Yhattrain,confounds(ji,:),family{2},betaY,interceptY);
            end
            Ytrain = [QYin ones(size(QYin,1),1)]; 
            b = pinv(Ytrain) * Yhattrain;
            predictedYp(J,:) = (predictedYp(J,:) - b(2)) / b(1);
        end
        
        %if ~strcmp(family{1},'gaussian') && perm==1,
        %    predictedYp0C(J,:) = predictedYp0(J,:); % deconfounded space
        %    if ~isempty(confounds) && ~isempty(betaY), predictedYp0(J,:) = ...
        %           nets_confound(predictedYp0(J,:),confXte,family{1},betaY); end % original space
        %end
        
        if perm==1
            stats.alpha(ifold) = options.alpha;
        end
        
    end
    
    % grotperms computed in deconfounded space
    if strcmp(family{2},'gaussian')  
        grotperms(perm) = sum((YC-predictedYpC).^2);
    elseif strcmp(family{2},'multinomial')
        grotperms(perm) = - 2 * sum(log(sum(YC .* predictedYpC,2)));  % MAL
    elseif strcmp(family{2},'poisson')
        grotperms(perm) = 2 * sum(YC.*log((YC+(YC==0)) ./ predictedYpC) - (YC - predictedYpC) );
    else % cox - in the current version there is no response deconfounding for family{1}="cox"
        grotperms(perm) = 0;
        failures = find(Yin(:,2) == 1)';
        for n=failures, grotperms(perm) = grotperms(perm) + ...
                log( predictedYp(n) / sum(predictedYp(Yin(:,1) >= Yin(n,1))) ); end
        grotperms(perm) = -2 * grotperms(perm);
    end
    
    if perm==1
        predictedY = predictedYp;
        predictedYmean = YinORIGmean; 
        predictedYC = predictedYpC;
        YoutC = YC;
        if strcmp(family{2},'gaussian')
            stats.dev = sum((YinORIG-predictedYp).^2);
            stats.nulldev = sum((YinORIG-YinORIGmean).^2);
            stats.corr = corr(YinORIG,predictedYp);
            stats.cod = 1 - stats.dev / stats.nulldev;
            if Nperm==1
                [~,pv] = corrcoef(YinORIG,predictedYp); stats.pval=pv(1,2);
                if corr(YinORIG,predictedYp)<0, pv = 1; end
            end
            if deconfounding(2)
                stats.dev_deconf = sum((YC-predictedYpC).^2);
                stats.nulldev_deconf = sum((YC-YCmean).^2);
                stats.corr_deconf = corr(YC,predictedYpC);
                if Nperm==1
                    [~,pvd] = corrcoef(YC,predictedYpC); stats.pval_deconf=pvd(1,2);
                end
            end
        elseif strcmp(family{2},'multinomial')
            stats.accuracy = mean(sum(YinORIG .* predictedYp,2));
            stats.dev = - 2 * sum(log(sum(YinORIG .* predictedYp,2)));
            stats.nulldev = - 2 * sum(log(sum(YinORIG .* YinORIGmean,2)));
            if deconfounding(2)
                stats.accuracy_deconf = mean(sum(YC .* predictedYpC,2));
                stats.dev_deconf = - 2 * sum(log(sum(YC .* predictedYpC,2)));
                stats.nulldev_deconf = - 2 * sum(log(sum(YC .* YCmean,2)));
            end
        elseif strcmp(family{2},'poisson')
            stats.dev = 2 * sum(YinORIG.*log((YinORIG+(YinORIG==0)) ./ predictedYp) - (YinORIG - predictedYp) );
            stats.nulldev = 2 * sum(YinORIG.*log((YinORIG+(YinORIG==0)) ./ YinORIGmean) - (YinORIG - YinORIGmean) );
            if deconfounding(2)
                stats.dev_deconf = 2 * sum(YC.*log((YC+(YC==0)) ./ predictedYpC) - (YC - predictedYpC) );
                stats.nulldev_deconf = 2 * sum(YC.*log((YC+(YC==0)) ./ YCmean) - (YC - YCmean) );
            end
        else % cox
            failures = find(YinORIG(:,2) == 1)';
            stats.dev = 0;
            for n=failures, stats.dev = stats.dev + log( predictedYp(n) / ...
                    sum(predictedYp(YinORIG(:,1) >= YinORIG(n,1))) ); end
            stats.dev = -2 * stats.dev;
            stats.nulldev = 0;
            for n=failures, stats.nulldev = stats.nulldev + ...
                    log( YinORIGmean(n) / sum(YinORIGmean(YinORIG(:,1) >= YinORIG(n,1))) ); end
            stats.nulldev = -2 * stats.nulldev;
        end
        if deconfounding(2), stats.cod_deconf = 1 - stats.dev_deconf / stats.nulldev_deconf; end
        
        if show_scatter && (strcmp(family{2},'gaussian') || strcmp(family{2},'poisson'))
            figure;  scatter(Yin,predictedYp);
        end
    else
        fprintf('Permutation %d \n',perm)
    end
end

if Nperm>1 
    stats.pval = sum(grotperms<=grotperms(1)) / (Nperm+1);
end
beta = squeeze(beta);

system(['rm -fr ',tmpnm]);

end

