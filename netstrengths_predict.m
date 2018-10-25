% netstrengths_predict - network strengths estimation (Rosemberg et al 2016, NN),
% using spline regression instead, and using LOO and permutation testing
%   Diego Vidaurre FMRIB Oxford, 2013-2014
%
% [predictedY,stats] = netstrengths_predict(Y,X,parameters);
% [predictedY,stats] = netstrengths_predict(Y,X,parameters,correlation_structure);
% [predictedY,stats] = netstrengths_predict(Y,X,parameters,correlation_structure,Permutations);
% [predictedY,stats] = netstrengths_predict(Y,X,parameters,correlation_structure,Permutations,confounds);
%
% INPUTS
% Y - data vector (samples X 1)
% X - design matrix (samples X features)
% parameters is a structure with:
%   + thrpval - threshold pvalue to include an edge in the positive/negative tail
%   + splineregr - if 1, spline regression will be performed using the
%                   network stregths; if 0 (default), linear regression will be used instead
%   + univariate - if 1, then each tail is regressed on the response
%                   separately (as in the paper); if 0, then both are used together (default to 0)
%   + CVscheme - number of cross-validation folds for model evaluation (0 for LOO)
%   + Nperm - number of permutations (set to 0 to skip permutation testing)
%   + verbose -  display progress?
% correlation_structure (optional) - A (Nsamples X Nsamples) matrix with
%                                   integer dependency labels (e.g., family structure), or 
%                                    A (Nsamples X 1) vector defining some
%                                    grouping: (1...no.groups) or 0 for no group
% Permutations (optional but must also have correlation_structure) - pre-created set of permutations
% confounds (optional) - 
%            features that potentially influence the inputs and the outputs 
%
% OUTPUTS
% predictedY - predicted response
% stats structure, with fields
%   + pval - permutation-based p-value, if permutation is run;
%            otherwise, correlation-based 
%   + dev - cross-validated deviance (= the sum of squared errors)
%   + cod - coeficient of determination
%   + dev_deconf - cross-validated deviance in the deconfounded space 
%   + cod_deconf - coeficient of determination in the deconfounded space 

function [predictedY,stats,predictedYC,YoutC,predictedYmean,Cin,grotperms] ...
    = netstrengths_predict(Yin,Xin,parameters,varargin)

if nargin<3, parameters = {}; end
if ~isfield(parameters,'CVscheme'), CVscheme=0;
else CVscheme = parameters.CVscheme; end
if ~isfield(parameters,'thrpval'), thrpval=0.01;
else thrpval = parameters.thrpval; end
if ~isfield(parameters,'univariate'), univariate=0;
else univariate = parameters.univariate; end
if ~isfield(parameters,'splineregr'), splineregr=0;
else splineregr = parameters.splineregr; end
if ~isfield(parameters,'interactions'), interactions=0;
else interactions = parameters.interactions; end
if ~isfield(parameters,'standardise'), standardise=0;
else standardise = parameters.standardise; end
if ~isfield(parameters,'Nperm'), Nperm=1;
else Nperm = parameters.Nperm; end
if ~isfield(parameters,'riemann'), riemann = length(size(Xin))==3;
else riemann = parameters.riemann; end
if ~isfield(parameters,'keepvar'), keepvar = 1;
else keepvar = parameters.keepvar; end
if ~isfield(parameters,'verbose'), verbose=0;
else verbose = parameters.verbose; end

if splineregr
    options_spl = struct('nlearn',1,...
        'alpha',1+(~univariate),...
        'M',10); % last is smoothing parameter, you could include it as an option
end

% put Xin in the right format, which depends on riemann=1
Xin = reshape_pred(Xin,riemann,keepvar); N = size(Xin,1); 

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
mx=mean(Xin);  sx=std(Xin);
Xin = Xin - repmat(mx,N,1);
if standardise
    Xin(:,sx>0) = Xin(:,sx>0) ./ repmat(sx(sx>0),N,1);
end

if (Nperm<2),  Nperm=1;  end;
cs=[];
if (nargin>3)
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
               for s1=ss,
                   for s2=ss
                       allcs = [allcs; [s1 s2]];
                   end
               end
            end
            % grotMZi and grotDZi need to be computer here
        end
    end
end
if ~exist('allcs','var'), allcs = []; end
PrePerms=0;
if (nargin>4)
    Permutations=varargin{2};
    if ~isempty(Permutations)
        PrePerms=1;
        Nperm=size(Permutations,2);
    end
end
confounds=[];
% get confounds, and deconfound Xin
if (nargin>5)
    confounds=varargin{3};
    confounds = confounds - repmat(mean(confounds),N,1);
    [~,Xin] = nets_deconfound(Xin,[],confounds,'gaussian');
end

YinORIG=Yin; YinORIGmean = zeros(size(Yin));
grotperms = zeros(Nperm,1+(~univariate));
if ~isempty(confounds),
    YC = zeros(size(Yin));
    YCmean = zeros(size(Yin));
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
                if rand<0.5, wt=[1 2]; else wt=[2 1]; end;
                PERM(grotMZi(ipe,1))=grotMZi(perm1(ipe),wt(1));
                PERM(grotMZi(ipe,2))=grotMZi(perm1(ipe),wt(2));
            end
            perm1=randperm(size(grotDZi,1));
            for ipe=1:length(perm1)
                if rand<0.5, wt=[1 2]; else wt=[2 1]; end;
                PERM(grotDZi(ipe,1))=grotDZi(perm1(ipe),wt(1));
                PERM(grotDZi(ipe,2))=grotDZi(perm1(ipe),wt(2));
            end
            from=find(PERM==0);  pto=randperm(length(from));  to=from(pto);  PERM(from)=to;
            Yin=YinORIG(PERM,:);
        else                   % pre-supplied permutation
            Yin=YinORIG(Permutations(:,perm),:);  % or maybe it should be the other way round.....?
        end
    end
    
    if univariate
        predictedYp = zeros(N,2);
        if ~isempty(confounds), predictedYpC = zeros(N,2); end
    else
        predictedYp = zeros(N,1);
        if ~isempty(confounds), predictedYpC = zeros(N,1); end
    end
    
    % create the inner CV structure 
    folds = cvfolds(Yin,'gaussian',CVscheme,allcs);
    
    for ifold = 1:length(folds) 
        
        if verbose, fprintf('CV iteration %d \n',ifold); end
        J = folds{ifold}; QTN = length(J);
        if isempty(J), continue; end
        
        ji=setdiff(1:N,J); QN = length(ji);
        X = Xin(ji,:); Y = Yin(ji,1);
        Xtest = Xin(J,:); 
        
        % computing mean of the response in the original space
        YinORIGmean(J) = mean(Y);
        
        % deconfounding business
        if ~isempty(confounds),
            [~,~,betaY,Y] = nets_deconfound([],Y,confounds(ji,:),'gaussian');
        end;
        
        % centering response
        my = mean(Y); Y = Y-my;
        YCmean(J) = my;

        % strength calculation
        unicoef = zeros(p,1); unipval = zeros(p,1);
        for j = 1:p
            [b,stats] = robustfit(X(:,j),Y);
            unicoef(j) = b(2);
            unipval(j) = stats.p(2);
        end
        if min(unipval) > thrpval  % seems like for some variables robustfit gives junk......
          [unicoef,unipval]=corr(X,Y,'type','Spearman');
        end        
        positive = (unipval < thrpval) & (unicoef > 0);
        negative = (unipval < thrpval) & (unicoef < 0);
        strength = zeros(QN,2); strengthtest = zeros(QTN,2);
        strength(:,1) = sum(X(:,positive),2);
        strength(:,2) = sum(X(:,negative),2);
        strengthtest(:,1) = sum(Xtest(:,positive),2);
        strengthtest(:,2) = sum(Xtest(:,negative),2);
               
        if splineregr && univariate
            for j = 1:2
                predictedYp(J,j) = RKHSEnsemble(strength(:,j),Y,strengthtest(:,j),options_spl,0) + ...
                    repmat(my,length(J),1);
            end
        elseif splineregr && ~univariate
            predictedYp(J,1) = RKHSEnsemble(strength,Y,strengthtest,options_spl,interactions) + ...
                repmat(my,length(J),1);
        elseif ~splineregr && univariate
            for j = 1:2 
                b = strength(:,j) \ Y;
                predictedYp(J,j) = strengthtest(:,j) * b + repmat(my,length(J),1);
            end
        else
            b = strength \ Y;
            predictedYp(J,1) = strengthtest * b + repmat(my,length(J),1);
        end
        
        predictedYpC(J,:) = predictedYp(J,:); 
        YC(J,:) = Yin(J,:); % predictedYpC and YC in deconfounded space
        if ~isempty(confounds), % in order to later estimate prediction accuracy in deconfounded space
            [~,~,~,YC(J,:)] = nets_deconfound([],YC(J,:),confounds(J,:),'gaussian',[],betaY);
            if ~isempty(betaY)
                predictedYp(J,:) = nets_confound(predictedYp(J,:),confounds(J,:),'gaussian',betaY); % original space
            end
        end
        
    end
    
    for j=1:size(predictedYp,2)
        grotperms(perm,j) = sum((YC-predictedYpC(:,j)).^2);
    end
    
    if perm==1
        predictedY = predictedYp;
        predictedYmean = YinORIGmean;
        predictedYC = predictedYpC;
        YoutC = YC;
        stats = {};
        for j=1:size(predictedYp,2)           
            stats.dev(j) = sum((YinORIG-predictedYp(:,j)).^2);
            stats.nulldev(j) = sum((YinORIG-YinORIGmean).^2);
            [~,pv] = corrcoef(YinORIG,predictedYp(:,j)); stats.pval(j) = pv(1,2);
        end
        if ~isempty(confounds),
            for j=1:size(predictedYp,2)
                stats.dev_deconf(j) = sum((YC-predictedYpC(:,j)).^2);
                stats.nulldev_deconf(j) = sum((YC-YCmean).^2);
                [~,pv] = corrcoef(YC,predictedYpC); stats.pval_deconf=pv(1,2);
            end
        end
        for j=1:size(predictedYp,2)
            stats.cod(j) = 1 - stats.dev(j) / stats.nulldev(j);
        end
        if ~isempty(confounds),
            for j=1:size(predictedYp,2)
                stats.cod_deconf(j) = 1 - stats.dev_deconf(j) / stats.nulldev_deconf(j); 
            end
        end
    else
        fprintf('Permutation %d \n',perm)
    end
end

if (Nperm>1), 
    for j=1:size(predictedYp,2)
        stats.pval(j) = sum(grotperms(:,j)<=grotperms(1,j)) / (Nperm+1); 
    end
end

end

