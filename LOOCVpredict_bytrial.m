function [predictedY,stats] = LOOCVpredict_bytrial (Xin,Yin,family,parameters,confounds)

if nargin<4, parameters = {}; end
if ~isfield(parameters,'alpha'), alpha = 0.001; 
else, alpha = parameters.alpha; end
if ~isfield(parameters,'standardize'), standardize = 1;
else, standardize = parameters.standardize; end

[T,N,P] = size(Xin); % no. time points, no. trials, no. channels
Q = size(Yin,3); % no. output features
predictedY = zeros(size(Yin));
meanY = zeros(size(Yin));

% Standardizing Xin
if standardize
    Xin = reshape(zscore(reshape(Xin,[T*N P])),[T N P]);
end

% get confounds, and deconfound Xin
if (nargin>4) && ~isempty(confounds)
    if size(confounds)==3
        NC = size(confounds,3); 
        confounds = zscore(reshape(confounds,[T*N NC]));
    else
        NC = size(confounds,2);
        confounds = zscore(confounds); 
    end
    Xin = reshape(Xin,[T*N P]);
    [~,~,Xin] = nets_deconfound(Xin,[],confounds,'gaussian');
    Xin = reshape(Xin,[T N P]);
    confounds = reshape(confounds,[T N NC]);
else
    confounds = []; 
end

for ifold = 1:N
    J = ifold;
    ji = setdiff(1:N,J);
    
    X = reshape(Xin(:,ji,:),[(N-1)*T P]);
    Y = reshape(Yin(:,ji,:),[(N-1)*T Q]);
    XJ = permute(Xin(:,J,:),[1 3 2]);
    
    % deconfounding
    if ~isempty(confounds)
        C = reshape(confounds(:,ji,:),[(N-1)*T NP]);
        [~,~,~,betaY,interceptY,Y] = nets_deconfound([],Y,C,'gaussian');
    end
     
    my = mean(Y);
    meanY(:,J,:) = repmat(my,T,1); 
    
    % estiamte the regression coefficients and predict
    if strcmp(family,'gaussian')
        Y = Y - repmat(my,size(Y,1),1);
        beta = (X' * X + alpha * eye(P)) * X' * Y;
        predictedY(:,J,:) = XJ * beta + repmat(my,size(XJ,1),1);
    elseif strcmp(family,'poisson')
        for j = 1:Q
            beta = glmfit(X,Y(:,j),'poisson');
            predictedY(:,J,j) = glmval(beta,XJ,'log');
        end
    else
        error('Distribution not recognised')
    end

    % back to original space
    if ~isempty(confounds)
        Ytmp = permute(predictedY(:,J,:),[1 3 2]);
        C = permute(confounds(:,J,:),[1 3 2]);
        predictedY(:,J,:) = nets_confound(Ytmp,C,'gaussian',betaY,interceptY); 
    end
    
    %disp(['Fold ' num2str(ifold)])
                  
end

stats = struct();
stats.corr = zeros(N,Q);
stats.cod = zeros(N,Q);
for J = 1:N
    for ii = 1:Q
        y = Yin(:,J,ii);
        yhat = predictedY(:,J,ii);
        my = meanY(:,J,ii);
        stats.corr(J,ii) = corr(y,yhat);
        if strcmp(family,'gaussian')
            dev = sum((y-yhat).^2); nulldev = sum((y-mean(my)).^2);
        elseif strcmp(family,'poisson')
            dev = 2 * sum(y.*log((y+(y==0)) ./ yhat) - (y - yhat) );
            nulldev = 2 * sum(y.*log((y+(y==0)) ./ my) - (y - my) );
        end
        stats.cod(J,ii) = 1 - dev / nulldev;
    end
end

    
end
