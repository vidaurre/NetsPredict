function [betaX,X,betaY,Y,r2X,r2Y] = nets_deconfound(X,Y,confX,family,betaX,betaY,tmpnm)
% r2X and r2Y are the explained variance of X and Y by the confounds 
% (only for gaussian family)

options = {}; options.intr = 0; options.standardize = 0; options.lambda_min = 1e-10; 
riemann = length(size(X))==3;

%if strcmp(family,'cox'), options.intr = false; end

if nargin<5, betaX = []; end
if nargin<6, betaY = []; end

if ~isempty(X) && ~riemann % standardizing X in riemannian space is not clear
    % We don't need to model the intercept because X has been demeaned
    if isempty(betaX)
        R = 0.00001 * eye(size(confX,2));  
        betaX = (confX' * confX + R) \ confX' * X;
    end
    res = X - confX*betaX;
    if nargin>4
        r2X = 1 - sum(res.^2) ./ sum(X.^2);
    end
    X = res;
end

if ~isempty(Y)
    % includes the intercept
    confX = [ones(size(confX,1),1) confX];
    switch family
        case 'gaussian'
            if isempty(betaY)
                R = 0.00001 * eye(size(confX,2)); R(1,1) = 0;
                betaY = (confX' * confX + R) \ confX' * Y;
            end
            res = Y - confX*betaY;
            if nargin>5
                r2Y = 1 - sum(res.^2) ./ sum(Y.^2);
            end
            Y = res;
        case 'multinomial' % I don't think this is ready, I need to revise
            if all(Y(:)==1 | Y(:)==0)
                betaY = []; % you could do it, but it's the same
            else 
                if isempty(betaY), betaY = nets_glmnet(confX, Y, family, 0, tmpnm, options); end
                Y = link_multinomial(Y) - nets_glmnetpredict(betaY,confX,betaY.lambda(end),'link');
                Y = inverselink_multinomial(Y);
            end
        case 'poisson' % I don't think this is ready, I need to revise
            if isempty(betaY), betaY = nets_glmnet(confX, Y, family, 0, tmpnm, options); end
            Y = link_poisson(Y) -  nets_glmnetpredict(betaY,confX,betaY.lambda(end),'link');
            Y = inverselink_poisson(Y);
        case 'cox'
            error('Deconfounding not yet implemented for family=cox')
    end
end

end



