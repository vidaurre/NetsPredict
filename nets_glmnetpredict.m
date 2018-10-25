function result = nets_glmnetpredict(object, newx, s, type, exact, offset)

% Adaptation of the glmnetPredict function from the Matlab glmnet package
%
% AUTHORS:
%    Algorithm was designed by Jerome Friedman, Trevor Hastie and Rob Tibshirani
%    Fortran code was written by Jerome Friedman
%    R wrapper (from which the MATLAB wrapper was adapted) was written by Trevor Hasite
%    The original MATLAB wrapper was written by Hui Jiang (14 Jul 2009),
%    and was updated and maintained by Junyang Qian (30 Aug 2013) junyangq@stanford.edu,
%    Department of Statistics, Stanford University, Stanford, California, USA.
%
%    Adaptation to fsl_nets by Diego Vidaurre,  Oxford University 
%

if nargin < 2 || isempty(newx)
    newx = [];
end
if nargin < 3
    s = [];
end
if nargin < 4 || isempty(type)
    type = 'link';
end
if nargin < 5 || isempty(exact)
    exact = false;
end
if nargin < 6
    offset = [];
end

typebase = {'link','response','coefficients','nonzero','class'};
typeind = find(strncmp(type,typebase,length(type)),1);
type = typebase{typeind};

if isempty(newx)
    if ~strcmp(type, 'coefficients') && ~strcmp(type, 'nonzero')
        error('You need to supply a value for ''newx''');
    end
end

%exact case: need to execute statements back in the parent environment
if (exact && ~isempty(s))
    which = ismember(s,object.lambda);
    if ~all(which)
        lambda = unique([object.lambda;reshape(s,length(s),1)]);
        %-----create a new variable in the parent environment
        vname = 'newlam';
        expr = sprintf('any(strcmp(''%s'', who))',vname);
        newname = vname;
        i = 0;
        while (evalin('caller',expr))
            i = i + 1;
            newname = [vname,num2str(i)];
            expr = sprintf('any(strcmp(who,''%s''))',newname);
        end
        parlam = newname;
        %-----
        assignin('caller', parlam, lambda);
        
        vname = 'temp_opt';
        expr = sprintf('any(strcmp(''%s'', who))',vname);
        newname = vname;
        i = 0;
        while (evalin('caller',expr))
            i = i + 1;
            newname = [vname,num2str(i)];
            expr = sprintf('any(strcmp(who,''%s''))',newname);
        end
        paropt = newname;
        
        if strcmp('[]',object.call{3})
            famcall = object.call{3};
        else
            famcall = sprintf('''%s''',object.call{3});
        end
        
        if ~strcmp('[]', object.call{4})
            evalin('caller', strcat(paropt,'=',object.call{4},';'));
            evalin('caller', strcat(paropt,'.lambda = ',parlam,';'));
            newcall = sprintf('glmnet(%s, %s, %s, %s)', ...
                object.call{1}, object.call{2}, famcall, paropt);
            object = evalin('caller', newcall);
        else
            evalin('caller', strcat(paropt,'.lambda = ',parlam,';'));
            newcall = sprintf('glmnet(%s, %s, %s, %s)', ...
                object.call{1}, object.call{2}, famcall, paropt);
            object = evalin('caller', newcall);
        end
        evalin('caller', sprintf('clearvars %s %s;',parlam,paropt));
    end
end


if strcmp(object.class,'elnet') || strcmp(object.class,'fishnet')
    a0 = transpose(object.a0);
    nbeta=[a0; object.beta];
    
    if (~isempty(s))
        lambda=object.lambda;
        lamlist=lambda_interp(lambda,s);
        nbeta=nbeta(:,lamlist.left).*repmat(lamlist.frac',size(nbeta,1),1) +nbeta(:,lamlist.right).*(1-repmat(lamlist.frac',size(nbeta,1),1));
    end
    
    if strcmp(type, 'coefficients')
        result = nbeta;
        return;
    end
    if strcmp(type, 'nonzero')
        result = nonzeroCoef(nbeta(2:size(nbeta,1),:), true);
        return;
    end
    
    result = [ones(size(newx,1),1), newx] * nbeta;
    if (object.offset)
        if isempty(offset)
            error('No offset provided for prediction, yet used in fit of glmnet');
        end
        if (size(offset,2)==2)
            offset = offset(:,2);
        end
        result = result + reshape(offset,size(result));
    end
end

if strcmp(object.class, 'lognet')
    a0 = object.a0;
    nbeta=[a0; object.beta];
    
    if (~isempty(s))
        lambda=object.lambda;
        lamlist=lambda_interp(lambda,s);
        nbeta=nbeta(:,lamlist.left).*repmat(lamlist.frac',size(nbeta,1),1) +nbeta(:,lamlist.right).*(1-repmat(lamlist.frac',size(nbeta,1),1));
    end
    
    if strcmp(type, 'coefficients')
        result = nbeta;
        return;
    end
    if strcmp(type, 'nonzero')
        result = nonzeroCoef(nbeta(2:size(nbeta,1),:), true);
        return;
    end
    
    result = [ones(size(newx,1),1), newx] * nbeta;
    if (object.offset)
        if isempty(offset)
            error('No offset provided for prediction, yet used in fit of glmnet');
        end
        if (size(offset,2)==2)
            offset = offset(:,2);
        end
        result = result + reshape(offset,size(result));
    end
    switch type
        case 'response'
            pp = exp(-result);
            result = 1./ (1+pp);
        case 'class'
            result = (result > 0) * 2 + (result <= 0) * 1;
    end
end

if strcmp(object.class, 'multnet') || strcmp(object.class,'mrelnet')
    if strcmp(object.class,'mrelnet')
        if strcmp(type, 'response')
            type = 'link';
        end
        object.grouped = true;
    end
    
    a0=object.a0;
    nbeta=object.beta;
    nclass=size(a0,1);
    nlambda=length(s);
    
    if (~isempty(s))
        lambda=object.lambda;
        lamlist=lambda_interp(lambda,s);
        for i=1:nclass
            kbeta=[a0(i,:); nbeta{i}];
            kbeta=kbeta(:,lamlist.left).*repmat(lamlist.frac',size(kbeta,1),1)+kbeta(:,lamlist.right).*(1-repmat(lamlist.frac',size(kbeta,1),1));
            nbeta{i}=kbeta;
        end
    else
        for i=1:nclass
            nbeta{i} = [a0(i,:);nbeta{i}];
        end
        nlambda = length(object.lambda);
    end
    if strcmp(type, 'coefficients')
        result = nbeta;
        return;
    end
    if strcmp(type, 'nonzero')
        if (object.grouped)
            result = nonzeroCoef(nbeta{1}(2:size(nbeta{1},1),:),true);
        else
            for i=1:nclass
                result{i}=nonzeroCoef(nbeta{i}(2:size(nbeta{i},1),:),true);
            end
        end
        return;
    end
    npred=size(newx,1);
    dp = zeros(nclass,nlambda,npred);
    for i=1:nclass
        fitk = [ones(size(newx,1),1), newx] * nbeta{i};
        dp(i,:,:)=dp(i,:,:)+reshape(transpose(fitk),1,nlambda,npred);
    end
    if (object.offset)
        if (isempty(offset))
            error('No offset provided for prediction, yet used in fit of glmnet');
        end
        if (size(offset,2) ~= nclass)
            error('Offset should be dimension%dx%d',npred,nclass)
        end
        toff = transpose(offset);
        for i = 1:nlambda
            dp(:,i,:) = dp(:,i,:) + toff;
        end
    end
    switch type
        case 'response'
            pp = exp(dp);
            psum = sum(pp,1);
            result = permute(pp./repmat(psum,nclass,1),[3,1,2]);
        case 'link'
            result=permute(dp,[3,1,2]);
        case 'class'
            dp=permute(dp,[3,1,2]);
            result = [];
            for i=1:size(dp,3)
                result = [result, softmax(dp(:,:,i))];
            end
    end
    
end

if strcmp(object.class,'coxnet')
    nbeta = object.beta;
    if (~isempty(s))
        lambda=object.lambda;
        lamlist=lambda_interp(lambda,s);
        nbeta=nbeta(:,lamlist.left).*repmat(lamlist.frac',size(nbeta,1),1) +nbeta(:,lamlist.right).*(1-repmat(lamlist.frac',size(nbeta,1),1));
    end
    if strcmp(type, 'coefficients')
        result = nbeta;
        return;
    end
    if strcmp(type, 'nonzero')
        result = nonzeroCoef(nbeta, true);
        return;
    end
    result = newx * nbeta;
    if (object.offset)
        if isempty(offset)
            error('No offset provided for prediction, yet used in fit of glmnet');
        end
        result = result + reshape(offset,size(result));
    end
    
    if strcmp(type, 'response')
        result = exp(result);
    end
end
        
end


function result = lambda_interp(lambda,s)
% lambda is the index sequence that is produced by the model
% s is the new vector at which evaluations are required.
% the value is a vector of left and right indices, and a vector of fractions.
% the new values are interpolated bewteen the two using the fraction
% Note: lambda decreases. you take:
% sfrac*left+(1-sfrac*right)

if length(lambda)==1 % degenerate case of only one lambda
    nums=length(s);
    left=ones(nums,1);
    right=left;
    sfrac=ones(nums,1);
else
    s(s > max(lambda)) = max(lambda);
    s(s < min(lambda)) = min(lambda);
    k=length(lambda);
    sfrac =(lambda(1)-s)/(lambda(1) - lambda(k));
    lambda = (lambda(1) - lambda)/(lambda(1) - lambda(k));
    coord = interp1(lambda, 1:length(lambda), sfrac);
    left = floor(coord);
    right = ceil(coord);
    sfrac=(sfrac-lambda(right))./(lambda(left) - lambda(right));
    sfrac(left==right)=1;
end
result.left = left;
result.right = right;
result.frac = sfrac;

end


function result = softmax(x, gap)
if nargin < 2
    gap = false;
end
d = size(x);
maxdist = x(:, 1);
pclass = repmat(1, d(1), 1);
for i =2:d(2)
    l = x(:, i) > maxdist;
    pclass(l) = i;
    maxdist(l) = x(l, i);
end
if gap
    x = abs(maxdist - x);
    x(1:d(1), pclass) = x * repmat(1, d(2));
    gaps = pmin(x);
end
if gap
    result = {pclass, gaps};
else
    result = pclass;
end

end


function result = nonzeroCoef(beta,bystep)

if nargin < 2
    bystep = false;
end

result = abs(beta)>0;    
if ~bystep
    result = any(result,2);
end

end