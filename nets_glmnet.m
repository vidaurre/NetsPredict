function fit = nets_glmnet(X, Y, family, screen, tmpnm, options)

if screen==1 && strcmp(family,'gaussian')
    fit = abs(Y'*zscore(X));
else
    if screen==1
        p = size(X,2);
        fit = zeros(p,1);
        for i=1:p
            if length(unique(Y))>1 && var(X(:,i))>0
                glmfit = glmnet(X(:,i), Y, family, options);
                fit(i) = glmfit.dev(end);
            else
                fit(i) = Inf;
            end
        end;        
    else
        fit = glmnet(X, Y, family, options);
    end
end
end