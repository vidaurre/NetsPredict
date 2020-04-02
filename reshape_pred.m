function Xin = reshape_pred(X,matrix_format,keepvar)
% Reshape predictors to have covariance format (matrix_format==1) 
% or vector format (matrix_format==0). 

%
% Usage:if matrix_format==1 and X is given vectorized,  
%       keepvar==1 means that X contains the variance; 
%       if matrix_format==0 and X is given in matrices format, 
%       keepvar dictates if we keep the diagonal or not
%       if matrix_format==0 and X is given vectorized,  
%       keepvar==0 means that X contains the variance and we want to remove
%       it; in this case, the variances are assumed to be ==1
%       
%       
% Diego Vidaurre, University of Oxford (2015)


N = size(X,1);
if matrix_format==0 && length(size(X))==3 % just vectorize the matrices
    Nnodes = size(X,2); 
    if keepvar
        Xin = zeros(N, Nnodes * (Nnodes+1) / 2);
    else
        Xin = zeros(N, Nnodes * (Nnodes-1) / 2);
    end
    for j=1:N
        grot = permute(X(j,:,:),[2 3 1]);
        Xin(j,:) = grot(triu(ones(Nnodes),~keepvar)==1);
    end
elseif matrix_format==1 && length(size(X))==2 % put in matrix format, and do riemann transform
    if keepvar==0
        Nnodes = (1 + sqrt(1+8*size(X,2))) / 2;
    else
        Nnodes = (-1 + sqrt(1+8*size(X,2))) / 2;
    end
    Xin = zeros(N, Nnodes, Nnodes);
    for j=1:N
        Xin(j,triu(ones(Nnodes),~keepvar)==1) = X(j,:);
        grot = permute(Xin(j,:,:),[2 3 1]);
        if keepvar
            Xin(j,:,:) = grot + grot' - diag(diag(grot));
        else
            Xin(j,:,:) = grot + grot' + eye(Nnodes);
        end
    end
elseif matrix_format==0 && length(size(X))==2 && keepvar==0 % remove the diagonal
    Nnodes = (-1 + sqrt(1+8*size(X,2))) / 2;
    Xin = zeros(N, Nnodes, Nnodes);
    for j=1:N
        ind1 = triu(ones(Nnodes),1)==1;
        ind2 = all(abs(X-1)>eps);
        Xin(j,ind1) = X(j,ind2);
        grot = permute(Xin(j,:,:),[2 3 1]);
        Xin(j,:,:) = grot + grot' + eye(Nnodes);
    end
    Xin = reshape_pred(Xin,0,0);
else
    Xin = X;
end

end