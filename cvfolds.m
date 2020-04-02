function folds = cvfolds(Y,family,CVscheme,allcs)

if nargin<4, allcs = []; end

[N,q] = size(Y);
if ~strcmp(family,'multinomial'), 
    q = length(unique(Y)); 
    if q<=3,
       Y = nets_class_vectomat(Y);  
    end
elseif strcmp(family,'multinomial') && q==1,
    Y = nets_class_vectomat(Y); q = size(Y,2);
end

if CVscheme==0, nfolds = N;
else nfolds = CVscheme;
end
folds = {}; ifold = 1;
grotDONE = zeros(N,1);
if strcmp(family,'multinomial') || q<=3   % stratified CV that respects the family structure
    counts = zeros(nfolds,q); Scounts = mean(Y); 
    for k = 1:nfolds
        if sum(grotDONE)==N, break; end
        folds{ifold} = [];
        while length(folds{ifold}) < ceil(N/nfolds)
            d = Inf(N,1); j=1;
            % find the family that best preserves the class proportions
            grotDONEI = grotDONE;
            while j<=N
                if (grotDONEI(j)==0)
                    Jj=[folds{ifold} j];
                    if (~isempty(allcs))  % leave out all samples related to the one in question
                        if size(find(allcs(:,1)==j),1)>0, Jj=[Jj allcs(allcs(:,1)==j,2)']; end
                    end
                    if length(Jj)>1, countsI = sum(Y(Jj,:)); % before: counts(LOfracI,:) + sum(Y(Jj,:));
                    else countsI = Y(Jj,:); % before: counts(LOfracI,:) + Y(Jj,:);
                    end
                    countsI = countsI / sum(countsI);
                    d(j) = sum( ( Scounts - countsI ).^2 ); % distance from the overall class proportions
                    grotDONEI(Jj) = 1;
                end
                j=j+1;
            end
            % and assign it to this fold
            [~,j] = min(d); j = j(1); 
            folds{ifold}=[folds{ifold} j];
            if (~isempty(allcs))  % leave out all samples related (according to cs) to the one in question
                if size(find(allcs(:,1)==j),1)>0, folds{ifold}=[folds{ifold} allcs(allcs(:,1)==j,2)']; end
            end
            grotDONE(folds{ifold})=1; counts(k,:) = sum(Y(folds{ifold},:));
            if k>1 && k<nfolds,
                if sum(grotDONE)>k*N/nfolds, break; end
            end
        end
        if ~isempty(folds{ifold}), ifold = ifold + 1; end
    end
else % standard CV respecting the family structure
    for k = 1:nfolds
        if sum(grotDONE)==N, break; end
        j=1;  folds{ifold} = [];
        while length(folds{ifold}) < ceil(N/nfolds) && j<=N
            if (grotDONE(j)==0)
                folds{ifold}=[folds{ifold} j];
                if (~isempty(allcs))  % leave out all samples related to the one in question
                    if size(find(allcs(:,1)==j),1)>0
                        folds{ifold}=[folds{ifold} allcs(allcs(:,1)==j,2)'];
                    end
                end
                grotDONE(folds{ifold})=1;
            end
            j=j+1;
            if k>1 && k<nfolds,
                if sum(grotDONE)>k*N/nfolds
                    break
                end
            end
        end
        if ~isempty(folds{ifold}), ifold = ifold + 1; end
    end
end


end
