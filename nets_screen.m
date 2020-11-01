function fit = nets_screen(X, Y, family)

q = size(Y,2);

if strcmpi(family,'gaussian') || (strcmpi(family,'multinomial') && q==1)
    fit = abs(Y'*zscore(X));
else  
    p = size(X,2);
    fit = zeros(p,1);
    for i = 1:p
        if strcmpi(family,'multinomial') && q>2
            [~,dev] = mnrfit(X,Y);
            fit(i) = -dev;
        elseif strcmpi(family,'poisson')
            [~,dev] = glmfit(X,Y,'poisson');
            fit(i) = -dev;     
        elseif strcmpi(family,'cox')
            error('Screening not yet implemented for Cox model')
        end
    end
end
