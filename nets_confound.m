function [Y] = nets_confound(Y,conf,family,betaY,my) 

switch family
    case 'gaussian'
        Y=Y+conf*betaY+my;
    case 'multinomial'
        if ~all(Y(:)==1 | Y(:)==0)
            Y = link_multinomial(Y) + nets_glmnetpredict(betaY,conf,betaY.lambda(end),'link');
            Y = inverselink_multinomial(Y);
        end
    case 'poisson'
        Y = link_poisson(Y) + nets_glmnetpredict(betaY,conf,betaY.lambda(end),'link');
        Y = inverselink_poisson(Y);
    case 'cox'
        error('Deconfounding not yet implemented for family=cox')
end

end





