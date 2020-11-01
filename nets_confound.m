function Y = nets_confound(Y,conf,family,betaY) 

switch family
    case 'gaussian'
        conf = [ones(size(conf,1),1) conf];
        Y = Y + conf*betaY;
    case 'multinomial' % not yet ready
        if ~all(Y(:)==1 | Y(:)==0)
            Y = link_multinomial(Y) + nets_glmnetpredict(betaY,conf,betaY.lambda(end),'link');
            Y = inverselink_multinomial(Y);
        end
    case 'poisson' % not yet ready
        Y = link_poisson(Y) + nets_glmnetpredict(betaY,conf,betaY.lambda(end),'link');
        Y = inverselink_poisson(Y);
    case 'cox'
        error('Deconfounding not yet implemented for family=cox')
end

end





