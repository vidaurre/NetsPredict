function y = inverselink_multinomial(x)
y = exp(x) ./ repmat(sum(exp(x),2),1,size(x,2));
end