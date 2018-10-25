function y = link_multinomial(x)
q = size(x,2);
arePr = sum( (x(:)==0) | (x(:)==1) ) == length(x(:));
y = zeros(size(x));
y(x==1) = 1; y(x==0) = -Inf; 
y(arePr,:) = log(x(arePr,:)); 
%expy = x(arePr,:);
% while true
% for j=1:q
%    y(arePr,j) = log ( x(arePr,j) .* sum(expy(:,setdiff(1:q,j)),2) ./ (1 - x(arePr,j))  );
%    expy(arePr,j) = exp(y(arePr,j));
% end
% xhat = exp(y(arePr,:)) ./ repmat(sum(exp(y(arePr,:)),2),1,q);
% d = (x(arePr,:) - xhat).^2;
% if sum(d(:))<1e-10,
%     break; 
% end
% end
y(arePr,:) = y(arePr,:) - repmat(mean(y(arePr,:),2),1,q);
end