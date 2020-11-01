b =  rand(4,1); b2 = 0.5 * rand(2,1); 
X = randn(100,4); C = randn(100,2); 
y = X * b + C * b2 + 0.25 * randn(100,1);
y2 = zeros(100,3); l1 = prctile(y,33); l2 = prctile(y,66); % 3 classes
for j = 1:100
    if y(j)<l1, y2(j,1) = 1; elseif y(j)<l2, y2(j,2) = 1; else, y2(j,3) = 1; end
end
y2b = y > 0; % 2 classes
y4 = min(floor(exp(y)),4); % poisson-like

options1 = struct('CVscheme',[2 2]); options1.Method = 'glmnet';
options2 = struct('CVscheme',[2 2]); options2.Method = 'lasso';
options3 = struct('CVscheme',[2 2]); options3.Method = 'ridge';
options4 = struct('CVscheme',[2 2]); options4.Method = 'unregularized';

s = nets_predict5(y,X,'gaussian',options1,[],[],C); disp(num2str(s.cod) )
s = nets_predict5(y2b,X,'multinomial',options1); disp([num2str(s.accuracy) ' ' num2str(s.baseline_accuracy) ]) 
s = nets_predict5(y2,X,'multinomial',options1); disp([num2str(s.accuracy) ' ' num2str(s.baseline_accuracy) ]) 
s = nets_predict5(y4,X,'poisson',options1); disp([num2str(s.dev) ' ' num2str(s.baseline_dev) ]) 

s = nets_predict5(y,X,'gaussian',options2,[],[],C); disp(num2str(s.cod))
s = nets_predict5(y2b,X,'multinomial',options2); disp([num2str(s.accuracy) ' ' num2str(s.baseline_accuracy) ]) 
s = nets_predict5(y4,X,'poisson',options2); disp([num2str(s.dev) ' ' num2str(s.baseline_dev) ])

s = nets_predict5(y,X,'gaussian',options3,[],[],C); disp(num2str(s.cod))

s = nets_predict5(y,X,'gaussian',options4,[],[],C); disp(num2str(s.cod))
s = nets_predict5(y2,X,'multinomial',options4); disp([num2str(s.accuracy) ' ' num2str(s.baseline_accuracy) ]) 

%%

options1 = struct('CVscheme',4); options1.Method = 'glmnet';
options2 = struct('CVscheme',4); options2.Method = 'lasso';
options3 = struct('CVscheme',4); options3.Method = 'ridge';
options4 = struct('CVscheme',4); options4.Method = 'unregularized';

[est,estdec] = nets_train(y,X,'gaussian',options1,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y) ) )
[est,estdec] = nets_train(y2b,X,'multinomial',options1,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y>0) ) )
[est,estdec] = nets_train(y2,X,'multinomial',options1,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(mean(sum(y2 .* yhat,2))))
[est,estdec] = nets_train(y4,X,'poisson',options1,[],C); 

[est,estdec] = nets_train(y,X,'gaussian',options2,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y) ) )
[est,estdec] = nets_train(y2b,X,'multinomial',options2,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y>0) ) )
[est,estdec] = nets_train(y4,X,'poisson',options2,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y>0) ) )

[est,estdec] = nets_train(y,X,'gaussian',options3,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y) ) )

[est,estdec] = nets_train(y,X,'gaussian',options4,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y) ) )
[est,estdec] = nets_train(y2b,X,'multinomial',options4,[],C);
yhat = nets_test(X,est,estdec,C);  disp(num2str(corr(yhat,y>0) ) )
[est,estdec] = nets_train(y2,X,'multinomial',options4,[],C); 
yhat = nets_test(X,est,estdec,C);  disp(num2str(mean(sum(y2 .* yhat,2))))

