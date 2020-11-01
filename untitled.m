Y = floor(4*rand(1200,1))+1;
group = []; for j = 1:10, group = [group; j*ones(100,1)]; end
group = [group; zeros(200,1)];
folds = cvfolds(Y,'multinomial',10,group);