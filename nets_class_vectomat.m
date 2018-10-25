% goes from a Nx1 vector of q different classes (Y) 
% to a Nxq matrix of dummy variables encoding the same 
function Ym = nets_class_vectomat(Y,classes)
N = length(Y); 
if nargin<2, classes = unique(Y); end
q = length(classes);
Ym = zeros(N,q);
for j=1:q, Ym(Y==classes(j),j) = 1; end 
