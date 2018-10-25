% goes from a Nxq matrix of dummy variables encoding the classes
% to a Nx1 vector of q different classes (Y) 
function Y = nets_class_mattovec(Ym,classes)
if nargin<2, q = size(Ym,2); classes = 1:q; 
else q = length(classes); 
end
Y = zeros(size(Ym,1),1);
for j=1:q, Y(Ym(:,j)>.5) = classes(j); end

