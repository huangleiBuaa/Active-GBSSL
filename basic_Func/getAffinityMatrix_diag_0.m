function [ W ] = getAffinityMatrix_diag_0( X,Theta )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

% X is the data Matrix,X is N*d,N is the number ,d is the feature dimension 
% Theta is a Parament ,base Gaussian kerner

%W is the AffinityMatrix,其diag全为0；

    [N,dim]=size(X);

    X2=sum(X.^2,2);
    distance=repmat(X2,1,N)+repmat(X2',N,1)-2*X*X';
    W=exp(-distance/Theta);
    %将其diag元素置0，其他元素值不变
    for i=1:N
       W(i,i)=0; 
    end

end



