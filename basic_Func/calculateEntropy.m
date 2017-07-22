function [ H ] = calculateEntropy( F_train )
%%CALCULATEENTROPY Summary of this function goes here
%   Detailed explanation goes here
%调用前保证F_train为归一的概率值。

% 输入F是一个N*C的矩阵，其中每行为每一个数据属于各类的概率值，其每行求和为1（可以放宽限制，但必须个元素大于0）
 
%   对F_train各行进行归一化处理，以使其为概率值；
%   F=F_train./repmat(sum(F_train,2),[1 size(F_train,2)]);%对更新的预测值进行归一化
%  [N,C]=size(F);
    epsilo=1e-300;%为了防止出现0*log0的情况；


    F_ep=F_train+epsilo;
    temp=F_ep.*log2(F_ep);
    H=-sum(sum(temp));
end

