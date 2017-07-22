function [ Risk ] = calculateRisk( F_train )
%CALCULATERISK Summary of this function goes here
%   Detailed explanation goes here
% 该方法用于计算期望风险系数。F_train是n*c矩阵，代表n个数据，每行代表该数据的概率分布
 % 结果返回每个数据的风险之和。
    [n,~]=size(F_train);
    Risk=0;
%     for i=1:n
%        temp=1-max(F_train(i,:));
%        Risk=Risk+temp;     
%     end
    %效率更高的解法
    F_trans=F_train';
    temp=1-max(F_trans);
    Risk=sum(temp);
    
end

