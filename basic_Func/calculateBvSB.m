function [ value ] = calculateBvSB( f )
%CALCULATEBVSB Summary of this function goes here
%   Detailed explanation goes here
%   输入f是一个C元向量，C的值为类别数
%   输出value为f的前两个最大值得差值。
    [v,~]=sort(f,'descend');
    value=v(1)-v(2);
end

