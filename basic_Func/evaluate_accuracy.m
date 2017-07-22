function [ accuracy ] = evaluate_accuracy( Y_train,F_train,set_L)
%EVALUATE_ACCURACY Summary of this function goes here
%   Detailed explanation goes here
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Y is groudtruch
%F is result(the confidence)
% set_L 为这些数据中已经标注了的数据，假定这批数据不参与评估
%result is the result(0,1 matrix)
Y=Y_train;
F=F_train;
Y(set_L,:)=[];
F(set_L,:)=[];

[lenU,C]=size(F);
[sorted,index]=sort(F,2,'descend');
max=index(:,1);
result=zeros(lenU,C);
for i=1: lenU
   result(i,max(i))=1; 
end
correct=0;
list=zeros(lenU,1);
for i=1:lenU
   if(Y(i,:)==result(i,:))
      correct=correct+1; 
      list(i)=1;
   end  
end
accuracy=correct/lenU;

end

