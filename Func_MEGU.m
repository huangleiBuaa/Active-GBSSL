function [ accu_F_U,accu_F_T ] = Func_MEGU( X_L,Y_L,X_U,Y_U,X_T,Y_T,Number_Iter )
%  X_L: the feature of the labeled data, which \in R^N*dim ,N is the number
%  of examples,and dim is dimension of the feature.

%  Y_L: the label of the labeled data, which in \in R^N*C
%  X_U: the feature of the unlabeled data (offline exmple pool)
%  Y_U: the label of the unlabeled data (for  evaluation)
%  X_T:  the feature of the unlabeled data (online exmple pool)
%  Y_T: the label of the unlabeled data (for  evaluation)

%  Number_Iter: the number of iteration.

    
%% 2.采用LGC算法来进行传播。计算初始的传播矩阵P_trans，预测值F_train,F_T以及其相应的归一化数值F_train_nom,F_T_nom;
    l=size(X_L,1);%标注训练集样例数
    u=size(X_U,1);%未标注数据训练集样例数
    t=size(X_T,1);%测试集样例数目
    C=size(Y_L,2);%类别数目
    n=l+u;
    
    X_train=[X_L;X_U];
    Y_train=[Y_L;Y_U];%训练集的goundTruth
    Set_train_L=1:l;%记录训练集中已被用户标注的数据在X_train中的Index，初始为标注集中的数据1:l；
    Set_test_L=[];%记录测试练集中已被用户标注的数据在X_test中的Index.初始为空
    Theta=144000;
    Alpha=0.89;
    
    disp('start to excute LGC...');
    W=getAffinityMatrix_diag_0(X_train,Theta);%构造affinity矩阵
    sumW=sum(W);
    isD=diag(sumW.^(-1/2));
    S=isD*W*isD;
    P_trans=(1-Alpha)*inv((eye(n)-Alpha*S));%构造传播矩阵
% 计算F_train    
    Y_unlabel=zeros(u,C);%将所有unlabeled的数据Y补上0；
    Y_0=[Y_L;Y_unlabel];
    F_train=P_trans*Y_0;% 训练集中利用LGC算法后得到的预测值

  % 对训练集中的未标注数据进行评估
    accu_F_U(1)=evaluate_accuracy(Y_train,F_train,Set_train_L);
     disp(strcat('the intional accuracy on training set is: ',num2str(accu_F_U(1))));
  % 计算F_test,并对其进行评估
     F_train_nom=F_train./repmat(sum(F_train,2),[1 size(F_train,2)]);%对预测值进行归一化
     
     DIST=distMat(X_T,X_train); 
     W_test_train=exp(-DIST/Theta);%构造Test集与训练集的权重矩阵，用于计算测试集的预测值。
    
     F_test = evaluate_testSet_new( W_test_train,Y_train,F_train_nom, Set_train_L);
     accu_F_T(1)=evaluate_accuracy(Y_T,F_test,Set_test_L);
      
     disp(strcat('the intional accuracy on test set is: ',num2str(accu_F_T(1))));
      
   % 清除不必要的变量
   clear W sumW isD S Y_unlabel Y_0 X_L Y_L X_U Y_U;
   % 在下一轮需要用的变量有 
   % 1.必要的数据集
   % X_train
   % Y_train 训练集的ground truth 在评估时要用的
   % X_T
   % Y_T
   % 2. 标识数据集的变量 
   % Set_train_L 记录训练集标注数据在训练集中的索引号
   % Set_test_L 记录测试集中被选中的标注数据在测试集中的索引号
   % 3.中间变量，可以减少计算以及每轮迭代后需要改变的量
   % P_trans 传播矩阵，用于增量更新时
   % F_train 对于训练集的预测值，不是概率形式，故有必要时对其求F_train_nom概率形式
   % F_test 对于测试集的预测值，是概率形式
   % W_test_train 测试集数据与训练集数据的权重矩阵，主要用于求测试集的预测值
   %
        
   
   %% 2 开始主动在线学习
    
  %  Number_Iter=10;
    for k=1:Number_Iter
        tic;
        disp(strcat('the Iterate: ',num2str(k)));
       
        %计算训练集中未标注数据提供给用户标注获得标签的条件期望熵。
        Set_train_U=setdiff(1:n,Set_train_L);
        u_train=length(Set_train_U);
        H_U_exp=zeros(1,u_train);%记录未标注数据的期望熵
        
        for i=1:u_train
          % Hx_exp=0;
           index=Set_train_U(i);
           % 计算X_U(index)的条件熵，由于X_U(index)的y值无法确定，只能采用求取经验条件熵的方式
           for j=1:C
               % 计算当将X_U(i)及其假定给的y值=j给定后，加入标注数据集，重新传播所得到的预测值F_train_plus
               F_train_plus=F_train;%注意这儿的跟新值用原始的未归一化数据，为了保证LGC算法的一致性
               F_train_plus(:,j)=F_train(:,j)+P_trans(:,index);%根据增量算法的线性法则，只需要更新第j列，值为P_trans的第i列
               F_train_plus_nom=F_train_plus./repmat(sum(F_train_plus,2),[1 size(F_train_plus,2)]);%对更新的预测值进行归一化
              % 考虑对X_train中未标注数据的条件熵
                 Hx_exp_j_train=calculateEntropy(F_train_plus_nom);
               % 考虑对X_test中未标注数据的条件熵 
 %                F_test_plus=evaluate_testSet_new( W_test_train,Y_train,F_train_plus_nom, Set_train_L);
 %               Hx_exp_j_test=calculateEntropy(F_test_plus);
                 Hx_exp_j=Hx_exp_j_train;
                % 对每一可能的类迭代计算H_U_exp；
                 H_U_exp(i)=H_U_exp(i)+F_train_nom(index,j)*Hx_exp_j;    
           
           end
        
        end
        
   
         %计算测试集中未标注数据提供给用户标注获得标签的条件期望熵。
       
         Set_test_U=setdiff(1:t,Set_test_L);
         u_test=length(Set_test_U);
         H_T_exp=zeros(1,u_test);%记录未标注数据的期望熵
          for i=1:u_test

              index=Set_test_U(i);
               x=X_T(index,:);%找到该测试数据的特征
              % 寻找其在训练集中的代理，并记录权重参数
              K_UL=5;%代理点的数目；
              
              DIST=distMat(X_train,x);%计算x与X_train之间的距离
              [~, IDX] = sort(DIST, 1);%排序
              i_KUL=IDX(1:K_UL);%选择距离最近的K_UL个点作为其代理点；
              W_ul=exp(-DIST(i_KUL)/Theta);%计算其权重；
              W_ul_nom=W_ul./sum(W_ul);%计算其归一化权重；


                % 计算x的条件熵，由于x的y值无法确定，只能采用求取经验条件熵的方式
               for j=1:C
               % 计算当将x及其假定给的y值=j给定后，加入标注数据集，重新传播所得到的预测值F_train_plus
                 F_train_plus=F_train;%注意这儿的跟新值用原始的未归一化数据，为了保证LGC算法的一致性
                 
                 for ii=1:K_UL
                    F_train_plus(:,j)=F_train(:,j)+W_ul_nom(ii)*P_trans(:,i_KUL(ii));%根据增量算法的线性法则，只需要更新第j列，值为P_trans的第i列
                 end
                 F_train_plus_nom=F_train_plus./repmat(sum(F_train_plus,2),[1 size(F_train_plus,2)]);%对更新的预测值进行归一化
              % 考虑对X_train中未标注数据的条件熵
                 Hx_exp_j_train=calculateEntropy(F_train_plus_nom);
               % 考虑对X_test中未标注数据的条件熵 
%                  F_test_plus=evaluate_testSet_new( W_test_train,Y_train,F_train_plus_nom, Set_train_L);
%                  Hx_exp_j_test=calculateEntropy(F_test_plus);
                 Hx_exp_j=Hx_exp_j_train;
                % 对每一可能的类迭代计算H_T_exp；
                 H_T_exp(i)=H_T_exp(i)+F_test(index,j)*Hx_exp_j;       
               end
              
          end
          [min_H_U,index_U]=min(H_U_exp);
          [min_H_T,index_T]=min(H_T_exp);
          
          % 选择数据给人标注，获得标签后加入到标注训练集，更新下一轮数据集
        
          if(min_H_T<min_H_U)
             % 从测试集中加入到标注训练集中
              example=Set_test_U(index_T);
              Set_test_L=[Set_test_L,example];%标识测试数据集中的已标注数据
             
              % 更新预测值；
              x=X_T(example,:);%找到该测试数据的特征
               DIST=distMat(X_train,x);%计算x与X_train之间的距离
              [~, IDX] = sort(DIST, 1);%排序
              i_KUL=IDX(1:K_UL);%选择距离最近的K_UL个点作为其代理点；
              W_ul=exp(-DIST(i_KUL)/Theta);%计算其权重；
              W_ul_nom=W_ul./sum(W_ul);%计算其归一化权重；
              
              y=Y_T(example,:);
              j=find(y==1);
              disp(strcat('the active label data is from test set: ',num2str(example),'--the class is:',num2str(j)));
              for ii=1:K_UL
                    F_train(:,j)=F_train(:,j)+W_ul_nom(ii)*P_trans(:,i_KUL(ii));%根据增量算法的线性法则，只需要更新第j列，值为P_trans的第i列
               end
              F_train_nom=F_train./repmat(sum(F_train,2),[1 size(F_train,2)]);%对更新的预测值进行归一化
              F_test = evaluate_testSet_new( W_test_train,Y_train,F_train_nom, Set_train_L); 
              % 在训练集和测试集上分别评估检验。
               accu_F_U(k+1)=evaluate_accuracy(Y_train,F_train,Set_train_L);
               accu_F_T(k+1)=evaluate_accuracy(Y_T,F_test,Set_test_L);
               disp(strcat('the accuracy on training data: ',num2str(accu_F_U(k+1)),...
                 '--the accuracy on test data:',num2str(accu_F_T(k+1))));
          else
             example=Set_train_U(index_U);
             Set_train_L=[Set_train_L,example];%标识训练集中的已标注数据
             
       
             y=Y_train(example,:);
             j=find(y==1);
             disp(strcat('the active label data is from training set: ',num2str(example),'--the class is:',num2str(j)));
             F_train(:,j)=F_train(:,j)+P_trans(:,example);%根据增量算法的线性法则，只需要更新第j列，值为P_trans的第i列
             F_train_nom=F_train./repmat(sum(F_train,2),[1 size(F_train,2)]);%对更新的预测值进行归一化
             F_test = evaluate_testSet_new( W_test_train,Y_train,F_train_nom, Set_train_L);
  
             % 在训练集和测试集上分别评估检验。
             accu_F_U(k+1)=evaluate_accuracy(Y_train,F_train,Set_train_L);
             accu_F_T(k+1)=evaluate_accuracy(Y_T,F_test,Set_test_L);
             disp(strcat('the accuracy on training data: ',num2str(accu_F_U(k+1)),...
                 '--the accuracy on test data:',num2str(accu_F_T(k+1))));
          
          end
          
          toc;
    end
    
end

