function [ accu_F_U,accu_F_T ] = Func_BvSB( X_L,Y_L,X_U,Y_U,X_T,Y_T,Number_Iter )

%  X_L: the feature of the labeled data, which \in R^N*dim ,N is the number
%  of examples,and dim is dimension of the feature.

%  Y_L: the label of the labeled data, which in \in R^N*C
%  X_U: the feature of the unlabeled data (offline exmple pool)
%  Y_U: the label of the unlabeled data (for  evaluation)
%  X_T:  the feature of the unlabeled data (online exmple pool)
%  Y_T: the label of the unlabeled data (for  evaluation)

%  Number_Iter: the number of iteration.

       l=size(X_L,1);%标注训练集样例数
    u=size(X_U,1);%未标注数据训练集样例数
    t=size(X_T,1);%测试集样例数目
    C=size(Y_L,2);%类别数目
    n=l+u;
    
    X_train=[X_L;X_U];
    Y_train=[Y_L;Y_U];%训练集的goundTruth
    Set_train_L=1:l;%记录训练集中已被用户标注的数据在X_train中的Index，初始为标注集中的数据1:l；
    Set_test_L=[];%记录测试练集中已被用户标注的数据在X_test中的Index.初始为空
    Theta=144400;
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
   %% 开始主动在线学习
     epsilo=1e-300;%为了防止出现0*log0的情况；

    for k=1:Number_Iter
     %   Hx_min=1e300;
        K_UL=5;
        disp(strcat('the Iterate: ',num2str(k)));
        %随机选择一个点；
        Set_train_U=setdiff(1:n,Set_train_L);
        Set_test_U=setdiff(1:t,Set_test_L);
        u_train=length(Set_train_U);
        u_test=length(Set_test_U);
        
        Hx_min_train=1e300;
        for i=1:u_train
            F_i=F_train_nom(i,:)+epsilo;
            H_i= calculateBvSB( F_i);%计算信心值最高的两类的差
            if (H_i<Hx_min_train)
               Hx_min_train=H_i;
               index_train=i;
            end
        end
     
       Hx_min_test=1e300;
        for i=1:u_test
            F_i=F_test(i,:)+epsilo;
            H_i= calculateBvSB( F_i);%计算信心值最高的两类的差
            if (H_i<Hx_min_test)
               Hx_min_test=H_i;
               index_test=i;
            end
        end 
          % 选择数据给人标注，获得标签后加入到标注训练集，更新下一轮数据集
          if(Hx_min_test<Hx_min_train)
             % 从测试集中加入到标注训练集中
              example=Set_test_U(index_test);
              Set_test_L=[Set_test_L,example];%标识测试数据集中的已标注数据
              disp(strcat('the active label data is from test set: ',num2str(example)));
              % 更新预测值；
              x=X_T(example,:);%找到该测试数据的特征
               DIST=distMat(X_train,x);%计算x与X_train之间的距离
              [~, IDX] = sort(DIST, 1);%排序
              i_KUL=IDX(1:K_UL);%选择距离最近的K_UL个点作为其代理点；
              W_ul=exp(-DIST(i_KUL)/Theta);%计算其权重；
              W_ul_nom=W_ul./sum(W_ul);%计算其归一化权重；
              
              y=Y_T(example,:);
              j=find(y==1);
              
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
              % 从训练集中选点
             example=Set_train_U(index_train);
             Set_train_L=[Set_train_L,example];%标识训练集中的已标注数据
             
             disp(strcat('the active label data is from training set: ',num2str(example)));
             y=Y_train(example,:);
             j=find(y==1);
             F_train(:,j)=F_train(:,j)+P_trans(:,example);%根据增量算法的线性法则，只需要更新第j列，值为P_trans的第i列
             F_train_nom=F_train./repmat(sum(F_train,2),[1 size(F_train,2)]);%对更新的预测值进行归一化
                    % 在训练集和测试集上分别评估检验。
               F_test = evaluate_testSet_new( W_test_train,Y_train,F_train_nom, Set_train_L); 
              % 在训练集和测试集上分别评估检验。
               accu_F_U(k+1)=evaluate_accuracy(Y_train,F_train,Set_train_L);
               accu_F_T(k+1)=evaluate_accuracy(Y_T,F_test,Set_test_L);
             disp(strcat('the accuracy on training data: ',num2str(accu_F_U(k+1)),...
                 '--the accuracy on test data:',num2str(accu_F_T(k+1))));
          
          end
          
    end
    
    
    


end

