  clear;
  %% load data
  load('dataSet\USPS\USPS_test.mat');%load the test data 
  load('dataSet\USPS\USPS_train.mat');
  %% the experiment is averaged over 20 sampled dataset for preventing the randomness
for index=1:20

    load(strcat('dataSet\USPS\label_10\USPS_',num2str(index)));
     
    % get the labeled data 
    Y_L=Y(sampleValue,:); 
    X_L=X(sampleValue,:);
    % get the unlabeled data
    Y_U=Y; 
    X_U=X;
    Y_U(sampleValue,:)=[];
    X_U(sampleValue,:)=[];
     disp(strcat('start to excute online Active---the dataset:',num2str(index)));
     
 %%
    % start 
    start_time = datestr(now,'yyyy-mm-dd_HH:MM:SS');   
    disp(['the time of start£º' start_time]);   
    
       Number_Iter=60;%the number of iterate
         [ accu_Local_U_E_max(index,:),accu_Local_T_E_max(index,:) ] = Func_MEB( X_L,Y_L,X_U,Y_U,X_T,Y_T,Number_Iter );
     [ accu_Local_U_BvSB_min(index,:),accu_Local_T_BvSB_min(index,:) ] = Func_BvSB( X_L,Y_L,X_U,Y_U,X_T,Y_T,Number_Iter );     
     [ accu_Local_U_R(index,:),accu_Local_T_R(index,:) ] = Func_random( X_L,Y_L,X_U,Y_U,X_T,Y_T,Number_Iter );
     [ accu_Global_U_E(index,:),accu_Global_T_E(index,:) ] = Func_MEGU( X_L,Y_L,X_U,Y_U,X_T,Y_T,Number_Iter );
     [ accu_Global_U_Risk(index,:),accu_Global_T_Risk(index,:) ] = Func_Risk( X_L,Y_L,X_U,Y_U,X_T,Y_T,Number_Iter );
     
     
     disp(['the time of end£º' datestr(now,'yyyy-mm-dd_HH:MM:SS:')]);  

 
end
