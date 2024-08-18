clear all
clc
close all

number_iteration=100;
load mnist_example_data
%%
CL=length(unique(LTra1));
input1.y=full(ind2vec(LTra1',CL)');
input1.data=DTra1;
input1.cl=CL;
input1.wo=6*CL;
input1.lr=1;
input1.cr=1/2;
input1.numi=number_iteration;
tic
output1=DFNNADC2(input1,'learning');
tt1=toc;
%%
input2=output1;
input2.data=DTes1;
tic
output2=DFNNADC2(input2,'testing');
tt2=toc;
Ye=output2.predy;
CM=confusionmat(LTes1,Ye);
L=length(CM(:,1));
Acc=sum(CM.*eye(L),'all')/sum(CM,'all')
BAcc=mean(diag(CM.*eye(L))./sum(CM,2))
