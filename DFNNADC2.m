function [output]=DFNNADC2(input,mode)
if  strcmp(mode,'learning')==1
    CL=input.cl;
    Wo=input.wo;
    NI=input.numi;
    data=input.data;
    y=input.y;
    N=size(data,1);
    Layer2.Sys=[];
    Layer1.Sys=[];
    Layer2.lr=input.lr;
    Layer1.lr=input.lr;
    Layer1.CR=input.cr;
    Layer2.CR=input.cr;
    Grad=zeros(size(y));
    GRAD=zeros(NI,1);
    for tt=1:1:NI
        seq=randperm(N);
        data=data(seq,:);
        y=y(seq,:);
        for kk=1:1:N
            Layer1.data=data(kk,:);Layer1.YS=Wo;
            [Layer1]=EFSTra_Forward(Layer1);
            Layer2.data=Layer1.Ye;Layer2.YS=CL;
            [Layer2]=EFSTra_Forward(Layer2);
            Grad(kk,:)=(y(kk,:)-Layer2.Ye);
            Layer2.data=Layer1.Ye;Layer2.Grad=Grad(kk,:);Layer2.RG=1;
            [Layer2]=EFSTra_Backward(Layer2);
            Layer1.data=data(kk,:);Layer1.Grad=Layer2.Grad;Layer1.RG=0;
            [Layer1]=EFSTra_Backward(Layer1);
        end
        GRAD(tt)=sum(mean(abs(Grad),1));
    end
    output.Layer1=Layer1;
    output.Layer2=Layer2;
end
if  strcmp(mode,'updating')==1
    CL=input.cl;
    Wo=input.wo;
    NI=input.numi;
    N0=input.pnumi;
    data=input.data;
    y=input.y;
    N=size(data,1);
    Layer1=input.Layer1;
    Layer2=input.Layer2;
    Grad=zeros(size(y));
    GRAD=zeros(NI,1);
    for tt=1:1:N0
        seq=randperm(N);
        data=data(seq,:);
        y=y(seq,:);
    end
    for tt=N0+1:1:NI+N0
        seq=randperm(N);
        data=data(seq,:);
        y=y(seq,:);
        for kk=1:1:N
            Layer1.data=data(kk,:);Layer1.YS=Wo;
            [Layer1]=EFSTra_Forward(Layer1);
            Layer2.data=Layer1.Ye;Layer2.YS=CL;
            [Layer2]=EFSTra_Forward(Layer2);
            Grad(kk,:)=(y(kk,:)-Layer2.Ye);
            Layer2.data=Layer1.Ye;Layer2.Grad=Grad(kk,:);Layer2.RG=1;
            [Layer2]=EFSTra_Backward(Layer2);
            Layer1.data=data(kk,:);Layer1.Grad=Layer2.Grad;Layer1.RG=0;
            [Layer1]=EFSTra_Backward(Layer1);
        end
        GRAD(tt-N0)=sum(mean(abs(Grad),1));
    end
    output.Layer1=Layer1;
    output.Layer2=Layer2;
end
if strcmp(mode,'testing')==1
    data1=input.data;
    Layer1=input.Layer1;
    Layer2=input.Layer2;
    N1=size(data1,1);
    Pred=zeros(N1,size(Layer2.Sys.A,1));
    for kk=1:1:N1
        Layer1.data=data1(kk,:);
        [Output1]=EFSTes(Layer1);
        Layer2.data=Output1.Ye;
        [Output2]=EFSTes(Layer2);
        %%
        Pred(kk,:)=Output2.Ye;
    end
    [~,predy]=max(Pred,[],2);
    output.predy=predy;
    output.soc=Pred;
end
end
function [y]=ActivationFunction(x,type,mode)
if strcmp(type,'sig')
    if strcmp(mode,'a')==1
        y=1./(1+exp(-1*x));
    end
    if strcmp(mode,'d')==1
        y=x.*(1-x);
    end
end
if strcmp(type,'relu')
    if strcmp(mode,'a')==1
        y=x;
        y(x<=0)=0;
    end
    if strcmp(mode,'d')==1
        y=zeros(size(x));
        y(x>0)=1;
    end
end
if strcmp(type,'tanh')
    if strcmp(mode,'a')==1
        y=tanh(x);
    end
    if strcmp(mode,'d')==1
        y=1-x.^2;
    end
end
end
function [centerlambda0,LocalDensity0]=ActivatingRules(ModelNumber,LocalDensity,threshold2)
[values,seq]=sort(LocalDensity,'descend');
LocalDensity0=zeros(ModelNumber,1);
centerlambda0=zeros(ModelNumber,1);
values=sum(triu(repmat(values,1,ModelNumber)),1);
a=find(values>=threshold2*sum(LocalDensity));
seq1=seq(1:1:a(1))';
LocalDensity0(seq1)=LocalDensity(seq1);
centerlambda0(seq1)=LocalDensity(seq1)./sum(LocalDensity(seq1));
end
function [Output]=EFSTes(Input)
datain=Input.data;
Output.Sys=Input.Sys;
L1=Input.Sys.L;
type=Input.Sys.type;
prototypes=Input.Sys.prototypes;
local_delta=Input.Sys.local_delta;
Global_mean=Input.Sys.Global_mean;
Global_X=Input.Sys.Global_X;
ModelNumber=Input.Sys.ModelNumber;
A=Input.Sys.A;
W=Input.Sys.W;
CM=Input.Sys.CM;
CL=Input.Sys.CL;
Global_mean1=Global_mean.*L1./(L1+1)+datain./(L1+1);
Global_X1=Global_X.*L1./(L1+1)+datain.^2./(L1+1);
Global_Delta1=abs(Global_X1-Global_mean1.^2);
[~,LocalDensity,~]=firingstrength(datain,ModelNumber,prototypes,local_delta,Global_Delta1,W);
[centerlambda,LocalDensity]=ActivatingRules(ModelNumber,LocalDensity,Input.Sys.threshold2);
Cdata=zeros(ModelNumber,Input.Sys.CW);
for tt=find(centerlambda'~=0)
    Cdata(tt,:)=ActivationFunction(datain*CM(:,:,tt),type,'a');
end
[Ye1,~]=OutputGeneration(Cdata,A,centerlambda,ModelNumber,CL,type);
Output.Ye=Ye1;
%%
end
function [Output]=EFSTra_Backward(Input)
datain=Input.data;
datain1=Input.compsdata;
Grad=Input.Grad;
Output.Sys=Input.Sys;
centerlambda=Input.Sys.centerlambda;
LocalDensity=Input.Sys.LocalDensity;
prototypes=Input.Sys.prototypes;
Global_Delta1=Input.Sys.Global_Delta1;
W=Input.Sys.W;
YeL=Input.Sys.YeL;
lr=Input.lr;
A=Input.Sys.A;
CM=Input.Sys.CM;
type=Input.Sys.type;
%%
Derive=ActivationFunction(YeL,type,'d');
seq=find(centerlambda~=0);
datain2=ActivationFunction(datain1,type,'d');
Xgrad=zeros(1,W);
%%
if Input.RG==1
    D1=sum(LocalDensity);
    C0=2*(prototypes-datain)./repmat(Global_Delta1,1,W);
    C0(isnan(C0))=0;
    C00=LocalDensity'*C0;
    C1=repmat(centerlambda,1,W).*C0-centerlambda/D1*C00;
    for jj=seq'
        temGD=Grad.*Derive(jj,:);
        temGA=temGD*A(:,2:end,jj);
        Xgrad=Xgrad+(centerlambda(jj)*temGA*(CM(:,:,jj).*repmat(datain2(jj,:),W,1))'+Grad*(YeL(jj,:)'*C1(jj,:)));
        A(:,:,jj)=A(:,:,jj)+lr*centerlambda(jj)*temGD'*[1,datain1(jj,:)];
        CM(:,:,jj)=CM(:,:,jj)+lr*centerlambda(jj)*datain'*(datain2(jj,:).*temGA);
    end
    Output.Grad=Xgrad;
end
if Input.RG==0
    for jj=seq'
        temGD=Grad.*Derive(jj,:);
        temGA=temGD*A(:,2:end,jj);
        A(:,:,jj)=A(:,:,jj)+lr*centerlambda(jj)*temGD'*[1,datain1(jj,:)];
        CM(:,:,jj)=CM(:,:,jj)+lr*centerlambda(jj)*datain'*(datain2(jj,:).*temGA);
    end
end
%%
Output.Sys.CM=CM;
Output.Sys.A=A;
Output.lr=lr;
%%
end
function [Output]=EFSTra_Forward(Input)
datain=Input.data;
CL=Input.YS;
if isempty(Input.Sys)
    type='sig';
    CR=Input.CR;
    W=length(datain);
    CW=ceil(CR*W);
    Output.Sys.center=datain;
    Output.Sys.prototypes=datain;
    Output.Sys.local_X=datain.^2;
    Output.Sys.local_delta=zeros(1,W);
    Output.Sys.Global_mean=datain;
    Output.Sys.Global_X=datain.^2;
    Output.Sys.Support=1;
    Output.Sys.ModelNumber=1;
    Output.Sys.A=(round(rand(CL,CW+1,1)))/(CW+1);
    Output.Sys.CM=VSRP(W,CW);
    Output.compsdata=ActivationFunction(datain*Output.Sys.CM,type,'a');
    Output.Ye=ActivationFunction([1,Output.compsdata]*Output.Sys.A',type,'a');
    Output.Sys.YeL=Output.Ye;
    Output.Sys.L=1;
    Output.Sys.W=W;
    Output.Sys.CW=CW;
    Output.Sys.CL=CL;
    Output.Sys.threshold1=exp(-3);
    Output.lr=Input.lr;
    Output.Sys.centerlambda=1;
    Output.Sys.LocalDensity=1;
    Output.Sys.Global_Delta1=0;
    Output.Sys.type=type;
    Output.Sys.threshold2=0.95;
else
    Input.Sys.L=Input.Sys.L+1;
    Output.Sys=Input.Sys;
    ii=Input.Sys.L;
    lr=Input.lr;
    type=Input.Sys.type;
    center=Input.Sys.center;
    prototypes=Input.Sys.prototypes;
    local_X=Input.Sys.local_X;
    local_delta=Input.Sys.local_delta;
    Global_mean=Input.Sys.Global_mean;
    Global_X=Input.Sys.Global_X;
    Support=Input.Sys.Support;
    ModelNumber=Input.Sys.ModelNumber;
    threshold1=Input.Sys.threshold1;
    A=Input.Sys.A;
    CW=Input.Sys.CW;
    W=Input.Sys.W;
    CM=Input.Sys.CM;
    CL=Input.Sys.CL;
    Global_mean=Global_mean.*(ii-1)./ii+datain./ii;
    Global_X=Global_X.*(ii-1)./ii+datain.^2./ii;
    Global_Delta=abs(Global_X-Global_mean.^2);
    [~,LocalDensity,Global_Delta1]=firingstrength(datain,ModelNumber,prototypes,local_delta,Global_Delta,W);
    LocalDensity(isnan(LocalDensity))=1;
    if max(LocalDensity)<threshold1
        %% new_cloud_add
        ModelNumber=ModelNumber+1;
        center(ModelNumber,:)=datain;
        prototypes(ModelNumber,:)=datain;
        local_X(:,:,ModelNumber)=datain.^2;
        Support=[Support,1];
        local_delta(:,:,ModelNumber)=zeros(1,W);
        Global_Delta1(ModelNumber,1)=sum(Global_Delta,'all')./2;
        A(:,:,ModelNumber)=(round(rand(CL,CW+1,1)))/(CW+1);
        CM(:,:,ModelNumber)=VSRP(W,CW);
        LocalDensity=[LocalDensity;1];
    else
        %% local_parameters_update
        [~,label0]=max(LocalDensity);
        Support(label0)=Support(label0)+1;
        center(label0,:)=((Support(label0)-1)*center(label0,:)+datain)/Support(label0);
        local_X(:,:,label0)=((Support(label0)-1)*local_X(:,:,label0)+datain.^2)/Support(label0);
        local_delta(:,:,label0)=abs(local_X(:,:,label0)-center(label0,:).^2);
        Global_Delta1(label0,1)=(sum(Global_Delta+local_delta(:,:,label0),'all')./2);
        LocalDensity(label0,1)=exp(-1*sum((datain-prototypes(label0,:)).^2,'all')/Global_Delta1(label0,1));
    end
    LocalDensity(isnan(LocalDensity))=1;
    [centerlambda,LocalDensity]=ActivatingRules(ModelNumber,LocalDensity,Input.Sys.threshold2);
    Cdata=zeros(ModelNumber,CW);
    for tt=find(centerlambda'~=0)
        Cdata(tt,:)=ActivationFunction(datain*CM(:,:,tt),type,'a');
    end
    [Ye1,YeL]=OutputGeneration(Cdata,A,centerlambda,ModelNumber,CL,type);
    %%
    Output.lr=lr;
    Output.Sys.centerlambda=centerlambda;
    Output.Sys.LocalDensity=LocalDensity;
    Output.Sys.center=center;
    Output.Sys.prototypes=prototypes;
    Output.Sys.local_X=local_X;
    Output.Sys.local_delta=local_delta;
    Output.Sys.Global_mean=Global_mean;
    Output.Sys.Global_Delta1=Global_Delta1;
    Output.Sys.Global_X=Global_X;
    Output.Sys.Support=Support;
    Output.Sys.ModelNumber=ModelNumber;
    Output.Sys.A=A;
    Output.Ye=Ye1;
    Output.compsdata=Cdata;
    Output.Sys.YeL=YeL;
    Output.Sys.CW=CW;
    Output.Sys.CM=CM;
end
end
function [centerlambda,LocalDensity,Global_Delta1]=firingstrength(datain,ModelNumber,center,local_delta,Global_Delta,W)
LocalDensity=zeros(ModelNumber,1);
Global_Delta1=zeros(ModelNumber,1);
for ii=1:1:ModelNumber
    datain1=sum((datain-center(ii,:)).^2,'all');
    Global_Delta1(ii,1)=sum((Global_Delta+local_delta(:,:,ii))/2,'all');
    LocalDensity(ii,1)=exp(-1*datain1./Global_Delta1(ii,1));
end
LocalDensity(isnan(LocalDensity))=1;
centerlambda=LocalDensity./sum(LocalDensity);
end
function [Ye,YeL]=OutputGeneration(datain,A,centerlambda,ModelNumber,CL,type)
Ye=zeros(1,CL);
YeL=zeros(ModelNumber,CL);
seq=find(centerlambda~=0);
for ii=seq'
    YeL(ii,:)=ActivationFunction([1,datain(ii,:)]*A(:,:,ii)',type,'a');
    Ye=Ye+YeL(ii,:)*centerlambda(ii);
end
end
function [M]=VSRP(W,K)
M=rand(W,K);
S=sqrt(W);A=(S*2-1)/(S*2);B=1/(S*2);
M(M<=B)=-1;
M(M>=A)=1;
M(M>B&M<A)=0;
M=M.*sqrt(S)/sqrt(K);
end