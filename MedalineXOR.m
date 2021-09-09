clc  % to clear the command window
clear  %to clear the workspace
 
in=[1 1;1 -1;-1 1;-1 -1];
t=[-1 1 1 -1];
 
insz=size(in,1);
 
nh=02;  %no of hidden nodes
 
wt=[0.05 0.2; 0.1 0.2];
%wt=rand(size(in,2),nh);
b1=[0.3; 0.15];
%b1=rand(1,nh);
%v=[0.5 0.5];
v=rand(1,nh)
b2=rand(1,nh)
 
 
lr=0.2;
sqrt=0;
error=0;
 
%madaline weight updation rule
for epoch =1:10
    for i=1:insz
        zin=wt*in(i,:)'+b1;
        z=threshold2(zin,0);
        yin=v*z+b2;
        y=threshold2(yin,0);
        
        if t(i)~=y
            dw=zeros(size(wt));
            db=zeros(size(b1));
            if t(i)<0
                loc=find(zin>0);
                for z=1:numel(loc)
                    dw(z,:)=lr*(t(i)-zin(loc(z,1)))*in(i,:);
                    db(z)=lr*(t(i)-zin(loc(z,1)));
                end
            else
                [val,loc]=min(abs(zin));
                for z=loc
                    dw(z,:)=lr*(t(i)-zin(loc,1))*in(i,:);
                    db(z)=lr*(t(i)-zin(loc,1));
                end
            end
            wt=wt+dw;
            b1=b1+db;
        end
        error = t(i) - yin;
        %sqrt = sqrt + error*error';
    end
    disp([wt b1])
end
 
str=sprintf('  final weights:');
disp(str);
disp(wt);
%disp(error);
sqrt = sqrt + error*error';
mse(epoch) = (sqrt/size(in,1));
fprintf('MSE = %.4f\n',mse(epoch));
 
plot(mse);
function[output]=threshold2(input,threshold)
if input>threshold
    output=1;
else
    output=-1;
end
end
