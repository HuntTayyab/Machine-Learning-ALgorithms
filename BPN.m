clc;
clear all;
close all;
%--------------input data------------------------
n = 3;
input = [1 1; 1 -1; -1 1; -1 -1];
target = [1 1 1 -1];
%-----------------weight initialization------------
disp("intiialised weight");
weight = rand(2,n);
v = rand(1,n);

bias = rand(1,n+1);
out = [];
for ep = 1 : 100
    for i = 1 : 4
        for j = 1 : n
            zin(j) = bias(j) + (input(i,:) * weight(:,j));
            z(j) = Bthresh(zin(j));    
        end
        yin(i) = bias(n+1) + (v * z');
       y(i) = Bthresh(yin(i));
       % Now compute the error portion sigma k
       sig(i) = (target(i) - y(i)) *  (y(i) * (1 - y(i)));

       for j = 1 : n
%--------- Now find the change in weights between output and hidden layers
           dv(j) = 0.25 * sig(i) * z(j);
%---------- Now compute error portion between hidden layer and input layer
si(j) = sig(i) * v(j) * z(j) * (1 - z(j)); 

%----------- Now compute the weight change between input and hidder layer
           w11(j) = 0.25 * si(j) * input(1);
           w21(j) = 0.25 * si(j) * input(2);
           b(j) = 0.25 * si(j);

%------------- Now compute the final weights of the network
            weight(1,j) = weight(1,j) + w11(j);
            weight(2,j) = weight(2,j) + w21(j);
            v(1,j) = v(1,j) + dv(j);
            bias(1,j) = bias(1,j) + b(j);
       end
db = 0.25 * sig(i);
       out = [out;[zin,z,yin(i),y(i),sig(i),dv,db,si,w11,w21,b,weight(1,:),weight(2,:),v,bias(1:n)]];
    end
    bias(1,(n+1)) = bias(1,(n+1)) + db;
    avg(ep) = mean(sig);
end
disp('Done');
plot(avg);
%---------------------------------------
function [output]=Bthresh(x)
    for i = 1:size(x)
        output(i) = 1/(1+exp(-x(i)));
    end
end
