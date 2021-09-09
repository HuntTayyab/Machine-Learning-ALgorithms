clc;
close all;
clear all;
 
in = 2;
hn = 3;
on = 1;
 
weight(:,:) = zeros(in,hn);
v(:,:) = zeros(on,hn);
 
beta1 = 0.7 * hn ^ (1/in);
beta2 = 0.7 * on ^ (1/hn);
 
rih = [-beta1 beta1];
rho = [-beta2 beta2];
 
bih = rand(1,hn) * range(rih) + min(rih);
bho = rand(1,on) * range(bih) + min(rho);
 
 
tc = [-bho bho];
 
weight = rand(in,hn) * range(tc) + min(tc);
v = rand(on,hn) * range(tc) + min(tc);
 
sumw = 0;
sumv = 0;
for i = 1 : in
    for j = 1 : hn
        sumw = sumw + (weight(i,j) * weight(i,j));
    end
end
for i = 1 : hn
    sumv = sumv + (v(i) * v(i));
end
 
sumw = sqrt(sumw);
sumv = sqrt(sumv);
 
for i = 1 : in
    for j = 1 : hn
        w(i,j) = weight(i,j) * beta1 / sumw;
        
    end
end
 
for i = 1 : hn
    v(i) = v(i) * beta2 / sumv;
end
 
input = [1 1; 1 -1; -1 1; -1 -1];
target = [1 -1 -1 -1];
 
bias = rand(1,hn+1);
for ep = 1 : 100
    for i = 1 : 4
        for j = 1 : hn
            zin(j) = bias(j) + (input(i,:) * weight(:,j));
            z(j) = Bthresh(zin(j));
        end
        yin(i) = bias(hn+1) + (v * z');
        y(i) = Bthresh(yin(i));
        
        % Now compute the error portion sigma k
        sig(i) = (target(i) - y(i)) *  (y(i) * (1 - y(i)));
        st(i) = (target(i) - y(i))^2;
        for j = 1 : hn
            % Now find the change in weights between output and hidden layers
            dv(j) = 0.25 * sig(i) * z(j);
            % Now compute error portion between hidden layer and input layer
            si(j) = sig(i) * v(j) * z(j) * (1 - z(j));
            
            % Now compute the weight change between input and hidder layer
            w11(j) = 0.25 * si(j) * input(1);
            w21(j) = 0.25 * si(j) * input(2);
            b(j) = 0.25 * si(j);
            
            % Now compute the final weights of the network
            weight(1,j) = weight(1,j) + w11(j);
            weight(2,j) = weight(2,j) + w21(j);
            v(1,j) = v(1,j) + dv(j);
            bias(1,j) = bias(1,j) + b(j);
        end
        db = 0.25 * sig(i);
        
    end


    bias(1,(hn+1)) = bias(1,(hn+1)) + db;
    avg(ep) = mean(st);
end
plot(avg,'r');
%------------------------------
function [output]=Bthresh(x)
    for i = 1:size(x)
        output(i) = 1/(1+exp(-x(i)));
    end
end