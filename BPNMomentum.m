clc;
clear all;
n = 2;
m=0.1;
 
input = [1 1; 1 -1; -1 1; -1 -1];
target = [1 -1 -1 -1];
weight = rand(2,n);
v = rand(1,n);
bias = rand(1,n+1);
 
dv = zeros(1:n);
delw1 = zeros(1:n);
delw2 = zeros(1:n);
b = zeros(1:n+1);
dbv = zeros(1:1);
for ep = 1:100
    for i = 1 : 4
        for j = 1 : n
            zin(j) = bias(j) + (input(i,:) * weight(:,j));
            z(j) = Bthresh(zin(j));
        end
        yin(i) = bias(n+1) + (v * z');
        y(i) = Bthresh(yin(i));
        % Now compute the error portion sigma k
        sig(i) = (target(i) - y(i)) *  (y(i) * (1 - y(i)));
        st(i) = (target(i) - y(i))^2;
        for j = 1 : n
            % Now find the change in weights between output and hidden layers
            dv(j) = 0.25 * sig(i) * z(j);
            % Now compute error portion between hidden layer and input layer
            si(j) = sig(i) * v(j) * z(j) * (1 - z(j));
            
            % Now compute the weight change between input and hidder layer
            if(j==1)
                delw1(j) = 0.25 * si(j) * input(1);
                delw2(j) = 0.25 * si(j) * input(2);
                b(j) = 0.25 * si(j);
            else
                delw1(j) = 0.25 * si(j) * input(1) +m*delw1(j-1);
                delw2(j) = 0.25 * si(j) * input(2)+m*delw2(j-1);
                b(j) = 0.25 * si(j)+m*b(j-1);
            end
            % Now compute the final weights of the network
            weight(1,j) = weight(1,j) + delw1(j);
            weight(2,j) = weight(2,j) + delw2(j);
            v(1,j) = v(1,j) + dv(j);
            bias(1,j) = bias(1,j) + b(j);
        end
        if(i==1)
            dbv(1) = 0.25 * sig(i);
        else
            dbv(1) = 0.25 * sig(i) + m*dbv(1);
        end
        
    end
    bias(1,(n+1)) = bias(1,(n+1)) + dbv;
    
    avg(ep) = mean(st);
end
disp(weight)
disp(v)
disp(bias)
disp('Done');
plot(avg);
hold on;

%------------------------------
function [output]=Bthresh(x)
    for i = 1:size(x)
        output(i) = 1/(1+exp(-x(i)));
    end
end

