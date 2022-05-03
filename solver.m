function [C,E,t] = solver(H,A,lambda)
n = size(A,1);
t = 0;
e = 1e-2;
max_iter = 100;
I = eye(n);
C = zeros(n);
E = C;
F = C;
r1 = 1;
r2 = 1;
Y1 = A;
Y2 = C;
D = H * ones(n,1);
phi = diag(D) - H;
inv_part = (2 * phi + (r1 + r2) * I) \ I;

while t < max_iter
    t = t + 1;
    
    % update C
    Ct = C;
    P1 = A - E + Y1 / r1;
    P2 = F - Y2 / r2;
    C = inv_part * (r1 * P1 + r2 * P2);
    
    % update E
    Et = E;
    E = r1 * (A - C) + Y1;
    E = E /(lambda + r1);
    E(H > 0) = 0;
    
    % update F
    Ft = F;
    F = C + Y2 / r2;
    F = min(max((F + F') / 2,0),1);
    
    % update Y
    Y1t = Y1;
    residual1 = A - C - E;
    Y1 = Y1t + r1 * residual1;
    
    Y2t = Y2;
    residual2 = C - F;
    Y2 = Y2t + r2 * residual2;
    
    diffC = abs(norm(C - Ct,'fro')/norm(Ct,'fro'));
    diffE = abs(norm(E - Et,'fro')/norm(Et,'fro'));
    diffF = abs(norm(F - Ft,'fro')/norm(Ft,'fro'));
    diffY1 = abs(norm(residual1,'fro')/norm(Y1t,'fro'));
    diffY2 = abs(norm(residual2,'fro')/norm(Y2t,'fro'));
    
    if max([diffC,diffE,diffF,diffY1,diffY2]) < e
        break;
    end
end