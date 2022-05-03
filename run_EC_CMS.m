function results = run_EC_CMS(A,B,k,lambda)

N = size(B,1);
n = numel(k);
results = zeros(N,n);
A = A - diag(diag(A));

[C,~,~] = solver(A,B,lambda);

for i = 1:n
    K = k(i);
    
    s = squareform(C - diag(diag(C)),'tovector');
    d = 1 - s;
    results(:,i) = cluster(linkage(d,'average'),'maxclust',K);
    
    disp(['Obtain ',num2str(K),' clusters.']);
end