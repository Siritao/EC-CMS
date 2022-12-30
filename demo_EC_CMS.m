% Ensemble Clustering via Co-association Matrix Self-enhancement
% This framework is based on the source code of
% TCYB--Locally Weighted Ensemble Clustering

clear;clc
dataName = 'Ecoli'; % You can switch to other datasets
M = 20; % Ensemble size
cntTimes = 20; % How many times will be run.
alpha = 0.75;
lambda = 0.01;

rng(1)
addpath(genpath(pwd))

para_theta = 0.4; % Parameter of LWEA

load([dataName,'.mat'],'members','gt');
clsNums = length(unique(gt));
[N, poolSize] = size(members);

% For each run, M base clusterings will be randomly drawn from the pool.
% Each row in bcIdx corresponds to an ensemble of M base clusterings.
bcIdx = zeros(cntTimes, M);
for i = 1:cntTimes
    tmp = randperm(poolSize);
    bcIdx(i,:) = tmp(1:M);
end

% Scores
NMI_LWEA = zeros(cntTimes, 1);
NMI = NMI_LWEA; % NMI of our model
ARI_LWEA = NMI_LWEA;
ARI = NMI_LWEA; % ARI of our model
F_LWEA = NMI_LWEA;
F = NMI_LWEA; % F-score of our model

for runIdx = 1:cntTimes
    % Construct the ensemble of M base clusterings
    % baseCls is an N x M matrix, each row being a base clustering.
    baseCls = members(:,bcIdx(runIdx,:));
    
    % Get all clusters in the ensemble
    [bcs, baseClsSegs] = getAllSegs(baseCls);
    
    % Compute ECI for LWEA
    ECI = computeECI(bcs, baseClsSegs, para_theta);
    % Compute LWCA
    LWCA = computeLWCA(baseClsSegs, ECI, M);
    
    % Perform LWEA
    resultsLWEA = runLWEA(LWCA, clsNums);
    NMI_LWEA(runIdx) = compute_nmi(resultsLWEA,gt);
    ARI_LWEA(runIdx) = RandIndex(resultsLWEA,gt);
    F_LWEA(runIdx) = compute_f(resultsLWEA,gt);
    
    % Perform our model
    CA = getCA(baseClsSegs, M);
    A = getHC(CA,alpha);
    results = run_EC_CMS(A,LWCA,clsNums,lambda);
    if min(results) == 0
        results = results + 1;
    end
    NMI(runIdx) = compute_nmi(results,gt);
    ARI(runIdx) = RandIndex(results,gt);
    F(runIdx) = compute_f(results,gt);
end

nmi=mean(NMI);
varnmi=std(NMI);
nmiLWEA=mean(NMI_LWEA);
varnmiLWEA=std(NMI_LWEA);

ari=mean(ARI);
varari=std(ARI);
ariLWEA=mean(ARI_LWEA);
varariLWEA=std(ARI_LWEA);

f=mean(F);
varf=std(F);
fLWEA=mean(F_LWEA);
varfLWEA=std(F_LWEA);

% save(['results_',dataName,'.mat']);
disp('**************************************************************');
disp(['** Average Performance over ',num2str(cntTimes),' runs on the ',dataName,' dataset **']);
disp(['Data size: ', num2str(N)]);
disp(['Ensemble size: ', num2str(M)]);
disp('Average NMI/ARI/F scores:');
disp(['EC_CMS : ',num2str(nmi),'  ',num2str(ari),...
    '  ',num2str(f)]);
disp(['LWEA   : ',num2str(nmiLWEA),'  ',num2str(ariLWEA),...
    '  ',num2str(fLWEA)]);
disp('**************************************************************');
disp('**************************************************************');