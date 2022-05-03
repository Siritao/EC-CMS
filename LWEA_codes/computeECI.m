%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
% This is a demo for the LWEA and LWGP algorithms. If you find this %
% code useful for your research, please cite the paper below.       %
%                                                                   %
% Dong Huang, Chang-Dong Wang, and Jian-Huang Lai.                  %
% "Locally weighted ensemble clustering."                           %
% IEEE Transactions on Cybernetics, 2018, 48(5), pp.1460-1473.      %
%                                                                   %
% The code has been tested in Matlab R2014a and Matlab R2015a on a  %
% workstation with Windows Server 2008 R2 64-bit.                   %
%                                                                   %
% https://www.researchgate.net/publication/316681928                %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ECI = computeECI(bcs, baseClsSegs, para_theta)

M = size(bcs,2);
ETs = getAllClsEntropy(bcs, baseClsSegs);
ECI = exp(-ETs./para_theta./M);


function Es = getAllClsEntropy(bcs, baseClsSegs)
% Get the entropy of each cluster w.r.t. the ensemble

baseClsSegs = baseClsSegs';

[~, nCls] = size(baseClsSegs);

Es = zeros(nCls,1);
for i = 1:nCls
    partBcs = bcs(baseClsSegs(:,i)~=0,:);
    Es(i) = getOneClsEntropy(partBcs);
end

function E = getOneClsEntropy(partBcs)
% Get the entropy of one cluster w.r.t the ensemble

% The total entropy of a cluster is computed as the sum of its entropy
% w.r.t. all base clusterings.

E = 0;
for i = 1:size(partBcs,2)
    tmp = sort(partBcs(:,i));
    uTmp = unique(tmp);
    
    if numel(uTmp) <= 1
        continue;
    end
    % else
    cnts = zeros(size(uTmp));
    for j = 1:numel(uTmp)
        cnts(j)=sum(sum(tmp==uTmp(j)));
    end
    
    cnts = cnts./sum(cnts(:));
    E = E-sum(cnts.*log2(cnts));
end

