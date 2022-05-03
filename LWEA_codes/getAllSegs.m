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

function [bcs, baseClsSegs] = getAllSegs(baseCls)

[N,M] = size(baseCls);
% n:    the number of data points.
% M:    the number of base clusterings.
% nCls:     the number of clusters (in all base clusterings).

bcs = baseCls;
nClsOrig = max(bcs,[],1);
C = cumsum(nClsOrig); 
bcs = bsxfun(@plus, bcs,[0 C(1:end-1)]);
nCls = nClsOrig(end)+C(end-1);
baseClsSegs=sparse(bcs(:),repmat([1:N]',1,M), 1,nCls,N);