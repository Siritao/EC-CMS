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

function LWCA=computeLWCA(baseClsSegs,ECI,M)
% Get locally weighted co-association matrix

baseClsSegs = baseClsSegs';
N = size(baseClsSegs,1);

% LWCA = (baseClsSegs.*repmat(ECI',N,1)) * baseClsSegs' / M;
LWCA = (bsxfun(@times, baseClsSegs, ECI')) * baseClsSegs' / M;

LWCA = LWCA-diag(diag(LWCA))+eye(N);