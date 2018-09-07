function [vp] = pnorm1(v1,v2,p)
vp=(abs(v1).^p+abs(v2).^p)^(1/p);
end