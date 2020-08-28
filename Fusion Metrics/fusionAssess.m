function res=fusionAssess(img1,img2,fused)

% function res=fusionAssess(im1,im2,fused)
%
% This function is to assess the fused image with different fusion
% assessment metrics.
% 
% im1   ---- input image one;
% im2   ---- input image two;
% fused ---- the fused image(s)
% res   ==== the metric value
%
% Z. Liu @ NRCC [Aug 21, 2009]
%

im1   =double(img1);
im2   =double(img2);

% Wang- NCIE $Q_{NCIE}$
Q(1)=metricWang(im1,im2,fused);
    
% Xydeas Gradient $Q_G$
Q(2)=metricXydeas(im1,im2,fused);

% FMI
Q(3)=fmi(im1,im2,fused);

% Piella  (need to select only one) $Q_S$
% Q(i,8)=index_fusion(im1,im2,fused{i});
Q(4)=metricPeilla(im1,im2,fused,1);


% VIFF metric
Q(5)=VIFF_Public(im1,im2,fused);

res=Q;
