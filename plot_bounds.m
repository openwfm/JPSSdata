function plot_bounds(lon,lat,U,L)
% Call:
% plot_bounds(lon,lat,U,L)
%
% Description:
% Plots the bounds to apply to SVM
%
% Inputs:
%   lon,lat  longitude and latitude coordinates
%   U        upper bounds
%   L        lower bounds
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH.
% Angel Farguell (angel.farguell@gmail.com), 2019-06-17
%-------------------------------------------------------------------------

uu = U;
uu(uu==max(uu(:))) = nan;
nu = sum(sum(~isnan(uu)));
ll = L;
ll(ll==min(ll(:))) = nan;
nl = sum(sum(~isnan(ll)));

figure
S = repmat(5,nu,1);
C = repmat([1,0,0],nu,1);
hu = scatter3(lon(~isnan(uu)), lat(~isnan(uu)), uu(~isnan(uu)), S, C, 'filled');
alpha = 0.7;
set(hu, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
hold on
S = repmat(5,nl,1);
C = repmat([0.2,0.7,0.2],nl,1);
hl = scatter3(lon(~isnan(ll)), lat(~isnan(ll)), ll(~isnan(ll)), S, C, 'filled');
alpha = 0.2;
set(hl, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)


end