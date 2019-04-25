function plot_svm(varargin)
% Call:
% plot_results(svm_file)
% plot_results(svm_file,zoom)
% plot_results(svm_file,zoom,result_file)
%
% Description:
% Plots the results in the output file from Support Vector Machine
%
% Inputs:
%   file     matlab output file from Support Vector Machine
%   zoom     boolean if plot zoom or not (default false)
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH.
% Angel Farguell (angel.farguell@gmail.com), 2019-03-21
%-------------------------------------------------------------------------

kk = 10;

if nargin < 1 || nargin > 3
   error('plot_svm(svm_file) or plot_svm(svm_file,zoom) or plot_svm(svm_file,zoom,result_file)');
end
svm_file = varargin{1};
r = load(svm_file);
uu = r.U;
uu(uu==max(uu(:))) = nan;
ll = r.L;
ll(ll==min(ll(:))) = nan;
if nargin < 2
    zoom = false;
else
    zoom = varargin{2};
end
if nargin > 2
    result_file = varargin{3};
    r2 = load(result_file);
    tt = (r2.T==r2.time_scale_num(2)-r2.time_scale_num(1));
    ll(tt) = nan;
    low = true;
else
    low = false;
end

figure
scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), 'r*')
hold on
scatter3(r.dxlon(~isnan(ll)), r.dxlat(~isnan(ll)), ll(~isnan(ll)), 'g*')

figure
scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), 'r*')
hold on
if low
    scatter3(r.dxlon(~isnan(ll)), r.dxlat(~isnan(ll)), ll(~isnan(ll)), 'g*')
    hold on
end
contour3(r.fxlon(1:kk:end,1:kk:end),r.fxlat(1:kk:end,1:kk:end),r.Z(1:kk:end,1:kk:end),100)
title('Support-vector machine: Fire detections vs fire arrival time')
xlabel('Longitude')
ylabel('Latitude')
zlabel('Time (days)')

if zoom
    zz = [min(r.Z(:))-.5 min(r.Z(:))+1.5];
    figure
    scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), 'r*')
    hold on
    if low
        scatter3(r.dxlon(~isnan(ll)), r.dxlat(~isnan(ll)), ll(~isnan(ll)), 'g*')
        hold on
    end
    contour3(r.fxlon(1:kk:end,1:kk:end),r.fxlat(1:kk:end,1:kk:end),r.Z(1:kk:end,1:kk:end),500)
    zlim(zz)
    caxis(zz)
    title('2 days envolving the ignition time')
    xlabel('Longitude')
    ylabel('Latitude')
    zlabel('Time (days)')
end

end
