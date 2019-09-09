function plot_svm(varargin)
% Call:
% plot_results(svm_file)
% plot_results(svm_file,result_file)
% plot_results(svm_file,result_file,zoom)
%
% Description:
% Plots the results in the output file from Support Vector Machine
%
% Inputs:
%   svm_file     matlab output file from Support Vector Machine
%   result_file  matlab output file from preprocessing of satellite data
%   zoom         boolean if plot zoom or not (default false)
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH.
% Angel Farguell (angel.farguell@gmail.com), 2019-03-21
%-------------------------------------------------------------------------

if nargin < 1 || nargin > 3
   error('plot_svm(svm_file) or plot_svm(svm_file,result_file) or plot_svm(svm_file,result_file,zoom)');
end
svm_file = varargin{1};
r = load(svm_file);
uu = r.U;
uu(uu==max(uu(:))) = nan;
ll = r.L;
ll(ll==min(ll(:))) = nan;
low = false;
zoom = false;
if nargin == 2
    result_file = varargin{2};
    r2 = load(result_file);
    tt = (r2.T==r2.time_scale_num(2)-r2.time_scale_num(1));
    ll(tt) = nan;
    low = true;
    figure
    scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), 'r.')
    hold on
    scatter3(r.dxlon(~isnan(ll)), r.dxlat(~isnan(ll)), ll(~isnan(ll)), 'g.')
end
if nargin == 3
    zoom = varargin{3};
end

dd = .1;
bb = [min(r.dxlon(~isnan(uu)))-dd, max(r.dxlon(~isnan(uu)))+dd, min(r.dxlat(~isnan(uu)))-dd, max(r.dxlat(~isnan(uu)))+dd];
figure
S = repmat(5,sum(sum(~isnan(uu))),1);
C = repmat([1,0,0],sum(sum(~isnan(uu))),1);
h1 = scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), S, C, 'filled');
alpha = 0.7;
set(h1, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
hold on
if low
    S = repmat(5,sum(sum(~isnan(ll))),1);
    C = repmat([0.2,0.7,0.2],sum(sum(~isnan(ll))),1);
    h2 = scatter3(r.dxlon(~isnan(ll)), r.dxlat(~isnan(ll)), ll(~isnan(ll)), S, C, 'filled');
    alpha = 0.2;
    set(h2, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
    hold on
else
    ml = logical((r.dxlon >= bb(1)).*(r.dxlon <= bb(2)).*(r.dxlat >= bb(3)).*(r.dxlat <= bb(4)));
    S = repmat(5,sum(sum(ml)),1);
    C = repmat([0.2,0.7,0.2],sum(sum(ml)),1);
    h3 = scatter3(r.dxlon(ml), r.dxlat(ml), ll(ml), S, C, 'filled');
    alpha = 0.2;
    set(h3, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
    hold on
end
contour3(r.fxlon,r.fxlat,r.Z,100)
xlim([bb(1),bb(2)])
ylim([bb(3),bb(4)])
title('Support-vector machine: Fire detections vs fire arrival time')
xlabel('Longitude')
ylabel('Latitude')
zlabel('Time (days)')

if zoom
    kk = 10;
    zz = [min(r.Z(:))-.5 min(r.Z(:))+1.5];
    figure
    scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), 'r.')
    hold on
    if low
        scatter3(r.dxlon(~isnan(ll)), r.dxlat(~isnan(ll)), ll(~isnan(ll)), 'g.')
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

figure
scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), 'r.')
hold on
contour3(r.fxlon,r.fxlat,r.Z,100)
title('Support-vector machine: Fire detections vs fire arrival time')
xlabel('Longitude')
ylabel('Latitude')
zlabel('Time (days)')

end
