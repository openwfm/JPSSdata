function plot_svm(svm_file)
% Call:
% plot_results(svm_file)
%
% Description:
% Plots the results in the output file from Support Vector Machine
%
% Inputs:
%   svm_file     matlab output file from Support Vector Machine
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH.
% Angel Farguell (angel.farguell@gmail.com), 2019-03-21
%-------------------------------------------------------------------------

r = load(svm_file);
if isfield(r,'U')
    uu = r.U;
    uu(uu==max(uu(:))) = nan;
    ll = r.L;
    ll(ll==min(ll(:))) = nan;

    dd = .1;
    bb = [min(r.dxlon(~isnan(uu)))-dd, max(r.dxlon(~isnan(uu)))+dd, min(r.dxlat(~isnan(uu)))-dd, max(r.dxlat(~isnan(uu)))+dd];
    figure
    S = repmat(5,sum(sum(~isnan(uu))),1);
    C = repmat([1,0,0],sum(sum(~isnan(uu))),1);
    h1 = scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), S, C, 'filled');
    alpha = 0.7;
    set(h1, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
    hold on
    ml = logical((r.dxlon >= bb(1)).*(r.dxlon <= bb(2)).*(r.dxlat >= bb(3)).*(r.dxlat <= bb(4)));
    S = repmat(5,sum(sum(ml)),1);
    C = repmat([0.2,0.7,0.2],sum(sum(ml)),1);
    h2 = scatter3(r.dxlon(ml), r.dxlat(ml), ll(ml), S, C, 'filled');
    alpha = 0.3;
    set(h2, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
    hold on
    contour3(r.fxlon,r.fxlat,r.Z,100)
    xlim([bb(1),bb(2)])
    ylim([bb(3),bb(4)])
    title('Support-vector machine: Satellite detections vs fire arrival time')
    xlabel('Longitude')
    ylabel('Latitude')
    zlabel('Time (days)')

    figure
    scatter3(r.dxlon(~isnan(uu)), r.dxlat(~isnan(uu)), uu(~isnan(uu)), 'r.')
    hold on
    contour3(r.fxlon,r.fxlat,r.Z,100)
    title('Support-vector machine: Fire detections vs fire arrival time')
    xlabel('Longitude')
    ylabel('Latitude')
    zlabel('Time (days)')
else
    ground = (r.y==-1);
    fire = (r.y==1);

    zmin = min(r.X(:,3));
    zmax = max(r.X(:,3));

    figure,
    S = repmat(5,sum(fire),1);
    C = repmat([1,0,0],sum(fire),1);
    h1 = scatter3(r.X(fire,1),r.X(fire,2),r.X(fire,3),S,C,'filled');
    alpha = 0.7;
    set(h1, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
    hold on
    S = repmat(5,sum(ground),1);
    C = repmat([0.2,0.7,0.2],sum(ground),1);
    h2 = scatter3(r.X(ground,1),r.X(ground,2),r.X(ground,3),S,C,'filled');
    alpha = 0.3;
    set(h2, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)
    hold on
    contour3(r.fxlon,r.fxlat,r.Z,100)
    zlim([zmin,zmax]);
    title('Support-vector machine: Satellite detections vs fire arrival time')
    xlabel('Longitude')
    ylabel('Latitude')
    zlabel('Time (days)')
    
    figure, 
    scatter3(r.X(fire,1), r.X(fire,2), r.X(fire,3), 'r.')
    hold on
    contour3(r.fxlon,r.fxlat,r.Z,100)
    zlim([zmin,zmax]);
    title('Support-vector machine: Fire detections vs fire arrival time')
    xlabel('Longitude')
    ylabel('Latitude')
    zlabel('Time (days)')
end

end
