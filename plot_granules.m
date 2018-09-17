function plot_granules(granules)
% Call:
% plot_granules(granules)
%
% Description:
% Plot all the granules in a matlab strucure. The matlab structure has to
% contain a layer of satellite product and date and the lon, lat and fire 
% information for each satellite product and date.
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH. 
% Angel Farguell (angel.farguell@gmail.com), 2018-08-24
%-------------------------------------------------------------------------

granule=fields(granules);
for ii=1:length(granule)
    plot_granule(granules.(granule{ii}));
end

end