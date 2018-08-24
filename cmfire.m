function cmfire()
% Call:
% cmFire()
%
% Description:
% Generates a colormap useful for fire detections data.
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH. 
% Angel Farguell (angel.farguell@gmail.com), 2018-08-24
%-------------------------------------------------------------------------

cmap = [[128,128,128];[128,128,128];[102,102,255];[255,255,255];[0,204,0];[128,128,128];[255,255,0];[255,165,0];[255,0,0]];
cmap = cmap/255;
colormap(cmap);
hcb=colorbar('h');
set(hcb,'YTick',[1:1:9]);
caxis([1 10]);
end