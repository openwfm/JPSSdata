function [ign_lons,ign_lats,ign_times] = ignitions(file)
% Call:
% [ign_lons,ign_lats,ign_times] = ignitions(file)
%
% Description:
% Prints the ingition coordinates and times from fire arrival time
%
% Inputs:
%   file     matlab file with:
%               fxlon,fxlat: lon-lat meshgrid
%               Z: fire arrival time mesh
%
% Developed in Matlab 9.2.0.556344 (R2017a) on MACINTOSH.
% Angel Farguell (angel.farguell@gmail.com), 2019-04-09
%-------------------------------------------------------------------------

format long
r = load(file);
A = r.Z;
% compute local minima
LM = zeros(size(A));
LM(2:end-1,2:end-1)=max(min(A(3:end,1:end-2),min(A(1:end-2,3:end),min(A(3:end,3:end),min(A(1:end-2,1:end-2),min(A(1:end-2,2:end-1),min(A(3:end,2:end-1),min(A(2:end-1,1:end-2),A(2:end-1,3:end))))))))-A(2:end-1,2:end-1),0);
% find where local minima
ii = find(LM>0);
% Longitudes, Latitudes, and Times
ign_lons = r.fxlon(ii)
ign_lats = r.fxlat(ii)
ign_time_num = A(ii)*double(r.tscale)+r.time_scale_num(1);
ign_times = datetime(ign_time_num,'ConvertFrom','epochtime','Epoch','1969-12-31 17:00:00')

end
