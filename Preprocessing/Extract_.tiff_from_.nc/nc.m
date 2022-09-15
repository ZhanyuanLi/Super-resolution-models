%Extracts climate data (".tiff" format) for each month from the annual data (".nc" format) for each climate factor.

clc;
clear;

%First read the data of one time point in the nc file in arcgis and export it as tif to provide the coordinate system for nc file batch export later.
maskname = 'G:\Preprocessing\nc_data_processing0\pv0\1.tif';
%The main purpose is to obtain R, the coordinate system geotiffread
[A,R] = readgeoraster(maskname);
info = geotiffinfo(maskname);

% Read nc file basic information
%The folder where the nc file is located
inpath = 'G:\Preprocessing\nc_data_processing0\2021\';  

%Get basic information about nc files
infile = strcat(inpath,'pv_hadukgrid_uk_1km_mon_202101-202112.nc');
%Read the variable named "_" from the data source
D = ncread(infile,'pv');   

%Read by time separately
for year = 2021:2021  %Annual cycle 
    for month = 1:12  %Monthly cycle
		%Month-by-month reading
        data = D(:,:,month);
		%Rotate 90° counterclockwise
        data = rot90(data);
		%Export path and name
        export2tif = ['G:\Preprocessing\data\2021\pv\pv_',int2str(year),'_',int2str(month),'.tif'];
		%Batch export to tif
        geotiffwrite(export2tif,data,R,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag);
    end
    disp(year);
end
