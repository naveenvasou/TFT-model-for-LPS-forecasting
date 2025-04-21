import xarray as xr
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.integrate import simpson

# Load track information for MDs, indexed by 'id'
MDs_track = pd.read_csv("raw/MDs_track.csv", index_col="id").drop(columns=['Unnamed: 0'])

# Function to compute spatial mean of a variable at given pressure levels within a radius from the center
def final_data(variable, levels, avg_radius):

    processed_data = []

    #defing MASK
    data = xr.open_dataset("raw/era5/"+variable+"1000_100/"+variable+"100"+"_MD.nc")
    x_vals = data['x'].values
    y_vals = data['y'].values
    
    x, y = np.meshgrid(x_vals, y_vals, indexing='ij')
    distance_from_center = np.sqrt(x**2 + y**2)
    circle_mask = distance_from_center <= avg_radius
    circle_mask_da = xr.DataArray(circle_mask, dims=['x', 'y'], coords={'x': x_vals, 'y': y_vals})


    for i in levels:
        data = xr.open_dataset("raw/era5/"+variable+"1000_100/"+variable+str(i)+"_MD.nc")
        print("LEVEL: ",i)
        temp = data['snap_'+variable].where(circle_mask_da).mean(dim={'x','y'})
        processed_data.append(list(temp.values))
    processed_data = np.array(processed_data)

    return processed_data

# Get vorticity data at multiple levels and assign to DataFrame
print("GETTING VO DATA...")
MDs_track["VO550"], MDs_track["VO750"], MDs_track["VO850"] = final_data("VO", [550,750,850], 4)

# Get potential vorticity and take the average of two levels
print("GETTING PV DATA...")
PV450, PV550 = final_data("PV", [450,550], 3)
MDs_track["PV"] = (PV450 + PV550)/2

# Get specific humidity at 850 hPa level
print("GETTING Q850 DATA...")
Q850 = final_data("Q", [850], 5)
MDs_track["Q850"] = Q850[0]

# Compute meridional gradient in specific humidity at 850 hPa using northern and southern averages
print("GETTING Q850_grad DATA...")
data = xr.open_dataset("raw/era5/Q1000_100/Q850_MD.nc")
sh_1 = data['snap_Q'].sel({"y":slice(-10,-5)}).mean(dim={'x','y'})
sh_2 = data['snap_Q'].sel({"y":slice(5,10)}).mean(dim={'x','y'})
sh_grad = (sh_2 - sh_1)/15   # North - South
MDs_track["Q850_grad"] = sh_grad

# Get 2m specific humidity (near surface)
print("GETTING Q2 DATA...")
data = xr.open_dataset("raw/era5/Q2/Q2_MD.nc")
x_vals = data['x'].values
y_vals = data['y'].values
x, y = np.meshgrid(x_vals, y_vals, indexing='ij')
distance_from_center = np.sqrt(x**2 + y**2)
circle_mask = distance_from_center <= 5
circle_mask_da = xr.DataArray(circle_mask, dims=['x', 'y'], coords={'x': x_vals, 'y': y_vals})
temp = data['snap_q2m'].where(circle_mask_da).mean(dim={'x','y'})
MDs_track["Q2"] = temp

# Compute 850 hPa U-wind (zonal) split into north and south of the center
print("GETTING snap_U DATA...")
data = xr.open_dataset("raw/era5/U1000_100/U850_MD.nc")
temp = data['snap_U'].sel({"x":slice(-10,10), "y":slice(-10,-0.25)}).mean(dim={'x','y'})
MDs_track["US_850"] = temp
temp = data['snap_U'].sel({"x":slice(-10,10), "y":slice(0.25,10)}).mean(dim={'x','y'})
MDs_track["UN_850"] = temp

# Compute 850 hPa V-wind (meridional) split into east and west of the center
print("GETTING snap_V DATA...")
data = xr.open_dataset("raw/era5/V1000_100/V850_MD.nc")
temp = data['snap_V'].sel({"x":slice(0.25,10), "y":slice(-10,10)}).mean(dim={'x','y'})
MDs_track["VE_850"] = temp
temp = data['snap_V'].sel({"x":slice(-10,-0.25), "y":slice(-10,10)}).mean(dim={'x','y'})
MDs_track["VW_850"] = temp

# Get 2-meter temperature
print("GETTING T2 DATA...")
data = xr.open_dataset("raw/era5/T2/T2_MD.nc")
x_vals = data['x'].values
y_vals = data['y'].values
x, y = np.meshgrid(x_vals, y_vals, indexing='ij')
distance_from_center = np.sqrt(x**2 + y**2)
circle_mask = distance_from_center <= 5
circle_mask_da = xr.DataArray(circle_mask, dims=['x', 'y'], coords={'x': x_vals, 'y': y_vals})
temp = data['snap_VAR_2T'].where(circle_mask_da).mean(dim={'x','y'})
MDs_track["T2"] = temp

# Compute Z_tilt: slope of geopotential height minima across levels
print("GETTING Z_tilt DATA...")
levels = [450, 550, 650, 750, 800, 850, 900, 950, 999]
min_x_lists = []

for i in levels:
    print(i)
    data = xr.open_dataset(f"raw/era5/Z1000_100/Z{i}_MD.nc")
    
    # Average over narrow latitudinal band around center
    snap_Z = data.snap_Z.sel({"y": slice(-2.5, 2.5)}).mean(dim={"y"})

    # Get x-location (longitude-like) of minimum geopotential height
    min_index = snap_Z.argmin(dim='x')
    min_x = snap_Z.isel(x=min_index).x

    min_x_lists.append(min_x)
    print("DONE")

# Fit linear model for each timestep to compute tilt (slope of geopotential height minima)
z_tilt = []
for i in range(33229):  # number of LPS-time points
    y_min = [min_x_lists[j][i] for j in range(len(levels))]  # x-min of Z at each level
    model = LinearRegression()
    x = np.array(levels).reshape(-1, 1)
    y = np.array(y_min)
    model.fit(x, y)
    z_tilt.append(model.coef_[0])  # slope of the fit

MDs_track["Z_tilt"] = z_tilt

# Compute vertically integrated Moist Static Energy (MSE)
print("GETTING MSE DATA...")
levels = [100, 150, 200, 250, 350, 450, 550, 650, 750, 800, 850, 900, 950, 999]
print("GETTING Q DATA FOR MSE...")
q = final_data("Q", levels, 5)
print("GETTING T DATA FOR MSE...")
T = final_data("T", levels, 5)
print("GETTING Z DATA  FOR MSE...")
Z = final_data("Z", levels, 5)

cp = 1004      # specific heat capacity at constant pressure (J/kg/K)
Lv = 2.5e6     # latent heat of vaporization (J/kg)

# Calculate MSE = cp*T + Z + Lv*q
mse = cp * T + Z + Lv * q

# Integrate MSE vertically using Simpson's rule
integrated_mse = simpson(y=mse, x=levels, axis=0)
MDs_track["integrated_mse"] = integrated_mse

# Extract geopotential data
print("GETTING Z250, Z550, Z850 DATA...")
Z250, Z550, Z850 = Z[3], Z[6], Z[10]
MDs_track["Z250"] = Z250
MDs_track["Z550"] = Z550
MDs_track["Z850"] = Z850

# Extract Rainfall data
print("GETTING Rainfall DATA...")
RF = xr.open_dataset("era5/raw/RF_MD.nc")
x_vals = RF['x'].values
y_vals = RF['y'].values
    
x, y = np.meshgrid(x_vals, y_vals, indexing='ij')
distance_from_center = np.sqrt(x**2 + y**2)
circle_mask = distance_from_center <= 5
circle_mask_da = xr.DataArray(circle_mask, dims=['x', 'y'], coords={'x': x_vals, 'y': y_vals})

RF = RF['snap_tp'].where(circle_mask_da).mean(dim={'x','y'})

MDs_track["RF"] = RF.values

