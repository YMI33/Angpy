# Ang's self defined function for plot
# By Zhiang Xie 3rd Jun 2022
import matplotlib.figure
class FigCustom(matplotlib.figure.Figure):
################################# subplot and 

########################################
    def add_subplotc(self,plist=None,paneldiv=(22,22),mproj=None):
        # generate panels
        import matplotlib.gridspec as gridspec
        import cartopy.crs as ccrs
        flab   = ('(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)')
        projmp = {'g':ccrs.Robinson(),'n':ccrs.NorthPolarStereo(),'s':ccrs.SouthPolarStereo()}
        pgrid  = gridspec.GridSpec(ncols=paneldiv[0], nrows=paneldiv[1], figure=self)
        
        if mproj in ['g','n','s']:
            projection = projmp[mproj]
        else:
            projection = mproj
        
        if plist is None:
            self.ax    = self.add_subplot(pgrid[:,:],projection=projection)
        else:
            if isinstance(plist,int): # preset panel position for specifical number of panels
                poslist   = self._PanelListForGivenNumber_(plist,pgrid)
            else: # user defined panel position
                poslist   = [pgrid[x,y] for (x,y) in plist ]
            axlst = []
            for k, subplt in enumerate(poslist): 
                ax = self.add_subplot(subplt,projection=projection)                    # Add axes
                ax.text(0,1.02,flab[k],fontsize=12,horizontalalignment='right',   # Add panel label
                        verticalalignment='bottom',transform=ax.transAxes)
                axlst.append(ax)

            self.ax    = axlst
        self.pgrid = pgrid
        self.proj  = projection
        return
    
    def StringMark(self, ax):
        if hasattr(self,'LeftString'):
            ax.text(0,1.02,self.LeftString,fontsize=14,horizontalalignment='left',   # Add string
                    verticalalignment='center',transform=ax.transAxes)
        if hasattr(self,'RightString'):
            ax.text(1,1.02,self.RightString,fontsize=14,horizontalalignment='right',   # Add string
                    verticalalignment='center',transform=ax.transAxes)
        if hasattr(self,'CenterString'):
            ax.set_title('')
            ax.text(0.5,1.02,self.CenterString,fontsize=14,horizontalalignment='center',   # Add string
                    verticalalignment='bottom',transform=ax.transAxes)
        return
    
    def _PanelListForGivenNumber_(self,plist,pgrid):
        # generate panel position for given panel number
        paneldiv = (22,22)
        colrow   = {2:(1,2),3:(2,2),4:(2,2),5:(3,2),6:(3,2),9:(3,3)}
        colint   = int((22-2*(colrow[plist][0]-1))/colrow[plist][0])
        rowint   = int((22-2*(colrow[plist][1]-1))/colrow[plist][1])
        poslist  = []
        row = 0; col = 0
        for k in range(plist):
            poslistk = pgrid[slice(row,row+rowint),slice(col,col+colint)]
            poslist.append(poslistk)
            col   = col + colint + 2
            if col > paneldiv[0]:
                col = 0
                row   = row + rowint + 2
        
        if ((plist % 2) == 1) and (plist < 9):
            poslist[-1] = pgrid[slice(-rowint,-1),slice(int(colint/2),-int(colint/2))]
        
        
        return poslist

################################# color setting ########################################
    def VariableColormap(self, varname, clevs):
        cbar = {'temp': [[80, 40, 150],[80, 50, 250],[50, 50, 255],[0, 150, 255],[60, 220, 180],[0, 250, 100],[80, 250, 100], 
                [150, 250, 100],[190, 250, 160],[230, 250, 200],[255, 255, 255],[250, 250, 200],[250, 240, 120],[250, 210, 0], 
                [250, 175, 0],[250, 120, 0],[250, 40, 0],[220, 0, 40],[200, 0, 100],[180, 0, 150],[250, 0, 250]],
                'ice_sheet_mean': [[255, 255, 255], [160, 230, 230],[100, 150, 230],  
                 [40, 100, 150], [150, 50, 150], [250, 50, 250] ],
                'ice_sheet_diff': [ [250, 0, 0], [230, 100, 40],[200, 150, 100],[230, 200, 160],[255, 255, 255],  [255, 255, 255], 
                          [255, 255, 255], [100, 150, 230], [40, 100, 150], [150, 50, 150], [250, 50, 250]],
                'icover': [[255, 255, 255], [160, 230, 230],[100, 150, 230],  
                 [40, 100, 150], [150, 50, 150], [250, 50, 250] ],
                'gmask': [ [0,0,0],[0,0,250],[255,255,255],[0,250,250],  [200,120,0],[255,255,0] ],
                'precip': [[ 245, 245, 245],[ 175, 237, 237],[ 152, 251, 152],[  67, 205, 128],[  59, 179, 113],[ 250, 250, 210],
                           [ 255, 255,   0],[ 255, 164,   0],[ 255,   0,   0],[ 205,  55,   0],[ 199,  20, 133],[ 237, 130, 237]],
                 'vel': [[255, 255, 255],[250, 250, 200],[250, 240, 120],[250, 210, 0], 
                [250, 175, 0],[250, 120, 0],[250, 40, 0],[220, 0, 40],[200, 0, 100],[180, 0, 150],[250, 0, 250]],
                 'velo': [[255, 255, 255],[250, 250, 200],[250, 240, 120],[250, 210, 0], 
                [250, 175, 0],[250, 120, 0],[250, 40, 0],[220, 0, 40],[200, 0, 100],[180, 0, 150],[250, 0, 250]]
               }
        keytab        = self.ColorCustom(clevs,cbar=cbar[varname])
        if varname == 'precip':
            keytab    = self.ColorNorm(clevs,cmap_base='BrBG',inv=False)
        self.colormap = keytab
        return
    
    def ColorNorm(self,clevs,cmap_base='RdBu',inv=True,omask=False,nowhite=False):
        import numpy as np
        import matplotlib as mpl
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        colormap_dict = {}
        levels = clevs
        num    = len(levels) 
        base   = cm.get_cmap(cmap_base, num)
        if(inv):
            cbase = base(-np.linspace(-1, 0, num))
        else:
            cbase = base(np.linspace(0, 1, num))

        if(omask):
            cbase  = cbase[int(num/2):,:]
            cbase[0,:] = [1,1,1,1]
            levels = levels[int(num/2):]
        elif(not nowhite):
            cbase[[int(num/2),int(num/2)+1],:] = [1,1,1,1]
            
        boundaries = levels
        colormap_dict['cmap'] = ListedColormap(cbase)
        colormap_dict['norm'] = mpl.colors.BoundaryNorm(boundaries, colormap_dict['cmap'].N)
        self.colormap         = colormap_dict
        return colormap_dict
    
    def ColorCustom(self,clevs, cbar=None):
        import numpy as np
        import matplotlib as mpl
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        colormap_dict = {}
        levels = clevs
        num    = len(levels) 
        if cbar is None:
            clist  = {'red'  : [ [0, 1, 1], [0.5, 1, 1], [1, 0, 0] ],
                      'green': [ [0, 0, 0], [0.5, 1, 1], [1, 0, 0] ],
                      'blue' : [ [0, 0, 0], [0.5, 1, 1], [1, 1, 1] ]
                     }
        else:
            carray  = np.array(cbar)/255.
            csize  = carray.shape[0] 
            divid  = np.linspace(0,1,csize)
            clist  = {'red'  : [list(x) for x in zip(divid,carray[:,0],carray[:,0])],
                    'green'  : [list(x) for x in zip(divid,carray[:,1],carray[:,1])],
                     'blue'  : [list(x) for x in zip(divid,carray[:,2],carray[:,2])]
                     }
        boundaries = levels
        colormap_dict['cmap'] = LinearSegmentedColormap('myclr',clist,num+1) 
        colormap_dict['norm'] = mpl.colors.BoundaryNorm(boundaries, colormap_dict['cmap'].N-2)
        self.colormap         = colormap_dict
        return colormap_dict

################################# plot setting ########################################
    def DrawGeoMap(self,var,axin=None,keytab={},gxout='pcolor'):
    # draw geocoordinate map, var is assumed to be an xarray
    #da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        import xarray as xr
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import matplotlib.path as mpath
        from cartopy.util import add_cyclic_point
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#         # add cyclic point
#         data,lon = add_cyclic_point(var_ori, coord=var_ori.lon)
#         cords    = {key:var_ori.coords[key].values for key in var_ori.coords }
#         cords['lon'] = lon
#         var      = xr.DataArray(data=data,dims=var_ori.dims,coords=cords,attrs=var_ori.attrs)
        
        # circle boundary
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        
        im    = None
        if gxout == 'pcolor':
            if hasattr(self,'colormap'):
                keytab.update(self.colormap)
            im = var.plot(ax=axin,transform=ccrs.PlateCarree(),**keytab, extend='both')
        elif gxout == 'contourf':
            if hasattr(self,'colormap'):
                keytab.update(self.colormap)
            im = var.plot.contourf(ax=axin,transform=ccrs.PlateCarree(),**keytab, extend='both')
        elif gxout == 'contour':
            im = var.plot.contour(ax=axin,transform=ccrs.PlateCarree(),**keytab)

        axin.coastlines(alpha=0.3)
        if isinstance(self.proj, ccrs.NorthPolarStereo) or isinstance(self.proj, ccrs.SouthPolarStereo):
            gl = axin.gridlines(alpha=0.3,draw_labels=False)
            axin.set_boundary(circle, transform=axin.transAxes)
        else:
            gl = axin.gridlines(alpha=0.3,draw_labels=True)
    #        axin.outline_patch.set_visible(False)
            gl.top_labels = False
    #        gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 14, 'color': 'black'}
            gl.ylabel_style = {'size': 14, 'color': 'black'}
            self.StringMark(axin)
        return axin, im
    
    def DrawProfile(self,varin,axin=None,keytab={},gxout='pcolor',zscale='log'):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.labelsize'] = 16
        # draw vertical profile
        var    = varin.copy()
        coord  = var.dims
        zname  = coord[0]; xname = coord[-1]
        # vertical coordinate
        zrev = False; 
        if var[zname].attrs['units'] == 'Pa':
            var = var.assign_coords({zname:var[zname]/100})
            var[zname].attrs.update({'units':'hPa','long_name':'Pressure'})
        elif var[zname].units == 'm':
            zrev = True; zscale = 'linear'
        
        keytab.update({'yscale':zscale,'yincrease':zrev}) 
        
        # draw plot
        if gxout == 'pcolor':
            if hasattr(self,'colormap'):
                keytab.update(self.colormap)
            im = var.plot(ax=axin,**keytab, extend='both')
        elif gxout == 'contourf':
            if hasattr(self,'colormap'):
                keytab.update(self.colormap)
            im = var.plot.contourf(ax=axin,**keytab, extend='both')
        elif gxout == 'contour':
            im = var.plot.contour(ax=axin,**keytab, extend='both')
            axin.clabel(im, inline=True, fontsize=12)
        
        # axes label setting
        #axin.set_xticks(np.linspace(-90,90,7))
        axin.set_yticks(np.linspace(900,100,9))
        axin.set(**FigCustom.AxesLabel(axin,var[xname],'x'))
        axin.set(**FigCustom.AxesLabel(axin,var[zname],'y'))
        axin.xaxis.label.set_size(14)
        axin.yaxis.label.set_size(14)
        axin.tick_params(labelsize=12)
        self.StringMark(axin)
      
        return axin, im            
      
    def AxesLabel(axin,xrarray,tickaxis):
        # set axes labels, specially for longitude, latitude
        # axes, coordinate variable and axis label (x/y) is required as input
        # the output is a dictionary with axis label, ticks and tick labels
        ticks_old = eval('axin.get_%sticks()'%tickaxis)
        tlabs_lst = eval('axin.get_%sticklabels()'%tickaxis)
        if xrarray.attrs['units'] == 'degrees_north':
            label     = 'Latitude'
            ticks_new = [ticks for ticks in ticks_old if np.abs(ticks)<90]
            ticks_lab = ['%.f$^o$N'%ticks       if ticks > 0                     else 
                         '%.f$^o$S'%(-ticks)    if ticks < 0                     else 
                          str(ticks)                          for ticks in ticks_new]
#         elif xrarray.attrs['units'] == 'degrees_east':
#             label     = 'Longitude'
#             ticks_new = ticks_old
#             ticks_lab = ['%.f$^o$E'%ticks       if ((ticks>  0) and (ticks<180)) else 
#                          '%.f$^o$W'%(-ticks)    if   ticks<  0                   else 
#                          '%.f$^o$W'%(360-ticks) if ((ticks>360) and (ticks<180)) else 
#                           str(ticks)                          for ticks in ticks_new]
        else:
            label     = '%s[%s]'%(xrarray.attrs['long_name'],xrarray.attrs['units'])
            ticks_new = ticks_old
            ticks_lab = [str(x) for x in ticks_new]
            
        tickdict = {'%sticks'%tickaxis:ticks_new,
                    '%sticklabels'%tickaxis:ticks_lab,
                    '%slabel'%tickaxis:label
                   }
        return tickdict 
    
    def show(self):
        from IPython.display import display
        display(self)
        return

# constant and function for Geoscience calculation 
import numpy as np
class GeoConst():
    # constant in Geoscience
    g = 9.8                            # gravitational acceleration [m s^-2]
    R_air = 287;                       # gas constant for dry air [J/mol/K]
    R_vapor = 461;                     # gas constant for water vapor [J/mol/K]
    Cp_air = 1004;                     # specific heat capacity of air [J/kg/K]
    cq_latent = 2.257e6;               # latent heat of condensation/evapoartion f water [J/kg]
    omega = 7.292e-5                   # the angular speed for Earth rotation [rad/s]
    earthr = 6.371e6                   # Earth radius [m]
    
class GeoFunction():
    def conform(data_from,data_to,drop=[]):
        # expand DataArray data (data_from) to a larger dimension data (dimension of data_to)
        if not isinstance(drop,list):
            dropdim = [drop]
        else:
            dropdim = drop
        dim_to   = data_to.coords
        dim_from = data_from.coords
        dim_exp  = list(set(dim_to.dims) - set(dim_from.dims) - set(dropdim))
        data_res = data_from.expand_dims({x:dim_to[x] for x in dim_exp})
        return data_res
    
    def Hint(data,lat):
        cst       = GeoConst() 
        lat_mdim  = GeoFunction.conform(lat,data)
        weight    = cst.earthr**2*np.cos(lat_mdim*np.pi/180.)


    def Vint(data,pt,pb,levname='plev'):
        import xarray as xr
        # vertical integration in weighted with air mass under pressure level coordinate (trapezoidal quadrature) 
        pb_mdim   = GeoFunction.conform(pb,data)                # integration bottom boundary
        pt_mdim   = GeoFunction.conform(pt,data)                # integration top boundary
        p_mdim    = GeoFunction.conform(data[levname],data)     # pressure level in data
        
        # pressure level replace maximum and minimum with bottom and top boundary
        p_mdim_bd = p_mdim.where(p_mdim >= pt_mdim, pt_mdim)   
        p_mdim_bd = p_mdim_bd.where(p_mdim <= pb_mdim, pb_mdim)
        
        p_edge    = (p_mdim_bd.isel({levname:slice(0,-2)}).to_numpy()+p_mdim_bd.isel({levname:slice(1,-1)}).to_numpy())/2. # middle point for pressure level in data
        pb_edge   = p_mdim_bd.copy()
        pt_edge   = p_mdim_bd.copy()
        pb_edge[{levname:slice(1,-1)}] = p_edge # integration interval lower edge
        pt_edge[{levname:slice(0,-2)}] = p_edge # integration interval upper edge
        p_weight  = pt_edge - pb_edge           # pressure weight for integration

        data_vint = -(data*p_weight).sum(levname)/GeoConst.g # vertical integration with air mass: -1/g int_{pb}^{pt} data dp

        return data_vint
    
    # function to calculate quantity
    def PotentialTemp(T,p):
        # calculate potential temperature
        cst   = GeoConst() 
        p0    = 100000.
        pwgt  = GeoFunction.conform(p,T)
        theta = T*(p0/p)**(cst.R_air/cst.Cp_air)
        
        return theta
        
        
    def LapseRate(T,p,atmtype='env'):
        cst       = GeoConst() 
        epsiron   = 0.622 # ratio of the molecular weight of water to that of dry air
        # calculate lapse rate by g/R dlnT/dlnp
        dlnTdlnp  = np.log(T).differentiate('plev')/np.log(p).differentiate('plev')
        #dlnTdlnp  = np.log(T).diff('plev')/np.log(p).diff('plev')
        lapse     = cst.g/cst.R_air*dlnTdlnp*1000 
        lapse.attrs = {'long_name':'Lapse Rate','units':'K km$^{-1}$'} 
        lapse_dry = cst.g/cst.Cp_air*1000 # units: K km$^{-1}$
        
        if atmtype=='dry':
            lapse     = lapse_dry
            #lapse.attrs = {'long_name':'Dry Lapse Rate','units':'K km$^{-1}$'} 
        elif atmtype=='moist':
            qs        = GeoFunction.SaturatedHumidity(T,p)
            lapse_wet = lapse_dry*(1+        cst.cq_latent   *qs/(           cst.R_air*T   )) / \
                                  (1+epsiron*cst.cq_latent**2*qs/(cst.Cp_air*cst.R_air*T**2))
            lapse     = lapse_wet
            lapse.attrs = {'long_name':'Moist Lapse Rate','units':'K km$^{-1}$'} 
        
        return lapse    
    
    def Magnus(T): # saturated vapor pressure
        # Magus equation for saturated (Bolton 1980), temperature unit: K, vapor pressure unit: Pa 
        Es = 611.2*np.exp(17.08085*(T-273.15)/(T-273.15+243.5))
        return Es
    
    def SaturatedHumidity(T,presure):
        # Specific humidity 
        p  = GeoFunction.conform(presure,T)
        Es = GeoFunction.Magnus(T)
        qs = 0.622*Es/(p-0.378*Es)
        return qs
    
    def StaticStability(T,p,atmtype='dry'):
        cst       = GeoConst()
        lapse     = GeoFunction.LapseRate(T,p,atmtype='env')
        stability = GeoFunction.LapseRate(T,p,atmtype=atmtype) - lapse
        stability.attrs = {'long_name':'%s Static Stability'%atmtype.title,'units':'K km$^{-1}$'}
        return stability
    
    def EqPotTemp(T,presure,state='Saturated'):
        # Juckers 2000 equivalent potential temperature
        cst     = GeoConst() 
        p       = GeoFunction.conform(presure,T)
        Es      = GeoFunction.Magnus(T) 
        Teq     = T + cst.R_air/cst.R_vapor*cst.cq_latent/(cst.Cp_air*p)*Es # specific humidity
        thetaeq = GeoFunction.PotentialTemp(Teq,p)
        return thetaeq
    
#     def Baroclinicity(T,U,p,plev_tb=[5e4,7e4]):
#         cst  = GeoConst() 
#         pot  = PotentialTemp(T,p)
#         lat  = T.lat
#         f    = 2*cst.omega*np.sin(lat/180*np.pi)
#         dU   = U.sel(plev=plev_tb[0],method='nearest')-U.sel(plev=plev_tb[1],method='nearest')
#         dlnT = pot.sel(plev=plev_tb[0],method='nearest')-pot.sel(plev=plev_tb[1],method='nearest')
#         sig  = 0.31*cst.g*f*np.abs(dU)/np.sqrt((plev_tb[0]-plev_tb[1])*dlnT
#         return
   
