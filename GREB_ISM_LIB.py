# GREB-ISM libary used for analysis
class Gvar:
    def __init__(self):
        from py3grads import Grads
        self.ga = Grads(verbose=False)
        self.fname = ''
        return
        
    def set(self,envlist):
        for k in range(len(envlist)):
            self.ga('set %s'%envlist[k])
        return

    def fopen(self,fname,num,fopen_flag=''):
        # change accordingly based on output directories
        self.fname = fname
        self.ga('./fopen.gs %s %d %s'%(self.fname,num,fopen_flag))
        return

    def hems(self,varname,label):
        if label=='g':
            axlist = [-180,180,-90,90]
        elif label=='n':
            axlist = [-180,180,0,90]
        elif label=='s':
            axlist = [-180,180,-90,0]
        else:
            axlist = label 
        cal_exp   = 'tloop(aave(%s,lon=%f,lon=%f,lat=%f,lat=%f))' \
                  % (varname,axlist[0],axlist[1],axlist[2],axlist[3])
        return self.ga.exp(cal_exp)
    
    def icev(self,area,var='iceh'):
        import numpy as np
        cord_env,vdim,varlist = self.cord()
        if var in set(varlist):
            # calculate ice volume, unit: 10^6 km^3
            if(area == 'g'):
               cal   = 'tloop(atot(%s,lon=-180,lon=180,lat=-90,lat=90)*6370*6370/1.e9)'%var
            elif(area == 'n'):
               cal   = 'tloop(atot(%s,lon=-180,lon=180,lat=0,lat=90)*6370*6370/1.e9)'%var
            elif(area == 's'):
               cal   = 'tloop(atot(%s,lon=-180,lon=180,lat=-90,lat=0)*6370*6370/1.e9)'%var
            else:
               cal   = 'tloop(atot(%s,lon=%f,lon=%f,lat=%f,lat=%f)*6370*6370/1.e9)'%(var,area[0],area[1],area[2],area[3])
            ice_volume = self.ga.exp(cal)
        else:
            ice_volume = np.arange(vdim[-1])*np.nan
        return ice_volume

    def xarray(self,var):
        import xarray as xr
        import numpy as np
        # cord_all,var  = self._ctlresolve(ctlinfo)
        if not isinstance(var,list):
            var = [var]                                                          # build list if input is one variable
        cord_env,vdim,varlist = self.cord()                                      # coordinate and variables information 
        # all variables in GrADS envrironment including self defined
        def_info      = self.ga('q define')[0]
        if def_info[0] == 'No Defined Variables':
            def_vars      = []
        else:
            def_vars      = [x.split()[0] for x in def_info[:-1]]
        varlist      += def_vars
        
        variable_list = []
        for v in var:
            if v in set(varlist):
                dvalue = np.reshape(self.ga.exp(v),vdim)                          # set variable 
            else:
                dvalue = np.zeros(vdim)*np.nan                                    # set NaNs if variable not defined in the file
                print('No \'%s\' in \'%s\', all NaNs instead'%(v,self.fname))
            dvalue = dvalue.transpose([3,2,0,1])                                  # change dimensions for [time,lev,lon,lat]
            if(dvalue.shape[-1]>cord_env['lon'].shape[0]):
                cord_env['lon']=np.append(cord_env['lon'],cord_env['lon'][0]+360) # full longitude case
            da = xr.DataArray(name=v, data=dvalue,
                          dims=['time','lev', 'lat', 'lon'],
                          coords=cord_env)
            variable_list.append(da)
        
        dataset = xr.merge(variable_list) 
        return dataset
    
    def _ctlresolve(self, ctlinfo):
        import re
        ctl_cord = {}
        var      = {}
        var_list = []
        var_strt = False
        for inline in ctlinfo:
            if inline.lower().startswith('xdef'):
                ctl_cord['lon'] = self._coordinate_variable_extract(inline) 
            elif inline.lower().startswith('ydef'):
                ctl_cord['lat'] = self._coordinate_variable_extract(inline) 
            elif inline.lower().startswith('zdef'):
                ctl_cord['lev'] = self._coordinate_variable_extract(inline) 
            elif inline.lower().startswith('tdef'):
                ctl_cord['time'] = self._coordinate_time_extract(inline) 
            elif inline.lower().startswith('vars'):
                tokens = inline.lower().split()
                var_strt = True; 
            elif inline.lower().startswith('endvars'):
                break
            elif var_strt:
                sdftab = re.compile('.*=>')
                tokens = inline.lower().split()
                var = {'name':sdftab.sub('',tokens[0]),
                       'lev':int(tokens[1]),
                       'comment': tokens[3:]}
                var_list.append(var)

        return ctl_cord,var_list

    def cord(self):
        import numpy as np
        ctlinfo_origin = self.ga('q ctlinfo')
        ctlinfo        = []
        fullstr        = '' 
        for it in ctlinfo_origin[0]:
            if it == '':
                ctlinfo.append(fullstr)
                continue
            try:
                float(it.split()[0])
            except ValueError:
                ctlinfo.append(fullstr)
                fullstr = ''
            fullstr = fullstr + it
        ctlinfo.remove('')
        cord_all,var0 = self._ctlresolve(ctlinfo)
        envinfo       = self.ga('q dim')
        env_cord = {}
        varlist  = [var['name'] for var in var0]
        vdim     = [0]*4
        for inline in envinfo[0]:
            if inline.lower().startswith('x'):
                lonlist,vdim[1]          = self._environment_coordinate_extract(inline,cord_all['lon']) 
                env_cord['lon']          = lonlist
                env_cord['lon'][:-1]     = [x - 360 if x >= lonlist[-1] else x for x in lonlist[:-1]]
            if inline.lower().startswith('y'):
                latlist = np.append(cord_all['lat'],[90,-90]) # deal with boundary point in latitude, latlist[-1]=-90 and latlist[end]=90 
                env_cord['lat'],vdim[0]  = self._environment_coordinate_extract(inline,latlist) 
            if inline.lower().startswith('z'):
                env_cord['lev'],vdim[2]  = self._environment_coordinate_extract(inline,cord_all['lev']) 
            if inline.lower().startswith('t'):
                env_cord['time'],vdim[3]  = self._environment_coordinate_extract(inline,cord_all['time']) 
            
        return env_cord, vdim, varlist

    def _coordinate_variable_extract(self,inline):
        import numpy as np
        tokens = inline.lower().split()
        num    = int(tokens[1])
        if   tokens[2] == 'linear': 
            start, intv = float(tokens[3]), float(tokens[4])
            cord_data = np.linspace(start,start + intv * (num-1),
                              num, dtype=np.float32)
                                
        elif tokens[2] == 'levels': 
            cord_data = np.array([x for x in tokens[3:]],dtype=np.float32) 

        else: raise Exception('error for '+tokens[0])
        return cord_data 
    
    def _environment_coordinate_extract(self,inline, cord_ctl):
        import numpy as np
        import math 
        tokens = inline.lower().split()
        vdim   = 0
        if   tokens[2] == 'varying': 
            start = math.floor(float(tokens[10]) - 1); end = math.ceil(float(tokens[12])) 
            if(start < 0):
                ind   = [x for x in range(start,end)]
                cord_data = cord_ctl[ind]
            else:
                
                cord_data = cord_ctl[start:end]
            vdim      = end - start 
                                
        elif tokens[2] == 'fixed': 
            start = int(tokens[8]) - 1 
            cord_data = np.array([cord_ctl[start]])
            vdim  = 1
        else: raise Exception('error for '+tokens[0])

        return cord_data, vdim 
    
    def _coordinate_time_extract(self,inline):
        import pandas as pd
        import numpy as np
        def gtime_conv(tstr):
            monlist = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06',
                       'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
            year = tstr[-4:]; mon = monlist[tstr[-7:-4]]; day = '01'; hour = '00' 
            if(len(tstr) == 9): 
                day = tstr[-9:-7]
            if(len(tstr) == 12): 
                day = tstr[-9:-7];hour = tstr[-12:-10]
            return '%s-%s-%s'%(year,mon,day)

        tokens = inline.lower().split()
        num    = int(tokens[1])
        if   tokens[2] == 'linear':
            intv_dict = {'yr':'Y','mo':'M','dy':'D','hrs':'H','mn':'min'}
            intv  = '%s%s'%(tokens[4][:-2],intv_dict[tokens[4][-2:]])
            cord_data = pd.period_range(gtime_conv(tokens[3]),periods=num,freq=intv)
                                
        elif tokens[2] == 'levels': 
            cord_data  = np.array([x for x in tokens[3:]],dtype='period') 

        else: raise Exception('error for '+tokens[0])
        return cord_data 

class ImParam:
    def ColorNorm(clevs,cmap_base='RdBu',inv=True,omask=False,nowhite=False):
        import numpy as np
        import matplotlib as mpl
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        picpar = {}
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
        picpar['cmap'] = ListedColormap(cbase)
        picpar['norm'] = mpl.colors.BoundaryNorm(boundaries, picpar['cmap'].N)
        return picpar
    
    def ColorCustom(clevs, cbar=None):
        import numpy as np
        import matplotlib as mpl
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        picpar = {}
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
        picpar['cmap'] = LinearSegmentedColormap('myclr',clist,num+1) 
        picpar['norm'] = mpl.colors.BoundaryNorm(boundaries, picpar['cmap'].N-2)
        return picpar

    def DrawGeoMap(var,axin=None,keytab={},mproj=None,subplt=(1,1,1),gxout='pcolor'):
    # draw geocoordinate map, var is assumed to be an xarray
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        if mproj is None:
            mproj = ccrs.Robinson()
        if axin is None:
            axin   = plt.subplot(subplt[0],subplt[1],subplt[2],projection=mproj)
        if gxout == 'pcolor':
            im = var.plot(ax=axin,transform=ccrs.PlateCarree(),**keytab, extend='both')
        elif gxout == 'contourf':
            im = var.plot.contourf(ax=axin,transform=ccrs.PlateCarree(),**keytab, extend='both')

        axin.coastlines(alpha=0.3)
        axin.gridlines(alpha=0.3)
        #axin.outline_patch.set_visible(False)
        return axin, im
        
    def DrawGeoMapSimple(var,keytab={},subkws=None):
    # draw geocoordinate map, var is assumed to be an xarray
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        if subkws is None:
            subkws = {'projection':ccrs.Robinson()}
        geoax,im = var.plot(transform=ccrs.PlateCarree(),**keytab, extend='both',subplot_kws=subkws)
        for axin in geoax.axes.flat:
            axin.coastlines(alpha=0.3)
            axin.gridlines()
        return im
        
