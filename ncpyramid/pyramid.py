# The MIT License (MIT)
# Copyright (c) 2016, 2017 by the ESA CCI Toolbox development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import time
from typing import Optional
import json

import xarray as xr
import numpy as np

MODE_LE = -1
MODE_EQ = 0
MODE_GE = 1


def write_pyramid(input_file: str,
                  output_dir: str = '.',
                  output_name: Optional[str] = None,
                  write_fr: bool = False,
                  tile_width: Optional[int] = None,
                  tile_height: Optional[int] = None):
    basename, ext = os.path.splitext(os.path.basename(input_file))
    if output_name is None or output_name.strip() == '':
        output_name = basename + '.pyramid'

    target_dir = os.path.join(output_dir, output_name)

    os.makedirs(target_dir, exist_ok=True)

    ds = xr.open_dataset(input_file)

    lon0, lat0, delta_lon, delta_lat = get_geo_rectangle_from_dataset(ds, eps=1e-4)
    print((lon0, lat0, delta_lon, delta_lat))

    w_max, h_max = -1, -1

    for var_name in ds.data_vars:
        var = ds[var_name]
        # print(var_name, var.dims, var.shape)
        if is_spatial_var(var):
            w, h = var.shape[-1], var.shape[-2]
            if w_max == -1 or (w >= w_max and h >= h_max):
                w_max = w
                h_max = h

    if w_max == -1 or h_max == -1:
        raise ValueError('no spatial variables found')

    print('maximum size: {w} x {h} cells'.format(w=w_max, h=h_max))

    selected_var_names = []
    for var_name in ds.data_vars:
        var = ds[var_name]
        if is_spatial_var(var):
            w, h = var.shape[-1], var.shape[-2]
            if w == w_max and h == h_max:
                selected_var_names.append(var_name)
            else:
                print('warning: variable {v} not included, wrong size'.format(v=var_name))
        else:
            print('warning: variable {v} not included, not spatial'.format(v=var_name))

    w_mode = MODE_GE
    if delta_lon == 360.:
        w_mode = MODE_EQ

    h_mode = MODE_GE
    if delta_lat == 180. or lat0 == -90.:
        h_mode = MODE_EQ

    (w, h), (tw, th), (nt0x, nt0y), nl = rect_subdivision(w_max, h_max, w_mode=w_mode, h_mode=h_mode,
                                                          tw_opt=min(w_max, tile_width or 512),
                                                          th_opt=min(h_max, tile_height or 512))

    print('number of pyramid levels: {nl}'.format(nl=nl))
    print('number of tiles at level zero: {nx} x {ny}'.format(nx=nt0x, ny=nt0y))
    print('pyramid tile size: {tw} x {th}'.format(tw=tw, th=th))
    print('image size at level zero: {w} x {h}'.format(w=nt0x * tw, h=nt0y * th))
    print('image size at level {k}: {w} x {h}'.format(k=nl - 1, w=w, h=h))

    ds.close()
    import math

    new_delta_lon = w * delta_lon / w_max
    new_delta_lat = h * delta_lat / h_max

    with open(os.path.join(target_dir, 'tiling-scheme.json'), 'w') as fp:
        json.dump(dict(numberOfLevelZeroTilesX=nt0x,
                       numberOfLevelZeroTilesY=nt0y,
                       tileWidth=tw,
                       tileHeight=th,
                       minimumLevel=0,
                       maximumLevel=nl - 1,
                       rectangle=dict(west=math.radians(lon0),
                                      south=math.radians(lat0),
                                      east=math.radians((180. + lon0 + new_delta_lon) % 360. - 180.),
                                      north=math.radians(lat0+new_delta_lat))
                       ), fp, indent=4)

    ds = xr.open_dataset(input_file, chunks=dict(lon=10*tw, lat=10*th))
    ds_orig = ds

    for var_name in selected_var_names:
        var = ds[var_name][...].chunk(dict(lon=tw, lat=th))
        var.encoding['chunksizes'] = get_chunk_sizes(var, tw, th)
        # print(downsampled_var.encoding)
        ds[var_name] = var

    t0 = time.clock()

    k = nl - 1
    if write_fr:
        print('writing full-res dataset at level {k}'.format(k=k))
        ds.to_netcdf(os.path.join(target_dir, 'L{k}.nc'.format(k=k)), format='netCDF4', engine='netcdf4')
        print('done after {dt} seconds'.format(dt=time.clock() - t0))
    else:
        print('write link to full-res dataset at level {k}'.format(k=k))
        with open(os.path.join(target_dir, 'L{k}.nc.lnk'.format(k=k)), 'w') as fp:
            fp.write(input_file)

    for i in range(1, nl):
        k = nl - 1 - i

        coords = dict(ds.coords)
        coords['lon'] = ds.coords['lon'][::2]
        coords['lat'] = ds.coords['lat'][::2]

        data_vars = dict()
        for var_name in selected_var_names:
            var = ds[var_name]
            downsampled_var = var[..., ::2, ::2]
            downsampled_var.encoding['chunksizes'] = get_chunk_sizes(var, tw, th)
            # print(downsampled_var.encoding)
            data_vars[var_name] = downsampled_var

        print('constructing lower-res dataset at level {k}'.format(k=k))
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_orig.attrs)
        print('writing lower-res dataset at level {k}...'.format(k=k))
        t1 = time.clock()
        ds.to_netcdf(os.path.join(target_dir, 'L{k}.nc'.format(k=k)), format='netCDF4', engine='netcdf4')
        print('done after {dt} seconds'.format(dt=time.clock() - t1))

    ds_orig.close()

    print('pyramid "{n}" written within {dt} seconds'.format(n=output_name, dt=time.clock() - t0))

    return target_dir


def get_geo_rectangle_from_dataset(ds: xr.Dataset, eps=1e-6):
    lon = ds.coords['lon']
    lat = ds.coords['lat']
    return get_geo_rectangle(lon, lat, eps=eps)


def get_geo_rectangle(lon, lat, eps=1e-6):
    dlon = np.gradient(lon)
    if (dlon.max() - dlon.min()) >= eps:
        lon = np.where(lon < 0., 360. + lon, lon)
        dlon = np.gradient(lon)
        if (dlon.max() - dlon.min()) >= eps:
            raise ValueError('coordinate variable "lon" not is not equi-distant')

    dlat = np.gradient(lat)
    if (dlat.max() - dlat.min()) >= eps:
        raise ValueError('coordinate variable "lat" not is not equi-distant')

    lon1 = lon[0] - 0.5 * dlon[0]
    lon2 = lon[-1] + 0.5 * dlon[0]
    if dlat[0] > 0.0:
        lat1 = lat[0] - 0.5 * dlat[0]
        lat2 = lat[-1] + 0.5 * dlat[0]
    else:
        lat1 = lat[-1] + 0.5 * dlat[0]
        lat2 = lat[0] - 0.5 * dlat[0]

    if lon1 < lon2:
        width = lon2 - lon1
    else:
        width = 360 + lon2 - lon1
    height = lat2 - lat1
    if abs(360. - width) < eps:
        lon1 = -180.
        width = 360.
    if abs(180. - height) < eps:
        lat1 = -90.
        height = 180.
    return lon1, lat1, width, height


def get_chunk_sizes(var, tile_width, tile_height):
    chunk_sizes = len(var.shape) * [1]
    chunk_sizes[-1] = tile_width
    chunk_sizes[-2] = tile_height
    return chunk_sizes


def is_spatial_var(var):
    shape = var.shape
    dims = var.dims
    return len(shape) >= 2 and len(shape) == len(dims) \
           and dims[-1] == 'lon' and dims[-2] == 'lat'


def rect_subdivision(w_min: int, h_min: int,
                     w_mode: int = MODE_EQ, h_mode: int = MODE_EQ,
                     tw_opt: Optional[int] = None, th_opt: Optional[int] = None,
                     tw_min: Optional[int] = None, th_min: Optional[int] = None,
                     tw_max: Optional[int] = None, th_max: Optional[int] = None,
                     nt0_max: Optional[int] = None,
                     nl_max: Optional[int] = None):
    subdivs_w = size_subdivisions(w_min, s_mode=w_mode, ts_opt=tw_opt, ts_min=tw_min, ts_max=tw_max, nt0_max=nt0_max,
                                  nl_max=nl_max)
    subdivs_h = size_subdivisions(h_min, s_mode=h_mode, ts_opt=th_opt, ts_min=th_min, ts_max=th_max, nt0_max=nt0_max,
                                  nl_max=nl_max)
    print(subdivs_w)
    print(subdivs_h)
    if not subdivs_w or not subdivs_h:
        raise ValueError(
            'size {w} x {h} does not allow for pyramid subdivision with given constraints'.format(w=w_min, h=h_min))

    w_max, tw, nt0_w, nl_w = subdivs_w[0]
    h_max, th, nt0_h, nl_h = subdivs_h[0]
    if w_mode:
        assert w_min == w_max and h_min == h_max

    if nl_w < nl_h:
        nl = nl_w
        nt0_h = h_max // (1 << (nl - 1)) // th
    elif nl_w > nl_h:
        nl = nl_h
        nt0_w = w_max // (1 << (nl - 1)) // tw
    else:
        nl = nl_w

    return (w_max, h_max), (tw, th), (nt0_w, nt0_h), nl


def size_subdivisions(s_act: int,
                      s_mode: int = MODE_EQ,
                      ts_opt: Optional[int] = None,
                      ts_min: Optional[int] = None,
                      ts_max: Optional[int] = None,
                      nt0_max: Optional[int] = None,
                      nl_max: Optional[int] = None):
    if s_act < 1:
        raise ValueError('invalid s_act')

    ts_min = ts_min or min(s_act, (ts_opt // 2 if ts_opt else 200))
    ts_max = ts_max or min(s_act, (ts_opt * 2 if ts_opt else 1200))
    nt0_max = nt0_max or 8
    nl_max = nl_max or 16

    if ts_min < 1:
        raise ValueError('invalid ts_min')
    if ts_max < 1:
        raise ValueError('invalid ts_max')
    if ts_opt < 1:
        raise ValueError('invalid ts_opt')
    if nt0_max < 1:
        raise ValueError('invalid nt0_max')
    if nl_max < 1:
        raise ValueError('invalid nl_max')

    subdivisions = []
    for ts in range(ts_min, ts_max + 1):
        s_max_min = s_act if s_mode else s_act - (ts - 1)
        s_max_max = s_act if s_mode else s_act + (ts - 1)
        for nt0 in range(1, nt0_max):
            s_max = nt0 * ts
            if s_max > s_max_max:
                break
            for nl in range(2, nl_max):
                s_max = (1 << (nl - 1)) * nt0 * ts
                ok = False
                if s_mode == MODE_GE:
                    if s_max >= s_act:
                        if s_max > s_max_max:
                            break
                        ok = True
                elif s_mode == MODE_LE:
                    if s_act >= s_max >= s_max_min:
                        ok = True
                elif s_max == s_act:
                    ok = True
                if ok:
                    rec = s_max, ts, nt0, nl
                    subdivisions.append(rec)

    # maximize nl
    subdivisions.sort(key=lambda rec: rec[3], reverse=True)
    if ts_opt:
        # minimize |ts - ts_opt|
        subdivisions.sort(key=lambda rec: abs(rec[1] - ts_opt))
    # minimize nt0 * ts
    subdivisions.sort(key=lambda rec: rec[2] * rec[1])
    # minimize s_max - s_min
    subdivisions.sort(key=lambda rec: rec[0] - s_act)

    return subdivisions
