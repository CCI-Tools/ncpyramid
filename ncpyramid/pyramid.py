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


import json
import os
import time
from typing import Optional

import xarray as xr

from ncpyramid.geospatialrect import get_geo_spatial_rect
from ncpyramid.tilingscheme import TilingScheme


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

    x_dim, y_dim = None, None
    w_max, h_max = -1, -1

    for var_name in ds.data_vars:
        var = ds[var_name]
        # print(var_name, var.dims, var.shape)
        dims = get_spatial_dims(var)
        if dims:
            w, h = var.shape[-1], var.shape[-2]
            if w_max == -1 or (w >= w_max and h >= h_max):
                w_max, h_max = w, h
                x_dim, y_dim = dims

    if w_max == -1 or h_max == -1:
        raise ValueError('no spatial variables found')

    print('maximum size: {w} x {h} cells'.format(w=w_max, h=h_max))

    selected_var_names = []
    for var_name in ds.data_vars:
        var = ds[var_name]
        dims = get_spatial_dims(var)
        if dims:
            w, h = var.shape[-1], var.shape[-2]
            if w == w_max and h == h_max:
                selected_var_names.append(var_name)
            else:
                print('warning: variable {v} not included, wrong size'.format(v=var_name))
        else:
            print('warning: variable {v} not included, not spatial'.format(v=var_name))

    geo_spatial_rect = get_geo_spatial_rect(ds.coords[x_dim], ds.coords[y_dim], eps=1e-4)
    print(geo_spatial_rect)

    tiling_scheme = TilingScheme.create(w_max, h_max, tile_width, tile_height, geo_spatial_rect)

    print(tiling_scheme)

    ds.close()
    with open(os.path.join(target_dir, 'tiling-scheme.json'), 'w') as fp:
        json.dump(tiling_scheme.to_cesium_json(), fp, indent=4)

    chunks = {x_dim: min(w_max, 10 * tiling_scheme.tile_width), y_dim: min(h_max, 10 * tiling_scheme.tile_height)}
    ds = xr.open_dataset(input_file, chunks=chunks)
    ds_orig = ds

    for var_name in selected_var_names:
        var = ds[var_name][...].chunk(chunks)
        var.encoding['chunksizes'] = get_chunk_sizes(var, tiling_scheme.tile_width, tiling_scheme.tile_height)
        # print(downsampled_var.encoding)
        ds[var_name] = var

    t0 = time.clock()

    k = tiling_scheme.num_levels - 1
    if write_fr:
        print('writing full-res dataset at level {k}'.format(k=k))
        ds.to_netcdf(os.path.join(target_dir, 'L{k}.nc'.format(k=k)), format='netCDF4', engine='netcdf4')
        print('done after {dt} seconds'.format(dt=time.clock() - t0))
    else:
        print('write link to full-res dataset at level {k}'.format(k=k))
        with open(os.path.join(target_dir, 'L{k}.nc.lnk'.format(k=k)), 'w') as fp:
            fp.write(input_file)

    for i in range(1, tiling_scheme.num_levels):
        k = tiling_scheme.num_levels - 1 - i

        coords = dict(ds.coords)
        coords[x_dim] = ds.coords[x_dim][::2]
        coords[y_dim] = ds.coords[y_dim][::2]

        data_vars = dict()
        for var_name in selected_var_names:
            var = ds[var_name]
            downsampled_var = var[..., ::2, ::2]
            downsampled_var.encoding['chunksizes'] = get_chunk_sizes(var,
                                                                     tiling_scheme.tile_width,
                                                                     tiling_scheme.tile_height)
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


def get_chunk_sizes(var: xr.DataArray, tile_width: int, tile_height: int):
    chunk_sizes = len(var.shape) * [1]
    chunk_sizes[-1] = tile_width
    chunk_sizes[-2] = tile_height
    return chunk_sizes


def get_spatial_dims(var: xr.DataArray):
    if var.ndim < 2:
        return None
    x_dim = var.dims[-1]
    y_dim = var.dims[-2]
    if x_dim == 'x' and y_dim == 'y':
        return x_dim, y_dim
    if x_dim == 'lon' and y_dim == 'lat':
        return x_dim, y_dim
    return None
