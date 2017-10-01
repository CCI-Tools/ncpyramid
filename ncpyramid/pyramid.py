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
from typing import Optional, Tuple, Any

import numpy as np
import xarray as xr
import math

GeoSpatialRect = Tuple[float, float, float, float]

MODE_LE = -1
MODE_EQ = 0
MODE_GE = 1


# TODO: handle correctly flipped y axis: if flipped, geo-width/height must grow in opposite direction
# TODO: handle correctly antimeridian

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

    geo_spatial_rect = get_geo_spatial_rect_from_dataset(ds, eps=1e-4)
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


class TilingScheme:
    def __init__(self,
                 num_levels: int,
                 num_level_zero_tiles_x: int,
                 num_level_zero_tiles_y: int,
                 tile_width: int,
                 tile_height: int,
                 geo_rectangle: Tuple[float, float, float, float]):
        self.num_levels = num_levels
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.num_level_zero_tiles_x = num_level_zero_tiles_x
        self.num_level_zero_tiles_y = num_level_zero_tiles_y
        self.geo_rectangle = geo_rectangle

    def num_tiles_x(self, level: int) -> int:
        return self.num_level_zero_tiles_x * (level << 1)

    def num_tiles_y(self, level: int) -> int:
        return self.num_level_zero_tiles_y * (level << 1)

    def width(self, level: int) -> int:
        return self.num_tiles_x(level) * self.tile_width

    def height(self, level: int) -> int:
        return self.num_tiles_x(level) * self.tile_width

    @property
    def min_width(self) -> int:
        return self.width(0)

    @property
    def min_height(self) -> int:
        return self.height(0)

    @property
    def max_width(self) -> int:
        return self.width(self.num_levels - 1)

    @property
    def max_height(self) -> int:
        return self.height(self.num_levels - 1)

    def __hash__(self) -> int:
        return self.num_levels \
               + 2 * self.tile_width \
               + 4 * self.tile_height \
               + 8 * self.num_level_zero_tiles_x \
               + 16 * self.num_level_zero_tiles_y \
               + hash(self.geo_rectangle)

    def __eq__(self, o: Any) -> bool:
        try:
            return self.num_levels == o.num_levels \
                   and self.tile_width == o.tile_width \
                   and self.tile_height == o.tile_height \
                   and self.num_level_zero_tiles_x == o.num_level_zero_tiles_x \
                   and self.num_level_zero_tiles_y == o.num_level_zero_tiles_y \
                   and self.geo_rectangle == o.geo_rectangle
        except AttributeError:
            return False

    def __str__(self):
        return '\n'.join(['number of pyramid levels: {nl}'.format(nl=self.num_levels),
                          'number of tiles at level zero: {nx} x {ny}'.format(nx=self.num_level_zero_tiles_x,
                                                                              ny=self.num_level_zero_tiles_y),
                          'pyramid tile size: {tw} x {th}'.format(tw=self.tile_width, th=self.tile_height),
                          'image size at level zero: {w} x {h}'.format(w=self.min_width, h=self.min_height),
                          'image size at level {k}: {w} x {h}'.format(k=self.num_levels - 1,
                                                                      w=self.max_width, h=self.max_height)])

    def __repr__(self):
        return 'TilingScheme(%s, %s, %s, %s, %s, %s)' % (
            self.num_levels, self.num_level_zero_tiles_x, self.num_level_zero_tiles_y,
            self.tile_width, self.tile_height, self.geo_rectangle)

    def to_cesium_json(self):
        lon0 = self.geo_rectangle[0]
        lat0 = self.geo_rectangle[1]
        delta_lon = self.geo_rectangle[2]
        delta_lat = self.geo_rectangle[3]
        return dict(numberOfLevelZeroTilesX=self.num_level_zero_tiles_x,
                    numberOfLevelZeroTilesY=self.num_level_zero_tiles_y,
                    tileWidth=self.tile_width,
                    tileHeight=self.tile_height,
                    minimumLevel=0,
                    maximumLevel=self.num_levels - 1,
                    rectangle=dict(west=math.radians(lon0),
                                   south=math.radians(lat0),
                                   east=math.radians((180. + lon0 + delta_lon) % 360. - 180.),
                                   north=math.radians(lat0 + delta_lat)))

    @classmethod
    def create(cls,
               w: int, h: int,
               tile_width: int, tile_height: int,
               geo_spatial_rect: Tuple[float, float, float, float],
               eps: float = 1e-6) -> 'TilingScheme':
        gsb_x1, gsb_y1, gsb_x2, gsb_y2 = geo_spatial_rect

        if gsb_x1 < gsb_x2:
            # crossing_antimeridan = False
            gsb_w = gsb_x2 - gsb_x1
        else:
            # crossing_antimeridan = True
            gsb_w = 360. + gsb_x2 - gsb_x1

        if gsb_y1 < gsb_y2:
            y_axis_flipped = True
            gsb_h = gsb_y2 - gsb_y1
        else:
            y_axis_flipped = False
            gsb_h = gsb_y1 - gsb_y2

        w_mode = MODE_GE
        if gsb_x1 == -180. and gsb_x2 == 180.:
            w_mode = MODE_EQ
        h_mode = MODE_GE
        if gsb_y1 == -90. and gsb_y1 == 90. or gsb_y1 == 90. and gsb_y2 == -90.:
            h_mode = MODE_EQ

        (w_new, h_new), (tw, th), (nt0x, nt0y), nl = pow2_2d_subdivision(w, h,
                                                                         w_mode=w_mode, h_mode=h_mode,
                                                                         tw_opt=min(w, tile_width or 512),
                                                                         th_opt=min(h, tile_height or 512))

        assert w_new >= w
        assert h_new >= h

        if w_new > w:
            gsb_w_new = w_new * gsb_w / w
            # We cannot adjust gsb_x1, because we expect x to increase with x indices
            # and hence we would later on have to read from negative x indexes
            gsb_x2_new = gsb_x1 + gsb_w_new
            if gsb_x2_new > 180.:
                gsb_x2_new = gsb_x2_new - 360.
        else:
            gsb_x2_new = gsb_x2

        if h_new > h:
            gsb_h_new = h_new * gsb_h / h
            if y_axis_flipped:
                # We cannot adjust gsb_y2, because we expect y to decrease with y indices
                # and hence we would later on have to read from negative y indexes
                gsb_y1_new = gsb_y1
                gsb_y2_new = gsb_y1_new + gsb_h_new
                if gsb_y2_new > 90.:
                    raise ValueError('illegal latitude coordinate range')
            else:
                # We cannot adjust gsb_y1, because we expect y to increase with y indices
                # and hence we would later on have to read from negative y indexes
                gsb_y2_new = gsb_y1
                gsb_y1_new = gsb_y2_new - gsb_h_new
                if gsb_y2_new < -90.:
                    raise ValueError('illegal latitude coordinate range')
        else:
            if y_axis_flipped:
                gsb_y1_new, gsb_y2_new = gsb_y1, gsb_y2
            else:
                gsb_y1_new, gsb_y2_new = gsb_y2, gsb_y1

        return TilingScheme(nl, nt0x, nt0y, tw, th, (gsb_x1, gsb_y1_new, gsb_x2_new, gsb_y2_new))


def _eq(x1, x2, eps):
    return abs(x2 - x1) < eps


def _adjust(x1, x2, x3, eps):
    return x3 if abs(x2 - x1) < eps else x1


def get_geo_spatial_rect_from_dataset(ds: xr.Dataset, eps: float = 1e-6):
    lon = ds.coords['lon']
    if lon.ndim > 1:
        lon = lon[(0,) * (lon.ndim - 1)]
    lat = ds.coords['lat']
    if lat.ndim > 1:
        lat = lat[(0,) * (lat.ndim - 1)]
    return get_geo_spatial_rect(lon, lat, eps=eps)


def get_geo_spatial_rect(x: np.ndarray, y: np.ndarray, eps: float = 1e-6):
    dx = np.gradient(x)
    if (dx.max() - dx.min()) >= eps:
        x = np.where(x < 0., 360. + x, x)
        dx = np.gradient(x)
        if (dx.max() - dx.min()) >= eps:
            raise ValueError('coordinate variable "lon" not is not equi-distant')

    dy = np.gradient(y)
    if (dy.max() - dy.min()) >= eps:
        raise ValueError('coordinate variable "lat" not is not equi-distant')

    # Outer boundaries are +/- half a cell size
    dx = dx[0]
    x1 = x[0] - 0.5 * dx
    x2 = x[-1] + 0.5 * dx
    x1 = _adjust(x1, -180., -180., eps)
    x1 = _adjust(x1, +180., -180., eps)
    x2 = _adjust(x2, -180., +180., eps)
    x2 = _adjust(x2, +180., +180., eps)

    dy = dy[0]
    y1 = y[0] - 0.5 * dy
    y2 = y[-1] + 0.5 * dy
    y1 = _adjust(y1, -90., -90., eps)
    y1 = _adjust(y1, +90., +90., eps)
    y2 = _adjust(y2, -90., -90., eps)
    y2 = _adjust(y2, +90., +90., eps)

    return x1, y1, x2, y2


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


def pow2_2d_subdivision(w: int, h: int,
                        w_mode: int = MODE_EQ, h_mode: int = MODE_EQ,
                        tw_opt: Optional[int] = None, th_opt: Optional[int] = None,
                        tw_min: Optional[int] = None, th_min: Optional[int] = None,
                        tw_max: Optional[int] = None, th_max: Optional[int] = None,
                        nt0_max: Optional[int] = None,
                        nl_max: Optional[int] = None):
    """
    Get a pyramidal quad-tree subdivision of a 2D image rectangle given by image width *w* and height *h*.
    We want all pyramid levels to use the same tile size *tw*, *th*. All but the lowest resolution level, level zero,
    shall have 2 times the number of tiles than in a previous level in x- and y-direction.

    As there can be multiple of such subdivisions, we select an optimum subdivision by constraints. We want
    (in this order):
    1. the resolution of the highest pyramid level, *nl* - 1, to be as close as possible to *w*, *h*;
    2. the number of tiles in level zero to be as small as possible;
    3. the tile sizes *tw*, *th* to be as close as possible to *tw_opt*, *th_opt*, if given;
    4. a maximum number of levels.

    :param w: image width
    :param h: image height
    :param w_mode: optional mode for horizontal direction, -1: *w_act* <= *w*, 0: *w_act* == *w*, +1: *w_act* >= *w*
    :param h_mode: optional mode for vertical direction, -1: *h_act* <= *h*, 0: *h_act* == *h*, +1: *h_act* >= *h*
    :param tw_opt: optional optimum tile width
    :param th_opt: optional optimum tile height
    :param tw_min: optional minimum tile width
    :param th_min: optional minimum tile height
    :param tw_max: optional maximum tile width
    :param th_max: optional maximum tile height
    :param nt0_max: optional maximum number of tiles at level zero of pyramid
    :param nl_max: optional maximum number of pyramid levels
    :return: a tuple ((*w_act*, *h_act*), (*tw*, *th*), (*nt0_x*, *nt0_y*), *nl*) with
             *w_act*, *h_act* being the final image width and height in the pyramids's highest resolution level;
             *tw*, *th* being the tile width and height;
             *nt0_x*, *nt0_y* being the number of tiles at level zero of pyramid in horizontal and vertical direction;
             and *nl* being the total number of pyramid levels.
    """
    w_act, tw, nt0_x, nl_x = pow2_1d_subdivision(w, s_mode=w_mode,
                                                 ts_opt=tw_opt, ts_min=tw_min, ts_max=tw_max,
                                                 nt0_max=nt0_max, nl_max=nl_max)
    h_act, th, nt0_y, nl_y = pow2_1d_subdivision(h, s_mode=h_mode,
                                                 ts_opt=th_opt, ts_min=th_min, ts_max=th_max,
                                                 nt0_max=nt0_max, nl_max=nl_max)
    if nl_x < nl_y:
        nl = nl_x
        nt0_y = h_act // (1 << (nl - 1)) // th
    elif nl_x > nl_y:
        nl = nl_y
        nt0_x = w_act // (1 << (nl - 1)) // tw
    else:
        nl = nl_x

    return (w_act, h_act), (tw, th), (nt0_x, nt0_y), nl


def pow2_1d_subdivision(s_act: int,
                        s_mode: int = MODE_EQ,
                        ts_opt: Optional[int] = None,
                        ts_min: Optional[int] = None,
                        ts_max: Optional[int] = None,
                        nt0_max: Optional[int] = None,
                        nl_max: Optional[int] = None):
    return pow2_1d_subdivisions(s_act,
                                s_mode=s_mode,
                                ts_opt=ts_opt,
                                ts_min=ts_min, ts_max=ts_max,
                                nt0_max=nt0_max, nl_max=nl_max)[0]


def pow2_1d_subdivisions(s: int,
                         s_mode: int = MODE_EQ,
                         ts_opt: Optional[int] = None,
                         ts_min: Optional[int] = None,
                         ts_max: Optional[int] = None,
                         nt0_max: Optional[int] = None,
                         nl_max: Optional[int] = None):
    if s < 1:
        raise ValueError('invalid s')

    ts_min = ts_min or min(s, (ts_opt // 2 if ts_opt else 200))
    ts_max = ts_max or min(s, (ts_opt * 2 if ts_opt else 1200))
    nt0_max = nt0_max or 8
    nl_max = nl_max or 16

    if ts_min < 1:
        raise ValueError('invalid ts_min')
    if ts_max < 1:
        raise ValueError('invalid ts_max')
    if ts_opt is not None and ts_opt < 1:
        raise ValueError('invalid ts_opt')
    if nt0_max < 1:
        raise ValueError('invalid nt0_max')
    if nl_max < 1:
        raise ValueError('invalid nl_max')

    subdivisions = []
    for ts in range(ts_min, ts_max + 1):
        s_max_min = s if s_mode == MODE_EQ or s_mode == MODE_GE else s - (ts - 1)
        s_max_max = s if s_mode == MODE_EQ or s_mode == MODE_LE else s + (ts - 1)
        for nt0 in range(1, nt0_max):
            s_max = nt0 * ts
            if s_max > s_max_max:
                break
            for nl in range(2, nl_max):
                s_max = (1 << (nl - 1)) * nt0 * ts
                ok = False
                if s_mode == MODE_GE:
                    if s_max >= s:
                        if s_max > s_max_max:
                            break
                        ok = True
                elif s_mode == MODE_LE:
                    if s >= s_max >= s_max_min:
                        ok = True
                else:  # s_mode == MODE_EQ:
                    if s_max == s:
                        ok = True
                    elif s_max > s:
                        break
                if ok:
                    rec = s_max, ts, nt0, nl
                    subdivisions.append(rec)

    if not subdivisions:
        return [(s, s, 1, 1)]

    # maximize nl
    subdivisions.sort(key=lambda r: r[3], reverse=True)
    if ts_opt:
        # minimize |ts - ts_opt|
        subdivisions.sort(key=lambda r: abs(r[1] - ts_opt))
    # minimize nt0
    subdivisions.sort(key=lambda r: r[2])
    # minimize s_max - s_min
    subdivisions.sort(key=lambda r: r[0] - s)

    return subdivisions
