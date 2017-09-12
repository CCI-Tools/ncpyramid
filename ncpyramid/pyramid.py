import os
import time

import numba as nb
import numpy as np
import xarray as xr


def write_pyramid(input_file, output_dir, output_name, tile_width, tile_height, level_count):
    basename, ext = os.path.splitext(os.path.basename(input_file))
    if output_name is None or output_name.strip() == '':
        output_name = basename + '.pyramid'

    target_dir = os.path.join(output_dir, output_name)

    os.makedirs(target_dir, exist_ok=True)

    ds = xr.open_dataset(input_file, chunks={})

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

    ts_count = 24
    tw_out = np.empty(ts_count, dtype=np.int32)
    th_out = np.empty(ts_count, dtype=np.int32)
    count = pyramid_subdivision(w_max, h_max,
                                min(tile_width, tile_height) // 4,
                                max(tile_width, tile_height) * 4,
                                tw_out, th_out)
    if count == 0:
        raise ValueError(
            'size does not allow for subdivision into any pyramid levels: {w} x {h}'.format(w=w_max, h=h_max))

    dx_min = 2 * w_max
    dy_min = 2 * h_max
    tw_best = -1
    th_best = -1
    for i in range(ts_count):
        tw = tw_out[i]
        th = th_out[i]
        if tw > 0:
            dx = abs(tw - tile_width)
            if dx < dx_min:
                dx_min = dx
                tw_best = tw
        if th > 0:
            dy = abs(th - tile_height)
            if dy < dy_min:
                dy_min = dy
                th_best = th

    if tw_best == -1 or th_best == -1:
        raise ValueError('failed to determine best tile size')

    print('number of pyramid levels: {c}'.format(c=count + 1))
    print('best tile sizes are: {tw} x {th}'.format(tw=tw_best, th=th_best))

    ds_orig = ds

    for var_name in selected_var_names:
        var = ds[var_name][...].chunk(dict(lat=th_best, lon=tw_best))
        var.encoding['chunksizes'] = get_chunk_sizes(var, tw_best, th_best)
        # print(downsampled_var.encoding)
        ds[var_name] = var

    print('writing hi-res dataset at level {c}'.format(c=count))
    t0 = time.clock()
    ds.to_netcdf(os.path.join(target_dir, 'L{c}.nc'.format(c=count)), format='netCDF4', engine='netcdf4')
    print('done after {dt} seconds'.format(dt=time.clock() - t0))

    for i in range(count):
        c = count - i - 1

        coords = dict(ds.coords)
        coords['lon'] = ds.coords['lon'][::2]
        coords['lat'] = ds.coords['lat'][::2]

        data_vars = dict()
        for var_name in selected_var_names:
            var = ds[var_name]
            downsampled_var = var[..., ::2, ::2]
            downsampled_var.encoding['chunksizes'] = get_chunk_sizes(var, tw_best, th_best)
            # print(downsampled_var.encoding)
            data_vars[var_name] = downsampled_var

        print('constructing lower-res dataset at level {c}'.format(c=c))
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_orig.attrs)
        print('writing lower-res dataset at level {c}...'.format(c=c))
        t1 = time.clock()
        ds.to_netcdf(os.path.join(target_dir, 'L{c}.nc'.format(c=c)), format='netCDF4', engine='netcdf4')
        print('done after {dt} seconds'.format(dt=time.clock() - t1))

    ds_orig.close()

    print('pyramid "{n}" written within {dt} seconds'.format(n=output_name, dt=time.clock() - t0))

    return target_dir


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


@nb.jit(nopython=True)
def pyramid_subdivision_count(s_max: int, ts: int, ntl0_max: int = 1):
    """
    Compute number of times *w* can be divided by 2 without remainder and while the result is still
    integer-dividable by *ts*.
    """
    count = 0
    s = s_max
    while s % 2 == 0 and s % ts == 0 and (s // ts) % 2 == 0 and (s // ts) > ntl0_max:
        s //= 2
        count += 1
    return count


@nb.jit(nopython=True)
def pyramid_subdivision(w_max: int, h_max: int,
                        ts_min: int, ts_max: int,
                        tw_out, th_out,
                        ntl0x_max: int = 1,
                        ntl0y_max: int = 1):
    size = ts_max - ts_min + 1

    cx = np.empty(ts_max - ts_min + 1, dtype=np.int32)
    cy = np.empty(ts_max - ts_min + 1, dtype=np.int32)
    for i in range(size):
        ts = ts_min + i
        cx[i] = pyramid_subdivision_count(w_max, ts, ntl0_max=ntl0x_max)
        cy[i] = pyramid_subdivision_count(h_max, ts, ntl0_max=ntl0y_max)

    cx_max = -1
    cy_max = -1
    for i in range(size):
        cx_max = max(cx[i], cx_max)
        cy_max = max(cy[i], cy_max)

    c = min(cx_max, cy_max)

    for ix in range(tw_out.size):
        tw_out[ix] = 0
    for iy in range(th_out.size):
        th_out[iy] = 0

    if c <= 0:
        return 0

    ix = 0
    iy = 0
    for i in range(size):
        if cx[i] >= c and ix < tw_out.size:
            tw_out[ix] = ts_min + i
            ix += 1
        if cy[i] >= c and iy < th_out.size:
            th_out[iy] = ts_min + i
            iy += 1

    return c
