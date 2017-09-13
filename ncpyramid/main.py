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


import argparse
import sys

from ncpyramid.pyramid import write_pyramid


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog=__package__,
                                     description="Generates pyramids from NetCDF files")
    parser.add_argument("input_file",
                        metavar='INPUT_FILE',
                        help="NetCDF input file (*.nc)")

    parser.add_argument("-d", "--output_dir",
                        metavar='OUTPUT_DIR',
                        default='.',
                        help="target directory for generated pyramid")

    parser.add_argument("-n", "--output_name",
                        metavar='OUTPUT_NAME',
                        default=None,
                        help="name of the pyramid; will be derived from INPUT_FILE if not given")

    parser.add_argument("-f", "--write_fr",
                        action='store_true',
                        help="also write out full resolution level; by default, a link to the original file is written")

    parser.add_argument("-W", "--tile_width",
                        metavar='TILE_WIDTH',
                        type=int,
                        default=None,
                        help="target tile width")

    parser.add_argument("-H", "--tile_height",
                        metavar='TILE_HEIGHT',
                        type=int,
                        default=None,
                        help="target tile height")

    args = parser.parse_args(args=args)

    generated_dir = write_pyramid(args.input_file,
                                  args.output_dir,
                                  args.output_name,
                                  args.write_fr,
                                  args.tile_width,
                                  args.tile_height)

    print('generated', generated_dir)


if __name__ == '__main__':
    main()
