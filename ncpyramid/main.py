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

    parser.add_argument("-W", "--tile_width",
                        metavar='TILE_WIDTH',
                        default=720,
                        help="target tile width")

    parser.add_argument("-H", "--tile_height",
                        metavar='TILE_HEIGHT',
                        default=720,
                        help="target tile height")

    parser.add_argument("-L", "--level_count",
                        metavar='LEVEL_COUNT',
                        default=5,
                        help="number of pyramid levels")

    args = parser.parse_args(args=args)

    generated_dir = write_pyramid(args.input_file,
                                  args.output_dir,
                                  args.output_name,
                                  args.tile_width,
                                  args.tile_height,
                                  args.level_count)

    print('generated', generated_dir)


if __name__ == '__main__':
    main()

