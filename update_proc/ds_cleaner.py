# -*- coding: utf-8 -*-
# ---------------------

from path import Path


def clean_ds(src_dir, dst_dir, buffer_size):
    # type: (str, str, int) -> None
    """
    :param src_dir: source directory containing cuts
    :param dst_dir: destination directory
        ->> old cuts will be moved here
    :param buffer_size: number of cuts you want to keep;
        ->> only the `buffer_size` newest elements will be kept
            in the source directory, while the others will be moved
            to the destination directory
    """

    # get all paths in the source directory
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.makedirs_p()
    all_paths = src_dir.files()

    # sort paths by date (ascending)
    # >> all_paths[0] is the path of the oldest image
    # >> all_paths[-1] is the path of the youngest image
    all_paths.sort(key=lambda p: p.basename())

    print(f'$> there are {len(all_paths)} cuts '
          f'in the source directory and buffer size is {buffer_size}')

    # keep the `buffer_size` oldest image
    if len(all_paths) < buffer_size:
        paths2keep = all_paths
        print(f'───$> we need to remove 0 cuts')
    else:
        _n = len(all_paths) - buffer_size
        print(f'───$> we need to remove the {_n} oldest cut(s)')
        paths2keep = all_paths[-buffer_size:]
    print('')

    # remove old files
    for p in all_paths:
        if p not in paths2keep:
            new_path = dst_dir / p.basename()
            cmd = f'mv "{p.abspath()}" "{new_path}'
            print(f'───$> {cmd}')


def main():
    clean_ds(
        src_dir='/goat-nas/Datasets/spal/progression_demo/train',
        dst_dir='/goat-nas/Datasets/spal/progression_trash',
        buffer_size=4992
    )


if __name__ == '__main__':
    main()
