import random
from typing import Set
from typing import Tuple

from path import Path

from day_recorder import DRElement
from day_recorder import DayRecorder


def __choice_perc(x, perc):
    # type: (Set[DRElement], float) -> Set[DRElement]
    """
    :param x: set of DRElement(s)
    :param perc: percentage of items to select
        >> value in range [0, 100]
    :return: a randomly drawn subset of `x`,
        with size equal to `perc`% of `x`;
        >> NOTE: minimum size is 1
    """
    size = int(round(len(x) * (perc / 100)))
    size = max(size, 1)
    random_subset = random.sample(list(x), size)
    return set(random_subset)


def daily_update_procedure(u_range, expl_perc):
    # type: (Tuple[int, int], float) -> None
    """
    :param u_range: uncertainty range w.r.t anomaly score (`s`);
        the range is defined by a tuple (s_min, s_max) such that:
        >> samples with `s < s_min` are definitely considered "GOOD"
        >> samples with `s > s_max` are definitely considered "BAD"
        >> samples with `s_min <= s <= s_max` cannot be they have an
           uncertain nature ==> we don't know for sure if they are
           "GOOD" or "BAD" samples
    :param expl_perc: exploration percentage; is the percentage of
        samples of the set G and B to be moved to U' for manual
        verification by the user
    """
    root_dir = Path(__file__).parent.parent / 'dummy_root'
    recorder = DayRecorder(root_dir=root_dir)

    # extracts the preliminary versions of the 3 sub-sets (G, B, U)
    # defined on the basis of the uncertainty range of the anomaly score
    subsets = {}
    for set_type in ['G', 'B', 'U']:
        subsets[set_type] = recorder.get_subset(
            set_type=set_type, u_range=u_range
        )

    print(f'$> DAILY PROCEDURE')
    print(f'----')

    print(f'$> initial sets cardinality:')
    print(f'───$> #G={len(subsets["G"])}, '
          f'#B={len(subsets["B"])}, '
          f'#U={len(subsets["U"])}')

    print(f'----')
    print(f'$> moving some elements from G and B to U')
    for set_type in ['G', 'B']:
        to_move = __choice_perc(subsets[set_type], perc=expl_perc)
        subsets[set_type] = subsets[set_type].difference(to_move)
        subsets['U'] = subsets['U'].union(to_move)
        print(f'───$> moving {expl_perc}% of {set_type} '
              f'({len(to_move)}) elements to U')

    print(f'$> current sets cardinality:')
    print(f'───$> #G={len(subsets["G"])}, '
          f'#B={len(subsets["B"])}, '
          f'#U={len(subsets["U"])}')

    print(f'----')
    print(f'$> updating training set with G')
    for element in subsets['G']:
        date_str, anomaly_score = element
        src_path = root_dir / 'daily_cuts' / date_str + '.jpg'
        dst_path = root_dir / 'train' / date_str + '.jpg'

        cmd = f'mv "{src_path.abspath()}" "{dst_path.abspath()}"'
        print(f'───$> {cmd} '
              f'(anomaly_score={anomaly_score:03d}) to training set')


if __name__ == '__main__':
    daily_update_procedure(u_range=(45, 150), expl_perc=5)
