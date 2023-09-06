
from path import Path
from dataset.ds_utils import mpath2info

# from_path = Path('/goat-nas/Datasets/spal/progression_demo/SPAL_container')
# to_path = Path('/goat-nas/Datasets/spal/progression_demo/SPAL_folder')
# # from_path = Path('/goat-nas/Datasets/spal/progression_demo/SPAL_folder')
# # to_path = Path('/goat-nas/Datasets/spal/progression_demo/SPAL_container')

# from_path = Path('C:\\Test\\img_src')
# to_path = Path('C:\\Test\\out_dst')
# from_path = Path('C:\\Test\\out_dst')
# to_path = Path('C:\\Test\\img_src')

from_path = Path('C:\\file_ex_tmp')
to_path = Path('C:\\file_ex')
# from_path = Path('C:\\file_ex')
# to_path = Path('C:\\file_ex_tmp')

infos = [mpath2info(fn) for fn in from_path.files()]
infos.sort(key=lambda d: d['datetime'])

for info in infos:
    print(info['original_name'])
    Path(info['original_name']).move(to_path)


# from_path_1 = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/DS/06/15')
# from_path_2 = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/DS/06/16')
#
# for fname in from_path_1.files():
#     print(fname)
#     fname.copy(in_path)
#
# for fname in from_path_2.files():
#     print(fname)
#     fname.copy(in_path)