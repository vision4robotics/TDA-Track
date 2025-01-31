from .nat2024_1 import NAT2024_Dataset
from .nut_l import NUT_LDataset
from .nat2021 import NAT2021_Dataset
from .darktrack2021 import DarkTrack2021_Dataset
from .uavdark135 import UAVDark135_Dataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'NAT2021' in name:
            dataset = NAT2021_Dataset(**kwargs)
        elif 'DarkTrack2021' in name:
            dataset = DarkTrack2021_Dataset(**kwargs)
        elif 'UAVDark135' in name:
            dataset = UAVDark135_Dataset(**kwargs)
        elif 'NAT2024' in name:
            dataset = NAT2024_Dataset(**kwargs)
        elif 'NUT' in name:
            dataset = NUT_LDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

