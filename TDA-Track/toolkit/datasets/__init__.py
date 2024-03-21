from .nut2024_40l import NUT2024_40LDataset
from .nut2024_60l import NUT2024_60LDataset
from .nut2024_100l import NUT2024_100LDataset


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
        if '40' in name:
            dataset = NUT2024_40LDataset(**kwargs)
        elif '60' in name:
            dataset = NUT2024_60LDataset(**kwargs)
        elif '100' in name:
            dataset = NUT2024_100LDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

