#%%
import os
import os.path as osp 
from glob import glob 
import re  # regular expression (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/



#%%
class BaseDataset(object):
    """ Base class of reid dataset
        - (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/datasets/bases.py
    """

    def get_imagedata_info(self, data):
        pids = []

        for _, pid in data:
            pids += [pid]

            
        pids = set(pids) # remove duplication
        
        num_pids = len(pids)
        num_imgs = len(data)


        return num_pids, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError



class BaseImageDataset(BaseDataset):
    """ Base class of image reid dataset
        - (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/datasets/bases.py
    """

    def print_dataset_statistics(self, gallery):
        num_gallery_pids, num_gallery_imgs = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------")
        print("  subset   | # ids | # images ")
        print("  ----------------------------")
        print(f"  gallery  | {num_gallery_pids:5d} | {num_gallery_imgs:8d} ")
        print("  ----------------------------")


#%%        
class DTC2021(BaseImageDataset):
    """ (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/datasets/Market1501.py
    """
    def __init__(self, data_dir = 'IMGs', verbose = True):
        super(DTC2021, self).__init__()

        self.dataset_dir = data_dir
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        
        gallery = self._process_dir(self.gallery_dir, relabel=False)


        if verbose:
            print("=> DTC2021 gallery loaded")
            self.print_dataset_statistics(gallery)


        self.gallery = gallery

        self.num_gallery_pids, self.num_gallery_imgs = self.get_imagedata_info(self.gallery)


    def _process_dir(self, data_dir, relabel=True):
        img_paths = glob(osp.join(data_dir, '*.png'))

    
        pattern = re.compile(r'([\d]+)_c([\S]+)_t')     # Regular expression ; (ref) https://wikidocs.net/4308
                                                        #                      (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/
        pid_container = set()
        for img_path in img_paths:
    
            pid, _ = map(str, pattern.search(img_path).groups()) # map(); (ref) https://dojang.io/mod/page/view.php?id=2286
            pid = int(pid)
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, cls_name = map(str, pattern.search(img_path).groups()) 
            pid = int(pid)

            if pid == -1: continue  # junk images are just ignored

            if relabel: pid = pid2label[pid]
#            dataset.append((img_path, pid, cls_name))
            dataset.append((img_path, pid))

        return dataset



#%%
if __name__ == '__main__' :
    """ test secssion
    """
    dataset = DTC2021("./data/IMGs")
    print(dataset.gallery)







