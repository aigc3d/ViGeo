from mmengine import TRANSFORMS

@TRANSFORMS.register_module()
class convert_to_tensor:
    """
    Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys, meta_keys):
        self.keys = keys
        self.meta_keys = meta_keys
    
    def __call__(self, sample):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """

        data = {}

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = sample[key]
        
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = sample[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'