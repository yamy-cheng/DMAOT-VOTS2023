import numpy as np

def transfer_mask(multi_mask, object_num):
    '''
    parameter:
        multi_mask: np.arr (h, w),  id=1,2,3...
    return:
        masks: List[mask1, mask2, ....]
    '''
    masks = []
    # exclude 0 (background)
    for obj_idx in range(1, object_num + 1):
        single_mask = np.zeros_like(multi_mask)
        single_mask[multi_mask==obj_idx] = 1
        masks.append(single_mask)

    return masks
if __name__ == "__main__":
    multi_mask = np.zeros((485, 485))
    # multi_mask[200:300, 250:300] = 1
    multi_mask[400:450, 250:300] = 2

    masks = transfer_mask(multi_mask, 2)
    print(masks[0].max())
    print(len(masks))