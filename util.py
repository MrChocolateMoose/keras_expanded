import math
import numpy as np
from PIL import Image
import itertools

def stitch_pil_images(pil_image_list, pil_image_size=None, margin=5, output_pil_image_size=None):

    if not pil_image_list:
        raise ValueError('empty pil_image_list')

    pil_image_mode = pil_image_list[0].mode

    # allow for pil image size to be automatically retrieved
    if pil_image_size == None:
        pil_image_size = pil_image_list[0].size

        all_sizes_equal = all(pil_image_size == cur_pil_image.size for cur_pil_image in pil_image_list)

        if not all_sizes_equal:
            raise ValueError("all PIL images in pil_image_list do not have the same size")

        all_modes_equal = all(pil_image_mode == cur_pil_image.mode for cur_pil_image in pil_image_list)

        if not all_modes_equal:
            raise ValueError("all PIL images in pil_image_list do not have the same mode")


    pil_image_list_len = len(pil_image_list)
    if output_pil_image_size == None:
        output_pil_image_dim = math.ceil(math.sqrt(pil_image_list_len))
        output_pil_image_size = (output_pil_image_dim, output_pil_image_dim)

    width = output_pil_image_size[0] * pil_image_size[0] + (output_pil_image_size[0] - 1) * margin
    height = output_pil_image_size[1] * pil_image_size[1] + (output_pil_image_size[1] - 1) * margin

    output_pil_image = Image.new(mode='RGBA', size=(width,height))

    for x in range(output_pil_image_size[0]):
        for y in range(output_pil_image_size[1]):

            cur_index = x * output_pil_image_size[0] + y

            # prevent out of bounds indicies from accessing our image list
            if cur_index >= pil_image_list_len:
                continue

            cur_pil_image = pil_image_list[cur_index]

            top = (pil_image_size[1] + margin) * y
            left = (pil_image_size[0] + margin) * x

            output_pil_image.paste(cur_pil_image, box=(top,left))

    return output_pil_image