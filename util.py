import math
import numpy as np
from PIL import Image
import matplotlib as plt
from sklearn import preprocessing

DEFAULT_BORDER_THICKNESS = 0
DEFAULT_CMAP_NAME = 'seismic'
DEFAULT_MARGIN = 5

def stitch_pil_image_from_values(value_list, size, cmap_name = DEFAULT_CMAP_NAME, margin = DEFAULT_MARGIN, border_thickness = DEFAULT_BORDER_THICKNESS):

    values = np.array(value_list).reshape(-1, 1)
    log_values = np.log(np.clip(values + 1, 1, np.inf))
    scaled_values = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(log_values)

    scaled_values = np.expand_dims(scaled_values, axis=1)

    cmap = plt.cm.get_cmap(cmap_name)

    cmap_values = cmap(scaled_values)

    cmap_pil_image_list = []
    for index in range(cmap_values.shape[0]):

        cmap_value = np.uint8(255 * cmap_values[index])

        # convert and resize heatmap
        cmap_pil_image = Image.fromarray(cmap_value, mode='RGB')
        cmap_pil_image = cmap_pil_image.resize(size)

        cmap_pil_image_list.append(cmap_pil_image)

    return stitch_pil_images(cmap_pil_image_list, margin = margin, border_thickness = border_thickness)



def stitch_pil_image_from_arrays(image_list, mode, margin = DEFAULT_MARGIN, border_thickness = DEFAULT_BORDER_THICKNESS):

    pil_image_list = []
    for image_array in image_list:

        pil_image = Image.fromarray(image_array,mode)
        pil_image = pil_image.transpose(Image.ROTATE_90)
        pil_image_list.append(pil_image)

    output_pil_image = stitch_pil_images(pil_image_list, margin = margin, border_thickness = border_thickness)

    return output_pil_image

#TODO: output_pil_image_size to output_pil_image_grid_size
def stitch_pil_images(pil_image_list, pil_image_size=None, margin=DEFAULT_MARGIN, border_thickness = DEFAULT_BORDER_THICKNESS, output_pil_image_size=None):

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

    width = (output_pil_image_size[0] * pil_image_size[0] + (output_pil_image_size[0] - 1) * margin) + (border_thickness * 2 * output_pil_image_size[0])
    height = (output_pil_image_size[1] * pil_image_size[1] + (output_pil_image_size[1] - 1) * margin) + (border_thickness * 2 * output_pil_image_size[1])

    output_pil_image = Image.new(mode='RGBA', size=(width,height))

    for x in range(output_pil_image_size[0]):
        for y in range(output_pil_image_size[1]):

            cur_index = x * output_pil_image_size[0] + y

            # prevent out of bounds indicies from accessing our image list
            if cur_index >= pil_image_list_len:
                continue

            cur_pil_image = pil_image_list[cur_index]

            left = (pil_image_size[0] + margin + (border_thickness * 2)) * x + border_thickness
            top = (pil_image_size[1] + margin + (border_thickness * 2)) * y + border_thickness

            output_pil_image.paste(cur_pil_image, box=(left, top))

    return output_pil_image