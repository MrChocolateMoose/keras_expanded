from keras.utils.visualize_util import plot
import os.path

def save_diagram(model, diagram_file_path, show_shapes=False, show_layer_names=True):

    diagram_dir = os.path.dirname(diagram_file_path)
    if not os.path.exists(diagram_dir):
        os.makedirs(diagram_dir)

    plot(model, to_file=diagram_file_path, show_shapes=show_shapes, show_layer_names=show_layer_names)
