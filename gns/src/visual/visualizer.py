import matplotlib.pyplot as plt
from matplotlib import animation


class Render:
    """
    Render rollout data into gif or vtk files.
    """
    def __init__(self, input_dir, input_name):
        """
        Initialize render class

        Args:
            input_dir (str): Directory where rollout.pkl are located
            input_name (str): Name of rollout `.pkl` file
        """

        # Files Control
        self.input_dir = input_dir
        self.input_name = input_name
        self.output_dir = input_dir
        self.output_name = input_name

    def render_GIF_animation(self):
        """
        Render `.gif` animation from `.pkl` trajectory data.

        Args:
            point_size (int): Size of particle in visualization
            timestep_stride (int): Stride of steps to skip.
            vertical_camera_angle (float): Vertical camera angle in degree
            viewpoint_rotation (float): Viewpoint rotation in degree

        Returns:
            gif format animation
        """
        # Init figures
        pass
