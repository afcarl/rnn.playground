from PIL import Image, ImageDraw
import numpy as np


class PathDataSet:

    def __init__(self, crop_size=(20, 20), bg_color="red", num_samples=10):
        self.num_samples = num_samples
        self.crop_size = crop_size
        self.bg_color = bg_color

    def _get_top_bottom_path_points(self, im_size):
        num_curves = np.random.random_integers(0, im_size[1]/10)
        points = [(np.random.random() * im_size[0], 0)]
        for i in range(num_curves):
            x_val = np.random.random() * im_size[0]
            y_val = np.random.random() * (im_size[1] - points[-1][1]) + points[-1][1]
            points.append((x_val, y_val))
        points.append((np.random.random() * im_size[0], im_size[1]))
        return points

    def _generate_positive_image(self, im_size):
        points = self._get_top_bottom_path_points(im_size)
        return self._draw_points_lines(points, im_size)

    def _draw_points_lines(self, points, im_size, rand=False):
        im = Image.new('L', im_size)
        draw = ImageDraw.Draw(im)
        for i in range(1, len(points)):
            start_point = points[i - 1]
            end_point = points[i]
            if (rand and np.random.random() > 0.5) or not rand:
                draw.line((start_point[0], start_point[1], end_point[0], end_point[1]), fill="aqua", width=5)

        #im.show()
        return np.array(im)

    def _generate_negative_image(self, im_size):
        points = self._get_top_bottom_path_points(im_size)
        return self._draw_points_lines(points, im_size, rand=True)

    def _generate_img(self, positive=True):
        im_size = (self.crop_size[0], np.random.randint(self.crop_size[1], 10 * self.crop_size[1]))
        if positive:
            img = self._generate_positive_image(im_size)
        else:
            img = self._generate_negative_image(im_size)
        #img_sequence = [img[i:i+self.crop_size[1]] for i in range(0, im_size[1], self.crop_size[1])]
        return img

    def generate_ds(self):
        x = []
        x.extend([self._generate_img(positive=True) for i in range(self.num_samples / 2)])
        x.extend([self._generate_img(positive=False) for i in range(self.num_samples / 2)])

        y = []
        y.extend([[1., 0.] for i in range(self.num_samples / 2)])
        y.extend([[0., 1.] for i in range(self.num_samples / 2)])
        return np.array(x), np.array(y)


#ds = PathDataSet()
#ds.generate_ds()