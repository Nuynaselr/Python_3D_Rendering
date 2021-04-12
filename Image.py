from PIL import Image
import numpy as np
import random
import re
import time


class My_Image:
    def __init__(self, height=2000, width=2000):
        self.height = height
        self.width = width
        self.matrix = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.scale = 0.2
        self.value_shift_image = (1, 1)
        self.file_path = 'Axe.obj'

    def get_image(self):
        return Image.fromarray(self.matrix)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_scalable_size(self):
        return (self.height * self.scale, self.width * self.scale)

    def save_image(self, name_image):
        temporary_image = self.get_image()
        temporary_image.save(name_image)

    def set_white(self):
        self.matrix.fill(255)

    def set_color(self, temp_color):
        for height in range(self.height):
            for width in range(self.width):
                self.matrix[height][width] = temp_color

    def chaos_color(self):
        for height in range(self.height):
            for width in range(self.width):
                self.matrix[height][width] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    def paint_points(self, list_points):
        for point in list_points:
            point_in_scale = self.get_scalable_point(point)
            self.matrix[point_in_scale[0]][point_in_scale[1]] = [214, 214, 214]

    def get_scalable_point(self, point):
        size_image = self.get_scalable_size()

        half_width = int(size_image[0] / 8)
        half_height = int(size_image[1] / 8)

        first_coordinate = point[0]
        second_coordinate = point[1]

        modified_first_coordinate = int(size_image[0] * self.value_shift_image[0]) + int((first_coordinate * half_width))
        modified_second_coordinate = int((float(second_coordinate) * half_height) + size_image[1] * self.value_shift_image[1])

        return (modified_first_coordinate, modified_second_coordinate)

    def paint_lines(self, info_object):
        list_points = info_object.get('point')
        for current_points in info_object.get('connect_points'):
            count_points = len(current_points)
            for j in range(count_points):
                first_point, second_point = list_points[current_points[j % count_points] - 1], list_points[current_points[(j + 1) % count_points] - 1]
                self.paint_line(first_point, second_point)

    def paint_line(self, first_point, second_point):
        if first_point[0] > second_point[0] and first_point[1] > second_point[1]:
            first_point, second_point = second_point, first_point

        modified_first_point = self.get_scalable_point(first_point)
        modified_second_point = self.get_scalable_point(second_point)

        step = 0
        while True:

            x = modified_first_point[0]*step*(-1) + modified_second_point[0]*step
            y = modified_first_point[1]*step*(-1) + modified_second_point[1]*step
            if modified_first_point[0] + int(x) == modified_second_point[0] and modified_first_point[1] + int(y) == modified_second_point[1]:
                break

            self.matrix[modified_first_point[0] + int(x)][modified_first_point[1] + int(y)] = [214, 214, 214]
            step += 0.001


    def parse_file(self):
        with open(self.file_path, 'r') as obj_file:
            data_point = obj_file.read()

        info_object = {'point': [],
                'connect_points': []}

        for line in data_point.split('\n'):
            try:
                data = re.split('\s+', line)
                type = data[0]
                list_value = data[1:]
            except:
                continue

            if type == 'v':
                info_object['point'].append([
                                # int(size_image[0] * self.value_shift_image[0]) +
                                # int((float(list_value[2]) * half_width)),
                                # int((float(list_value[0]) * half_height)
                                #      + size_image[1] * self.value_shift_image[1]
                                #     )
                                float(list_value[2]), float(list_value[0])
                            ])
            elif type == 'f':
                list_point_for_connection = []
                for values in list_value:
                    list_point_for_connection.append(int(values.split('/')[0]))
                info_object['connect_points'].append(
                    tuple(list_point_for_connection)
                )

        return info_object

    def get_barycentric_coordinates(self, point):
        first_point = point[0]
        second_point = point[1]
        third_point = point[2]

        min_max_x = [min([first_point[0], second_point[0], third_point[0]]),
                     max([first_point[0], second_point[0], third_point[0]])]

        min_max_x = [min([first_point[1], second_point[1], third_point[1]]),
                     max([first_point[1], second_point[1], third_point[1]])]



        lambda_0 = ((x1 - x2)*(y - y2) - (y1 - y2)(x - x2)) / ((x1 - x2)*(y0 - y2) - (y1 - y2) (x0 - x2))



if __name__ == '__main__':
    test_image = My_Image()

    test_image.file_path = 'Axe.obj'
    test_image.scale = 0.2
    test_image.value_shift_image = (1.8, 1)

    # test_image.set_white()
    # test_image.save_image('test_image.png')
    # test_image.set_color([255, 0, 0])
    # test_image.save_image('test_image_1.png')
    # test_image.chaos_color()
    # test_image.save_image('test_image_2.png')

    # list_points = [(500, 1000), (2000, 1000)]
    #
    # test_image.paint_line(list_points[0], list_points[1])
    # test_image.save_image('test_line.png')

    ### Draw Axe

    # info_object = test_image.parse_file()

    # test_image.paint_points(info_object['point'])
    # test_image.paint_lines(info_object)
    # test_image.save_image('test_axe.png')

    test_info_object = {
        'point': [[-2.005891, -0.216794], [-1.208645, -0.530387], [-1.207215, -0.745505]],
        'connect_points': [[1, 2, 3]]
    }
    test_image.paint_points(test_info_object['point'])
    test_image.paint_lines(test_info_object)
    test_image.save_image('test_rectangle.png')






