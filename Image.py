import math

from PIL import Image
import numpy as np
import random
import re
import time


class My_Image:
    def __init__(self, height=2000, width=2000, depth=2000):
        self.height = height
        self.width = width
        self.depth = depth
        self.matrix = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        self.scale = 0.2
        self.value_shift_image = (1, 1)
        self.file_path = 'Axe.obj'
        self.matrix_scale = np.array([
            [self.width, 0, self.width/2],
            [0, self.height, self.height/2],
            [0, 0, 1]
        ])

    def get_image(self):
        return Image.fromarray(self.matrix)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_scalable_size(self):
        return (self.height * self.scale, self.width * self.scale, self.depth * self.scale)

    def get_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

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

    def check_image_borders(self, point):
        if (point[0] >= 0 and point[0] <= self.width) \
                and (point[1] >= 0 and point[1] <= self.height):
            return False, point

        if point[0] < 0:
            point[0] = 0
        elif point[0] > self.width:
            point[0] = self.width

        if point[1] < 0:
            point[1] = 0
        elif point[1] > self.height:
            point[1] = self.height

        return True, point

    def paint_points(self, list_points, color=[214, 214, 214]):
        for point in list_points:
            _, point = self.check_image_borders(point)
            self.matrix[point[0]][point[1]] = color

    # def get_scalable_point(self, point, type=int):
    #     # print(self.matrix_scale)
    #     column_point = np.array([[point[0], point[1], point[2]]]).T
    #     scale_point = np.dot(self.matrix_scale, column_point).T
    #     return scale_point.astype(type)

    def get_scalable_point(self, point, type=int):
        size_image = self.get_scalable_size()

        half_width = (size_image[0] / 8)
        half_height = (size_image[1] / 8)
        half_depth = (size_image[2] / 8)

        first_coordinate = point[0]
        second_coordinate = point[1]
        third_coordinate = point[2]

        modified_first_coordinate = size_image[0] * self.value_shift_image[0] + first_coordinate * half_width
        modified_second_coordinate = second_coordinate * half_height + size_image[1] * self.value_shift_image[1]
        modified_third_coordinate = third_coordinate * half_depth

        if type == float:
            modified_first_coordinate = round(modified_first_coordinate, 2)
            modified_second_coordinate = round(modified_second_coordinate, 2)
            return [modified_first_coordinate, modified_second_coordinate, modified_third_coordinate]

        elif type == int:
            return [int(modified_first_coordinate), int(modified_second_coordinate), int(modified_third_coordinate)]

    def get_scalable_list(self, list_points, type=int):
        ending_list = []
        for i in list_points:
            ending_list.append(self.get_scalable_point(i, type))
        return ending_list

    def paint_lines(self, info_object):
        list_points = info_object.get('point')
        for current_points in info_object.get('connect_points'):
            count_points = len(current_points)
            for j in range(count_points):
                first_point, second_point = list_points[current_points[j % count_points] - 1], list_points[current_points[(j + 1) % count_points] - 1]
                self.paint_line(first_point, second_point)

    def paint_line(self, first_point, second_point):
        x1, y1, _ = self.get_scalable_point(first_point)
        x2, y2, _ = self.get_scalable_point(second_point)

        dx = x2 - x1
        dy = y2 - y1

        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

        if dx < 0: dx = -dx
        if dy < 0: dy = -dy

        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy

        x, y = x1, y1

        error, t = el / 2, 0

        self.paint_points([[x, y]])

        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            self.paint_points([[x, y]])

    def parse_file(self):
        with open(self.file_path, 'r') as obj_file:
            data_point = obj_file.read()

        info_object = {'point': [],
                       'connect_points': [],
                       'normal': [],
                       'normal_value': []}

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
                                float(list_value[0]), float(list_value[1]), float(list_value[2])
                            ])
            elif type == 'f':
                for values in [[list_value[0], list_value[1 + i], list_value[2 + i]] for i in range(len(list_value) - 2)]:
                    info_object['connect_points'].append(
                        tuple([int(i.split('/')[0]) for i in values])
                    )
                    info_object['normal'].append(
                        tuple([int(i.split('/')[2]) for i in values])
                    )

            elif type == 'vn':
                info_object['normal_value'].append([
                    # int(size_image[0] * self.value_shift_image[0]) +
                    # int((float(list_value[2]) * half_width)),
                    # int((float(list_value[0]) * half_height)
                    #      + size_image[1] * self.value_shift_image[1]
                    #     )
                    float(list_value[0]), float(list_value[1]), float(list_value[2])
                ])


        return info_object

    # def draw_triangles(self, info_object):
    #     for points in info_object.get('connect_points'):
    #         point_coordinate = [info_object.get('point')[point - 1] for point in points]
    #         normal_point_coordinate = self._normal(point_coordinate[0], point_coordinate[1], point_coordinate[2])
    #         value_angle_of_incidence = self._angle_of_incidence(normal_point_coordinate)
    #         if value_angle_of_incidence < 0:
    #             self._draw_barycentric_coordinates(point_coordinate, color=(0, 64*value_angle_of_incidence, 0))

    def draw_triangles(self, info_object):
        for points, normal in zip(info_object.get('connect_points'), info_object.get('normal')):
            point = [info_object.get('point')[point - 1] for point in points]
            normal_point_coordinate = self._normal(point[0], point[1], point[2])
            value_angle_of_incidence = self._angle_of_incidence(normal_point_coordinate)
            if value_angle_of_incidence < 0:
                self._draw_barycentric_coordinates(point, normal, info_object.get('normal_value'))

    # def _draw_barycentric_coordinates

    # def _draw_barycentric_coordinates(self, points, color=[127, 127, 127]):
    #     _, first_point = self.check_image_borders(self.get_scalable_point(points[0], type=float))
    #     _, second_point = self.check_image_borders(self.get_scalable_point(points[1], type=float))
    #     _, third_point = self.check_image_borders(self.get_scalable_point(points[2], type=float))
    #
    #     min_max_x = [min([first_point[0], second_point[0], third_point[0]]),
    #                  max([first_point[0], second_point[0], third_point[0]])]
    #
    #     min_max_y = [min([first_point[1], second_point[1], third_point[1]]),
    #                  max([first_point[1], second_point[1], third_point[1]])]
    #
    #     step = 1
    #
    #     z_buffer = np.full((self.height, self.width), np.inf, dtype=np.uint8)
    #
    #     for value_x in np.arange(min_max_x[0], min_max_x[1]+step, step):
    #         for value_y in np.arange(min_max_y[0], min_max_y[1]+step, step):
    #
    #             try:
    #                 x = int(value_x)
    #                 y = int(value_y)
    #
    #                 x0, y0, _ = first_point
    #                 x1, y1, _ = second_point
    #                 x2, y2, _ = third_point
    #
    #                 lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    #                 lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    #                 lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    #
    #                 ### Check sum barycentric coordinates == 0
    #                 # sum_lambda = lambda0 + lambda1 + lambda2
    #                 # print(f'lambda_0 {lambda0} lambda_1 {lambda1} lambda_2 {lambda2} Sum: {sum_lambda}')
    #
    #                 if lambda0 > 0 and lambda1 > 0 and lambda2 > 0:
    #                     z_streak = lambda0*first_point[2] + lambda1*second_point[2] + lambda2*third_point[2]
    #                     if z_buffer[x][y] > z_streak:
    #                         self.paint_points([[x, y]], color=color)
    #                         z_buffer[x][y] = z_streak
    #
    #             except ZeroDivisionError:
    #                 pass

    def _draw_barycentric_coordinates(self, points, normal, normal_value, color=[127, 127, 127]):
        _, first_point = self.check_image_borders(self.get_scalable_point(points[0], type=float))
        _, second_point = self.check_image_borders(self.get_scalable_point(points[1], type=float))
        _, third_point = self.check_image_borders(self.get_scalable_point(points[2], type=float))

        min_max_x = [min([first_point[0], second_point[0], third_point[0]]),
                     max([first_point[0], second_point[0], third_point[0]])]

        min_max_y = [min([first_point[1], second_point[1], third_point[1]]),
                     max([first_point[1], second_point[1], third_point[1]])]

        step = 1

        z_buffer = np.full((self.height, self.width), np.inf, dtype=np.uint8)

        for value_x in np.arange(min_max_x[0], min_max_x[1]+step, step):
            for value_y in np.arange(min_max_y[0], min_max_y[1]+step, step):

                try:
                    x = int(value_x)
                    y = int(value_y)

                    x0, y0, _ = first_point
                    x1, y1, _ = second_point
                    x2, y2, _ = third_point

                    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
                    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
                    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))

                    ### Check sum barycentric coordinates == 0
                    # sum_lambda = lambda0 + lambda1 + lambda2
                    # print(f'lambda_0 {lambda0} lambda_1 {lambda1} lambda_2 {lambda2} Sum: {sum_lambda}')

                    if lambda0 > 0 and lambda1 > 0 and lambda2 > 0:

                        z_streak = lambda0*first_point[2] + lambda1*second_point[2] + lambda2*third_point[2]

                        if z_buffer[x][y] > z_streak:
                            normal0 = normal_value[normal[0] - 1]
                            normal1 = normal_value[normal[1] - 1]
                            normal2 = normal_value[normal[2] - 1]

                            l0 = self._angle_of_incidence(normal0)
                            l1 = self._angle_of_incidence(normal1)
                            l2 = self._angle_of_incidence(normal2)

                            color = (0, 255 * (lambda0 * l0 + lambda1 * l1 + lambda2 * l2), 0)

                            self.paint_points([[x, y]], color=color)
                            z_buffer[x][y] = z_streak

                except ZeroDivisionError:
                    pass

    def _normal(self, point_1, point_2, point_3):  # нормаль к треугольнику по трем вершинам

        normal_vector = []

        edge_A_x = point_2[0] - point_1[0]
        edge_A_y = point_2[1] - point_1[1]
        edge_A_z = point_2[2] - point_1[2]

        edge_B_x = point_3[0] - point_1[0]
        edge_B_y = point_3[1] - point_1[1]
        edge_B_z = point_3[2] - point_1[2]

        mormal_x = edge_A_y * edge_B_z - edge_A_z * edge_B_y
        normal_vector.append(mormal_x)

        mormal_y = edge_A_z * edge_B_x - edge_A_x * edge_B_z
        normal_vector.append(mormal_y)

        mormal_z = edge_A_x * edge_B_y - edge_A_y * edge_B_x
        normal_vector.append(mormal_z)

        return normal_vector

    def _angle_of_incidence(self, n_vec):
        cos_angle = 0.0
        try:
            l_vec = [1, 1, 1]

            norma_n = math.sqrt(n_vec[0] ** 2 + n_vec[1] ** 2 + n_vec[2] ** 2)
            norma_l = math.sqrt(l_vec[0] ** 2 + l_vec[1] ** 2 + l_vec[2] ** 2)

            scalar_multi = n_vec[0] * l_vec[0] + n_vec[1] * l_vec[1] + n_vec[2] * l_vec[2]
            norma_multi = norma_n * norma_l

            cos_angle = scalar_multi / norma_multi
        except:
            pass

        return cos_angle

    def _rotate(self, point, alfa, betta, gamma):
        cos_alfa = math.cos(math.radians(alfa))
        sin_alfa = math.sin(math.radians(alfa))

        cos_betta = math.cos(math.radians(betta))
        sin_betta = math.sin(math.radians(betta))

        cos_gamma = math.cos(math.radians(gamma))
        sin_gamma = math.sin(math.radians(gamma))

        matrix_1 = np.array([[1, 0, 0], [0, cos_alfa, sin_alfa], [0, -sin_alfa, cos_alfa]])
        matrix_2 = np.array([[cos_betta, 0, sin_betta], [0, 1, 0], [-sin_betta, 0, cos_betta]])
        matrix_3 = np.array([[cos_gamma, sin_gamma, 0], [-sin_gamma, cos_gamma, 0], [0, 0, 1]])

        matrix_multi = matrix_1.dot(matrix_2)
        matrix_result = matrix_multi.dot(matrix_3)

        return np.round(np.dot(matrix_result, np.array(point).T).T, 3).tolist()

    def rotate_matrix(self, matrix, angel):
        for index in range(len(matrix)):
            matrix[index] = self._rotate(matrix[index], angel[0], angel[1], angel[2])

        return matrix


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

    info_object = test_image.parse_file()

    info_object['point'] = test_image.rotate_matrix(info_object['point'], [90, 90, 0])

    test_image.paint_points(test_image.get_scalable_list(info_object['point']))
    test_image.paint_lines(info_object)
    test_image.draw_triangles(info_object)
    test_image.save_image('test_axe.png')


    # test_info_object = {
    #     'point': [[-2.005891, -0.216794, -0.216794], [-1.208645, -0.530387, -0.216794], [-1.607215, -0.745505, -0.216794], [-1.007215, 0.445505, -0.216794]],
    #     'connect_points': [[1, 2, 3]]
    # }
    #
    # print(test_image.get_scalable_point(test_info_object.get('point')[0]))
    # test_image.new_get_scalable_point(test_info_object.get('point')[0])

    # test_info_object1 = {
    #     'point': [[-2.005891, -0.216794], [-1.208645, -0.530387], [-1.607215, -0.745505], [-1.007215, 0.445505]],
    #     'connect_points': [[3, 4, 1]]
    # }
    # test_info_object_2 = {
    #     'point': [[-2.005891, -0.216794], [-1.208645, -0.530387], [-1.007215, 0.445505]],
    #     'connect_points': [[1, 2, 3]]
    # }
    #
    # test_info_object['point'] = test_image.triangulation(test_info_object['point'])
    # test_image.paint_points(test_image.get_scalable_list(test_info_object['point']))
    # test_image.paint_lines(test_info_object)
    #
    # test_info_object1['point'] = test_image.triangulation(test_info_object1['point'])
    # test_image.paint_points(test_image.get_scalable_list(test_info_object1['point']))
    # test_image.paint_lines(test_info_object1)
    # test_image.get_barycentric_coordinates(test_info_object['point'])
    # test_image.get_barycentric_coordinates(test_info_object_2['point'])
    #
    # test_image.save_image('test_rectangle.png')


