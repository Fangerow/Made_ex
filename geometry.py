import cv2 as cv
import numpy as np
from random import randint


class Geometry:
    def __init__(self):
        self.image = np.full((320, 320, 3), fill_value=255)

    @staticmethod
    def get_random_color():
        return (randint(0, 255), randint(0, 255), randint(0, 255))

    @staticmethod
    def get_random_thickness():
        return randint(0, 1)

    def __creare_circle(self, num_sir, shape=320):
        if not num_sir:
            return self.image
        bor = shape
        cash = []
        for square in range(num_sir):
            color = self.get_random_color()
            thickness = self.get_random_thickness()
            k_ = randint(25, 100)
            z, b, k = randint(25, bor - 25), randint(25, bor - 25), k_
            if z in cash or b in cash:
                continue
            if z + k_ >= bor or b + k_ >= bor or z - k <= 1 or b - k <= 1:
                k = 15
            cash.append(z)
            cash.append(b)
            self.image = cv.circle(self.image, (z, b), radius=k, color=color, thickness=thickness)
        return self.image

    def __creare_rectangle(self, num_req, shape=320):
        if not num_req:
            return self.image
        bor = shape
        cash = []
        for rectangle in range(num_req):
            color = self.get_random_color()
            thickness = self.get_random_thickness()
            k_1, k_2 = randint(25, 100), randint(25, 100)
            if abs(k_1 - k_2) < 5:
                continue
            z, b, k1, k2 = randint(25, bor - 25), randint(25, bor - 25), k_1, k_2
            if z in cash or b in cash:
                continue

            if z + k_1 >= bor or b - k_2 <= 1:
                k1, k2 = 15, 10
            cash.append(z)
            cash.append(b)
            self.image = cv.rectangle(self.image, (z, b - k2), (z + k1, b), color, thickness)
        return self.image

    def __creare_paral(self, num_paral, shape=320):
        if not num_paral:
            return self.image
        bor = shape
        cash = []
        for rectangle in range(num_paral):
            color = self.get_random_color()
            thickness = self.get_random_thickness()
            k_, d_ = randint(25, 100), randint(25, 100)
            a, b, k, d = randint(25, bor - 25), randint(25, bor - 25), k_, d_
            if a in cash or b in cash:
                continue

            if a + 2 * k_ >= bor or a + d_ >= bor or b + 2 * k_ >= bor:
                k = 20
                d = 15
            cash.append(a)
            cash.append(b)
            points = np.array([[[a, b], [a, b + k], [a + d, b + 2 * k], [a + d, b + k]]], np.int32)
            self.image = cv.polylines(self.image, [points], True, color, thickness)
        return self.image

    def creare_quadrate(self, num_sq, shape=320):
        if not num_sq:
            return self.image
        bor1 = shape
        cash = []
        for square in range(num_sq):
            color = self.get_random_color()
            thickness = self.get_random_thickness()
            k_ = randint(25, 100)
            z, b, k = randint(25, bor1 - 25), randint(25, bor1 - 25), k_
            if z in cash or b in cash:
                continue
            if z + k_ >= bor1 or b - k_ <= 1:
                k = 15
            cash.append(z)
            cash.append(b)
            self.image = cv.rectangle(self.image, (z, b - k), (z + k, b), color, thickness)
        return self.image

    def add_geom(self):
        num_sq, num_cir, num_rec, num_par = randint(11, 18), randint(0, 2), randint(0, 1), randint(0, 1)
        self.image = self.creare_quadrate(num_sq)
        self.image = self.__creare_circle(num_cir)
        self.image = self.__creare_rectangle(num_rec)
        self.image = self.__creare_paral(num_par)
        return self.image, num_sq
