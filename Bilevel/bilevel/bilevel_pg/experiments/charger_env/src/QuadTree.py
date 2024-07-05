

import numpy as np

# tree.py

import numpy as np

# 定义四叉树节点类
class QuadTreeNode:
    def __init__(self, boundary, capacity):
        self.boundary = boundary  # [x_min, x_max, y_min, y_max]
        self.capacity = capacity
        self.chargers = []
        self.divided = False

    def subdivide(self):
        x_min, x_max, y_min, y_max = self.boundary
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        self.northwest = QuadTreeNode([x_min, x_mid, y_mid, y_max], self.capacity)
        self.northeast = QuadTreeNode([x_mid, x_max, y_mid, y_max], self.capacity)
        self.southwest = QuadTreeNode([x_min, x_mid, y_min, y_mid], self.capacity)
        self.southeast = QuadTreeNode([x_mid, x_max, y_min, y_mid], self.capacity)
        self.divided = True

    def insert(self, charger):
        if not self._contains(self.boundary, charger):
            return False

        if len(self.chargers) < self.capacity:
            self.chargers.append(charger)
            return True

        if not self.divided:
            self.subdivide()

        if self.northwest.insert(charger): return True
        if self.northeast.insert(charger): return True
        if self.southwest.insert(charger): return True
        if self.southeast.insert(charger): return True

    def _contains(self, boundary, charger):
        x, y = charger['x'], charger['y']
        x_min, x_max, y_min, y_max = boundary
        return x_min <= x <= x_max and y_min <= y <= y_max

    def query(self, range, found):
        if not self._intersects(range, self.boundary):
            return

        for charger in self.chargers:
            if self._contains(range, charger):
                found.append(charger)

        if self.divided:
            self.northwest.query(range, found)
            self.northeast.query(range, found)
            self.southwest.query(range, found)
            self.southeast.query(range, found)

    def _intersects(self, range, boundary):
        rx_min, rx_max, ry_min, ry_max = range
        x_min, x_max, y_min, y_max = boundary
        return not (rx_min > x_max or rx_max < x_min or ry_min > y_max or ry_max < y_min)
