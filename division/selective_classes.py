class relation:
    def __init__(self, me, you, similar):
        self.me = me
        self.you = you
        self.similar = similar


class Segmentation_node:
    def __init__(self, x1, y1, x2, y2, v):
        self.x = x1
        self.y = y1
        self.next_x = x2
        self.next_y = y2
        self.v = v

    def __lt__(self, other):
        return self.v < other.v