from src.augmentors.augmentor import SAugmentor


class Identity(SAugmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, x, OP1, OP2, num_n, num_e, num_t, index):
        return x, OP1, OP2, num_n, num_e, num_t, index
