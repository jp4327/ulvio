import os


class Parameters():
    def __init__(self):
        self.devices = [0]
        self.n_processors = 4

        self.imu_per_image = 10
        self.imu_int_prev = 0

        # Data Preprocessing
        self.img_w = 512   # original size is about 1226
        self.img_h = 256   # original size is about 370

        # Data Augmentation
        self.is_hflip = True
        self.is_color = False
        self.flag_imu_aug = False
        self.is_crop = False

        self.mlp_hidden_size = 128
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.mlp_dropout_out = 0.2
        self.clip = None
        self.batch_norm = True

        self.imu_hidden_size = 64 # imu channel size
        self.visual_f_len = 378 # latent visual feature size
        self.imu_f_len = 128 # latent inertial feature size

        self.dropout = 0
        self.imu_prev = 0

        # Training
        self.decay = 5e-6
        self.batch_size = 16
        self.pin_mem = True

par = Parameters()
