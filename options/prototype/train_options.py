from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays

        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')

        # for training (Note: in train_errnet.py, we mannually tune the training protocol, but you can also use following setting by modifying the code in errnet_model.py)
        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')


        # data augmentation
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--stage',type=int,default=0,help='train_stage')
        self.parser.add_argument('--Dlayers', type=int, default=5, help='train_stage')
        self.parser.add_argument('--skip', action='store_true')
        self.parser.add_argument('--fullreal', action='store_true')
        self.parser.add_argument('--indoor', action='store_true')
        self.parser.add_argument('--dadataset', action='store_true')
        self.parser.add_argument('--dcp', action='store_true')
        self.parser.add_argument('--darts', action='store_true')
        self.parser.add_argument('--continue', action='store_true')
        self.parser.add_argument('--holygrail', action='store_true')
        self.parser.add_argument('--Debug', action='store_true')
        self.parser.add_argument('--concise', action='store_true')

        self.isTrain = True
