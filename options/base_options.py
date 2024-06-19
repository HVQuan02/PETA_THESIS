import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--seed', default=2024, help='seed for randomness')
        parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
        parser.add_argument('--num_classes', type=int, default=23)
        parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec'])
        parser.add_argument('--dataset_path', type=str, default='/kaggle/input/thesis-cufed/CUFED')
        parser.add_argument('--split_dir', type=str, default='/kaggle/input/cufed-full-split')
        parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--album_clip_length', type=int, default=32)
        parser.add_argument('--remove_model_jit', type=int, default=None)
        parser.add_argument('--use_transformer', type=int, default=1)
        parser.add_argument('--transformers_pos', type=int, default=1)
        parser.add_argument('--threshold', type=float, default=0.75)
        parser.add_argument('-v', '--verbose', action='store_true', help='show details')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt