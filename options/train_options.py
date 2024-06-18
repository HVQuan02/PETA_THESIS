from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--resume', default=None, help='checkpoint to resume training')
        parser.add_argument('--train_batch_size', type=int, default=5, help='train batch size')
        parser.add_argument('--val_batch_size', type=int, default=20, help='validate batch size')
        parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
        parser.add_argument('--lr_policy', type=str, default='cosine', choices=['cosine', 'step', 'multi_step', 'onecycle'])
        parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
        parser.add_argument('--lr_gamma', type=float, default=0.5)
        parser.add_argument('--lr_step', type=int, default=10)
        parser.add_argument('--lr_milestones', nargs="+", type=int, default=[30, 60, 90], help='milestones of learning decay')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd optimizer')
        parser.add_argument('--warmup_epoch', type=int, default=10, help='number of warmup epochs')
        parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train')
        parser.add_argument('--save_dir', default='/kaggle/working/weights', help='directory to save checkpoints')
        parser.add_argument('--loss', type=str, default='asymmetric', help='loss function')
        parser.add_argument('--patience', type=int, default=10, help='patience of early stopping')
        parser.add_argument('--min_delta', type=float, default=0.1, help='min delta of early stopping')
        parser.add_argument('--threshold', type=float, default=90, help='val mAP threshold of early stopping')

        return parser