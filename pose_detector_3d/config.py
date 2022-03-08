from yacs.config import CfgNode as CN


_C  =  CN()

# General arguments
_C.dataset = 'infiniteform'
_C.actions = '*' # which actions to test on. * means all
_C.eval_checkpoint = 'checkpoints/2022-03-03_20-49-48/ckpt_best.pth.tar'  # checkpoint to evaluate from
#_C.eval_checkpoint = 'checkpoints/ckpt_linear.pth.tar'  # checkpoint to evaluate from
_C.eval_file_keypoints = 'data/data_2d_infiniteform_0305.npy' # file containing 2D keypoints that we will run eval on to get 3D keypoints
_C.eval_save_dir = 'predictions'
_C.train_checkpoint = '' # checkpoint to continue training from
_C.train_dataset = ''
_C.device = 'cuda'
_C.save_every = 10
_C.max_norm = True  # whether to use the max norm constraint on the weights of each layer


# Model arguments
_C.batch_size = 128
_C.linear_size = 1024
_C.epochs = 200
_C.num_workers = 8
_C.lr = 1e-3
_C.lr_decay = 100000
_C.lr_gamma = 0.96 # gamma of lr_decay