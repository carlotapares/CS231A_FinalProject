from torch.utils.tensorboard import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, learning_rate, duration, iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_testing(self, mpjpe, epoch):
        self.add_scalar("testing.mpjpe", mpjpe, epoch)