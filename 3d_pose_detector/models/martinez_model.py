import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class LinearBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(LinearBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p_dropout)
        )

    def forward(self, x):
        out = self.model(x)
        return out + x


class MartinezModel(nn.Module):
    def __init__(self, input_size, output_size, linear_size=1024, num_blocks=2, p_dropout=0.5):
        super(MartinezModel, self).__init__()

        self.num_blocks = num_blocks

        # pre-process
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p_dropout)
        )

        self.linear_blocks = []
        for _ in range(num_blocks):
            self.linear_blocks.append(LinearBlock(linear_size, p_dropout))
        self.linear_blocks = nn.ModuleList(self.linear_blocks)

        # final output from the model head
        self.model_head = nn.Linear(linear_size, output_size)

    def forward(self, x):
        # pre-processing
        out = self.preprocess(x)

        # linear blocks
        for i in range(self.num_blocks):
            out = self.linear_blocks[i](out)

        out = self.model_head(out)

        return out
