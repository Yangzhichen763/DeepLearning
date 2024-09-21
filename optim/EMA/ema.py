

class EMA:
    """
    EMA: Exponential Moving Average
    对模型的参数做平均，以求提高测试指标并增加模型鲁棒性
    """
    def __init__(self, model, decay=0.9999):
        assert 0.0 < decay < 1.0, "Decay must be between 0.0 and 1.0"

        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """
        注册模型参数
        一般在 EMA 的初始化之后紧接着进行
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()  # param.detach().clone().data

    def forward(self):
        self.update()

    def update(self):
        """
        更新 shadow weights
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average  # .clone()

    def apply_shadow(self):
        """
        保存当前的模型参数（用于恢复），并应用 shadow weights 作为模型参数
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        恢复上一次的模型参数
        不会影响到原来优化过程，对使用 EMA 进行 validate 模型有大用
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]

        self.backup = {}


if __name__ == '__main__':
    import torch
    _model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(_model.parameters(), lr=0.1)

    # 初始化
    ema = EMA(_model, 0.999)
    ema.register()

    # 训练过程中，更新完参数后，同步 update shadow weights
    def train():
        optimizer.step()
        ema.update()

    # eval 之前，apply shadow weights；eval 之后，恢复原来模型的参数
    def evaluate():
        ema.apply_shadow()
        ema.restore()
