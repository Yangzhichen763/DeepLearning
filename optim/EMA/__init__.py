from torch.optim.swa_utils import AveragedModel


# 官方 pytorch 库中提供了 Averaging Weights 算法的实现，可以直接使用。
# 使用方法可以参考：https://zhuanlan.zhihu.com/p/479898259
class ExponentialMovingAverage(AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avh_model_param, model_param, num_averaged):
            return decay * avh_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)
