from torch import nn


class MessageFunction(nn.Module):
    """
  Module which computes the message for a given interaction.
  """

    def compute_message(self, raw_messages):
        return None


class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension, message_dimension, device):
        super(MLPMessageFunction, self).__init__()
        # Sequential 有序的容器
        self.mlp = nn.Sequential(    
            # nn.Linear(in_features, out_features) - 每个输出样本的大小 - 每个输入样本的大小
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.LeakyReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
            nn.LeakyReLU(),
        )
        self.device = device

    def compute_message(self, raw_messages):
        messages = self.mlp(raw_messages)

        return messages


class IdentityMessageFunction(MessageFunction):

    def compute_message(self, raw_messages):
        return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension, device):
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension, device)
    elif module_type == "identity":
        return IdentityMessageFunction()
