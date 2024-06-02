import torch
import math
import torch.nn.functional as F

class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_size = 8, max_length=5):
        self.hidden_size = hidden_size

        # For positional encoding
        num_timescales = self.hidden_size // 2  ##一半余弦，一半正弦
        max_timescale = 10.0
        min_timescale = 1.0 ##max_timescale min_timescale是时间尺度的上下界

        ##以上：计算时间尺度

        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)

        print("log_timescale_increment: ", log_timescale_increment)

        ##感觉是（max_timescale-min_timescale）取对数。
        ##在对数空间中相邻时间尺度之间的增量
        ##计算时间尺度的增量

        self.inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)

        print("inv_timescales: ", self.inv_timescales)

        ##计算时间尺度的值  
        ##inv_timescales是时间尺度的倒数

        # register_buffer('inv_timescales', inv_timescales)
 
    def get_position_encoding(self, x):
        max_length = x.size()[1] ##x传入的参数是input B*N*D，所以这里取的是N
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        ##position=tensor([0.，1.,2.,……，max_length-1.])，维度是N
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        ##position.unsqueeze(1)后维度是N*1，inv_timescales.unsqueeze(0)后维度是1*D/2
        ##所以scaled_time维度是N*D/2
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        ##signal维度是N*D
        signal = signal.view(1, max_length, self.hidden_size)
        ##把signal拉成(1*N*D)
        return signal
        
if __name__ == "__main__":

    pos_encoding = PositionalEncoding(hidden_size=8, max_length=5)
    encoding = pos_encoding.get_position_encoding(torch.randn(1, 5, 8))
    print("shape is: ", encoding.shape)
    print("encoding is: ", encoding)
