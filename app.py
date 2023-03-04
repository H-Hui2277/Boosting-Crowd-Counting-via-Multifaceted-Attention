import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import math
from models.vgg_c import vgg19_trans

class Counter(object):
    '''@d 初始化counter
    @p state_dir 模型state_dict保存的路径
    @p device 使用的设备'0'或者'cpu'
    '''
    def __init__(self, state_dir=None, device='cpu') -> None:
        self.device = torch.device(device)
        # init model
        self.model = vgg19_trans()
        self.model.to(self.device)
        self.model.eval()
        if state_dir != None:
            self.model.load_state_dict(torch.load(state_dir, self.device))
        # init transform
        self.trans = Compose([
            ToTensor(), 
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    '''@d 计算一张图中的人数
    @p img tensor
    '''
    def __call__(self, img):
        # load img.
        img = img if isinstance(img, torch.Tensor) \
            else self.trans(img)
        c, h, w = img.size()
        assert c == 3
        img = img.view(1, c, h, w).to(self.device)
        h, w = int(h), int(w)
        img_blocks = []
        if h >= 3584 or w >= 3584:
            h_stride = int(math.ceil(1.0 * h / 3584))
            w_stride = int(math.ceil(1.0 * w / 3584))
            h_step = h // h_stride
            w_step = w // w_stride
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * h_step
                    if i != h_stride - 1:
                        h_end = (i + 1) * h_step
                    else:
                        h_end = h
                    w_start = j * w_step
                    if j != w_stride - 1:
                        w_end = (j + 1) * w_step
                    else:
                        w_end = w
                    img_blocks.append(img[:, :, h_start:h_end, w_start:w_end])
            with torch.no_grad():
                crowd_count = 0.0
                for _, img_block in enumerate(img_blocks):
                    output = self.model(img_block)[0]
                    crowd_count += torch.sum(output).item()
        else:
            with torch.no_grad():
                outputs = self.model(img)[0]
                crowd_count = torch.sum(outputs).item()
        return crowd_count
    
