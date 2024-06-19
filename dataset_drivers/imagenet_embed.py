import torch
from torchvision import models

class Resnet50_embed:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)


    def get_output(self, video, conv4d=False):
        assert len(video.shape) == 4
        if video.shape[1:] == (224, 224, 3):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2)).to(torch.float32)

        assert video.shape[-1] == 224 and video.shape[-2] == 224 and video.shape[-3] == 3
        if conv4d:
            resnet_model = torch.nn.Sequential(*(list(self.model.children())[:-3])).to(self.device)
        else:
            resnet_model = torch.nn.Sequential(*(list(self.model.children())[:-1])).to(self.device)
        resnet_model.eval()
        
        prev_t = 0
        length = video.shape[0]
        this_video = torch.Tensor().long().to('cpu')
        for j in range(100000):
            upper_bound = min((j + 1) * 50, length)
            video_chunk = video[prev_t:min((j + 1) * 50, length)].to(self.device)
            base_output = resnet_model(video_chunk)
            this_video = torch.cat((this_video, base_output.detach().cpu()), 0)
            torch.cuda.empty_cache()
            prev_t = upper_bound
            if (j + 1) * 50 >= length:
                break
        return this_video.squeeze().detach().cpu().numpy()