import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def jsondataset_collate_fn(datas):
    """
       data: is a list of tuples with (example, label, length)
    """
    batch_videos = []
    batch_times = []
    floaterize = False
    for video, times, _ in datas:
        if type(video) == str:
            pass
        else:
            floaterize = True
            video = video.to(device)
        batch_videos.append(video)
        batch_times.append(times)
    if floaterize:
        batch_videos = [item.float() for item in batch_videos]
    return batch_videos, batch_times