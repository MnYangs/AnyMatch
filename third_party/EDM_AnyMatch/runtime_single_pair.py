import torch
import cv2

from src.edm.edm import EDM
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.utils.plotting import make_matching_figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

torch.cuda.set_device(0)

config = get_cfg_defaults()
data_cfg_path = "configs/data/scannet_test_1500.py"
main_cfg_path = "configs/edm/indoor/edm_base.py"
config.merge_from_file(main_cfg_path)
config.merge_from_file(data_cfg_path)

config.EDM.COARSE.MCONF_THR = 0.2
config.EDM.COARSE.BORDER_RM = 2

_config = lower_config(config)
matcher = EDM(config=_config["edm"]).cuda()
state_dict = torch.load("weights/edm_outdoor.ckpt")["state_dict"]
matcher.load_state_dict(state_dict)
matcher = matcher.eval().cuda()

# Load example images
img0_pth = "assets/scannet_sample_images/scene0707_00_15.jpg"
img1_pth = "assets/scannet_sample_images/scene0707_00_45.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640, 480))  # input size shuold be divisible by 32
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.0
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.0

batch = {"image0": img0, "image1": img1}

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
total_ms = 0
warmup_num = 10
test_num = 100
with torch.no_grad():
    for _ in range(warmup_num):
        matcher(batch)
        
    print(f"run single pair {test_num} times..")
    
    torch.cuda.synchronize()
    for i in range(test_num):
        start_event.record()
        matcher(batch)
        end_event.record()
        torch.cuda.synchronize()
        total_ms += start_event.elapsed_time(end_event)

print("avg_ms: ", total_ms / test_num)


# Draw result
mkpts0 = batch['mkpts0_f'].cpu().numpy()
mkpts1 = batch['mkpts1_f'].cpu().numpy()
mconf = batch['mconf'].cpu().numpy()
color = cm.jet(mconf)
text = [
    'EDM',
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
plt.show()