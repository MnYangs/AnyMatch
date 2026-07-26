from third_party.EDM_AnyMatch.src.config.default import _CN as cfg

cfg.TRAINER.CANONICAL_BS = 8*4  #  4*4
cfg.TRAINER.CANONICAL_LR = 2e-3  #  2e-3
cfg.TRAINER.WARMUP_STEP = int(4125*8 / cfg.TRAINER.CANONICAL_BS * 2)  # 18600,37300,3050
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [3,6,9]    #原来为[8, 12, 16, 20, 24]
cfg.TRAINER.EPI_ERR_THR = 1e-4

cfg.EDM.COARSE.MCONF_THR = 0.05  #原来为0.05
cfg.EDM.FINE.SIGMA_THR = 1e-6   #原来为1e-6
cfg.EDM.COARSE.BORDER_RM = 0

# Top-K should not exceed grid_size = TEST_RES_H / 8 * TEST_RES_W / 8
# The recommended value is approximately grid_size * 0.35 for Megadepth
# cfg.EDM.COARSE.TOPK = int(832 / 8 * 832 / 8 * 0.35)  # 3786 for train & LO-RANSAC test
cfg.EDM.COARSE.TOPK = int(832 / 8 * 832 / 8 * 0.35)  # 7258 for test  train：1152
