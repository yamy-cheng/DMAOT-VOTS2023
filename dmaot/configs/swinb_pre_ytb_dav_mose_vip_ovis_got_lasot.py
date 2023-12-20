import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='SwinB_DeAOTL',stage='default'):
        super().__init__(exp_name, model)
        if stage == 'default':
            self.STAGE_NAME = 'PRE_YTB_DAV'
        else:
            self.STAGE_NAME = stage
        if self.STAGE_NAME == 'PRE':
            self.DATASETS = ['static']

            self.DATA_DYNAMIC_MERGE_PROB = 1.0

            self.TRAIN_LR = 4e-4
            self.TRAIN_LR_MIN = 2e-5
            self.TRAIN_WEIGHT_DECAY = 0.03
            self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
            self.TRAIN_AUX_LOSS_RATIO = 0.1

            self.init_dir(data='/home/cym/datasets',root='/home/cym/proj/VOS03-de',eval='./')

            self.DATA_PRE_STRONG_AUG = True
            self.DATA_TPS_PROB = 0.3
            self.DATA_TPS_SCALE = 0.02


        elif self.STAGE_NAME == 'PRE_YTB_DAV':

            self.DATASETS = ['youtubevos', 'davis2017', 'lasot', 'got10k', 'mose', 'vipseg', 'ovis']
        
            self.init_dir(data='/home/cym/datasets',root='/home/cym/proj/VOS03-de',eval='./')   # modify
            self.DATA_YTB_REPEAT = 4
            self.DATA_DAVIS_REPEAT = 15
            self.DATA_LASOT_REPEAT = 6
            self.DATA_MOSE_REPEAT = 8
            self.DATA_VIPSEG_REPEAT = 1
            self.DATA_GOT10K_REPEAT = 1
            self.DATA_OVIS_REPEAT = 8

            self.DATA_DYNAMIC_MERGE_PROB_OVIS = 0.2
            self.DATA_DYNAMIC_MERGE_PROB_MOSE = 0.2
            self.DATA_DYNAMIC_MERGE_PROB_VIP = 0
            self.DATA_DYNAMIC_MERGE_PROB_LASOT = 0.5
            self.DATA_DYNAMIC_MERGE_PROB_GOT10K = 0.5

            pretrain_exp = self.EXP_NAME
            pretrain_stage = 'PRE'
            pretrain_ckpt = 'save_step_100000.pth'
            self.PRETRAIN_FULL = True  # if False, load encoder only
            self.PRETRAIN_MODEL = os.path.join(self.DIR_ROOT, 'result',
                                            pretrain_exp, pretrain_stage,
                                            'ema_ckpt', pretrain_ckpt)
            
            self.TRAIN_SAVE_MED_STEP = 10000
            self.TRAIN_START_SAVE_MED_RATIO = 0.4

            self.DATA_RANDOM_GAUSSIAN_BLUR = 0.3
            self.DATA_RANDOM_GRAYSCALE = 0.2
            self.DATA_RANDOM_COLOR_JITTER = 0.8

