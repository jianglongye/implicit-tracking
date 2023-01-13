from yacs.config import CfgNode as CN


def get_default_cfg():
    _C = CN()
    _C.VERSION = 2

    _C.DATA_DIR = "./data/waymo/processed/validation/"
    _C.DATASET = "waymo"
    _C.OUTPUT_DIR = "./output/debug"
    _C.OBJECT_LIST_PATH = "./data/waymo/splits/easy_list.json"
    _C.CKPT_PATH = "/root/code/shape-reconstruction/experiments/point_sdf/partial_point_weighted/ckpts/latest.pth"

    _C.EXP = CN()
    _C.EXP.USE_MEAN_CODE = False
    _C.EXP.UPDATE_NUM = 500
    _C.EXP.WO_ENCODER = False
    _C.EXP.RAW_PC = False
    _C.EXP.USE_DETECTION = False
    _C.EXP.MEAN_CODE_PATH = (
        " /root/code/shape-reconstruction/experiments/point_sdf/partial_point_weighted/code/mean.npz"
    )

    _C.DEBUG = CN()
    _C.DEBUG.IS_DEBUG = False

    _C.SHAPE_MODEL = CN()
    _C.SHAPE_MODEL.CODE_DIM = 512
    _C.SHAPE_MODEL.HIDDEN_DIM = 512
    _C.SHAPE_MODEL.POINT_FEAT_DIMS = [3, 64, 256, 512]
    _C.SHAPE_MODEL.DECODER_DIMS = [1024, 512, 256, 256, 1]
    _C.SHAPE_MODEL.USE_RES_DECODER = False

    _C.MOTION_MODEL = CN()
    _C.MOTION_MODEL.AVG_WEIGHT = 0.5
    _C.MOTION_MODEL.PREV_WEIGHT = 0.5

    _C.SEARCH_CODE = CN()
    _C.SEARCH_CODE.IF_FINETUNE = True
    _C.SEARCH_CODE.IF_ENCODE = False
    _C.SEARCH_CODE.FINETUNE_FIRST_FRAME = False
    _C.SEARCH_CODE.PREV_REG_WEIGHT = 5
    _C.SEARCH_CODE.INIT_REG_WEIGHT = 10
    _C.SEARCH_CODE.ZERO_REG_WEIGHT = 5
    _C.SEARCH_CODE.ITER_NUM = 100
    _C.SEARCH_CODE.LR = 1e-3
    _C.SEARCH_CODE.POLICY = "agg"  # 'prev', 'prev_init', 'agg'

    _C.LOSS = CN()
    _C.LOSS.SHAPE_LOSS_WEIGHT = 1.0
    _C.LOSS.SHAPE_LOSS_TYPE = "l1"
    _C.LOSS.USE_CD = True
    _C.LOSS.USE_SHAPE = True
    _C.LOSS.USE_MC = False
    _C.LOSS.USE_MP = False
    _C.LOSS.CD_WEIGHT = 0.4
    _C.LOSS.CD_LOSS_TYPE = "smooth_l1"
    _C.LOSS.CD_SMOOTH_L1_LOSS_BETA = 0.04

    _C.CD_LOSS = CN()
    _C.CD_LOSS.USE_PREV_PC = True
    _C.CD_LOSS.USE_INIT_PC = True
    _C.CD_LOSS.USE_AGG_PC = True
    _C.CD_LOSS.PREV_PC_WEIGHT = 1.0
    _C.CD_LOSS.INIT_PC_WEIGHT = 1.0
    _C.CD_LOSS.AGG_PC_WEIGHT = 1.0
    _C.CD_LOSS.PREV_PC_DIR = "double"
    _C.CD_LOSS.INIT_PC_DIR = "double"
    _C.CD_LOSS.AGG_PC_DIR = "x-y"
    _C.CD_LOSS.MERGE_INIT_TO_PREV = False
    _C.CD_LOSS.MERGE_AGG_TO_PREV = False
    _C.CD_LOSS.MaxPCNum = 2000

    _C.OPTIM = CN()
    _C.OPTIM.OPTIMIZER = "SGD"
    _C.OPTIM.INIT_LR = 0.1
    _C.OPTIM.GAMMA = 0.5
    _C.OPTIM.MILESTONES = [100, 200, 300, 400]
    _C.OPTIM.ITER_NUM = 300

    return _C.clone()
