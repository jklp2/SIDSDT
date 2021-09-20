from .prototype import Model as proModel
from .basic import Model as basicModel
from .basic_stage import  Model as basic_stageModel
from .basic_stage import  Model_final_stage as basic_stageModel_final
from .demo import Model as demo_model
from .basic_unet import Model as basic_unetModel
from .basic_darts import Model as basic_dartsModel
from .basic_stage_unrolled import Model_final_stage as basic_stageModel_final_unrolled
from .darts_stage import Model_final_stage as darts_stageModel_final
from .Composite_Reuse_Autocoder import  Model as CRAModel
from .Composite_Reuse_Autocoder_unrolled import  Model as CRAModel_unrolled
from .Composite_Reuse_Autocoder_unrolled import Model_final_stage as CRAModel_unrolled_final
from .Composite_Reuse_Autocoder_unrolled import Model_S2S as CRAModel_unrolled_S2S

from .FHA import Model_overall as FHAModel
from .FHA import Model_cyclegan as FHAModel_cyc
from .FHA import Model_cycenhance as CYC_enhance
from .Semi import Model as Semi
from .Composite_Reuse_Autocoder_unrolled import Model_final_distill as CRA_final_distill

def prototype():
    return proModel()

def basic():
    return basicModel()

def basic_stage():
    return basic_stageModel()

def basic_stage_final():
    return basic_stageModel_final()

def demo():
    return demo_model()

def basic_unet():
    return basic_unetModel()

def basic_darts():
    return basic_dartsModel()

def basic_stage_final_unrolled():
    return basic_stageModel_final_unrolled()

def darts_stage_final():
    return darts_stageModel_final()

def cra():
    return CRAModel()

def cra_unrolled():
    return CRAModel_unrolled()

def cra_unrolled_s2s():
    return CRAModel_unrolled_S2S()

def cra_unrolled_final():
    return CRAModel_unrolled_final()

def fha():
    return FHAModel()

def fha_cyc():
    return FHAModel_cyc()

def semi():
    return Semi()

def cyc_enhance():
    return CYC_enhance()

def distill():
    return CRA_final_distill()