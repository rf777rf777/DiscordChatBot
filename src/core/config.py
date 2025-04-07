from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

from diffusers import PNDMScheduler, LMSDiscreteScheduler
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from diffusers import UniPCMultistepScheduler,DPMSolverSDEScheduler

class Settings:
    #API基本資料設定
    PROJECT_NAME_zh_TW: str = "隨機塔羅"
    PROJECT_NAME_EN_US: str = "Random Tarot"
    VERSION: str = "1.0.0"

    #router設定
    ROUTER_NAME_SDImage, ROUTER_Description_SDImage = ('SDImage', 'Stable Diffusion Image')

    #Stable Diffusion Samplers
    SD_SAMPLERS = {
        "ddim": {  # DDIM (Denoising Diffusion Implicit Models)
            "cls": DDIMScheduler,
            "kwargs": {}
        },
        "pndm": {  # PNDM (Pseudo Numerical Methods for Diffusion Models)
            "cls": PNDMScheduler,
            "kwargs": {}
        },
        "lms": {  # LMS (Laplacian Pyramid Sampling)
            "cls": LMSDiscreteScheduler,
            "kwargs": {}
        },
        "euler": {  # Euler
            "cls": EulerDiscreteScheduler,
            "kwargs": {}
        },
        "euler_a": {  # Euler A (Ancestral)
            "cls": EulerAncestralDiscreteScheduler,
            "kwargs": {}
        },
        "unipc": {  # UniPC (Unified Predictor-Corrector)
            "cls": UniPCMultistepScheduler,
            "kwargs": {}
        },
        "dpmpp_2m": {  # DPM++ 2M
            "cls": DPMSolverMultistepScheduler,
            "kwargs": {
                "use_karras_sigmas": False,
                "algorithm_type": "dpmsolver++"
            }
        },
        "dpmpp_2m_karras": {  # DPM++ 2M Karras
            "cls": DPMSolverMultistepScheduler,
            "kwargs": {
                "use_karras_sigmas": True,
                "algorithm_type": "dpmsolver++"
            }
        },
        "dpmpp_2m_sde": {  # DPM++ 2M SDE Karras
            "cls": DPMSolverSDEScheduler,
            "kwargs": {
                "use_karras_sigmas": True,
                "algorithm_type": "dpmsolver++_sde"
            }
        }
    }
settings = Settings()