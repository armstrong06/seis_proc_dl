from apply_to_continuous.apply_models_config import CFG
from apply_to_continuous import apply_models_clean
applier = apply_models_clean(CFG)
applier.apply_to_data()