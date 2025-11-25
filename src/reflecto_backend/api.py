from pathlib import Path

import numpy as np
from reflecto.io_utils import load_xrr_dat
from reflecto.physics_utils import tth2q
from reflecto.simulate.simul_genx import ParamSet

from .core.inference import XRRInferenceEngine

WEIGHTS_DIR = Path(r"data/models")

def ai_guess(tths: np.ndarray, refl: np.ndarray, wavelen:float = 1.54) -> tuple[list[ParamSet], ParamSet]:

    inference_engine: XRRInferenceEngine = XRRInferenceEngine(exp_dir=WEIGHTS_DIR / "251126_6out")
    qs: np.ndarray = tth2q(tths, wavelen)
    preds: np.ndarray = inference_engine.predict(qs, refl)
    pred_f_d, pred_f_sig, pred_f_sld = preds[0], preds[1], preds[2]
    pred_s_d, pred_s_sig, pred_s_sld = preds[3], preds[4], preds[5]

    return [ParamSet(pred_f_d, pred_f_sig, pred_f_sld)], ParamSet(pred_s_d, pred_s_sig, pred_s_sld)


if __name__ == "__main__":
    data_file: Path = Path("./data") / "example_data" / "#1.dat"
    tths, refl = load_xrr_dat(data_file)
    film_params, sio2_param = ai_guess(tths, refl)
    print("SiO2:", sio2_param)
    print("Films")
    for param in film_params:
        print(param)