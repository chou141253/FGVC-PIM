import torch
import torch.nn.functional as F
import warnings

from models.builder import MODEL_GETTER
from utils.costom_logger import timeLogger

warnings.simplefilter("ignore")

@torch.no_grad()
def plot():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser("PIM-FGVC Heatmap Generation")
    parser.add_argument("--img", default="", type=str)
    parser.add_argument("--pretrained", default="", type=str)
    parser.add_argument("--model_type", default="swin-t", type=str)
    args = parser.parse_args()

    build_model()

