import logging
import sys
import os
from pathlib import Path
from matplotlib import cm
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()


DIR2CACHE = ""
if not Path(DIR2CACHE).is_dir():
    DIR2CACHE = "~/.cachedir"

SCRIPT_NAME = Path(sys.argv[0]).stem

# PATHS
ROOT_DIR = Path(os.getenv("DIR2ROOTDIR", Path(os.path.dirname(os.path.abspath(__file__)))/".."))

DIR2DATA = ROOT_DIR / "data"
DIR2SEMI_SIMULATED_DATA = DIR2DATA / "datasets"

DIR2ACIC_2016 = DIR2SEMI_SIMULATED_DATA / "acic_2016"
DIR2ACIC_2018 = DIR2SEMI_SIMULATED_DATA / "acic_2018"
DIR2ACIC_2018_SCALING = DIR2ACIC_2018 / "scaling"
PATH2ACIC_2018_X = DIR2ACIC_2018 / "x.csv"
PATH2ACIC_2018_PARAMS = DIR2ACIC_2018_SCALING / "params.csv"
DIR2ACIC_2018_F = DIR2ACIC_2018_SCALING / "factuals"
DIR2ACIC_2018_CF = DIR2ACIC_2018_SCALING / "counterfactuals"

DIR2TWINS = DIR2SEMI_SIMULATED_DATA / "TWINS"
DIR2IHDP = DIR2SEMI_SIMULATED_DATA / "IHDP"


DIR2NOTES = (
    ROOT_DIR
    / "/../papiers/papiers_matthieu/selecting_predictive_models_for_causal_inference/research_journal/notes_img"
)
DIR2PAPER_IMG = (
    ROOT_DIR / "../how_to_select_causal_models/images/"
)


DIR2FIGURES = ROOT_DIR / "figures" 
DIR2FIGURES.mkdir(exist_ok=True, parents=True)

DIR2FIGURES_T = DIR2FIGURES / "test"
DIR2FIGURES_T.mkdir(exist_ok=True, parents=True)

DIR2REPORTS = DIR2DATA / "analysis_tables"
DIR2REPORTS.mkdir(exist_ok=True, parents=True)

DIR2EXPES = DIR2DATA / "experiences"
DIR2EXPES.mkdir(exist_ok=True, parents=True)

# Graphics
sns.set(font_scale=1.7, style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.figsize": (14, 6),
        # "font.size": 20,
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)

CAUSAL_DF_COLUMNS = ["idx", "y_0", "y_1", "mu_0", "mu_1", "e", "y", "a"]

# Colors
TAB_COLORS = [cm.tab20(i) for i in range(20)]
untreated_color = TAB_COLORS[2]
treated_color = TAB_COLORS[0]

COLOR_MAPPING = {0: untreated_color, 1: treated_color}
LABEL_MAPPING = {0: "Control", 1: "Treated"}
LS_MAPPING = {0: "dotted", 1: "dashed"}

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Test
LOGGER_OUT = logging.getLogger("LOGGER_NAME")
