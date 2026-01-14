import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["font.monospace"] = ["FreeMono"]
matplotlib.rcParams['figure.figsize'] = (12, 10)
matplotlib.rcParams['figure.dpi'] = 600
matplotlib.rcParams["axes.labelsize"] = 18
matplotlib.rcParams["axes.titlesize"] = 18
matplotlib.rcParams["legend.fontsize"] = 16
matplotlib.rcParams["figure.titlesize"] = 30
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
hist_params = {
    "histtype": "step",
    "linewidth": 5,
    "cumulative": True,
    "density": True
}