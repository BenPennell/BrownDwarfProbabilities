import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})
#matplotlib.rcParams['figure.figsize'] = (12, 10)
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