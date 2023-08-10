import matplotlib.pyplot as plt

# Setting global parameters for font sizes
plt.rcParams["font.size"] = 15  # Default font size
plt.rcParams["axes.labelsize"] = 15  # Axes label font size
plt.rcParams["axes.titlesize"] = 16  # Axes title font size
plt.rcParams["xtick.labelsize"] = 15  # X-tick label font size
plt.rcParams["ytick.labelsize"] = 15  # Y-tick label font size
plt.rcParams["legend.fontsize"] = 15  # Legend font size
plt.rcParams["figure.titlesize"] = 17  # Set font size for suptitle
plt.rcParams["figure.titleweight"] = "bold"  # Set font weight for suptitle

# Setting global parameters for font family
font_name = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = font_name
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['text.usetex'] = True
