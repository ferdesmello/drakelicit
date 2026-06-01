

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
import json

#-------------------------------------------------------------------------------------------
print("Hello. Starting!")

#-------------------------------------------------------------------------------------------
# Read and parse the JSON file
with open("computed.json", "r") as file:
    parsed_data = json.load(file)

# Drill straight down into 'community_N'
community_data = parsed_data["community_N"]

# Extract your target X and Y arrays
Ns = community_data["log_N_grid"]
freq = community_data["pdf"]


#-------------------------------------------------------------------------------------------
# Make the figures
print("Making the figures...")

# Custom dark gray color between dimgray and black
custom_dark_gray = (0.2, 0.2, 0.2)

# Create a figure object with size 6x8 inches and 2 subfigs
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(Ns, freq, marker='o', linestyle='-', color='b', label='test')

# Design
ax.set_xlabel("$N$", fontsize=12, color=custom_dark_gray)
ax.set_ylabel("Frequency", fontsize=10, color=custom_dark_gray)
ax.set_title("PDF of number of civilizations in our Galaxy", fontsize=10, pad=10, color=custom_dark_gray)
#ax.grid(True, linestyle=":", linewidth="1")

# Axes
#ax.set_xscale("log")

ax.minorticks_on()
ax.tick_params(which="major", direction="out", length=4, color=custom_dark_gray)
ax.tick_params(which="minor", direction="in", length=0)
ax.tick_params(which="both", bottom=True, top=False, left=True, right=False)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color(custom_dark_gray)
ax.spines['bottom'].set_color(custom_dark_gray)
ax.tick_params(axis='both', colors=custom_dark_gray)
                 
#-------------------------------------------------------------------------------------------
# Adjust the vertical and horizontal spacing, so there are no overlapings
plt.tight_layout()

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
print("All done!")

#-------------------------------------------
# Call plt.show() to make the graphics appear.
plt.show()