import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import csv
import os

# All our data-files are saved in the folder data
os.chdir('../data')
files = os.listdir('.')


# This is the list, where all our data will be saved in
data_list = []

# We only want our data-files, not some random hidden file
for element in files:
    if element[0] == '.':
        files.remove(element)

# Now we get all our different parameters
for element in files:
    flip = int(element[0])
    cooling_parameter = float(element[2:10])
    data_list.append([flip, cooling_parameter])
    with open(element, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            data_list[-1].append(float(row[1]))

data_list.sort(key=lambda tup: tup[1])

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

l_min = 4292.14


def Cumulative(x):
    return 0.5*(1 + erf(x/np.sqrt(2)))


def Errf(length, mean, sterr):
    return (Cumulative((length - mean)/sterr)) / \
           (1 - Cumulative((l_min - mean)/sterr))


def Distribution(x, mean, sterr):
    return 1/(sterr*np.sqrt(2*np.pi))*np.exp(-(x - mean)*(x - mean) /
                                            (2*sterr*sterr))


def GaussianDistribution(length, mean, sterr):
    variance = sterr*sterr
    return 1/np.sqrt(2*np.pi*variance)*np.exp(-(length - mean) *
                    (length - mean)/(2*variance))*2/(np.sqrt(variance) *
                    (1 - erf((l_min - mean)/np.sqrt(2*variance))))


flip_1 = np.array([x[1:] for x in data_list if x[0] == 1])
flip_2 = np.array([x[1:] for x in data_list if x[0] == 2])

#print(flip_1)

fig, ax = plt.subplots()
ax.set_xlabel('Weglänge [km]')
ax.set_ylabel('Wahrscheinlichkeit')
ax.grid()
for element in flip_1:
    values = np.sort(element[1:])
    points = values.size
    values, indices = np.unique(values, return_counts=True)
    cumul = np.cumsum(indices/points)
    popt, pcov = curve_fit(Errf, values, cumul, [4500, 1000])
    x = np.linspace(l_min, 6500, 500)
    ax.plot(x, GaussianDistribution(x, popt[0], popt[1]),
            label=r'$C=%.3f, \mu=%.1f, \sigma=%.1f$' % (element[0], popt[0], popt[1]))

ax.set_xlim(4000, 6500)
ax.set_ylim(0., 4*1e-5)
ax.legend(loc='best')
fig.tight_layout()
fig.savefig('../plt/distribution_1.pdf', transparent=True, dpi=300)

fig, ax = plt.subplots()
ax.set_xlabel('Weglänge [km]')
ax.set_ylabel('Wahrscheinlichkeit')
ax.grid()
for element in flip_2:
    values = np.sort(element[1:])
    points = values.size
    values, indices = np.unique(values, return_counts=True)
    cumul = np.cumsum(indices/points)
    popt, pcov = curve_fit(Errf, values, cumul, [4500, 1000])
    x = np.linspace(l_min, 6500, 500)
    ax.plot(x, GaussianDistribution(x, popt[0], popt[1]),
            label=r'$C=%.3f, \mu=%.1f, \sigma=%.1f$' % (element[0], popt[0], popt[1]))

ax.set_xlim(4000, 6500)
ax.set_ylim(0., 4*1e-5)
ax.legend(loc='best')
fig.tight_layout()
fig.savefig('../plt/distribution_2.pdf', transparent=True, dpi=300)
