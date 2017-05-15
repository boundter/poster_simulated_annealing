import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import os

plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams['font.size'] = 20

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
    with open(element, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        time_sum = 0
        line_count = 0
        for row in csv_reader:
            time_sum += float(row[0])
            line_count += 1.
        mean_time = time_sum/line_count
        flip = int(element[0])
        cooling_parameter = float(element[2:10])
        data_list.append([flip, cooling_parameter, mean_time])

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------


def FitFunc(x, a, b):
    return a*np.tan(b*x)


flip_1 = [[x[1], x[2]*1e-6] for x in data_list if x[0] == 1]
flip_2 = [[x[1], x[2]*1e-6] for x in data_list if x[0] == 2]

flip_1 = np.transpose(np.array(flip_1))
flip_2 = np.transpose(np.array(flip_2))

popt_1, pcov = curve_fit(FitFunc, flip_1[0], flip_1[1])
popt_2, pcov = curve_fit(FitFunc, flip_2[0], flip_2[1])

fig, ax = plt.subplots()

x = np.linspace(0.45, 1., 500, endpoint=False)

ax.set_xlabel('Cooling-Parameter')
ax.set_ylabel('Simulationsdauer [s]')

ax.plot(x, FitFunc(x, popt_1[0], popt_1[1]), label=r'$%.2f\tan(%.2f x)$' %
                                                   (popt_1[0], popt_1[1]))
ax.plot(x, FitFunc(x, popt_2[0], popt_2[1]), label=r'$%.2f\tan(%.2f x)$' %
                                                   (popt_1[0], popt_1[1]))
ax.plot(flip_1[0], flip_1[1], 'o', label='Typ 1')
ax.plot(flip_2[0], flip_2[1], 'o', label='Typ 2')

ax.set_xlim(0.45, 1.)
ax.set_ylim(0., 25.)
ax.legend(loc='best')

os.chdir('../plt')
fig.tight_layout()
fig.savefig('time.pdf', transparent=True, dpi=300)
