import os 
import matplotlib
from matplotlib import pyplot as plt

file_path = os.path.join(os.path.sep, 'Users', 'student', 'GitHub', 'bmi203_final')

crossval_fname = 'crossval_roc.txt'

crossval_pass = []
crossval_roc = []
with open(os.path.join(file_path, crossval_fname)) as f:
    for line in f:
        lineval = line.strip().split()
        pass_num = int(lineval[0])
        roc = float(lineval[1])

        crossval_pass.append(pass_num)
        crossval_roc.append(roc)

fig, ax = plt.subplots()
plt.scatter(crossval_pass, crossval_roc)
ax.set_ylim(0,1)
ax.set_ylabel('roc score')
ax.set_xlabel('trial')
plt.title('25 cross-validation tests')
plt.savefig(os.path.join(file_path, 'figures/crossval.png'), dpi=300)