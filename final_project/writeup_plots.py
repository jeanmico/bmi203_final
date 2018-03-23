import os 
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict

file_path = os.path.join(os.path.sep, 'Users', 'student', 'GitHub', 'bmi203_final')


lambda_dct = defaultdict(list)
lambda_name = 'lambda_roc.txt'
with open(os.path.join(file_path, lambda_name)) as f:
    for line in f:
    	lineval = line.strip().split()

    	lambda_reg = float(lineval[0])
    	error = float(lineval[1])

    	lambda_dct[lambda_reg].append(error)

xlist = [a*100 for a in range(1, 11)]

fig, ax = plt.subplots()
for lambda_reg, values in lambda_dct.items():
	print(values)
	plt.plot(xlist, values, '-o', label=str(lambda_reg), alpha=.5)
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('pass')
ax.set_ylabel('loss')
plt.title('regularization optimization')
plt.savefig(os.path.join(file_path, 'figures/lambda_reg.png'), dpi=300)


hidden_fname = 'hidden_roc.txt'

hidden_pass = []
hidden_roc = []
with open(os.path.join(file_path, hidden_fname)) as f:
    for line in f:
        lineval = line.strip().split()
        pass_num = int(lineval[0])
        roc = float(lineval[1])

        hidden_pass.append(pass_num)
        hidden_roc.append(roc)

fig, ax = plt.subplots()
plt.scatter(hidden_pass, hidden_roc)
ax.set_ylim(0,1)
ax.set_ylabel('roc score')
ax.set_xlabel('hidden nodes')
plt.title('optimizing hidden nodes')
plt.savefig(os.path.join(file_path, 'figures/hidden.png'), dpi=300)


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








