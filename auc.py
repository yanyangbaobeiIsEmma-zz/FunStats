# simple python script to calculate AUC for binary classifcation
# input: true labels, predicted scores
# label: 1 - positivel; 0 - negative

### THIS AUC FUNCTION IS TOO NAIVE, ONLY FOR WITHOUT TIE SCORES IN DIFFERENT LABELS SCENARIOS.... TRY USE 
### http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html INSTEAD

import numpy as np

def auc(labels, scores, posLabel = 1):

	sortIndex = np.argsort(scores)
	# rank the labels by descending order of scores
	mylabels = np.array(labels)[sortIndex[::-1]]
	nPos = sum(mylabels == posLabel)
	nNeg = len(mylabels) - nPos
	pos = 0
	neg = 0
	area = 0
	for i in range(len(mylabels)):
		if mylabels[i] == 0:
			neg += 1
			area += pos
		else:
			pos += 1
	return area  * 1.0 / (nPos * nNeg)
