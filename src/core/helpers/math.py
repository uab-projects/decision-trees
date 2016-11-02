import math

def H(self,amounts):
	acc=0
	total_classes = sum(amounts)
	for el in amounts:
		if el != 0:
			prob = el/total_classes
			acc += prob * math.log(prob,2)
	return -acc
