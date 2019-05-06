if __name__ == "__main__":

	Y = None
	Y_hat = None

none = []
mild = []
moderate = []
moderately_severe = []
severe = []


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np


for i in range(33):
	if(0 <= Y[i] < 5):
		none.append((Y[i], Y_hat[i]))
	if(5 <= Y[i] < 10):
		mild.append((Y[i], Y_hat[i]))
	if(10 <= Y[i] < 15):
		moderate.append((Y[i], Y_hat[i]))
	if(15 <= Y[i] < 20):
		moderately_severe.append((Y[i], Y_hat[i]))
	if(20 <= Y[i] <= 24):
		severe.append((Y[i], Y_hat[i]))

none = np.array(none)
mild = np.array(mild)
moderate = np.array(moderate)
moderately_severe = np.array(moderately_severe)
severe = np.array(severe)

	
none_c_over = []
for i in range(none.shape[0]):
	if(none[i][1] > none[i][0]):
		none_c_over.append(none[i][1] - none[i][0])




none_c_under = []
for i in range(none.shape[0]):
	if(none[i][1] < none[i][0]):
		none_c_under.append(none[i][1] - none[i][0])



none_RMSE = np.sqrt(mean_squared_error(none[:,0], none[:,1]))
none_MAE = mean_absolute_error(none[:,0], none[:,1])











mild_c_over = []
for i in range(mild.shape[0]):
	if(mild[i][1] > mild[i][0]):
		mild_c_over.append(mild[i][1] - mild[i][0])




mild_c_under = []
for i in range(mild.shape[0]):
	if(mild[i][1] < mild[i][0]):
		mild_c_under.append(mild[i][1] - mild[i][0])



mild_RMSE = np.sqrt(mean_squared_error(mild[:,0], mild[:,1]))
mild_MAE = mean_absolute_error(mild[:,0], mild[:,1])








moderate_c_over = []
for i in range(moderate.shape[0]):
	if(moderate[i][1] > moderate[i][0]):
		moderate_c_over.append(moderate[i][1] - moderate[i][0])




moderate_c_under = []
for i in range(moderate.shape[0]):
	if(moderate[i][1] < moderate[i][0]):
		moderate_c_under.append(moderate[i][1] - moderate[i][0])



moderate_RMSE = np.sqrt(mean_squared_error(moderate[:,0], moderate[:,1]))
moderate_MAE = mean_absolute_error(moderate[:,0], moderate[:,1])










moderately_severe_c_over = []
for i in range(moderately_severe.shape[0]):
	if(moderately_severe[i][1] > moderately_severe[i][0]):
		moderately_severe_c_over.append(moderately_severe[i][1] - moderately_severe[i][0])




moderately_severe_c_under = []
for i in range(moderately_severe.shape[0]):
	if(moderately_severe[i][1] < moderately_severe[i][0]):
		moderately_severe_c_under.append(moderately_severe[i][1] - moderately_severe[i][0])



moderately_severe_RMSE = np.sqrt(mean_squared_error(moderately_severe[:,0], moderately_severe[:,1]))
moderately_severe_MAE = mean_absolute_error(moderately_severe[:,0], moderately_severe[:,1])







severe_c_over = []
for i in range(severe.shape[0]):
	if(severe[i][1] > severe[i][0]):
		severe_c_over.append(severe[i][1] - severe[i][0])




severe_c_under = []
for i in range(severe.shape[0]):
	if(severe[i][1] < severe[i][0]):
		severe_c_under.append(severe[i][1] - severe[i][0])



severe_RMSE = np.sqrt(mean_squared_error(severe[:,0], severe[:,1]))
severe_MAE = mean_absolute_error(severe[:,0], severe[:,1])












print(none_RMSE, none_MAE, np.std(none_c_over), np.std(none_c_under))
print(mild_RMSE, mild_MAE, np.std(mild_c_over), np.std(mild_c_under))
print(moderate_RMSE, moderate_MAE, np.std(moderate_c_over), np.std(moderate_c_under))
print(moderately_severe_RMSE, moderately_severe_MAE, np.std(moderately_severe_c_over), np.std(moderately_severe_c_under))
print(severe_RMSE, severe_MAE, np.std(severe_c_over), np.std(severe_c_under))