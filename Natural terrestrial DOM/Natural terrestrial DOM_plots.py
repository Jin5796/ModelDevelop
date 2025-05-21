# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 21:59:18 2023

@author: Ada
"""
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import numpy as np
import os
import shap
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline 

# apply Regression
from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
    
from scipy.stats  import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 
 
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut

home = 'F:/AASecondPaper/5_Model_develop/1022New/Terrestrial/'
 

nDesc = 5 
modelname="GBRT"
model = GradientBoostingRegressor(random_state=123,n_estimators= 100,
                                  max_depth= 8, learning_rate=0.1,
                                  min_samples_leaf = 7,
                                  subsample= 0.8) #GradientBoostingRegressor

feature_selected =   ['CrippenLogP', 'ErGFP285_RDKit', 'MATS2m', 'PatternFP730_RDKit', 'Mor29m_R']

 

target = 'logKdoc'
    

strat_train_set = pd.read_excel(home + 'Train_terrestrial.xlsx')
strat_test_set = pd.read_excel(home + 'Test_terrestrial.xlsx')
   

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
    
])


new_names = {
             'CrippenLogP':'CrippenLogP', 'ErGFP285_RDKit':'ErGFP285',
            'MATS2m':'MATS2m', 'PatternFP730_RDKit':'PatternFP730', 'Mor29m_R':'Mor29m'}


train_X = strat_train_set[feature_selected].rename(columns=new_names)
train_Y = strat_train_set[target].copy()
vaild_X = strat_test_set[feature_selected].rename(columns=new_names)
vaild_Y = strat_test_set[target].copy()

predictors0 = train_X
responses = train_Y
predictors_vaild0= vaild_X
responses_vaild= vaild_Y

predictors = num_pipeline.fit_transform(predictors0)
feature_names  = num_pipeline.get_feature_names_out()
predictors = pd.DataFrame(predictors, columns=feature_names)


predictors_vaild = num_pipeline.transform(predictors_vaild0)
predictors_vaild = pd.DataFrame(predictors_vaild, columns=feature_names)


model.fit(predictors, responses)
predictions = model.predict(predictors)
predictions_vaild = model.predict(predictors_vaild)


predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
predictions_vaild_df = pd.DataFrame(predictions_vaild, columns=['Predictions_Valid'])

folder_path = 'F:/AASecondPaper/5_Model_develop/1022New/Terrestrial/1115/'
file_path = os.path.join(folder_path, 'predictions_results.xlsx')

with pd.ExcelWriter(file_path) as writer:
    
    predictions_df.to_excel(writer, sheet_name='Train_Predictions', index=False)
    predictions_vaild_df.to_excel(writer, sheet_name='Valid_Predictions', index=False)
    
    
    
MSE = mean_squared_error(responses,predictions)
R2 = r2_score(responses,predictions)
MSE_vaild = mean_squared_error(responses_vaild,predictions_vaild)
R2_vaild = r2_score(responses_vaild,predictions_vaild)

RMSE = np.sqrt(MSE)
RMSE_vaild = np.sqrt(MSE_vaild)  


##########################williams_plot################################################# 
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

resid_train = (responses - predictions) / np.sqrt(np.mean((responses - predictions)**2))
leverage_train = np.diag(np.dot(np.dot(predictors, np.linalg.inv(np.dot(predictors.T,predictors))), predictors.T))

resid_test = (responses_vaild - predictions_vaild)/ np.sqrt(np.mean((responses_vaild - predictions_vaild)**2))
leverage_test = np.diag(np.dot(np.dot(predictors_vaild, np.linalg.inv(np.dot(predictors.T,predictors))), predictors_vaild.T))


n_train, p = predictors.shape
n_test = predictors_vaild.shape[0]
h_boundary = 3 * ((p + 1) / n_train)


fig, ax = plt.subplots(figsize=(10, 8))


ax.scatter(leverage_train, resid_train,edgecolor='gray', s=125,linewidth=0.5,label='Training Set')

ax.scatter(leverage_test, resid_test,edgecolor='gray',s=125, linewidth=0.5,label='Vaildation Set')

ax.set_ylim(-5,5)

ax.set_xlim(0, max(np.max(leverage_train), np.max(leverage_test)) + 0.1)

ax.set_xlabel("Leverage($h_i$)", fontsize=25)
ax.set_ylabel("Standardized Residuals($δ$)", fontsize=25)
ax.tick_params(axis='both', labelsize=20)

ax.axhline(y=3, color='gray', linestyle='--', linewidth=1.5)
ax.axhline(y=-3, color='gray', linestyle='--', linewidth=1.5)
ax.axvline(x=h_boundary, color='gray', linestyle='--', linewidth=1.5)

plt.text(ax.get_xlim()[1]-0.1, ax.get_ylim()[1]-0.5,'Model = '+str(modelname), fontsize=20)
plt.text(ax.get_xlim()[1]-0.1, ax.get_ylim()[1]-1.2,'$n_{Desc}$ = '+str(nDesc), fontsize=20)
plt.text(ax.get_xlim()[1]-0.1, ax.get_ylim()[1]-1.9,r'$\mathit{h^*}=%.3f$' % h_boundary, fontsize=20)
ax.text(h_boundary + 0.02, -7.2, '${h^*}$', fontsize=20)
plt.savefig(home +'1115' +str(modelname) +'_' + str(nDesc) + 'AD.png',dpi=300, format='png')
plt.show()




##########################scatterplot#################################
limmax = 9.0
limmin = 0

plt.figure(figsize=(4, 4))
plt.xlim(limmin, limmax)
plt.ylim(limmin, limmax)

plt.scatter(responses.values,predictions,edgecolor='gray', linewidth=0.5,label='Training Set')
plt.scatter(responses_vaild.values,predictions_vaild,edgecolor='gray', linewidth=0.5,label='Vaildation Set')

x_vals = np.array([-1.5,9])
y_vals = x_vals * 1 + 1
plt.plot(x_vals, y_vals, color='gray', linestyle='--',linewidth=1)
x_vals2 = np.array([-1.5,9])
y_vals2 = x_vals2 * 1 -1
plt.plot(x_vals2, y_vals2, color='gray', linestyle='--',linewidth=1)
y_vals3 = x_vals2
plt.plot(x_vals, y_vals3, color='black', linestyle='-',linewidth=0.5)

tick_interval = 1 
plt.xticks(np.arange(limmin, limmax + tick_interval, tick_interval))
plt.yticks(np.arange(limmin, limmax + tick_interval, tick_interval))


plt.xlabel('Observed log $K$$_{DOC}$', fontsize=12, usetex=False)
plt.ylabel('Predicted log $K$$_{DOC}$', fontsize=12)
plt.text(limmin+0.2,limmax-0.8,'Model = '+str(modelname))
plt.text(limmin+0.2,limmax-1.6,'$RMSE$$_{train}$ = '+str(round(RMSE,3)))
plt.text(limmin+0.2,limmax-2.4,'$RMSE$$_{test}$ = '+str(round(RMSE_vaild,3)))
plt.text(limmin+0.2,limmax-3.1,'$n$$_{\mathrm{Desc}}$ = '+str(nDesc))
plt.savefig(home +'1115' + str(modelname) +'_' + str(nDesc) + 'plot.png',dpi=300, format='png')

plt.show()



############################SHAP###############################
  
import matplotlib.ticker as ticker 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 28 
plt.rcParams["xtick.labelsize"] = 24  
plt.rcParams["ytick.labelsize"] = 24 
plt.rcParams["axes.labelsize"] = 28  


plot_size = [10, 7.5]


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(predictors)

if isinstance(shap_values, np.ndarray):
    shap_values = shap.Explanation(values=shap_values,
                                   data=predictors,
                                   base_values=np.array([explainer.expected_value] * len(predictors)),
                                   feature_names=predictors.columns)
 
plt.figure(figsize=plot_size)
shap.plots.beeswarm(shap_values, max_display=12,
                    order=shap_values.abs.mean(0),
                    show=False)

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))

plt.xlim(None, None)
plt.savefig(home +'1115/' + str(modelname) + '_' + str(nDesc) + 'shap.jpg', dpi=600, format='jpg',
           bbox_inches='tight', pad_inches=0.1)
plt.clf()
plt.show()


 
#############################3D#################################

from mpl_toolkits.mplot3d import Axes3D
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
 
feature_names = ['CrippenLogP', 'ErGFP285', 'MATS2m',  'Mor29m']

from matplotlib.colors import LinearSegmentedColormap

for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
       
        plt.rcParams['font.family'] = 'Times New Roman'
        fig = plt.figure(figsize=(15, 6))
        
        # Create first subplot for 3D surface plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        # Select two features for analysis
        features = (feature_names[i], feature_names[j])
        
        # Calculate partial dependence values
        pdp = partial_dependence(model, predictors, features=features, kind="average", grid_resolution=12)
        XX, YY = np.meshgrid(pdp["values"][0], pdp["values"][1])
        Z = pdp.average[0].T
        
        custom_camp = LinearSegmentedColormap.from_list('custom',['#F2F9FF',  '#CCE4FF', '#B3D9FF',
                                                                  '#99CCFF', '#66B3FF',
                                                                #  '#3399FF',
                                                                  '#00A0FF'
                                                                  ,'#0061FF'

 ],N=256)
        # Create 3D surface plot
        surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=custom_camp, edgecolor="k")
        ax.set_xlabel(features[0], fontdict={'fontsize': 20}, labelpad=15)
        ax.set_ylabel(features[1], fontdict={'fontsize': 20}, labelpad=15)
        ax.view_init(elev=22, azim=122)
        # Increase tick label fontsize
        ax.tick_params(axis='both', which='major', labelsize=19)
        
        # Create second subplot for top-down view
        ax2 = fig.add_subplot(1, 2, 2)
        # Get the x and y axis limits
        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        
        # Create a grid of points
        x = np.linspace(xlim[0], xlim[1], Z.shape[1])
        y = np.linspace(ylim[0], ylim[1], Z.shape[0])
        X, Y = np.meshgrid(x, y)
        # Plot the contour lines
        im = ax2.imshow(Z[::-1], cmap=custom_camp, extent=[np.min(XX), np.max(XX), np.min(YY), np.max(YY)], aspect='auto')
        ax2.set_xlabel(features[0], fontdict={'fontsize': 20}, labelpad=5)
        ax2.set_ylabel(features[1], fontdict={'fontsize': 20}, labelpad=5)
        # Increase tick label fontsize for grid plot
        ax2.tick_params(axis='both', which='major', labelsize=19)  # Increase fontsize for ticks
       
    
        cbar = fig.colorbar(im, ax=ax2)
        cbar.set_label('Partial Dependence Value', fontsize=22, labelpad=15)
        
           
        
        cbar.ax.tick_params(labelsize=19) 
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.2, left=0.02, right=0.95, top=0.98, bottom=0.12)
        plt.savefig(home + '1115/' + features[0]+features[1] + '.jpg', dpi =600,format='jpg', bbox_inches = 'tight',pad_inches=0)
        plt.show()

 


 
