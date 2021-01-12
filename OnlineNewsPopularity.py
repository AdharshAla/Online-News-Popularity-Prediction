#Import required librarie----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib as mp
import seaborn as sns
import sklearn as sl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, fbeta_score, roc_curve, auc, roc_auc_score, precision_score
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from multiprocessing import Process, Lock, Manager
from sklearn.neural_network import MLPClassifier 

##To increase the max rows and columns displayed on console
pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth',10)
pd.set_option('display.width',None)
#Data Analysis---------------------------------------------------------------------------------

#read listings csv file into popularity_df dataframe
popularity_df = pd.read_csv("OnlineNewsPopularity.csv")

#Get total number of rows and columns
popularity_df.shape

#Display list of columns in the data set
popularity_df.columns

#Trim the left spaces in each column name
popularity_df.columns = popularity_df.columns.str.lstrip()

#Display first 5 rows of data
popularity_df.head()

#Display various datatypes of the columns in the dataset
popularity_df.info()

#Print describe the dataset
popularity_df.describe()

#List the count of Nans in each column of popularity_df
popularity_df.isna().sum()

#Adding A New column 'day' based on day wise flags
popularity_df.loc[popularity_df['weekday_is_monday']==1,'day'] = 'Monday'
popularity_df.loc[popularity_df['weekday_is_tuesday']==1,'day'] = 'Tuesday'
popularity_df.loc[popularity_df['weekday_is_wednesday']==1,'day'] = 'Wednesday'
popularity_df.loc[popularity_df['weekday_is_thursday']==1,'day'] = 'Thursday'
popularity_df.loc[popularity_df['weekday_is_friday']==1,'day'] = 'Friday'
popularity_df.loc[popularity_df['weekday_is_saturday']==1,'day'] = 'Saturday'
popularity_df.loc[popularity_df['weekday_is_sunday']==1,'day'] = 'Sunday'
popularity_df['day'].fillna('Other', inplace=True)

# Adding A New column 'category' based on different category flags
popularity_df.loc[popularity_df['data_channel_is_lifestyle']==1,'category'] = 'data_lifestyle'
popularity_df.loc[popularity_df['data_channel_is_entertainment']==1,'category'] = 'data_entertainment'
popularity_df.loc[popularity_df['data_channel_is_bus']==1,'category'] = 'data_bus'
popularity_df.loc[popularity_df['data_channel_is_socmed']==1,'category'] = 'data_socmed'
popularity_df.loc[popularity_df['data_channel_is_tech']==1,'category'] = 'data_tech'
popularity_df.loc[popularity_df['data_channel_is_world']==1,'category'] = 'data_world'
popularity_df['category'].fillna('Other', inplace=True)

#Extracting the date and year from 'url' attribute 
popularity_df['published_date']= (popularity_df.url.str.split('/').str[3] + '-' + popularity_df.url.str.split('/').str[4] + '-' + popularity_df.url.str.split('/').str[5])
popularity_df['published_year']= (popularity_df.published_date.str.split('-').str[0])
popularity_df['published_year'] = popularity_df['published_year'].astype(float)

# Adding A New Column 'popularity' based on share value
popularity_df.loc[popularity_df['shares'] < 1400,'popularity'] = 'unpopular'
popularity_df.loc[popularity_df['shares'] >= 1400,'popularity'] = 'popular'

#count of number of articles posted on different days
popularity_df.groupby(['day']).size().reset_index(name='counts')

#Data Visualization---------------------------------------------------------------------------------
#1. Bar plot of count of number of articles posted on different days
cols = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
               'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']
fig = plt.figure(figsize = (16,5))
plt.title("Number of articles over different days category", fontsize = 16)
plt.ylabel("articles count", fontsize = 12)
plt.xlabel("Different days", fontsize = 12)
popularity_df[cols].sum().plot(kind='bar',color = list('rgbkymc'))

#Visualize the feature of different day of week
cols = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
               'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']
unpop=popularity_df[popularity_df['shares']<1400]
pop=popularity_df[popularity_df['shares']>=1400]
unpop_day = unpop[cols].sum().values
pop_day = pop[cols].sum().values

fig = plt.figure(figsize = (16,5))
plt.title("Count of popular/unpopular news over different day of week", fontsize = 16)
plt.bar(np.arange(len(cols)), pop_day, width = 0.3, align="center", color = 'g', \
          label = "popular")
plt.bar(np.arange(len(cols)) - 0.3, unpop_day, width = 0.3, align = "center", color = 'b', \
          label = "unpopular")
plt.xticks(np.arange(len(cols)), cols)
plt.ylabel("Count", fontsize = 12)
plt.xlabel("Days of week", fontsize = 12)
    
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.savefig("days.pdf")
plt.show()


#2. Pie Chart to illustrate the counts of articles based on various categories
cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
               'data_channel_is_tech', 'data_channel_is_world']
popularity_df[cols].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title("Percentage of articles count over different categories", fontsize = 16)
fig = plt.figure(figsize = (16,10))

#Average number of shares over different article category
cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
               'data_channel_is_tech', 'data_channel_is_world']
results = [popularity_df[popularity_df[x] == 1]['shares'].mean() for x in cols]
plt.title("Average number of shares over different article category", fontsize = 16)
plt.ylabel("Average shares", fontsize = 12)
plt.xlabel("Different category", fontsize = 12)
sns.barplot(x=cols, y=results)
sns.set(rc={'figure.figsize':(16, 10)});

# Visualize the feature of different article category
cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
               'data_channel_is_tech', 'data_channel_is_world']
unpop_chan = unpop[cols].sum().values
pop_chan = pop[cols].sum().values
fig = plt.figure(figsize = (16,5))
plt.title("Count of popular/unpopular news over different article category", fontsize = 16)
plt.bar(np.arange(len(cols)), pop_chan, width = 0.3, align="center", color = 'g', \
          label = "popular")
plt.bar(np.arange(len(cols)) - 0.3, unpop_chan, width = 0.3, align = "center", color = 'b', \
          label = "unpopular")
plt.xticks(np.arange(len(cols)), cols)

plt.ylabel("Count", fontsize = 12)
plt.xlabel("Differen category", fontsize = 12)
    
plt.legend(loc = 'upper center')
plt.tight_layout()
plt.savefig("chan.pdf")
plt.show()


#3. Average shares count for number of images/videos in the articles
a = popularity_df.groupby('num_videos')['shares'].mean()
b = popularity_df.groupby('num_imgs')['shares'].mean()
b = pd.DataFrame(b)
b.index.names = ['count']
b.columns = ['mean_shares_img']
a = pd.DataFrame(a)
a.index.names = ['count']
a.columns = ['mean_shares_videos']
frames = [a, b]
sns.set(rc={'figure.figsize':(25, 8)})
c = pd.merge(a, b, on='count', how='outer')
c.plot(kind='bar')
plt.title("Average shares count for number of images/videos", fontsize = 16)
plt.ylabel("shares count", fontsize = 12)
plt.xlabel("videos/images count", fontsize = 12)


#4. #Density plot
fig = plt.figure(figsize = (16,10))
sns.jointplot(x='n_tokens_content', y='shares', data = popularity_df);

# Average number of shares count for weekend
popularity_df.groupby('is_weekend')['shares'].mean().plot(kind='bar', color = list('mc')) 
plt.ylabel('shares')
plt.title("Average number of shares count for a weekend/ not weekend", fontsize = 16)
plt.ylabel("Average number of shares count", fontsize = 12)
plt.xlabel("weekend or not", fontsize = 12)
sns.set(rc={'figure.figsize':(16, 10)});

#5. Box Plot
plt.figure(figsize=(20,8))
sns.boxplot(x="category", y="num_keywords", hue="is_weekend",
                 data = popularity_df, palette="Set3")

#6. Violin plot between different days with number of images
plt.figure(figsize=(16,8))
sns.violinplot(x="day", y="num_imgs", hue="published_year",
                 data = popularity_df, palette="Set3")

#7. pointplot to compare average shares w.r.t number of words in title for both the years
df_group = popularity_df.groupby(['n_tokens_title','published_year']).mean()['shares'].reset_index(name='avg_shares')
plt.figure(figsize=(20,8))
ax = sns.pointplot(df_group['n_tokens_title'],df_group['avg_shares'], hue=df_group['published_year'],
                   data = popularity_df,
                   markers=["o", "x"],
                   linestyles=["-", "--"])
plt.title('Average shares w.r.t. Number of words in title', fontsize=20)
plt.xlabel('Number of words in title', fontsize=18)
plt.ylabel('Average shares count', fontsize=16)
plt.show()

#8. stripplot between shares count and number of keywords
plt.figure(figsize=(20,12))
sns.stripplot(x="num_keywords",y="shares",data= popularity_df,jitter=True,hue="published_year")
plt.title("Shares count w.r.t. number of keywords", fontsize = 16)
plt.ylabel("shares count", fontsize = 12)
plt.xlabel("Number of keywords", fontsize = 12)

#Drop these unwanted columns after data analysis
popularity_df.drop("day",axis=1)
popularity_df.drop("category",axis=1)
popularity_df.drop("published_date",axis=1)
popularity_df.drop("published_year",axis=1)

#9. #Heatmap
f, ax = plt.subplots(figsize=(10, 8))
corr = popularity_df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
#PCA ----------------------------------------------------------------------------------------
#K-Means Clustering
X = popularity_df2.drop(["shares"],axis=1)
y = popularity_df2["shares"]
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)
#Converting the transformed X into Data Frame
X = pd.DataFrame(X, columns = ["n_tokens_title", 
                                "n_tokens_content","n_unique_tokens","num_hrefs","num_self_hrefs","num_imgs","num_videos", 
                                "average_token_length","num_keywords","data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed",
                                "data_channel_is_tech","data_channel_is_world","kw_min_min", "kw_max_min","kw_min_max","kw_max_max",
                                "kw_avg_max","kw_min_avg","kw_max_avg","kw_avg_avg","self_reference_min_shares","self_reference_max_shares",
                                "self_reference_avg_sharess","weekday_is_monday","weekday_is_tuesday","weekday_is_wednesday","weekday_is_thursday","weekday_is_friday",
                                "weekday_is_saturday","weekday_is_sunday","is_weekend","LDA_00","LDA_01","LDA_02","LDA_03","LDA_04",
                                "global_subjectivity","global_sentiment_polarity","global_rate_positive_words","global_rate_negative_words",
                                "rate_positive_words","rate_negative_words","avg_positive_polarity","min_positive_polarity","max_positive_polarity",
                                "avg_negative_polarity","min_negative_polarity","max_negative_polarity",
                                "title_subjectivity","title_sentiment_polarity","abs_title_subjectivity","abs_title_sentiment_polarity"])

# Principal Component Analysis (PCA) is performed with 2 number of components  
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal_component_1', 'principal_component_2'])

#Plot a scatter plot between the two components 'principal_component_1', 'principal_component_2'
ax = sns.scatterplot(x="principal_component_1", y="principal_component_2", data=principalDf)

#K-means clustering
from sklearn import cluster
k_means=cluster.KMeans(n_clusters=3)
k_means.fit(principalDf)
print(k_means.labels_)
df2 = pd.DataFrame(k_means.labels_,columns=['label'])
kmeans_df = pd.concat([popularity_df2, df2], axis=1)
kmeans_df.head()

kmeans_df['label'].value_counts()

#Diving the data set into 3 clusters based on the labels
Cluster0 = kmeans_df[kmeans_df.label==0]
Cluster0.drop('label',axis=1)

Cluster1 = kmeans_df[kmeans_df.label==1]
Cluster1.drop('label',axis=1)

Cluster2 = kmeans_df[kmeans_df.label==2]
Cluster2.drop('label',axis=1)
#-----------------------------------------------------------------------------------------------------

#Clustering the data into 5 clusters with K-Means clustering-----------------------------------------
df = pd.read_csv("OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()
df.head()

sharesclusters = df.shares.values.reshape(-1,1)

#fit the data with k-means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0).fit(sharesclusters)
labels = kmeans.labels_
df['clusters'] = labels
#Gets the count of each cluster
df['clusters'].value_counts()

df = df.drop('shares',axis=1)
df['popularity']= df['clusters']

features_top15= ['LDA_00','LDA_02','is_weekend','weekday_is_friday','weekday_is_monday','weekday_is_thursday',
                 'weekday_is_tuesday','weekday_is_wednesday','LDA_04', 'LDA_01','LDA_03','n_non_stop_unique_tokens',
                'n_unique_tokens','avg_positive_polarity','avg_negative_polarity']

X = df[features_top15]
y = df['popularity']

#Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Transform the data using Min Max Scaler
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)
#-------------------------------------------------------------------------------
# NaiveBayes
nbpipe = make_pipeline(preprocessing.MinMaxScaler(),GaussianNB())
parameters_nb = {}
nb = GridSearchCV(nbpipe,param_grid= parameters_nb,cv=5)
nb.fit(X_train,y_train)

nb.best_estimator_

nbscore = nb.score(X_test,y_test)

print ('NAIVE BAYES SCORE: ', nbscore)

#---------------------------------------------------------------------------------
#RandomForest
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(preprocessing.MinMaxScaler(), 
                         RandomForestClassifier())
hyperparameters = {'randomforestclassifier__n_estimators': [10,20,50,100,250,500],
                    'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
                  'randomforestclassifier__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline,param_grid= hyperparameters,cv=5)

clf.fit(X_train,y_train)

clf.best_estimator_

clf.predict(X_test)

rfscore = clf.score(X_test,y_test)

print ('RANDOM FOREST SCORE: ', rfscore)
#---------------------------------------------------------------------------------

# Logistic Regression

lrpipeline = make_pipeline(preprocessing.MinMaxScaler(), 
                         LogisticRegression('lbfgs'))
parameters_LR = {"logisticregression__penalty": ['l1','l2'],
              "logisticregression__C": [0.1,0.5,1.,2.,2.5,5]}
lr = GridSearchCV(lrpipeline,param_grid= parameters_LR,cv=5)
lr.fit(X_train,y_train)

lr.best_estimator_

lrscore = lr.score(X_test,y_test)

print ('LOGISTIC REGRESSION SCORE: ', lrscore)
#---------------------------------------------------------------------------------

# Neural Networks
neuralnets = make_pipeline(preprocessing.MinMaxScaler(), 
                         MLPClassifier())
parameters_NN = {"mlpclassifier__activation": ['logistic','tanh','relu'],
              "mlpclassifier__learning_rate": ['constant', 'invscaling', 'adaptive']}
nn = GridSearchCV(neuralnets,param_grid= parameters_NN,cv=5)
nn.fit(X_train,y_train)

nn.best_estimator_

nnscore = nn.score(X_test,y_test)

print ('NEURAL NETWORKS SCORE: ', nnscore)
#---------------------------------------------------------------------------------

#SVM
SVCpipeline = make_pipeline(preprocessing.MinMaxScaler(), SVC())
parameters_SVC = {}
svc = GridSearchCV(SVCpipeline,param_grid= parameters_SVC,cv=5)
svc.fit(X_train,y_train)

svcscore = svc.score(X_test,y_test)

svc.best_estimator_

print ('SVM SCORE: ', svcscore)
#---------------------------------------------------------------------------------

#K-NeighborsClassifier
knnpipe = make_pipeline(preprocessing.MinMaxScaler(),KNeighborsClassifier())
parameters_knn = {}
knn = GridSearchCV(knnpipe,param_grid= parameters_knn,cv=5)
knn.fit(X_train,y_train)

knn.best_estimator_

knnscore = knn.score(X_test,y_test)

print ('KNN SCORE: ', knnscore)
#---------------------------------------------------------------------------------

#DecisionTreeClassifier
dtcpipe = make_pipeline(preprocessing.MinMaxScaler(),KNeighborsClassifier())
parameters_dtc = {}
dtc = GridSearchCV(dtcpipe,param_grid= parameters_dtc,cv=5)
dtc.fit(X_train,y_train)

dtc.best_estimator_

dtcscore = dtc.score(X_test,y_test)

print ('DECISION TREE SCORE: ', dtcscore)
#---------------------------------------------------------------------------------

#Adaboost
adapipeline = make_pipeline(preprocessing.MinMaxScaler(), 
                         AdaBoostClassifier())
parameters_ADA = {"adaboostclassifier__n_estimators": [100,200,300,400],
              "adaboostclassifier__learning_rate": [0.1,0.5,1]}
ada = GridSearchCV(adapipeline,param_grid= parameters_ADA,cv=5)
ada.fit(X_train,y_train)

adascore = ada.score(X_test,y_test)

ada.best_estimator_

print ('ADAPTIVE BOOSTING SCORE: ', adascore)

#Converting shares variable to categorical value-------------------------------------------------------
popularity_df1 = popularity_df
# Drop the features that are not effective on predicting the share counts
popularity_df1 = popularity_df1.drop("url",axis=1)
popularity_df1 = popularity_df1.drop("timedelta",axis= 1)
popularity_df1 = popularity_df1.drop("kw_avg_min",axis= 1)
popularity_df1 = popularity_df1.drop("n_non_stop_unique_tokens",axis= 1)
popularity_df1 = popularity_df1.drop("n_non_stop_words",axis= 1)

# handle target attrubte to binary
popular = popularity_df.shares >= 1400
unpopular = popularity_df.shares < 1400
popularity_df1.loc[popular,'shares'] = 1
popularity_df1.loc[unpopular,'shares'] = 0
#---------------------------------------------------------------------------------
#Test and Train data split
X = popularity_df1.drop(["shares"],axis=1)
y = popularity_df1['shares']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Performing Classication modelling on Raw Data-----------------------------------------------
#Logistic Regression
def logred(X_train, y_train,X_test,y_test,return_dict_logred):

    logreg = LogisticRegression(solver = 'lbfgs')
    logreg.fit(X_train, y_train)

    y_train_predicted = logreg.predict(X_train)
    y_test_predicted = logreg.predict(X_test)

    print("------Classification Report Train Data---------------")
    print(classification_report(y_train, y_train_predicted))
    
    print("------Classification Report Test Data---------------")
    print(classification_report(y_test, y_test_predicted))
     
    conf_mat_logred = confusion_matrix(y_test, y_test_predicted)
    
    print("------Confusion Matrix---------------")
    print(pd.crosstab(y_test, y_test_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True))
    ax = sns.heatmap(conf_mat_logred, annot=True, cmap='Blues', fmt='d') #notation: "annot" not "annote"
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    # Compute ROC curve and AUC (Area under the Curve)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_test_predicted)
    
    roc_auc_logred = auc(false_positive_rate, true_positive_rate)
    return_dict_logred['roc_auc_logred']=roc_auc_logred
    return_dict_logred['conf_mat_logred']=conf_mat_logred
    
    ## Plot ROC Curve
    plt.title("Logistic Regression")
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc_logred)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
#Performing Classication modelling on Raw Data-----------------------------------------------
    
#Random Forest
def rf(X_train, y_train,X_test,y_test,return_dict_rf):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    y_train_predicted = rf.predict(X_train)
    y_test_predicted = rf.predict(X_test)
    
    print("------Classification Report Train Data---------------")
    print(classification_report(y_train, y_train_predicted))
    
    print("------Classification Report Test Data---------------")
    print(classification_report(y_test, y_test_predicted))
    
    conf_mat_rf = confusion_matrix(y_test, y_test_predicted)
    
    print("------Confusion Matrix---------------")
    print(pd.crosstab(y_test, y_test_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True))
    ax = sns.heatmap(conf_mat_rf, annot=True, cmap='Blues', fmt='d') #notation: "annot" not "annote"
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    # Compute ROC curve and AUC (Area under the Curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_test_predicted)
    
    roc_auc_rf = auc(false_positive_rate, true_positive_rate)
    return_dict_rf['roc_auc_rf']=roc_auc_rf
    return_dict_rf['conf_mat_rf']=conf_mat_rf

    
    ## Plot ROC Curve
    plt.title("Random Forest")
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc_rf)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
#Performing Classication modelling on Raw Data-----------------------------------------------
# SVM
def svm(X_train, y_train,X_test,y_test,return_dict_svm):
    from sklearn.svm import SVC

    clf = SVC()
    clf.fit(X_train, y_train)

    y_train_predicted = clf.predict(X_train)
    y_test_predicted = clf.predict(X_test)
    
    print("------Classification Report Train Data---------------")
    print(classification_report(y_train, y_train_predicted))
    
    print("------Classification Report Test Data---------------")
    print(classification_report(y_test, y_test_predicted))
    
    global conf_mat_svm
    conf_mat_svm = confusion_matrix(y_test, y_test_predicted)
    print("------Confusion Matrix---------------")
    print(pd.crosstab(y_test, y_test_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True))
    ax = sns.heatmap(conf_mat_svm, annot=True, cmap='Blues', fmt='d') #notation: "annot" not "annote"
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    # Compute ROC curve and AUC (Area under the Curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_test_predicted)
    global roc_auc_svm
    roc_auc_svm = metrics.auc(false_positive_rate, true_positive_rate)
    return_dict_svm['roc_auc_svm'] = roc_auc_svm
    return_dict_svm['conf_mat_svm'] = conf_mat_svm

    ## Plot ROC Curve
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc_svm)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
#------------------------------------------------------------------------------------------------
manager=Manager()
return_dict_logred=manager.dict()
return_dict_rf=manager.dict()
return_dict_svm=manager.dict()

#Calling logistic regression model
logred(X_train, y_train,X_test,y_test,return_dict_logred)

#Calling Random Forest Model
rf(X_train, y_train,X_test,y_test,return_dict_rf)

#Calling Support Vector Machine
svm(X_train,y_train,X_test, y_test,return_dict_svm)
#------------------------------------------------------------------------------------------------
#Transforming the target variable using log transform and convert it into binary with median value as threshold
popularity_df2 = popularity_df
# Drop the features that are not effective on predicting the share counts
popularity_df2 = popularity_df2.drop("url",axis=1)
popularity_df2 = popularity_df2.drop("timedelta",axis= 1)
popularity_df2 = popularity_df2.drop("kw_avg_min",axis= 1)
popularity_df2 = popularity_df2.drop("n_non_stop_unique_tokens",axis= 1)
popularity_df2 = popularity_df2.drop("n_non_stop_words",axis= 1)

#Applied Log transformation on target variable 'shares'
popularity_df2["shares"] = np.log(popularity_df2["shares"])
#Applied cuberoot transforation on below variableas as the max values of these variables are vey high
popularity_df2["n_unique_tokens"] = np.cbrt(popularity_df2["n_unique_tokens"])
popularity_df2["self_reference_min_shares"] = np.cbrt(popularity_df2["self_reference_min_shares"])
popularity_df2["kw_max_min"] = np.cbrt(popularity_df2["kw_max_min"])

# Distribution of 'shares' variable after log transformation
plt.figure(figsize=(18,5))
popularity_df2.shares.hist(bins=50)

# handle target attrubte to binary after log transformation
popular = popularity_df2.shares >= 7.24
unpopular = popularity_df2.shares < 7.24

popularity_df2.loc[popular,'shares'] = 1
popularity_df2.loc[unpopular,'shares'] = 0
#--------------------------------------------------------------------------------------------
#Classification Modelling on Nomalized data with 80% as training data
#Test and Train data split
X = popularity_df2.drop(["shares"],axis=1)
y = popularity_df2['shares']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rfclassifier = RandomForestClassifier(random_state=0)
rfclassifier.fit(X_train, y_train)
rfclassifier.predict(X_test)
rfclassifier.score(X_train, y_train) , rfclassifier.score(X_test, y_test)

rfclassifier.feature_importances_

feat_imp = pd.Series(rfclassifier.feature_importances_,X_train.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances',figsize = (15,8))
plt.ylabel('Feature Importance Score')

import pickle
filename = 'rfclassifier.pkl'
pickle.dump(rfclassifier, open(filename, 'wb'))
#-----------------------------------------------------------------------------------

#Using RFEC for feature selection for each of the three algorithms with CV = 10
#ADA BOOST with cross validation = 10
estimator = AdaBoostClassifier(random_state=0)
selector = RFECV(estimator, step=1, cv=10)
selector = selector.fit(X_train, y_train)
selector.ranking_

#Logistic Regression with cv=10
estimator_LR = LogisticRegression(solver = 'lbfgs',random_state=0)
selector_LR = RFECV(estimator_LR, step=1, cv=10)
selector_LR = selector_LR.fit(X_train, y_train)
selector_LR.ranking_

#Random Forest with cv=10
estimator_RF = RandomForestClassifier(random_state=5)
selector_RF = RFECV(estimator_RF, step=1, cv=10)
selector_RF = selector_RF.fit(X_train, y_train)
selector_RF.ranking_
#--------------------------------------------------------------------------------------
# Plot the cv score vs number of features for Logistic Regression algorithm
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector_LR.grid_scores_) + 1), selector_LR.grid_scores_)
plt.show()

print(X_train.columns.values[selector_LR.ranking_==1].shape[0])
print(X_train.columns.values[selector_LR.ranking_==1])
features_LR = X[X.columns.values[selector_LR.ranking_==1]]

# Plot the cv score vs number of features for ADA Boost
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.savefig('RFE_ADA.pdf')
plt.show()

print(X_train.columns.values[selector.ranking_==1].shape[0])
print(X_train.columns.values[selector.ranking_==1])
features_ADA = X[X.columns.values[selector.ranking_==1]]


# Plot the cv score vs number of features for Random Forest algorithm
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector_RF.grid_scores_) + 1), selector_RF.grid_scores_)
plt.savefig('RFE_RF.pdf')
plt.show()

print(X_train.columns.values[selector_RF.ranking_==1].shape[0])
print(X_train.columns.values[selector_RF.ranking_==1])
features_RF = X[X.columns.values[selector_RF.ranking_==1]]
#----------------------------------------------------------------------------------------
#Splitting Normalized data into training and testing sets with training data as 90%
X_train_ADA, X_test_ADA, y_train_ADA, y_test_ADA = train_test_split(features_ADA, y, test_size = 0.1, random_state = 0)

X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(features_LR, y, test_size = 0.1, random_state = 0)

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(features_RF, y, test_size = 0.1, random_state = 0)

print("Training set has {} samples.".format(X_train_ADA.shape[0]))
print("Testing set has {} samples.".format(X_test_ADA.shape[0]))
#---------------------------------------------------------------------------------------

#Calculating the evaluation methods
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''   
    results = {}
    
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time

    results['train_time'] = end-start
        
    # Get predictions on the first 4000 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:4000])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end-start
            
    # Compute accuracy on the first 4000 training samples
    results['acc_train'] = accuracy_score(y_train[:4000],predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # Compute F-score on the the first 4000 training samples
    results['f_train'] = fbeta_score(y_train[:4000],predictions_train,beta=1)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=1)
    
    # Compute AUC on the the first 4000 training samples
    results['auc_train'] = roc_auc_score(y_train[:4000],predictions_train)
        
    # Compute AUC on the test set
    results['auc_test'] = roc_auc_score(y_test,predictions_test)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    print("{} with accuracy {}, F1 {} and AUC {}.".format(learner.__class__.__name__,\
          results['acc_test'],results['f_test'], results['auc_test']))
    # Return the results
    return results

#--------------------------------------------------------------------------------------
# Method to evaluate the results of models and plot the graphs
import matplotlib.patches as mpatches
def evaluate(results,name):

    # Create figure
    fig, ax = plt.subplots(2, 4, figsize = (16,7))

    # Constants
    bar_width = 0.3
    colors = ['#00A0A0', '#A00000', '#00A000']
   
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'auc_train','pred_time', 'acc_test',\
                                    'f_test', 'auc_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//4, j%4].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//4, j%4].set_xticks([0.45, 1.45, 2.45])
                ax[j//4, j%4].set_xticklabels(["1%", "10%", "100%"])
                ax[j//4, j%4].set_xlim((-0.1, 3.0))
               
    # Add labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[0, 3].set_ylabel("AUC")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    ax[1, 3].set_ylabel("AUC")
    ax[1, 0].set_xlabel("Training Set Size")
    ax[1, 1].set_xlabel("Training Set Size")
    ax[1, 2].set_xlabel("Training Set Size")
    ax[1, 3].set_xlabel("Training Set Size")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[0, 3].set_title("AUC on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    ax[1, 3].set_title("AUC on Training Subset")
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[0, 3].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    ax[1, 3].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches,  bbox_to_anchor = (-1.4, 2.54),\
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.savefig(name)
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------
#Training the models with refined hyperparameters
samples_1 = int(X_train_ADA.shape[0]*0.01)
samples_10 = int(X_train_ADA.shape[0]*0.1)
samples_100 = X_train_ADA.shape[0]

clf_A = AdaBoostClassifier(random_state=0,learning_rate=0.5,n_estimators=300)
clf_B = LogisticRegression(random_state=0, C=2.5)
clf_C = RandomForestClassifier(random_state=0, n_estimators=500)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        if clf == clf_A:
            results[clf_name][i] = \
            train_predict(clf, samples, X_train_ADA, y_train_ADA, X_test_ADA, y_test_ADA)
        elif clf == clf_B:
            results[clf_name][i] = \
            train_predict(clf, samples, X_train_LR, y_train_LR, X_test_LR, y_test_LR)
        else:
            results[clf_name][i] = \
            train_predict(clf, samples, X_train_RF, y_train_RF, X_test_RF, y_test_RF)

#------------------------------------------------------------------------------------------
            # Method for grid search
def gridsearch(clf,parameters,X_train, y_train, X_test, y_test):
    scorer = make_scorer(roc_auc_score)
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
    best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
    print(clf.__class__.__name__)
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print( "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions,beta=1)))
    print("AUC on testing data: {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=1)))
    print("Final AUC on the testing data: {:.4f}".format(roc_auc_score(y_test, best_predictions)))

    print(best_clf)

#----------------------------------------------------------------------------------------------
# Hyperparameters for the grid search
from sklearn.metrics import make_scorer 
from sklearn.model_selection import GridSearchCV
parameters_RF = {"n_estimators": [10,20,50,100,250,500]}
parameters_LR = {"penalty": ['l1','l2'],
              "C": [0.1,0.5,1.,2.,2.5,5]}
parameters_ADA = {"n_estimators": [100,200,300,400],
              "learning_rate": [0.1,0.5,1]}
#---------------------------------------------------------------------------------------------
# GRID SEARCH RANDOM FOREST
gridsearch(clf_C,parameters_RF,X_train_RF, y_train_RF, X_test_RF, y_test_RF)
# GRID SEARCH LOGISTIC REGRESSION
gridsearch(clf_B,parameters_LR,X_train_LR, y_train_LR, X_test_LR, y_test_LR)
# GRID SEARCH ADAPTIVE BOOSTING
gridsearch(clf_A,parameters_ADA,X_train_ADA, y_train_ADA, X_test_ADA, y_test_ADA)