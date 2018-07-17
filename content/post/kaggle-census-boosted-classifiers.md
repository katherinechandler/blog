---
date: 2018-07-10T10:58:08-04:00
description: "Some fun with boosted classification algorithms"
title: "Classifying Household Income from US Census Data"
author: 'Katherine Chandler'
featured_image: '/images/census.jpg'
---

With this project I dug into classification algorithms and selected a data set from the ['Adult Census Income'](https://www.kaggle.com/uciml/adult-census-income/home) Kaggle Challenge. The data supplied for this challenge data was extracted from the 1994 census and contains aggregated demographic information (associated with a class 'weight') for US households. The prediction task is a binary classification to determine if a given household's annual income is above $50,000 a year.

In this blog I present a high-level overview of my analysis. For in-depth feature selection and hyper-parameter tuning code, see my [Census Income Classification](https://github.com/katherinechandler/Census_Kaggle/tree/master) notebook in this repo.

My approach for this analysis was as follows:

1. Initial exploratory analysis
2. Initial assessment of ML classification estimators
    - Quality was assessed by calculating precision/recall scores and comparing ROC curves
3. Recursive feature selection (RFE) using estimator shortlist to rank feature importance
4. Re-assess short-listed models with new feature set
5. Perform parameter tuning using GridSearchCV on best models
6. Identify best model

The best model identified in this analysis was the GradientBoostingClassifier (accuracy = 87.4%), with LightGBMClassifier coming in a close second (accuracy = 87.3%). Both of these algorithms are 'boosted' algorithms, which means they use multiple weakly-correlated features to build predictions. The success of this type of model indicates that while some demographic features may be strong indicators of household income, the complete story of household income in the US is made up of many weakly correlated but significant factors.

### Initial exploratory analysis

The census data provided with this Kaggle Challenge was nice and clean, so after minimal data wrangling I did some exploration. This data set contains 32,561 observations, each of which represents a weighted demographic group described by 14 variables (defined [here](https://www.census.gov/prod/2003pubs/censr-5.pdf)). Of the 32,561 demographic groups described in this data set, only 7,841 groups (24.1%) are in the 'greater than $50K annually' category. 

<figure>
  <img src="/images/census_images/census_fig1.png" alt="Fig1" style='width:75%'>
  <figcaption>Figure 1. The ratio of demographic groups in each income category</figcaption>
</figure>

The income class data tells us that if we were to always predict 'less than $50K annually', the model would be correct 75.9% of the time. This imbalance makes it easy to overestimate the quality of a model when predicting the minority class, so the accuracy score of potential classification models needs to be considered in the context of this 75.9%-baseline.

I plotted a Pearson's correlation matrix to determine possible relationships between the variables and the prediction target (income).

<figure>
  <img src="/images/census_images/census_fig3.png" alt="Fig1" style='width:75%'>
  <figcaption>Figure 2. Pearson's correlation heatmap</figcaption>
</figure>

Examining this heat-map shows some small to medium correlations in the data. Age-income are weakly correlated (&rho; = 0.234), as are capital_gain-income (&rho; = 0.223) and capital_lost-income (&rho; = 0.151). These correlations make intuitive sense, as people who are older often have higher earning power, and households with higher disposable income are more likely to invest money and experience capital gains and losses. Similarly, there is an expected correlation between income-hours_per_week (&rho; = 0.230), indicating that people who work more hours often make more money than those who work fewer hours. In addition to these small correlations, the Pearson's correlation test found a medium-strength correlation between education and income (&rho; = 0.335), again fitting the intuitive assumptions about income potential.

Demographic features like race, country of origin, or marital/familial status were not addressed in the correlation matrix since they are categorical variables, but I suspected that the relationship between categorical demographic factors and income might be subtle. This idea that multiple weak factors might provide predictive power lead me to include several 'boosted' algorithms in my initial assessment of Machine Learning (ML) estimators.

### Initial assessment of ML classification estimators
I generated a list of 8 classification algorithms for a preliminary screen. All of these algorithms are in the sklearn package except the lgb.LGBMClassifier, which is a boosted tree model documented [here](https://lightgbm.readthedocs.io/en/latest/index.html#).
\
\
`
estimator_list = [LogisticRegression, RandomForestClassifier, SGDClassifier, 
                  lgb.LGBMClassifier, ExtraTreesClassifier, 
                  GradientBoostingClassifier, SVC, LinearSVC]
`
\
\
I assessed the algorithms using `sklearn.metrics` (`classification_report`, `precision_score`, and `recall_score`) in my training-cross validation functions and plotted the Receiver Operating Characteristics (ROC) curve of each estimator.

<figure>
  <img src="/images/census_images/census_fig5.png" alt="Fig1" style='width:200%'>
  <figcaption>Figure 3. The ROC curve of selected estimators in predicting income class</figcaption>
</figure>

The initial exploration of models shows that 1) precision is highest for LGBMClassifier (0.787) and SVC (0.828), 2) recall is highest for LGBMClassifier (0.645) and SGDClassifier (0.831), and 3) the ROC curve of the LGBMClassifier and GradientBoostingClassifier look nearly equivalently good. 

As I suspected, boosted models work well for this classification task.

### Recursive feature selection (RFE) using estimator shortlist to rank feature importance

I narrowed down to a 'short-list' of 4 classifiers to perform feature selection.
\
\
`estimator_shortlist = [LGBMClassifier, 
                      SVC, 
                      SGDClassifier, 
                      GradientBoostingClassifier]`\
\
I performed RFE for each estimator and saved the ranking scores to a pandas data frame.

```
def rank_features(X, y, estimator, feature_list):
    est = estimator()
    rfe = RFE(est)
    rfe = rfe.fit(X,y)
    keys = feature_list
    values = rfe.ranking_
    dictionary = dict(zip(keys, values))
    scores = pd.DataFrame.from_dict(dictionary, orient='index')
    scores.columns = ['{}_rank'.format(estimator.__name__)]
    return(scores)
```

I evaluated feature rankings for each estimator. Features consistently ranked as important are likely going to be relevant for building a final model and provide insight into the nature of the data. Features that are consistently ranked as important are `'workclass', 'marital_status', 'occupation', 'capital_gain', 'capital_loss', 'education_num', 'hours_per_week', 'gender'`, and `'race'`. `'Age'` is ranked as important by the LGBMClassifier and GradientBoostingClassifier models. Specific `'native_country'` features are ranked as important to individual models, but overall these features are less important.

Based on this assessment, I removed 'native_country' from the feature list and re-assessed my estimator shortlist using precision/recall assessment and ROC curve comparisons as described above.

### Re-assess short-listed models with new feature set

There is only a slight change in performance for the estimators after feature refinement. A few of the estimators get slightly better but some actually get poorer. 

`Precision` : \
`'GradientBoostingClassifier': 0.7990654205607477 (before)`\
`'GradientBoostingClassifier': 0.79858657243816256 (after)`\
`'LGBMClassifier': 0.78727841501564133 (before)`\
`'LGBMClassifier': 0.78296988577362414 (after)`\
`'SGDClassifier': 0.51455026455026454 (before)`\
`'SGDClassifier': 0.41291263906620462 (after)`\
`'SVC': 0.82787958115183247 (before)`\
`'SVC': 0.8098591549295775 (after)`\

`Recall`:\
`'GradientBoostingClassifier': 0.58461538461538465 (before)`\
`'GradientBoostingClassifier': 0.57948717948717954 (after)`\
`'LGBMClassifier': 0.64529914529914534 (before)`\
`'LGBMClassifier': 0.64444444444444449 (after)`\
`'SGDClassifier': 0.83119658119658124 (before)`\
`'SGDClassifier': 0.96752136752136753 (after)`\
`'SVC': 0.54059829059829057 (before)`\
`'SVC': 0.54059829059829057(after)``\

I continued model refinement using the smaller feature set (native_country removed) to make the models generalizable, but the decrease in precision/recall was considered. The ROC curve was plotted for the 4 short-listed estimators using the reduced feature list.

<figure>
  <img src="/images/census_images/census_fig6.png" alt="Fig1" style='width:200%'>
  <figcaption>Figure 4. The ROC curve of selected estimators in predicting income class, 
  features reduced</figcaption>
</figure>

Based on this analysis, the two most promising classifiers are the GradientBoostingClassifier and the LGBMClassifier. Both of these algorithms are 'boosted' algorithms, which means they use multiple weakly-correlated features to classify the target. The GradientBoostingClassifier and the LGBMClassifier were selected for further tuning using GridSearchCV.

### Perform parameter tuning using GridSearchCV on best models
I tuned the hyper parameters of the GradientBoostingClassifier following the process outlined in this [blog post](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/). The approach is to 1) hold learning rate constant, 2) optimize n_estimators (number of trees), 3) optimize other tree parameters, then 4) use the newly determined parameters to optimize learning rate and n_estimators in a combined GridSearch. This model is slow to train, so doing a succession of GridSearches for different sets of parameters is faster than doing one large search on a wide range of parameters. After I determined 'reasonable' values for the hyper-parameters, I did a final combined GridSearch using a tight range of the 'optimized' parameters to capture any unexpected interactions between parameters. See Section 5 of my [Census Income Classification](https://github.com/katherinechandler/Census_Kaggle/tree/master) notebook for more detailed tuning information. The parameters of the tuned GradientBoostingClassifier are indicated below:

```
GradientBoostingClassifier(learning_rate = 0.01,
                           n_estimators=700, 
                           max_depth=16, 
                           min_samples_split=450, 
                           min_samples_leaf= 2, 
                           max_features= 44, 
                           subsample = 1.0)
```
The performance of the tuned model was compared to a model using default parameter settings.

`GradientBoostingClassifier performance:`\
`The precision_score is 0.7985865724381626, default params`\
`The recall_score is 0.5794871794871795, default params`\
`The accuracy_score is 0.8642645101852799, default params`\

`The precision_score is 0.7893639207507821, tuned parameters`\
`The recall_score is 0.647008547008547, tuned parameters`\
`The accuracy_score is 0.874091513972771, tuned parameters`\

In this final tuned model the accuracy score and recall score have increased significantly. The precision score, however, has gone down slightly. The tuned algorithm is identifying more of the relevant (greater than $50K households) targets, but at the cost of also identifying slightly more false cases (less than $50K) as positives. There is a tradeoff between precision and recall, but in this case the overall increase in model accuracy makes a precision decrease acceptable.

The LightGBMClassifier was tuned using a similar approach, though larger grids could be searched with GridSearchCV since this algorithm trains quickly. The parameters of the tuned LightGBMClassifier are indicated below:

```
lgb.LGBMClassifier(learning_rate=0.05,
                   n_estimators=250,
                   reg_alpha=0.001,  
                   num_leaves=18,
                   max_depth=11,
                   min_data_in_leaf=7)
```

The performance of the tuned LGBMClassifier model was compared to a model using default parameter settings. As with the GradientBoostingClassifier, accuracy and recall had increased but precision score slightly decreased.

`LGBMClassifier performance:`\
`The precision_score is 0.78296988577362, default parameters`\
`The recall_score is 0.6444444444444445, default parameters`\
`The accuracy_score is 0.87204422151704, default parameters`\

`The precision_score is 0.7904761904761904, tuned parameters`\
`The recall_score is 0.6384615384615384, tuned parameters`\
`The accuracy_score is 0.8728631384993346, tuned parameters`

### Identify the best model

After testing a range of algorithms and parameters, the two best models were boosted classifiers: the GradientBoostingClassifier and the LGBMClassifier. After tuning algorithm hyper-parameters, the accuracy of the GradientBoostingClassifier was 87.4% and the accuracy of the LGBMClassifier was a close 87.3%. The GradientBoostingClassifier is the best algorithm for this task, correctly classifying 64.7% of households making more than $50K annually (recall_score = 0.6470). Of positive cases identified by this model, 78.9% are true positives (precision_score is 0.7894).