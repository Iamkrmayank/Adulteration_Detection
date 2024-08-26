1.Dataset
The dataset used in this study pertains to the analysis of jeera powder (cumin powder) images. Initially, macro images of jeera powder were captured to extract relevant statistical features. These features were then compiled into a structured dataset. The objective of this dataset is to utilize the extracted features for classification tasks, potentially to distinguish between different qualities or types of jeera powder.  Data Collection
The images of jeera powder were obtained using a high-resolution camera capable of capturing macro images. This ensured detailed and high-quality images, which are essential for accurate feature extraction. The images were processed using image analysis techniques to derive several statistical features. 
Data Extraction
Data extraction refers to the process of retrieving specific data from various sources to transform it into a format suitable for further processing and analysis. This involves identifying, retrieving, and transforming data so that it can be used for various purposes such as analysis, reporting, and integration into databases or applications.
 Dataset Features The dataset comprises the following columns, each representing a specific statistical attribute extracted from the jeera powder images:  mean_intensity: The average intensity of the pixels in the image. This feature indicates the overall brightness of the image. 
median_intensity: The median value of the pixel intensities in the image. This is useful for understanding the central tendency of the pixel values, particularly in the presence of outliers. 
mode_intensity: The most frequently occurring pixel intensity value in the image. This feature can highlight the most common pixel value, giving insights into the most dominant intensity level. 
std_intensity: The standard deviation of the pixel intensities. This measure indicates the spread or variability of the intensity values around the mean, reflecting the contrast in the image. 
var_intensity: The variance of the pixel intensities. Similar to standard deviation, this feature represents the degree of dispersion in the intensity values. 
skewness: A measure of the asymmetry of the distribution of pixel intensities. This can help in understanding the distribution shape and whether there are more dark or light pixels. 
kurtosis: A measure of the "tailedness" of the distribution of pixel intensities. High kurtosis indicates more pixels with extreme intensity values, either very dark or very bright. 
Outcome: The target variable, which contains binary values (0 or 1). This column indicates the classification result based on the extracted features. For instance, it may denote different quality grades of the jeera powder.
Class Imbalance and SMOTE Application Initially, the 'Outcome' column exhibited a significant class imbalance with the following distribution:  Class 0: 25 instances Class 1: 225 instances 
To address this imbalance and ensure that the machine learning models are not biased towards the majority class, we applied the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE generates synthetic samples for the minority class (Class 0 in this case) by interpolating between existing samples. This helps to create a more balanced dataset without merely duplicating existing records.  Post-SMOTE Application After applying SMOTE, the dataset achieved a balanced distribution of the 'Outcome' values, resulting in equal representation of both classes. This balanced dataset is crucial for training robust and unbiased models, improving their ability to generalize and perform well on unseen data.
Class 0: 225 instances Class 1: 225 instances
Link for Dataset: https://github.com/Iamkrmayank/Adulteration_Detection/blob/main/Jeera_New.xlsx
2.Data Preparation
To ensure the quality and reliability of the dataset, several preprocessing steps were undertaken:
(a). Data Exploration 
This involves understanding the dataset by summarizing its main characteristics often using visual methods. It includes inspecting the data types, checking for missing values, and computing basic statistics.
•  Checking data types and structure: df.info()
•  Summarizing statistics: df.describe()
(b). Data Preprocessing
This involves cleaning and transforming raw data into a format that can be used for modelling. It includes handling missing values, encoding categorical variables, and feature extraction or creation.
	Handling Missing Values by Removal of Incomplete Records: df.dropna()

(c)  Data Scaling:.
This involves standardizing or normalizing the features so that they have similar scales, which is important for many machine learning algorithms.
•  Standard scaling (zero mean, unit variance): StandardScaler()
•  Min-max scaling (scaling features to a fixed range, e.g., 0 to 1): MinMaxScaler()

 












3.Proposed Model
Stacking classifier

 


An ensemble learning technique called stacking classifier leverages the strengths of multiple base models to improve prediction accuracy. It works by combining predictions from many base models, each trained with different hyperparameters or techniques, to create a robust meta model. 
Using the base models' diverse viewpoints on the dataset, this meta model—also called a blender or aggregator—skillfully learns to synchronise the predictions produced by the base models. 
Stacking improves predictive performance and offers a flexible approach to complex categorization tasks by leveraging the diversity among base models.
A stacking classifier is an ensemble method that combines multiple machine learning models to improve predictive performance
1. Training Base Models (QDA, MLP Classifier, AdaBoost)
	QDA (Quadratic Discriminant Analysis)
	MLP Classifier (Multi-Layer Perceptron)
	AdaBoost (Adaptive Boosting)
These base models are trained on the same training set independently. Mathematically, if 80% is the training set and 20% is the target variable:
QDA:         y_QDA^^=QDA(x)
				MLP CLassifier:  y_MLP^^=MLP(x)
				AdaBoost∶ y_Ada^^=AdaBoost(x)
2. Creating a New Training Set
The predictions from the base models are used to create a new training set. This new training set is typically composed of the outputs (predictions) of the base models. If there are N training samples and k base models, the new training set 0.80 will have 8 features (each feature being the prediction of a base model):

 
3. Training the Meta Model (Logistic Regression)
The new training set 0.80 is used to train a meta-model (in this case, Logistic Regression). The meta-model learns how to combine the predictions of the base models to make the final prediction. Mathematically:
Meta Model: y_Meta^^=  LogisticRegression(x)
4. Making Final Predictions
The final predictions are made by the meta-model. The new data (test set) is passed through the base models to get their predictions, which are then used as input for the meta-model to produce the final predictions. If X_test is the test set:
(a) Get predictions from base models:
y_QDA^^  (X test) ,y_MLP^^  (X test)  ,〖         y〗_Ada^^  (X test)
(b) Form the new test set for the meta-model:
 
(c) Make final predictions with the meta-model:
Meta Model: y_Meta^^=  LogisticRegression(x)
Logistic Regression
Logistic regression models the probability that a given input X belongs to a certain class y. For binary classification, y can take on values 0 or 1. Instead of modelling y directly, logistic regression models the probability that y = 1 using the logistic function (also called the sigmoid function).
Logistic Function (Sigmoid Function)
The logistic function is defined as:
σ(z)=1/(1+ e^(-z) )
where z is a linear combination of the input features. The logistic function maps any real-valued number into the range [0, 1], making it suitable for probability estimation.
Logistic Regression Model
In logistic regression, the probability that y = 1given the input features x = {x1,x2,x3,}
P(y=1│x)=σ(z)=1/(1+e^z )

where
	z= W^T x+b= w_1 x_1+ w_2 x_2+ w_3 x_3+⋯+ w_n X_n+b 
	w is the vector of weights (coefficients).

	b is the bias term (intercept).

	x is the input feature vector.











GaussianNB
Gaussian Naive Bayes (GaussianNB) is a classification algorithm that applies the principles of Bayes' theorem with the assumption of independence among predictors, and it assumes that the continuous features follow a Gaussian (normal) distribution. Here's an explanation of Gaussian Naive Bayes with relevant equations:
Bayes' Theorem
Bayes' theorem provides a way to calculate the posterior probability P(y│X)  
from the prior probability P(y) , the likelihood P(X|y) , and the evidence P(X) :
P(y│X)=(P(X│y).P(y))/(P(X))
where
	P(y∣X) is the posterior probability of class y given the feature vector X.

	P(X∣y) is the likelihood of the feature vector X given class y.

	P(y) is the prior probability of class y.

	P(X)is the evidence or the total probability of the feature vector X.
Naive Bayes Assumption
The "naive" assumption in Naive Bayes is that all features X_i  are independent given the class y:
P(X│y)= ∏_1^n▒〖P(X_i 〗|y)
where 〖X = (X〗_1,X_2,X_3….,X_(n) ) is the feature vector. 
Gaussian Naive Bayes
In Gaussian Naive Bayes, we assume that the continuous features follow a Gaussian distribution. The probability density function of the Gaussian distribution is given by:
P(X_i│y)=1/√(2πσ_(y,i)^2 ) exp⁡(-(X_i-μ_(y,i) )^2/(2σ_(y,i)^2 ))
where:
	μ_(y,i) the mean of the feature X_i for class y.
	σ_(y,i) is the standard deviation of the feature X_i for class y.
Combining the Equations
Using Bayes' theorem and the naive assumption, the posterior probability P(y∣X) can be expressed as:
P(y│X)∝P(y).∏_1^n▒〖P(X_i 〗|y)
Substituting the Gaussian distribution into this equation:
P(y│X)∝P(y).∏_1^n▒1  1/√(2πσ_(y,i)^2 ) exp⁡(-(X_i-μ_(y,i) )^2/(2σ_(y,i)^2 ))
Classification
To classify a new instance X, we compute the posterior probability for each class and choose the class with the highest posterior probability:
y^^=arg⁡〖  〗 〖max〗_y  P(y|X)
In practice, since we are multiplying many probabilities, it is numerically more stable to work with the logarithms of the probabilities. Thus, we often use the log-posterior:
log⁡〖P(y│X)〗 α log⁡P(y)  + ∑_1^n▒〖logP(X_i |y)〗
Substituting the Gaussian distribution:
log⁡〖P(y│X)〗 α log⁡P(y)-∑_1^n▒〖log⁡(√(2πσ_(y,i)^2+〖(X_i-μ_(y,i))〗^2/(2σ_(y,i)^2 ))〗











XGBoost
XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.
Overview of Gradient Boosting
Gradient Boosting is a technique where new models are trained to correct the errors made by previous models. The key idea is to build the model in a stage-wise fashion, minimizing the loss function by adding weak learners (usually decision trees) to the ensemble.
Objective Function
The objective function in XGBoost combines a loss function that measures the model's fit and a regularization term that penalizes the complexity of the model to prevent overfitting.
The general form of the objective function is:
Objective= ∑_1^n▒〖L(y_i 〗,y_i^^)+ ∑_1^n▒〖Ω(f_k)〗
where:
	L is the loss function, such as mean squared error (MSE) for regression or log loss for classification.
	Ω is the regularization term for the model complexity.
	y_i  is the true label.
	y_i^^  is the predicted label.
	f_k  is the k-th tree in the ensemble.

Regularization Term
The regularization term Ω(f_k )  typically includes the number of leaves in the tree and the sum of the squared leaf weights, which helps control the complexity of the model:
Ω(f)= γT+  1/2 λ∑_(j=1)^T▒ω_j^2 
where:
	T is the number of leaves.
	ω_j  is the weight of the j-th leaf.
	Υ and λ  are regularization parameters
Additive Training
In Gradient Boosting, we add one new function (tree) at a time to minimize the objective. If y_i^(^(t))   is the prediction of the ensemble at iteration t, the model is updated as follows:
y_i^(^(t+1))=y_i^(^(t))+ f_t (x_i)
where f_t is the new tree added at iteration t. 
Taylor Expansion
XGBoost uses a second-order Taylor expansion to approximate the loss function for efficient optimization. The objective function for adding the new tree f_t  becomes:
〖Objective〗^((t))≈∑_1^n▒〖[g_i f_t (x_i )+1/2〗 〖h_i f_t (x_i )〗^2]+Ω(f_t) 
where:
	g_i=(∂L(y_i,y_i^(^^((t)) )))/(∂y_i^(^(t)) ) is the first-order gradient (residual).
	 h_i=  (∂^2 L(y_i,y_i^(^^((t)) )))/(∂y_i^(^(t)2) ) is the second-order gradient (Hessian).
Tree Structure
Each tree is constructed to minimize the objective function. The score of a tree is calculated based on the gradients and Hessians. For a given structure q (which assigns each data point to a leaf), the optimal weight w_j for each leaf j is:
ω_j=-(∑_(i ϵ I_j)▒g_i )/(∑_(i ϵ I_j)▒〖h_i+ λ〗)
where I_j   is the set of indices of data points assigned to leaf j.
The corresponding objective reduction for each leaf is:
Reduction= -1/2 ∑_(j=1 )^T▒〖(∑_(i ϵ I_j)▒〖g_i)〗〗^2/(∑_(i ϵ I_j)▒〖h_i+ λ〗)+γT



AdaBoost
AdaBoost, short for Adaptive Boosting, is an ensemble learning algorithm that combines multiple weak classifiers to create a strong classifier. It iteratively trains weak classifiers on different subsets of the training data, assigning higher weights to misclassified samples at each iteration. This focus on challenging examples allows subsequent weak classifiers to improve their performance. In each iteration, a weak classifier is trained on the data with current sample weights, aiming to minimize classification error. 
The algorithm then calculates classifier weights based on weak classifier errors, updating sample weights to prioritize misclassified examples. After normalization, the weak classifiers' predictions are combined using a weighted majority vote, with their weights considered. 
This process is repeated for a specified number of iterations, producing a final prediction by aggregating the weak classifiers' predictions. AdaBoost effectively reduces error by aligning weak models into a strong one, making it a powerful technique in machine learning.
Overview of AdaBoost
AdaBoost works by iteratively adding weak classifiers to form a strong classifier. It adjusts the weights of misclassified samples to focus more on the harder cases in subsequent iterations.
Initialization 
	Initialize Weights: Each training sample (x_i,y_i) is assigned an initial weight w_(i ), such that all weights sum up to 1. Typically, the weights are initially uniformly:
ω_i^((1))=1/N,for i=1,2,3,…,N
where N is the number of training samples.

Iterative Process
For t=1,2,3,….,T (where T  is the total number of iterations):
1. Train Weak Classifier: Train a weak classifier h_t (x) using the current weights ω_i^((t)).
2. Compute Weighted Error: Calculate the weighted error ϵ_t  of the weak classifier.
ϵ_t=∑_1^N▒〖ω_i^((t) ) I〗(h_t (x_i)≠y_i)
    where I  is the indicator function that is 1 if h_t (x_i )≠y_i and 0 is otherwise.
3.Compute Classifier Weight: Calculate the weight  of the weak classifier based on its error:
				α_t=1/2  ln⁡〖((1-∈_t))/∈_t 〗
4.Update Weights: Update the weights of the training samples. Increase the weights of the misclassified samples and decrease the weights of the correctly classified samples:
ω_i^((t+1))=ω_i^((t))  exp⁡〖(α_t I(h_t (x_i)〗≠y_i))
5.Normalize Weights: Normalize the weights so that they sum to 1:
				ω_i^((t+1))=(ω_i^((t+1)))/(∑_(j=1)^N▒ω_j^((t+1)) )
Final Strong Classifier
The final strong classifier H(x) is a weighted majority vote of the T weak classifiers:
H(x)=sign(∑_(t=1)^T▒〖α_t h_t (x〗))

















Result and Discussion
In our analysis, we used various machine learning models to predict our target variable. The models evaluated and their corresponding accuracies are summarized in the table provided.
Model Name 	Accuracy(%)	Precision	Recall	F1-Score
Logistic Regression	68.88	0.69	0.69	0.69
GaussianNB	73.33	0.73	0.73	0.73
Voting Classifier	86.66	0.87	0.87	0.87
XGBoost Classifier 	90.00	0.89	0.91	0.90
LightGBM	91.11	0.91	0.91	0.91
Adaboost	93.33	0.93	0.93	0.93
Random Forest Classifier	93.33	0.93	0.93	0.93
Gradient Boosting Classifier	94.44	0.94	0.94	0.94
Stacking Classifier	97.76	0.97	0.97	0.97

we achieved the highest overall accuracy using a Stacking Classifier, an advanced ensemble learning technique. In our stacking approach, we used the following base models:
	Quadratic Discriminant Analysis (QDA)
	MLPClassifier (Multi-layer Perceptron) with a neural network architecture of two hidden layers (100 and 50 neurons, respectively) and a maximum of 1000 iterations.
	AdaBoostClassifier with 100 estimators.
The final estimator for stacking was Logistic Regression. This combination allowed us to leverage the strengths of different types of classifiers:
	QDA handles data with distinct class distributions effectively.
	MLPClassifier captures complex patterns through neural networks.
	AdaBoost focuses on difficult-to-classify instances.
By stacking these models and using logistic regression as the final estimator, we effectively captured diverse patterns and interactions in the data, resulting in improved predictive performance. The stacking classifier outperformed all individual models, demonstrating the power of combining different machine learning techniques to achieve higher accuracy and robustness in predictions.
 

The ROC curve in our code is generated using the following steps:
	Train-Test Split: Split the dataset into training and testing sets.
	Feature Scaling: Scale the features using StandardScaler.
	Define Base Models: Define the base models for stacking (QDA, MLPClassifier, and AdaBoostClassifier).
	Define Final Estimator: Define the final estimator for stacking (Logistic Regression).
	Create and Train Stacking Classifier: Create and train the stacking classifier.
	Make Predictions: Make predictions on the test set.
	Compute Accuracy and Confusion Matrix: Calculate the accuracy and confusion matrix to evaluate performance.
	Compute ROC Curve and AUC:
	Calculate the probabilities for the positive class.
	Compute the false positive rate and true positive rate using roc_curve.
	Compute the area under the curve using auc.
	Plot the ROC curve.
ROC Curve Interpretation
	Orange Line (ROC Curve): Represents the trade-off between TPR and FPR at different threshold values. The closer this curve follows the left-hand border and then the top border of the ROC space, the better the performance of the classifier.
	Diagonal Line (Baseline): Represents a random classifier (AUC = 0.5). If your ROC curve is above this line, your classifier is better than random guessing.
	AUC = 0.98: Indicates excellent performance. An AUC of 0.98 means the model has a 98% chance of distinguishing between a positive and a negative instance correctly.


