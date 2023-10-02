We are given three datasets
train.csv - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
test.csv - Test set. 1 day of ads to for testing your model predictions.
sampleSubmission.csv - Sample submission file in the correct format, corresponds to the All-0.5 Benchmark.
Each row in the dataset is one clict to the web page. We are predicting the click-through rate of each user to a certain page.

The data fields in the given files are
id - ad identifier
click - 0/1 for non-click/click
hour - format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC
C1 - anonymized categorical variable
banner_pos - the visited banner position
site_id - the visited site id
site_domain - the visited site domain
site_category - the visited site category
app_id - the using app id
app_domain - the using app domain
app_category - the using app category
device_id - the using device id device_ip - the using device ip
device_model - the using device model
device_type - the using device type
device_conn_type - the using device connection type
C14-C21 - anonymized categorical variables

The data is downloaded from the following link:
https://www.kaggle.com/competitions/avazu-ctr-prediction/data
