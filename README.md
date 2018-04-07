# silver-_sparro
Assignment 2 For ML Engineers Freshers/Interns: Predicting correctness Index

We are sharing training data consisting of 3 fields (Date, Time, Bill Total) extracted from receipt images clicked from a mobile phone using a DL algorithm. Data consists of Ground Truth values as well as Predictions along with confidence score. 

Data is as follows:

    Image_Id: unique name of image
    gt_total, gt_time, gt_date : manually extracted receipt fields
    pred_total, pred_time, pred_date : fields predicted by DL algorithm
    conf_total, conf_time, conf_date : confidence score of DL algorithm


Training file consists of 5000 rows and validation file consists of 500 rows.

Task1: 

Train a model (ML/DL or anything else) to output 1 if the prediction is correct and 0 if the prediction is incorrect.

    a) List down features created and used (if using ML techniques)

    b) Get recall and precision above 90% on validation set provided. We shall check accuracy on test set available separately with us. 

Bonus Task: Create a model to output 1 if all 3 fields are predicted correctly and 0 if 2 or fewer fields are predicted correctly with above 90% precision and recall.
