# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

from sklearn.utils import class_weight
import numpy as np
import _config as config


def classweights(
    labels_dict,
    wts_use,
    trainlabels,
    balance=config.class_balance,
    setclassimportance=config.setclassimportance,
    num_train_imgs=0,
    train_cat_cts=[],
):
    """
    This function creates a dictionary of class weights needed for input for the CNN, where the class index is the key, and the value is the class weight for that class. 

    wts_use (str): if set to "yes", then class weights will be set and returned based on whether set balance var to True or set setweights
    trainlabels: list of labels which will be used to calculate the weights. Used in any scenario where class weights are required
    Note: Either balance OR setclassimportance must be used if using weights
    balance (boolean): if True, will do it based on unequal balance of number imgs per class.
    setclassimportance (list): if balance is False, it will use this list of predefined (predetermined) percentages that add up to 100% representing mportance by class. N. Note, when using balance = True, it's assuming each class is equal, so 1/6 = 16.67% per class. When using this instead, set it so that dry, wet, etc are the percentages we want (i.e. usually to underweight obs).
    num_train_imgs (int): if using setclassimportance, this is needed as it is the total number of images in the training set and used in calculation of weights.
    train_cat_cts (list): count of images in each class, used for weight calculation. Should be alphabetical

    Returns dictionary of each class and its corresponding weight
    """
    # # Initialize the LabelEncoder to convert from string classes to values (0-5 if 6-class)
    # le = preprocessing.LabelEncoder()
    # # Fit the encoder on the unique class labels (it automatically maps strings to numbers)
    # le.fit(trainlabels)
    # # Transform the string labels to numeric values (0-5)
    # labels_ind = le.transform(trainlabels)

    # print(trainlabels)
    # print(labels_dict)


    # labels_ind = [labels_dict[catname] for catname in trainlabels]
    labels_ind = trainlabels
    # try doing class weighting on trainlables instead of labels_ind (values)
    if wts_use == "yes":
        if balance:
            print(
                "using class weights based solely on unequal balance of imgs per class, i.e. gives each class equal importance/influence in model build via loss function"
            )
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(labels_ind),
                y=labels_ind,
            )
        else:
            print(
                "using class weights as defined by the given importance/influence of each class, which sum to 1 in total. Doing the normal way is 1/6 per class (if 6 classes) but we may want to force severe snow to make up 1/2 of total influence, which we can do using this method. All classes just have to sum to 1. "
            )
            print(setclassimportance)
            class_weights = []
            for i in range(0, len(train_cat_cts)):
                print(num_train_imgs)
                ifeq = setclassimportance[i] * num_train_imgs
                print(ifeq)
                print(train_cat_cts[i])
                wt = ifeq / train_cat_cts[i]
                print(wt)
                class_weights.append(wt)
        class_weight_set = dict(zip(np.unique(labels_ind), class_weights))
    else:
        class_weight_set = None
    print(f"Class weight set is {class_weight_set}")

    return class_weight_set
