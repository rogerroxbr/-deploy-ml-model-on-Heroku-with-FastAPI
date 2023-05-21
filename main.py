"""
Put the code for your API here.
Author : Roger de Tarso
Date : 20th may 2023
"""
from starter.train_model import trainer, get_data, batch_inference


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

if __name__ == "__main__":
    DATA_PATH = "data/cleaned_data.csv"
    MODEL_PATH = "model/random_forest_model_with_encoder_and_lb.pkl"
    print(MODEL_PATH)

    # Get the splitted data
    train_data, test_data = get_data(DATA_PATH)
    # Training the model on the train data
    trainer(train_data, MODEL_PATH, CAT_FEATURES)
    # evaluating the model on the test data
    precision, recall, f_beta = batch_inference(
        test_data, MODEL_PATH, CAT_FEATURES
    )
