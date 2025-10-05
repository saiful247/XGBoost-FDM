# Machine Learning Classification Project

## Overview

This project is a machine learning application that implements an XGBoost multiclass classification model. The model has been trained with Random Over Sampling (ROS) to handle class imbalance and is deployed as a web application.

## Deployed Application

You can access the live deployed application here: [https://fdm-app-deploy-saiful-1088335055755.asia-southeast1.run.app/](#) <!-- Replace this placeholder with the actual URL when available -->

## Project Structure

```
├── requirements.txt       # Python dependencies
├── app/
│   ├── app.py             # Flask web application
│   ├── feature_columns.pkl    # Pickled feature columns used by the model
│   ├── final_multiclass_xgboost_ros_model.pkl    # Trained XGBoost model
│   └── label_encoder.pkl  # Label encoder for target classes
```

## Installation

To set up this project locally, follow these steps:

1. Clone this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

To run the application:

```
cd app
python app.py
```

The web application will start and be available at the URL shown in the terminal output.

## Dependencies

The project relies on several libraries which are specified in the `requirements.txt` file, including:

- Flask (web framework)
- XGBoost (machine learning model)
- scikit-learn (machine learning utilities)
- pandas (data manipulation)
- numpy (numerical operations)

## Model Information

The model is a multiclass XGBoost classifier trained with Random Over Sampling to handle class imbalance. The model artifacts include:

- Trained model file (`final_multiclass_xgboost_ros_model.pkl`)
- Feature columns (`feature_columns.pkl`)
- Label encoder for the target classes (`label_encoder.pkl`)

## API Endpoints

The application likely exposes API endpoints for making predictions. Refer to the code in `app.py` for detailed information about the available endpoints and their usage.

## License

[Add appropriate license information here]

## Contributors

[Add contributor information here]

## Acknowledgements

[Add acknowledgements here]
