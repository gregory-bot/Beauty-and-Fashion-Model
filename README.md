# Beauty and Fashion Recommendation System

## Overview
This project aims to solve a critical problem in the beauty industry: 76% of consumers struggle to find makeup and skincare products that suit their skin tone, type, and concerns, often wasting money on ineffective items. Our solution provides personalized product recommendations that:

- Match the user's skin tone (with a focus on darker/African tones where existing AI often fails).
- Fit their skin type, concerns, allergies, preferences, and budget.
- Prioritize products that actually work based on real efficacy data.

The system is designed to be trustworthy, fair, and highly accurate, outperforming generic AI models by avoiding misleading outputs through controlled, explainable logic.

## Solution
The platform uses a hybrid AI pipeline:

1. **Computer Vision Model**: Analyzes user-uploaded face/skin images to detect skin tone, type, and concerns.
2. **Recommendation Engine**: Matches extracted features to a curated product database for accurate, budget-aware suggestions.

## Why Build Our Own Model?
Initially, we used Gemini for this task. However, due to high costs associated with tokens and transactions, we decided to develop our own model. This approach will reduce costs and allow us to create a more tailored solution.

## File Structure
The project is organized as follows:

```
README.md
requirements.txt
api/
    main.py
    schemas.py
    services.py
data/
    processed/
    raw/
        dataset_scin_app_questions.csv
        dataset_scin_cases.csv
        dataset_scin_label_questions.csv
        dataset_scin_labels.csv
models/
    recommender/
        features.py
        train.py
    vision/
        dataset.py
        model.py
        train.py
notebooks/
    data_exploration.ipynb
    model_evaluation.ipynb
```

## Development Steps

### 1. Data Preparation
- Explore and preprocess the datasets located in the `data/raw/` directory.
- Clean and structure the data for training the models.

### 2. Computer Vision Model
- Develop a model to analyze user-uploaded images and extract features such as skin tone, type, and concerns.
- Use the `models/vision/` directory for dataset handling, model definition, and training scripts.

### 3. Recommendation Engine
- Build a recommendation engine that matches the extracted features to the product database.
- Use the `models/recommender/` directory for feature engineering and training the recommendation model.

### 4. API Development
- Develop APIs to serve the models and handle user interactions.
- Use the `api/` directory for API endpoints, schemas, and services.

### 5. Evaluation and Optimization
- Use the notebooks in the `notebooks/` directory for data exploration and model evaluation.
- Optimize the models for accuracy, fairness, and cost-efficiency.

## Getting Started

### Prerequisites
- Python 3.8+
- Install dependencies using:

```
pip install -r requirements.txt
```

### Running the Project
1. Prepare the data by placing raw datasets in the `data/raw/` directory.
2. Train the models using the scripts in the `models/` directory.
3. Start the API server:

```
python api/main.py
```

4. Access the API endpoints to interact with the system.

## Contributing
Contributions are welcome! Please follow the standard Git workflow:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Submit a pull request.

## License
This project is licensed under the MIT License.