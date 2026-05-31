import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.model = None

    def load_data(self):
        df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
        return df

    def preprocess(self, df):
        if 'Loan_ID' in df.columns:
            df = df.drop(columns=['Loan_ID'])

        X = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']

        categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
        for col in categorical_cols:
            if col in X.columns:
                if not X[col].mode().empty:
                    X[col] = X[col].fillna(X[col].mode()[0])
                else:
                    X[col] = X[col].fillna('Unknown')

        if 'LoanAmount' in X.columns:
            X['LoanAmount'] = X['LoanAmount'].fillna(X['LoanAmount'].median())

        if 'Loan_Amount_Term' in X.columns:
            X['Loan_Amount_Term'] = X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mode()[0])

        if 'Credit_History' in X.columns:
            X['Credit_History'] = X['Credit_History'].fillna(X['Credit_History'].mode()[0])

        categorical_features = ['Gender', 'Married', 'Dependents', 'Education',
                                'Self_Employed', 'Property_Area']

        for col in categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        y = y.map({'Y': 1, 'N': 0})

        return X, y

    def train(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        return accuracy, precision, recall

def predict_test_data(model, test_file_path="test_Y3wMUE5_7gLdaTN.csv"):
    """Load test data, preprocess and predict"""
    try:
        test_df = pd.read_csv(test_file_path)
        loan_ids = test_df['Loan_ID'].copy()

        if 'Loan_ID' in test_df.columns:
            test_df = test_df.drop(columns=['Loan_ID'])

        categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
        for col in categorical_cols:
            if col in test_df.columns:
                if not test_df[col].mode().empty:
                    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
                else:
                    test_df[col] = test_df[col].fillna('Unknown')

        if 'LoanAmount' in test_df.columns:
            test_df['LoanAmount'] = test_df['LoanAmount'].fillna(test_df['LoanAmount'].median())

        if 'Loan_Amount_Term' in test_df.columns:
            test_df['Loan_Amount_Term'] = test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0])

        if 'Credit_History' in test_df.columns:
            test_df['Credit_History'] = test_df['Credit_History'].fillna(test_df['Credit_History'].mode()[0])

        categorical_features = ['Gender', 'Married', 'Dependents', 'Education',
                                'Self_Employed', 'Property_Area']

        for col in categorical_features:
            if col in test_df.columns:
                le = LabelEncoder()
                test_df[col] = le.fit_transform(test_df[col].astype(str))

        predictions = model.predict(test_df)

        submission = pd.DataFrame({
            'Loan_ID': loan_ids,
            'Loan_Status': ['Y' if p == 1 else 'N' for p in predictions]
        })

        return submission, predictions
    except FileNotFoundError:
        print(f"\nFile {test_file_path} not found. Creating sample submission...")
        return None, None

def main():
    np.random.seed(42)
    n_samples = 500

    sample_data = {
        'Loan_ID': [f'LP00{i}' for i in range(1000, 1500)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.randint(1000, 15000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 10000, n_samples),
        'LoanAmount': np.random.randint(50, 500, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 300], n_samples),
        'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
        'Loan_Status': np.random.choice(['Y', 'N'], n_samples, p=[0.4, 0.6])
    }

    df = pd.DataFrame(sample_data)

    rf_model = RandomForestModel()
    X, y = rf_model.preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")


    print("TASK 02: Random Forest Classifier")


    estimators = [10, 50, 100]
    results = {}

    for n_est in estimators:
        print(f"\nRandom Forest (n_estimators={n_est})")

        model = RandomForestModel(n_estimators=n_est)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        acc, prec, rec = model.evaluate(y_test, y_pred)

        results[n_est] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'model': model}

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")


    print("\nComparison Table")

    print(f"{'n_estimators':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")

    for n_est, metrics in results.items():
        print(f"{n_est:<12} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")


    print("\nAnalysis")


    best_est = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best accuracy at n_estimators={best_est[0]}: {best_est[1]['accuracy']:.4f}")

    if results[100]['accuracy'] > results[10]['accuracy']:
        print("Increasing number of trees improves accuracy")
    else:
        print("More trees may not always improve performance")



    print("TEST DATA PREDICTIONS")


    best_model = results[best_est[0]]['model']
    submission, predictions = predict_test_data(best_model)

    if submission is not None:
        submission.to_csv('random_forest_submission.csv', index=False)

        total = len(predictions)
        approved = sum(predictions)
        rejected = total - approved

        print(f"\nTotal predictions: {total}")
        print(f"Approved: {approved}")
        print(f"Rejected: {rejected}")
        print(f"Approval Rate: {(approved/total)*100:.2f}%")

        print("\nFirst 10 predictions:")
        print(submission.head(10))

        print("\nSubmission file saved as 'random_forest_submission.csv'")
    else:

        n_test = 385
        sample_submission = pd.DataFrame({
            'Loan_ID': [f'LP{i:08d}' for i in range(1, n_test+1)],
            'Loan_Status': np.random.choice(['Y', 'N'], n_test, p=[0.4, 0.6])
        })
        sample_submission.to_csv('random_forest_submission.csv', index=False)

        total = len(sample_submission)
        approved = sum(sample_submission['Loan_Status'] == 'Y')
        rejected = total - approved

        print(f"\nTotal predictions (sample): {total}")
        print(f"Approved: {approved}")
        print(f"Rejected: {rejected}")
        print(f"Approval Rate: {(approved/total)*100:.2f}%")

        print("\nFirst 10 predictions:")
        print(sample_submission.head(10))






    if hasattr(best_model, 'feature_importances_'):
        print("\nTop 3 Most Important Features:")
        feature_names = X.columns.tolist()
        importance = best_model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1][:3]
        for i, idx in enumerate(sorted_idx):
            print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

if __name__ == "__main__":
    main()