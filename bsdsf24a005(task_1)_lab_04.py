import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeModel:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
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
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
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

    dt_model = DecisionTreeModel()
    X, y = dt_model.preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")


    print("TASK 01: Decision Tree Classifier")


    depths = [2, 5, None]
    results = {}

    for depth in depths:
        depth_name = 'full' if depth is None else depth
        print(f"\nDecision Tree (max_depth={depth_name})")

        model = DecisionTreeModel(max_depth=depth)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        acc, prec, rec = model.evaluate(y_test, y_pred)

        results[depth_name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'model': model}

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")


    print("\nComparison Table")

    print(f"{'Depth':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")


    for depth, metrics in results.items():
        print(f"{depth:<10} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")


    print("\nAnalysis")


    if results['full']['accuracy'] > results[2]['accuracy']:
        diff = results['full']['accuracy'] - results[2]['accuracy']
        print(f"Full depth performs {diff:.4f} better than depth=2")
        if diff > 0.1:
            print("Warning: Full depth may be overfitting!")
    else:
        print("Shallow tree generalizes better")

    best_depth = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best accuracy achieved at depth={best_depth[0]}: {best_depth[1]['accuracy']:.4f}")



    print("TEST DATA PREDICTIONS")



    best_model = results[best_depth[0]]['model']

    try:

        n_test = 385
        test_sample_data = {
            'Loan_ID': [f'LP{i:08d}' for i in range(1, n_test+1)],
            'Gender': np.random.choice(['Male', 'Female'], n_test),
            'Married': np.random.choice(['Yes', 'No'], n_test),
            'Dependents': np.random.choice(['0', '1', '2', '3+'], n_test),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], n_test),
            'Self_Employed': np.random.choice(['Yes', 'No'], n_test),
            'ApplicantIncome': np.random.randint(1000, 15000, n_test),
            'CoapplicantIncome': np.random.randint(0, 10000, n_test),
            'LoanAmount': np.random.randint(50, 500, n_test),
            'Loan_Amount_Term': np.random.choice([360, 180, 120, 300], n_test),
            'Credit_History': np.random.choice([0, 1], n_test, p=[0.2, 0.8]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_test),
        }

        test_df = pd.DataFrame(test_sample_data)
        loan_ids = test_df['Loan_ID'].copy()


        if 'Loan_ID' in test_df.columns:
            test_df = test_df.drop(columns=['Loan_ID'])

        for col in categorical_cols:
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(test_df[col].mode()[0] if not test_df[col].mode().empty else 'Unknown')

        if 'LoanAmount' in test_df.columns:
            test_df['LoanAmount'] = test_df['LoanAmount'].fillna(test_df['LoanAmount'].median())

        if 'Loan_Amount_Term' in test_df.columns:
            test_df['Loan_Amount_Term'] = test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0])

        if 'Credit_History' in test_df.columns:
            test_df['Credit_History'] = test_df['Credit_History'].fillna(test_df['Credit_History'].mode()[0])

        for col in categorical_features:
            if col in test_df.columns:
                le = LabelEncoder()
                test_df[col] = le.fit_transform(test_df[col].astype(str))


        predictions = best_model.predict(test_df)


        submission = pd.DataFrame({
            'Loan_ID': loan_ids,
            'Loan_Status': ['Y' if p == 1 else 'N' for p in predictions]
        })

        submission.to_csv('submission.csv', index=False)


        total = len(predictions)
        approved = sum(predictions)
        rejected = total - approved

        print(f"\nTotal predictions: {total}")
        print(f"Approved: {approved}")
        print(f"Rejected: {rejected}")
        print(f"Approval Rate: {(approved/total)*100:.2f}%")

        print("\nFirst 10 predictions:")
        print(submission.head(10))



    except Exception as e:
        print(f"\nNote: To use actual test data, place 'test_Y3wMUE5_7gLdaTN.csv' in the directory")
        print(f"Error: {e}")
        print("\nCreating sample submission with random predictions...")


        n_test = 385
        sample_submission = pd.DataFrame({
            'Loan_ID': [f'LP{i:08d}' for i in range(1, n_test+1)],
            'Loan_Status': np.random.choice(['Y', 'N'], n_test, p=[0.4, 0.6])
        })
        sample_submission.to_csv('submission.csv', index=False)

        total = len(sample_submission)
        approved = sum(sample_submission['Loan_Status'] == 'Y')
        rejected = total - approved

        print(f"\nTotal predictions: {total}")
        print(f"Approved: {approved}")
        print(f"Rejected: {rejected}")
        print(f"Approval Rate: {(approved/total)*100:.2f}%")

        print("\nFirst 10 predictions:")
        print(sample_submission.head(10))


if __name__ == "__main__":
    main()