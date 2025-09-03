from src.preprocessor import preprocess_data
from src.model import train_model
from src.visualize import plot_confusion_matrix, plot_cv_scores

def main():
    # 1. Preprocess dataset
    X, y = preprocess_data("data/dataset.csv")  # make sure dataset.csv is in project root

    # 2. Train model
    model, acc, report, cm, cv_scores = train_model(X, y)

    # 3. Print results
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)
    print("\nCross-validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    # 4. Visualize results
    plot_confusion_matrix(cm)
    plot_cv_scores(cv_scores)

if __name__ == "__main__":
    main()
