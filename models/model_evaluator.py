import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.figure_factory as ff

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = list(set(y_true))
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Viridis',
        showscale=True,
        annotation_text=cm.astype(str)
    )
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    return fig

def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print(report)

def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
