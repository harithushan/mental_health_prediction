import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

def plot_training_results(results_df):
    fig = px.bar(
        results_df,
        x='model_name',
        y='accuracy',
        title='Model Accuracy Comparison',
        labels={'accuracy': 'Accuracy', 'model_name': 'Model'}
    )
    return fig

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