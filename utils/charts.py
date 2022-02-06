import pandas as pd
import plotly.express as px


def build_roc(fpr, tpr, thresholds, auc_result):
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr}, index=thresholds)
    roc = px.area(df, x='fpr', y='tpr', title=f'ROC Curve (AUC={auc_result:.4f})',
                  labels=dict(fpr='False Positive Rate', tpr='True Positive Rate'), hover_data={'threshold': df.index})
    roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    roc.update_yaxes(scaleanchor="x", scaleratio=1)
    roc.update_xaxes(constrain='domain')
    return roc


def build_precision_recall(precision, recall, thresholds, auc_result):
    df = pd.DataFrame({'recall': recall, 'precision': precision}, index=thresholds)
    precision_recall = px.area(df, x='recall', y='precision', title=f'Precision-Recall Curve (AUC={auc_result:.4f})',
                               labels=dict(recall='Recall', precision='Precision'), hover_data={'threshold': df.index})
    precision_recall.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
    precision_recall.update_yaxes(scaleanchor="x", scaleratio=1)
    precision_recall.update_xaxes(constrain='domain')
    return precision_recall
