import plotly.express as px


def build_roc(df, auc_results):
    roc = px.area(df, x='fpr', y='tpr', title=f'ROC Curve',
                  labels=dict(fpr='False Positive Rate', tpr='True Positive Rate'),
                  hover_data={'threshold': df.thresholds},
                  facet_col='class')
    roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    roc.update_yaxes(scaleanchor="x", scaleratio=1)
    roc.update_xaxes(constrain='domain')
    roc.for_each_annotation(lambda a: a.update(text=a.text +
                                                    " (AUC=" +
                                                    str(round(auc_results[a.text.split('=')[-1]], 2)) +
                                                    ")"))
    return roc


def build_precision_recall(df, auc_results):
    precision_recall = px.area(df, x='recall', y='precision', title=f'Precision-Recall Curve',
                               labels=dict(recall='Recall', precision='Precision'),
                               hover_data={'threshold': df.thresholds},
                               facet_col='class')
    precision_recall.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
    precision_recall.update_yaxes(scaleanchor="x", scaleratio=1)
    precision_recall.update_xaxes(constrain='domain')
    precision_recall.for_each_annotation(lambda a: a.update(text=a.text +
                                                                 " (AUC=" +
                                                                 str(round(auc_results[a.text.split('=')[-1]], 2)) +
                                                                 ")"))
    return precision_recall
