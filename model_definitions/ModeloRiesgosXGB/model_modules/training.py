from teradataml import (
    DataFrame,
    GLM,
    ScaleFit,
    RandomSearch,
    XGBoost, 
    ScaleTransform
)
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import numpy as np
import random
import pandas




def plot_feature_importance(fi, img_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(kind="barh").set_title("Feature Importance")
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)

    print ("Scaling using InDB Functions...")
    
    #scaler = ScaleFit(
    #    data=train_df,
    #    target_columns = feature_names,
    #    scale_method = context.hyperparams["scale_method"],
    #    miss_value = context.hyperparams["miss_value"],
    #    global_scale = context.hyperparams["global_scale"].lower() in ["true", "1"],
    #    multiplier = context.hyperparams["multiplier"],
    #    intercept = context.hyperparams["intercept"]
    #)


    #scaled_train = ScaleTransform(
    #    data=train_df,
    #    object=scaler.output,
    #    accumulate = [target_name,entity_key]
    #)
    
   # scaler.output.to_sql(f"scaler_${context.model_version}", if_exists="replace")
   # print("Saved scaler")
    
    print("Starting training...")

    
    # Definición de la grilla de búsqueda de hiperparámetros

    XGB_params = {"input_columns":feature_names,
              "response_column" : target_name,
              "max_depth":tuple(random.randrange(3, 50) for i in range(10)),
              "lambda1" : tuple(round(random.uniform(0.001, 1.0), 3) for i in range(10)),
              "model_type" : "classification",
              "num_boosted_trees": 72,
              "shrinkage_factor":tuple(round(random.uniform(0.001, 1.0), 3) for i in range(10)),
              "iter_num":(50, 200, 500, 1000)}
    
    eval_params = {"id_column": entity_key,
               "model_type": "classification",
               "accumulate": target_name,
               "object_order_column": ['task_index', 'tree_num', 'iter', 'class_num', 'tree_order']}
    
    # Inicio de la búsqueda en grilla
    
    print ("Inicio de la búsqueda en grilla")
    rs_obj = RandomSearch(func=XGBoost, params=XGB_params, n_iter=4)
    
    # Inicio de la optimización con RandomSearch

    print ("Inicio de la optimización con RandomSearch")
    rs_obj.fit(data=train_df,
           verbose=1, frac=0.85,
           **eval_params
            )
    
    # Id del mejor modelo

    modelid = rs_obj.best_model_id
    print (modelid)
    print(rs_obj.best_score_)  # Mejor score 
    print(rs_obj.best_params_) # Hyperparametros)
    #model.result.to_sql(f"model_${context.model_version}", if_exists="replace")    
    print("Saved trained model xyz")

    # Calculate feature importance and generate plot
    
    #model_pdf = model.result.to_pandas()[['predictor','estimate']]
    #predictor_dict = {}
    
    #for index, row in model_pdf.iterrows():
    #    if row['predictor'] in feature_names:
    #        value = row['estimate']
    #        predictor_dict[row['predictor']] = value
    
    #feature_importance = dict(sorted(predictor_dict.items(), key=lambda x: x[1], reverse=True))
    #keys, values = zip(*feature_importance.items())
    #norm_values = (values-np.min(values))/(np.max(values)-np.min(values))
    #feature_importance = {keys[i]: float(norm_values[i]*1000) for i in range(len(keys))}
    #plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")

    #print(feature_names)
    #print(target_name)
    #print(feature_importance)
    #print(context)
    
    #record_training_stats(
    #    train_df,
    #    features=feature_names,
    #    targets=[target_name],
    #    categorical=[target_name],
    #    feature_importance=feature_importance,
    #    context=context
    #)
