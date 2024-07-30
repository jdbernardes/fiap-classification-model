from fiap_classification_model.extractor.extract_data import ExtractUCIData
from fiap_classification_model.ml_models.classification_model import ClassificationModel
from sklearn.neighbors import KNeighborsClassifier

data_source_id = 545
result_path = "fiap_classification_model/data/result.csv"
edata = ExtractUCIData(data_source_id)
X = edata.retrieve_x()
Y = edata.retrieve_y()
cmodel = ClassificationModel(X, Y, result_path)
knn = KNeighborsClassifier(n_neighbors=7)
model_list = cmodel.train_model(knn)
cmodel.run_predict(
    model=model_list[0],
    X_test=model_list[1],
    y_test=model_list[2],
)
# cross_mode = 'str'
# cmodel.train_with_cross_validation(mode=cross_mode, model=knn)

# for result in results:
#     print(result)
