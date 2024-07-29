from fiap_classification_model.extractor.extract_data import ExtractUCIData
from fiap_classification_model.ml_models.classification_model import ClassificationModel
from sklearn.neighbors import KNeighborsClassifier

data_source_id = 545
edata = ExtractUCIData(data_source_id)
X = edata.retrieve_x()
Y = edata.retrieve_y()
cmodel = ClassificationModel(X, Y)
knn = KNeighborsClassifier(n_neighbors=7)
results = cmodel.run_model(knn)


for result in results:
    print(result)
