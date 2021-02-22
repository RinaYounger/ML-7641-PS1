from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
import math
import graphviz
from sklearn import tree

if __name__ == "__main__":


    data = load_boston()
    model = DecisionTreeRegressor(max_depth=3)
    # train_x, test_x, train_y, test_y = train_test_split(data["data"], data["target"], random_state=0)
    model.fit(data["data"], data["target"])
    y_pred = model.predict(data["data"])
    print("RMSE")
    print(np.sqrt(mean_squared_error(data["target"], y_pred)))

    # model.fit(train_x, train_y)
    # y_train_pred = model.predict(train_x)
    # print("Training RMSE")
    # print(np.sqrt(mean_squared_error(train_y, y_train_pred)))
    # y_pred = model.predict(test_x)
    # print("Testing RMSE")
    # print(np.sqrt(mean_squared_error(test_y, y_pred)))

    dot_data = tree.export_graphviz(model,feature_names=data.feature_names, out_file=None)
    graph = graphviz.Source(dot_data, format='png')
    graph.render("DT Rrgressor")








