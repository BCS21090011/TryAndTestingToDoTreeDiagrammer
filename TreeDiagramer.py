import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from matplotlib import pyplot as plt

class TreeDiagramer():
    def __init__(self,dataset: pd.DataFrame):
        self.OriDataset: pd.DataFrame = dataset.copy()
        self.Dataset: pd.DataFrame = dataset.copy()
        self.DataValRepresentation: dict = {}

        self.Features: list[str] = self.OriDataset.columns.tolist()

    def OneHotMapping(self, columnsToApply: list[str]):
        changesCol: list[str] = []
        varList: list[list[str]] = []

        for col in columnsToApply:
            if not pd.api.types.is_numeric_dtype(self.Dataset[col]):
                colNames: np.ndarray = self.Dataset[col].unique()
                encoder: OneHotEncoder = OneHotEncoder()

                changesCol.append(col)
                varList.append(colNames.tolist())

                encodedData: pd.DataFrame = pd.DataFrame(encoder.fit_transform(self.Dataset[[col]]).toarray())
                encodedData.columns = colNames

                self.Dataset = self.Dataset.join(encodedData)
                self.Dataset.drop(col, axis=1, inplace=True)

        self.OneHotDataVal: dict = dict(zip(changesCol, varList))

    def Mapping(self, columnsToApply: list[str]=None):
        changesCol: list[str] = []
        varDict: list[dict] = []

        for col in columnsToApply:
            if all((type(var) == bool) for var in self.Dataset[col]):
                changesCol.append(col)

                var: np.ndarray = self.Dataset[col].unique()  # Get the unique values in the pd.Series.
                changes: dict = {}

                if True in var:
                    changes.update({True: 1})
                if False in var:
                    changes.update({False: 0})

                self.Dataset[col] = self.Dataset[col].map(changes)

                varDict.append(changes)

            if not pd.api.types.is_numeric_dtype(self.Dataset[col]):  #This code uses the is_numeric_dtype function from the pandas.api.types module to check if the data type of the values in col is numeric.
                changesCol.append(col)

                var: np.ndarray = self.Dataset[col].unique()  # Get the unique values in the pd.Series.
                numVar: list[int] = list(range(len(var)))  # This will create a list with values from 0 until the number of variables of the column.

                changes: dict = dict(zip(var, numVar))
                self.Dataset[col] = self.Dataset[col].map(changes)

                varDict.append(changes)

        self.DataValRepresentation.update(dict(zip(changesCol, varDict)))

    def TreeDiagramingDataset(self, outputCol: str, uselessCol: list[str]=None, saveFileName: str=None):
        if uselessCol != None:
            self.Dataset.drop(uselessCol, axis=1, inplace=True)

            for col in uselessCol:
                self.Features.remove(col)

        if not all(pd.api.types.is_numeric_dtype(self.Dataset[col].dtype) for col in self.Dataset):
            self.Mapping(columnsToApply=self.Dataset.columns)

        self.output: pd.Series = self.Dataset[outputCol].copy()
        self.Dataset.drop(outputCol, axis=1, inplace=True)
        self.Features.remove(outputCol)

        self.treeDgrm: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier()
        self.treeDgrm = self.treeDgrm.fit(self.Dataset.values, self.output.values)

        tree.plot_tree(self.treeDgrm, feature_names=self.Dataset.columns)

        if saveFileName != None:
            plt.savefig(saveFileName)

    def Predict(self, inputs: list[int]):
        return self.treeDgrm.predict(inputs)

if __name__ == "__main__":
    import IOIwrote as IO
    retryPredict: bool = True
    outputCol: str = "Able to join the company vacation"

    data: dict = {
        "Age": [44, 25, 27, 32, 23, 34, 33, 26, 22, 27, 45, 63, 58, 65],
        "Years worked": [10, 1, 2, 3, 0, 3, 23, 21, 3, 6, 33, 43, 41, 39],
        "Gender": ["Male", "Female", "Other", "Male", "Female", "Other", "Male", "Female", "Other", "Male", "Male", "Female", "Other", "Male"],
        "Working hour": [10, 9, 9, 11, 9.5, 9, 9, 9, 9, 1, 7, 8.5, 8, 8],
        "Able to join the company vacation": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No"]
    }

    df: pd.DataFrame = pd.DataFrame(data)
    obj: TreeDiagramer = TreeDiagramer(dataset=df)
    obj.OneHotMapping(columnsToApply=["Gender"])
    obj.TreeDiagramingDataset(outputCol=outputCol, saveFileName="TreeDiagram.png")

    while retryPredict == True:
        userInput: list[list] = [[]]

        for col in obj.Features:
            print(f"Column: [{col:^16}]:")

            if col in obj.OneHotDataVal:
                oneHotInput: list[int] = [0] * len(obj.OneHotDataVal[col])  #Initialize a list with size of len(obj.OneHotDataVal[col] and the values is 0.
                for i in range(len(obj.OneHotDataVal[col])):
                    print(f"Index: [{i:>3}]:[{obj.OneHotDataVal[col][i]:^16}]")

                userOneHotIndex: int = IO.ReadInt(qstStr="Input the index: ", inMin=0, inMax=len(obj.OneHotDataVal[col]))
                oneHotInput[userOneHotIndex] = 1  #Set the column corresponding to the variable to 1.
                userInput[0].extend(oneHotInput)

            elif col in obj.DataValRepresentation:
                for key, val in obj.DataValRepresentation[col].items():
                    print(f"Index: [{val:>3}]:[{key:^16}]")

                userInput[0].append(IO.ReadInt(qstStr="Input the index: ", inMin=0, inMax=len(obj.DataValRepresentation[col])))

            else:
                userInput[0].append(IO.ReadFloat(qstStr="Input: "))

        predctOutput: int = int(obj.Predict(inputs=userInput))
        predctOutputVar: str = ""

        if outputCol in obj.DataValRepresentation:
            for key, value in obj.DataValRepresentation[outputCol].items():
                if predctOutput == value:
                    predctOutputVar = key

        print(f"The prediction output is: {predctOutput} ({predctOutputVar})")

        retryPredict = IO.YNDecision(decisionStr="Try to predict again? (Y/N)\n")