import pandas as pd
import numpy as np


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X: pd.DataFrame, y:pd.Series, verbose=False):
        X_copy = X.copy()               # создадим копию нашей матрицы фичей, что бы не изменить изначальный
        X_copy.insert(0, 'base', 1)     # допишем слева столбик из 1 для свободного члела
        w = np.ones(X_copy.shape[1])    # составим вектор весов, заполненный из 1

        if verbose:                   # Напишем подсчет ошибки до начала обучения
            initial_MSE = np.mean((np.dot(X_copy, w) - y)**2)
            print(f'start | loss: {initial_MSE}')
            
        for i in range(self.n_iter): # Напишем цикл обучения
            pred = np.dot(X_copy, w) # cчитаем предасказания модели 
            error = pred - y         # вычисляем ошибку (предсказания - реальные значения)
            grad = (2/len(X_copy)) * np.dot(error, X_copy) # вычисляем градиенты по весам
            w -= self.learning_rate * grad  # обновляем веса 

            if verbose and i % 10 == 0:     # выводим логи 
                loss = np.mean(error**2)
                print(f'Iteration {i}| loss: {loss}')
        # сохроняем веса
        self.weights = w 

    def predict(self, X: pd.DataFrame)->np.array:
        X_copy = X.copy()            # делаем копию матрицы фичей, чтобы не менять изначальный датафрейм
        X_copy.insert(0, 'base', 1)  # допишем слева столбик из 1 для свободного члена
        predict = X_copy.dot(self.weights) # делаем предсказания X_copy @ self.weights
        return predict 
    
    def get_coef(self):
        return self.weights[1:] # метод для вывода весов начиная с 1го заначения
