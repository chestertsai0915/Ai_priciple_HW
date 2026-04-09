import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import os
random_seed = 42
np.random.seed(random_seed)
class TrainBase:
    def __init__(self):
        self.weights = None
        self.intercept = None
        
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

class LinearRegression(TrainBase):
    def fit(self, X, y, learning_rate=0.01, T=1000,batch_size=1):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0
        
        # T 次更新
        for t in range(T):
            # 隨機選點
            batch_indices = np.random.choice(n_samples, size=batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices].flatten()

            # 值和誤差 
            y_pred_batch = np.dot(X_batch, self.weights) + self.intercept
            error = y_pred_batch - y_batch

            # 更新參數
            self.weights -= learning_rate * np.dot(X_batch.T, error) / batch_size
            
            self.intercept -= learning_rate * np.mean(error)

            if (t + 1) % 200 == 0:
                logger.info(f"Iteration {t+1}/{T} completed.")

    def predict(self, X):
        return X @ self.weights + self.intercept
    
class LogisticRegression(TrainBase):
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, learning_rate=0.75, T=1000, batch_size=32):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0

        for t in range(T):
            batch_indices = np.random.choice(n_samples, size=batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices].flatten()

            y_pred_batch = self.predict_proba(X_batch)
            error = y_pred_batch - y_batch

            self.weights -= learning_rate * np.dot(X_batch.T,error) / batch_size
            self.intercept -= learning_rate * np.mean(error)

    def predict_proba(self, X):
        linear_combination = X @ self.weights + self.intercept
        return self.sigmoid(linear_combination)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

def compute_mse(prediction, ground_truth):
    return np.mean((prediction - ground_truth) ** 2)

def compute_logistic_loss(prediction, ground_truth):
    logistic_loss = np.mean(-ground_truth * np.log(prediction) - (1 - ground_truth) * np.log(1 - prediction))
    return logistic_loss

def hw1(learning_rate=0.01, T=1000, batch_size=1):
    try:
        x_df = pd.read_csv('hw1_dataset\Problem 1\Averaged homework scores.csv')
        y_df = pd.read_csv('hw1_dataset\Problem 1\Final exam scores.csv')
        
        X_all = x_df.values.reshape(-1, 1) 
        y_all = y_df.values.reshape(-1)
    except Exception as e:
        logger.error(f"讀取失敗: {e}")
        return

    # 分割原始資料
    train_x_orig, test_x_orig = X_all[:400], X_all[400:]
    train_y, test_y = y_all[:400], y_all[400:]

    # 資料縮放：全部除以 100
    train_x = train_x_orig / 100.0
    test_x = test_x_orig / 100.0

    model = LinearRegression()
    model.fit(train_x, train_y, learning_rate=learning_rate, T=T, batch_size=batch_size)

    y_test_pred = model.predict(test_x)
    test_mse = compute_mse(y_test_pred, test_y)
    
    logger.info(f"Final Weights: {model.weights}")
    logger.info(f"Final Intercept: {model.intercept:.4f}")
    logger.info(f"Testing MSE: {test_mse:.4f}")

    # 繪圖 
    plt.figure(figsize=(10, 6))
    # 畫點：使用原始 X 座標 (0~100)
    plt.scatter(test_x_orig, test_y, color='black', alpha=0.6, marker='*', s=80, linewidths=1.5, label='Testing dataset')
    
    line_x_orig = np.linspace(test_x_orig.min(), test_x_orig.max(), 100).reshape(-1, 1)
    line_y = model.predict(line_x_orig / 100.0)
    
    plt.plot(line_x_orig, line_y, color='red', linewidth=2, label='Linear regression result')
    
    plt.xlabel('Averaged Homework Scores')
    plt.ylabel('Final Exam Scores')
    plt.title(f'Part 1: SGD Linear Regression (Testing MSE: {test_mse:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()

def hw2(learning_rate=0.75, T=1000, batch_size=32):
    epsilon = 1e-15
    try:
        X1_= pd.read_csv(r'D:\code\python\人工智慧原理\hw1_dataset\Problem 2\Averaged homework scores.csv', header=None)
        X2_= pd.read_csv(r'D:\code\python\人工智慧原理\hw1_dataset\Problem 2\Final exam scores.csv', header=None)
        Y_= pd.read_csv(r'D:\code\python\人工智慧原理\hw1_dataset\Problem 2\Results.csv', header=None)

    except Exception as e:
        logger.error(f"讀取失敗: {e}")
        return
    X= np.column_stack((X1_, X2_))
    Y= np.array(Y_)
    # 分割原始資料
    train_x_orig, test_x_orig = X[:400], X[400:]
    train_y, test_y = Y[:400], Y[400:]

    # 資料縮放：全部除以 100
    X_train_mean= train_x_orig.mean(axis=0)
    X_train_std= train_x_orig.std(axis=0)
    X_train_norm= (train_x_orig - X_train_mean) / X_train_std

    model = LogisticRegression()
    model.fit(X_train_norm, train_y, learning_rate=learning_rate, T=T, batch_size=batch_size)
    X_test_norm = (test_x_orig - X_train_mean) / X_train_std
    y_test_pred = model.predict(X_test_norm)
    y_test_pred  = np.clip(y_test_pred , epsilon, 1 - epsilon)
    test_loss = compute_logistic_loss(y_test_pred, test_y)

    logger.info(f"Final Weights: {model.weights}")
    logger.info(f"Final Intercept: {model.intercept:.4f}")
    
    logger.info(f"Testing Logistic Loss: {test_loss:.4f}")

    # 繪圖
    Y_test=  test_y.reshape(-1)
    plt.figure(figsize=(8, 6))

    # 畫出預測為錄取的點 
    plt.scatter(X_test_norm[Y_test == 1][:, 0], 
                X_test_norm[Y_test == 1][:, 1], 
                color='red', label='Predicted: Admitted (1)', alpha=0.7)

    # 畫出預測為未錄取的點 
    plt.scatter(X_test_norm[Y_test == 0][:, 0], 
                X_test_norm[Y_test == 0][:, 1], 
                color='blue', marker='x', label='Predicted: Rejected (0)', alpha=0.7)

    
    x_vals = np.array([X_test_norm[:, 0].min() - 0.5, X_test_norm[:, 0].max() + 0.5])
    # 確保 w[1] 不為 0 以避免除以零錯誤
    if model.weights[1] != 0:
        y_vals = -(model.weights[0] / model.weights[1]) * x_vals - (model.intercept / model.weights[1])
        plt.plot(x_vals, y_vals, '--', color='green', linewidth=2, label='Decision Boundary')

    plt.title("Logistic Regression Hard Decisions (Normalized)")
    plt.xlabel("Averaged Homework Score (Normalized)")
    plt.ylabel("Final Exam Score (Normalized)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

   
    plt.xlim(X_test_norm[:, 0].min() - 0.5, X_test_norm[:, 0].max() + 0.5)
    plt.ylim(X_test_norm[:, 1].min() - 0.5, X_test_norm[:, 1].max() + 0.5)

    plt.show()



if __name__ == '__main__':
    #hw1(learning_rate=0.01, T=1000, batch_size=32)
    hw2(learning_rate=0.75, T=1000, batch_size=1)
    