# 第5章：机票价格预测项目 - 预测与分析机票价格

本项目是一个端到端的机器学习实战项目，旨在通过机器学习技术预测和分析机票价格。项目涵盖了从数据收集、预处理、模型训练到最终使用 Flask 框架进行 Web 应用部署的完整工作流。

## 📌 项目内容

本项目分为两个主要部分，分别涵盖了模型构建和应用部署。

### 1️⃣ 机票价格预测 (数据分析与模型训练)
位于 `1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices` 目录下。

本部分主要关注数据科学工作流：
-   **项目介绍与目标**：利用历史机票数据进行价格趋势分析与预测。
-   **数据处理**：包含数据清洗、特征工程（日期时间处理、分类变量编码）和探索性数据分析 (EDA)。
-   **模型训练**：使用机器学习算法（如 XGBoost、随机森林等）进行模型训练和特征选择。
-   **评估与洞察**：评估模型性能，并从数据中挖掘影响票价的关键因素。

**核心文件：**
-   `Project_1_Flight_Fare_Price_Prediction.ipynb`: 包含完整分析过程的 Jupyter Notebook。
-   `Data_Train.xlsx`: 训练数据集，包含 10,000+ 条航班记录。
-   `Test_set.xlsx`: 测试数据集。

### 2️⃣ 使用 Flask 框架部署机票价格预测模型
位于 `2. Deploying the Flight Fare Prediction Model with Flask Framework Making Prediction` 目录下。

本部分展示了如何将训练好的模型转化为实际可用的 Web 服务：
-   **Web 应用部署**：基于 Flask 框架构建后端服务。
-   **用户交互界面**：提供用户友好的 Web 界面，支持实时输入航班信息并获取价格预测。
-   **功能特性**：
    -   **用户系统**：支持用户注册、登录及权限管理（管理员与普通用户）。
    -   **航班管理**：用户可以添加、编辑和删除航班记录。
    -   **数据可视化**：提供 `/analysis` 页面，展示基于数据的图表分析。
    -   **实时预测**：集成训练好的模型 (`flight_xgb.pkl`) 进行在线推断。

**核心文件：**
-   `app.py`: Flask 应用入口，处理路由和业务逻辑。
-   `models.py`: 定义数据库模型 (User, Flight, Prediction)。
-   `templates/`: HTML 模板文件，包含主页、登录注册、航班列表等页面。
-   `static/`: 静态资源文件。

## 🚀 快速开始

### 环境要求
请确保安装了 Python 3.x，并安装以下主要依赖库：
-   Flask
-   pandas
-   numpy
-   scikit-learn
-   matplotlib
-   seaborn

### 如何运行

1.  **进入 Web 项目目录**：
    ```bash
    cd "2. Deploying the Flight Fare Prediction Model with Flask Framework Making Prediction"
    ```

2.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```

3.  **运行应用**：
    ```bash
    python app.py
    ```

4.  **访问应用**：
    打开浏览器访问 `http://localhost:5000` 即可使用系统。

## ⚠️ 注意事项
-   **模型兼容性**：项目中的模型文件 `flight_xgb.pkl` 是基于特定版本的 `scikit-learn` (0.22.1) 训练的。如果遇到版本不兼容错误，建议重新运行 Notebook 重新训练模型，或降级 `scikit-learn` 版本。
-   **数据库**：项目使用 SQLite/MySQL (取决于配置)，首次运行会自动创建所需的数据表。

## 📂 项目结构概览

```
Section 5/
├── 1. Flight Fare Prediction Project-1 ... /  # 数据分析与模型训练
│   ├── Project_1_Flight_Fare_Price_Prediction.ipynb
│   └── Data_Train.xlsx
├── 2. Deploying the Flight Fare Prediction ... /  # Web 应用部署
│   ├── app.py              # Flask 应用主程序
│   ├── models.py           # 数据库模型
│   ├── templates/          # 前端模板
│   ├── static/             # 静态文件
│   └── requirements.txt    # 项目依赖
└── Readme.md               # 项目说明文档
```
