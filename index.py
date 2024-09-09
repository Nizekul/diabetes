import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import numpy as np

data, meta = arff.loadarff('diabetes.arff')

df = pd.DataFrame(data)
df['class'] = df['class'].str.decode('utf-8')

label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

learning_rates = [0.1, 0.01, 0.001]
hidden_neurons = [(3,), (5,), (7,)]

# Definir o número de execuções
n_runs = 30

# Armazenar os resultados
all_results = []

for run in range(n_runs):
    print(f"Execução {run+1}/{n_runs}")
    
    # Loop sobre taxas de aprendizado e número de neurônios
    for lr in learning_rates:
        for neurons in hidden_neurons:
            # Criar o modelo MLP
            mlp = MLPClassifier(hidden_layer_sizes=neurons, learning_rate_init=lr, max_iter=1000, random_state=run)
            
            # Treinar o modelo
            mlp.fit(X_train, y_train)
            
            # Fazer previsões
            y_pred = mlp.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Armazenar resultados
            all_results.append({
                'run': run + 1,
                'learning_rate': lr,
                'neurons': neurons,
                'mse': mse,
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix
            })

for result in all_results:
    print(f"Execução: {result['run']}, Taxa de aprendizado: {result['learning_rate']}, Neurônios: {result['neurons']}")
    print(f"Erro Médio Quadrático (MSE): {result['mse']}, Acurácia: {result['accuracy']}")
    print(f"Matriz de Confusão:\n{result['confusion_matrix']}")
    print("-" * 50)

avg_mse = np.mean([result['mse'] for result in all_results])
avg_accuracy = np.mean([result['accuracy'] for result in all_results])

print(f"MSE Médio após {n_runs} execuções: {avg_mse}")
print(f"Acurácia Média após {n_runs} execuções: {avg_accuracy}")
