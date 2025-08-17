# Previsão de Série Temporal com LSTM

## Redes Neurais 2025.1  

Este projeto tem como objetivo realizar previsão do preço de fechamento do Bitcoin utilizando redes neurais do tipo **LSTM (Long Short-Term Memory)**, aplicadas em dados temporais históricos.

---

## 📂 Estrutura do Projeto

- `data-bitcoin_timedata_v2.csv` : Conjunto de dados com preços históricos do Bitcoin (`date` e `close`).  
- `notebook.ipynb` : Notebook principal com todo o pipeline, incluindo análise exploratória, treinamento do modelo LSTM, avaliação e extrapolação.  
- `best_model_state_dict.pth` : Modelo treinado com o melhor desempenho salvo durante o treino (gerado após execução do notebook).

---

## 🛠 Tecnologias Utilizadas

- Python 3.x  
- Pandas, Numpy  
- Matplotlib, Seaborn  
- Statsmodels (análise de autocorrelação e correlação)  
- Scikit-learn (MinMaxScaler)  
- PyTorch (LSTM, treinamento e avaliação)  
- tqdm (barra de progresso)

---

## 🔍 Pipeline do Projeto

1. **Leitura e pré-processamento dos dados**  
   - Conversão da coluna `date` para datetime  
   - Normalização do preço de fechamento (`close`) usando `MinMaxScaler`  
   - Divisão dos dados em `train`, `valid` e `test`  

2. **Análise exploratória**  
   - Visualização da série temporal  
   - Análise de correlação entre defasagens (`lags`)  
   - Pairplot para visualização conjunta das variáveis defasadas  

3. **Preparação de dados para LSTM**  
   - Criação de janelas temporais (`window_size`) para X (features) e y (target)  
   - Geração de `DataLoaders` para treino, validação e teste  

4. **Modelagem com LSTM**  
   - Definição da arquitetura LSTM com dropout e camada linear final  
   - Treinamento com `Adam`, monitoramento da loss e `early stopping`  
   - Scheduler de taxa de aprendizado opcional (`StepLR`, `CosineAnnealingLR`, etc.)  

5. **Avaliação do modelo**  
   - Métricas utilizadas: RMSE, MAE, MAPE  
   - Visualização de previsões no conjunto de teste (`next day`)  
   - Extrapolação iterativa para previsão futura

---

## ⚙ Configuração do Modelo

Exemplo de hiperparâmetros utilizados no projeto:

```python
config = {
    "window_size": 7,
    "hidden_size": 80,
    "num_layers": 1,
    "dropout_rate": 0.1,
    "n_epochs": 300,
    "learning_rate": 5e-3,
    "optimizer": torch.optim.Adam,
    "loss_fn": torch.nn.MSELoss(),
    "patience": 20,
    "bs_train": 16,
    "bs_val_test": 256,
    "use_lr_scheduler": True,
    "scheduler_type": "StepLR",
    "scheduler_params": {"StepLR": {"step_size": 20, "gamma": 0.7}}
}
