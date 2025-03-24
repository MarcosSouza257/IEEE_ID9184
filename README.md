# IEEE_ID9184
Brazilian Stock Market Forecast with Heterogeneous Data Integration for a Set of Stocks

## Estrutura do Código


### Descrição das Pastas e Arquivos

#### Pastas
- **data_collection/**:  
  Contém o código-fonte para coleta e extração dos dados brutos que serão utilizados no projeto. Aqui estão scripts ou notebooks responsáveis por acessar APIs, bancos de dados ou outras fontes de dados.

- **data/**:  
  Armazena os arquivos de dados processados e prontos para uso no treinamento e teste dos modelos. Inclui conjuntos de dados limpos, normalizados ou pré-processados.

- **output/**:  
  Pasta destinada ao armazenamento das saídas geradas pelo projeto, como resultados de treinamentos, métricas de desempenho, gráficos e outros arquivos relevantes.

- **docs/**:  
  Contém a documentação adicional do projeto, incluindo guias de uso, explicações técnicas, diagramas e qualquer outro material que ajude a entender o funcionamento do projeto.

#### Arquivos
- **config.py**:  
  Arquivo de configuração centralizado, onde são definidos caminhos de pastas, parâmetros globais e outras configurações utilizadas ao longo do projeto. Facilita a manutenção e a adaptação do código.

- **utils.py**:  
  Contém funções utilitárias reutilizáveis em diferentes partes do projeto, como funções de pré-processamento de dados, cálculos matemáticos ou auxiliares para manipulação de arquivos.

- **utils_backtest.py**:  
  Inclui o código-fonte específico para a realização do backtest, como funções de simulação, cálculo de métricas de desempenho e geração de relatórios.

- **models.py**:  
  Contém a implementação dos 4 modelos utilizados no projeto, incluindo a definição, treinamento e avaliação de cada um. Pode incluir modelos de machine learning, deep learning ou outras abordagens.

- **main.ipynb**:  
  Notebook principal que executa o treinamento dos modelos. Aqui são carregados os dados, configurados os modelos e realizadas as etapas de treinamento e avaliação.

- **backtest.ipynb**:  
  Notebook dedicado à execução do backtest. Inclui a aplicação dos modelos treinados em dados históricos, a análise de desempenho e a geração de resultados.

#### Dependências
- **requirements.txt**:  
  Lista todas as bibliotecas e dependências necessárias para executar o projeto. Para instalar, utilize o comando:  




### Como Executar o Código

1. Instale as dependências necessárias:


## Instalação

1.  Clone o repositório:

    ```bash
    git clone https://github.com/MarcosSouza257/IEEE_ID9184.git
    ```

2.  Navegue até o diretório do projeto:

    ```bash
    cd IEEE_ID9184
    ```
3. Crie um ambiente virtual (opcional, mas recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```


4.  Instale as dependências:

    ```bash
    pip install -r requirements.txt


## Como Usar 






