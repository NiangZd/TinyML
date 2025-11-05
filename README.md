# TinyML - Reconhecimento de Comandos de Voz com TensorFlow Lite

Este projeto demonstra o uso de TinyML aplicando Machine Learning em dispositivos com poucos recursos. Ele consiste em treinar um modelo de rede neural convolucional (CNN) para reconhecer comandos de voz ("yes" e "no") e ruído de fundo, e depois usar o modelo convertido em TensorFlow Lite (TFLite) para inferência em áudios individuais.

---

## Funcionalidades

* Treinamento de um modelo CNN para reconhecimento de voz (classes: "yes", "no", "background_noise").
* Aplicação de data augmentation para aumentar a robustez e generalização do modelo.
* Conversão do modelo para TensorFlow Lite, otimizado para inferência em dispositivos limitados.
* Inferência em tempo real de áudios individuais, retornando a classe mais provável e a confiança da predição.

---

## Estrutura do Projeto

.
├── backend/
│   ├── train_model.py      # Código para treinar o modelo e gerar custom_model.tflite
│   └── predict_audio.py    # Código para realizar inferência em arquivos de áudio
├── custom_data/            # Pasta com seus dados de áudio
│   ├── yes/
│   ├── no/
│   └── _background_noise_/
├── test_yes.wav            # Exemplo de áudio de teste (sim)
├── test_no.wav             # Exemplo de áudio de teste (não)
└── README.md

---

## Requisitos

* Python 3.8 ou superior
* TensorFlow 2.x
* Librosa
* NumPy
* SoundFile (sf)

### Instalação de Dependências

Instale as bibliotecas necessárias com o seguinte comando:

pip install tensorflow librosa numpy soundfile

---

## Como Treinar o Modelo

1. Organize seus Dados: Coloque seus arquivos de áudio no formato .wav e organize-os nas pastas correspondentes:
    * custom_data/yes/
    * custom_data/no/
    * custom_data/_background_noise_/

2. Execute o Script de Treinamento:

python backend/train_model.py

O script irá:
* Aplicar data augmentation aos dados de áudio.
* Transformar os áudios em Mel-Spectrogramas (representação visual da frequência).
* Treinar e validar a CNN.
* Salvar o modelo treinado e convertido como custom_model.tflite.

---

## Como Testar/Inferir Áudios

1. Coloque seus Áudios de Teste: Coloque os arquivos de áudio que deseja testar na pasta raiz do projeto (ex.: test_yes.wav, test_no.wav).

2. Execute o Script de Inferência:

python backend/predict_audio.py

O script processará cada áudio, transformando-o em Mel-Spectrograma. O modelo TFLite fará a predição e mostrará:

* Label mais provável
* Confiança da predição (Probabilidade da classe mais alta)
* Probabilidades de todas as classes

---

## Sobre o TinyML

Este projeto exemplifica o fluxo completo do TinyML:

1. Pré-processamento eficiente de dados de áudio.
2. Treinamento de um modelo (CNN) eficiente.
3. Conversão para o formato otimizado TensorFlow Lite.
4. Inferência em dispositivos leves, sem a necessidade de computadores potentes ou conexão com a internet, permitindo a Inteligência Artificial na ponta (Edge AI).
