import tensorflow as tf
import numpy as np
import soundfile as sf 
import os
import librosa

# --------------------------------------------------------
# CONFIGURAÇÕES DO SISTEMA
# --------------------------------------------------------

MODEL_PATH = "custom_model.tflite"
LABELS = ["yes", "no", "_background_noise_"]
SAMPLING_RATE = 16000
# Parâmetros para transformar o áudio em uma “imagem” sonora chamada Mel-Spectrograma
NUM_MEL_BINS = 64              # Quantos "blocos" de frequências queremos
LOWER_EDGE_HERTZ = 80.0        # Frequência mínima (grave)
UPPER_EDGE_HERTZ = 7600.0      # Frequência máxima (aguda)

# --------------------------------------------------------
# CARREGAMENTO DO MODELO TFLITE
# --------------------------------------------------------
try:
    # Cria o interpretador do modelo (essa é a “máquina” que entende o arquivo .tflite)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    
    # Prepara o modelo para uso (aloca espaço na memória)
    interpreter.allocate_tensors()

    # Pega informações sobre o formato esperado de entrada e saída do modelo
    input_details = interpreter.get_input_details()   # Tamanho e tipo da entrada (ex: imagem 124x64x1)
    output_details = interpreter.get_output_details() # Tamanho e tipo da saída (ex: 3 classes)

    print(f"Modelo carregado de {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar o modelo TFLite: {e}")
    print("Verifique se o arquivo 'custom_model.tflite' existe e foi gerado corretamente.")
    exit()

# --------------------------------------------------------
# PRÉ-PROCESSAMENTO DO ÁUDIO
# --------------------------------------------------------

def preprocess_audio(file_path):
    """
    Essa função pega um arquivo de áudio (.wav),
    ajusta ele para o formato que o modelo entende
    (16 kHz, 1 segundo de duração, e transforma em Mel-Spectrograma).
    """

    try:
        # Carrega o áudio e ajusta a taxa para 16 kHz
        data, _ = librosa.load(file_path, sr=SAMPLING_RATE)
    except Exception as e:
        print(f"Erro ao carregar ou resamplear '{file_path}': {e}")
        # Se der erro, retorna uma matriz “vazia” do tamanho que o modelo espera para evitar bugs
        return np.zeros((1, 124, NUM_MEL_BINS, 1), dtype=np.float32)

    # Ajusta o áudio para ter exatamente 1 segundo de duração
    # Se for menor, preenche com zeros. Se for maior, corta o excesso.
    data = np.pad(data, (0, max(0, SAMPLING_RATE - len(data))), 'constant')
    data = data[:SAMPLING_RATE]

    # Calcula o STFT (Transformada de Fourier no tempo)
    # Isso transforma o som em uma representação de frequências ao longo do tempo
    waveform = tf.convert_to_tensor(data, dtype=tf.float32)
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)  # pegamos apenas o módulo (intensidade)

    # Converte o espectrograma para a escala "Mel"
    # Essa escala é inspirada na forma como o ouvido humano percebe sons
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS,
        spectrogram.shape[-1],
        SAMPLING_RATE,
        LOWER_EDGE_HERTZ,
        UPPER_EDGE_HERTZ
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)  # aplica logaritmo (compressão)

    # Ajusta o formato final (como se fosse uma imagem)
    # O modelo espera algo como (1, altura, largura, canais)
    output_tensor = mel_spectrogram[..., tf.newaxis]  # adiciona canal (como uma imagem preto e branco)
    output_tensor = output_tensor[tf.newaxis, ...]    # adiciona o “batch” (como se fosse 1 amostra)

    # Retorna o resultado pronto para o modelo
    return output_tensor.numpy().astype(np.float32)

# --------------------------------------------------------
# FUNÇÃO DE PREDIÇÃO
# --------------------------------------------------------

def predict_audio(file_path):
    """
    Essa função faz a parte “mágica”:
    pega o arquivo de áudio, processa ele e pergunta ao modelo:
    “O que você acha que é isso?”
    """

    # Se o arquivo não existir, nem tenta
    if not os.path.exists(file_path):
        print(f"\nArquivo não encontrado: {file_path}")
        return

    print(f"\nTestando arquivo: {file_path}")

    # Passo 1: Pré-processar o áudio (transformar em Mel-Spectrograma)
    input_data = preprocess_audio(file_path)
    if input_data.size == 0:
        return

    try:
        # Passo 2: Alimentar o modelo com o dado processado
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Passo 3: Executar o modelo (fazer a predição)
        interpreter.invoke()

        # Passo 4: Pegar a resposta do modelo (as probabilidades)
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    except ValueError as ve:
        # Isso acontece se o formato do dado não combina com o modelo
        print(f"\n❌ ERRO DE DIMENSÃO: {ve}")
        print(f"Dimensão gerada: {input_data.shape}")
        print(f"Esperada pelo modelo: {input_details[0]['shape']}")
        return

    # --------------------------------------------------------
    # MOSTRA RESULTADOS DETALHADOS
    # --------------------------------------------------------

    print("-" * 30)
    for label, score in zip(LABELS, output_data):
        # Mostra quanto o modelo acha que é cada categoria
        print(f"{label:<20} → {score:.6f}")

    # Pega a classe mais provável (a de maior pontuação)
    label = LABELS[np.argmax(output_data)]
    confidence = float(np.max(output_data))

    print("-" * 30)
    print(f"Resultado final:")
    print(f"Palavra mais provável: {label}")
    print(f"Confiança: {confidence:.4f}")
    print("-" * 30)

    # Retorna o resultado como um dicionário, se quiser usar em outro lugar
    return {"label": label, "confidence": confidence}

# --------------------------------------------------------
# EXECUÇÃO DE TESTE
# --------------------------------------------------------
if __name__ == '__main__':
    # Lista de arquivos para testar (coloque seus próprios .wav aqui)
    TEST_FILES = ["test_yes.wav", "test_no.wav"]

    # Testa cada arquivo e mostra o resultado
    for f in TEST_FILES:
        predict_audio(f)

    print("\n✅ Teste de predição finalizado.")
