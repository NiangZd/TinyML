import tensorflow as tf
import numpy as np
import os
from functools import reduce

assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

# ------------------------
# CONFIGURAÇÕES PRINCIPAIS
# ------------------------
DATA_DIR = '../custom_data'  # Pasta que contém subpastas: yes/, no/, _background_noise_/
LABELS = ['yes', 'no', '_background_noise_']  # Classes que nosso modelo vai reconhecer
SAMPLING_RATE = 16000  # Todos os áudios serão convertidos para 16 kHz
AUGMENT_TIMES = 10  # Quantas vezes vamos criar versões "aumentadas" de cada áudio

# ------------------------
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# ------------------------

def decode_audio(audio_binary):
    """
    Recebe um arquivo WAV em binário e transforma em um vetor de áudio.
    tf.audio.decode_wav retorna [samples, canais]; removemos a dimensão de canais.
    """
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    return audio

def pad_or_trim(waveform):
    """
    Ajusta a duração do áudio para 1 segundo (16.000 samples).
    Se for menor, adiciona zeros (padding). Se for maior, corta (trimming).
    """
    desired_length = SAMPLING_RATE
    waveform_length = tf.shape(waveform)[0]
    waveform = tf.cond(
        waveform_length < desired_length,
        lambda: tf.pad(waveform, [[0, desired_length - waveform_length]]),  # adiciona zeros no final
        lambda: waveform[:desired_length]  # corta o excesso
    )
    return waveform

def get_label(file_path):
    """
    Recebe o caminho de um arquivo e retorna a label.
    Por exemplo: '../custom_data/yes/audio1.wav' -> 'yes'
    """
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    """
    Combina as funções acima:
    - Lê o arquivo
    - Decodifica e ajusta duração
    - Retorna waveform + label
    """
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    waveform = pad_or_trim(waveform)
    label = get_label(file_path)
    return waveform, label

# ------------------------
# FUNÇÃO PARA MEL-SPECTROGRAMA
# ------------------------

def get_spectrogram(waveform):
    """
    Transforma o áudio (waveform) em Mel-Spectrograma.
    Um Mel-Spectrograma é uma imagem que representa o áudio no tempo e nas frequências,
    que é mais fácil para redes neurais processarem.
    """
    # Passo 1: calcula o STFT (Short-Time Fourier Transform)
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)  # magnitude do espectro

    # Passo 2: converte para escala Mel (mais próxima da percepção humana)
    num_mel_bins = 64
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], SAMPLING_RATE, 80.0, 7600.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate([num_mel_bins]))

    # Passo 3: aplica logaritmo para reduzir grande variação de valores
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Passo 4: adiciona dimensão de canal (como uma imagem: altura x largura x canal)
    mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)
    return mel_spectrogram

def get_spectrogram_and_label_id(audio, label):
    """
    Converte áudio em Mel-Spectrograma e label em índice numérico.
    Por exemplo, 'yes' -> 0, 'no' -> 1, '_background_noise_' -> 2
    """
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(tf.cast(tf.equal(label, LABELS), tf.int32))
    return spectrogram, label_id

# ------------------------
# DATA AUGMENTATION (AUMENTO DE DADOS)
# ------------------------
def augment_waveform(waveform):
    """
    Cria pequenas variações do áudio original para aumentar o dataset:
    - Adiciona ruído
    - Altera o volume
    - Desloca levemente no tempo
    """
    waveform = waveform + tf.random.normal(tf.shape(waveform), stddev=0.02)
    waveform = waveform * tf.random.uniform([], 0.8, 1.2)
    shift = tf.random.uniform([], -0.1, 0.1) * SAMPLING_RATE
    waveform = tf.roll(waveform, tf.cast(shift, tf.int32), axis=0)
    return waveform

# ------------------------
# FUNÇÃO PRINCIPAL: TREINAMENTO E CONVERSÃO PARA TFLITE
# ------------------------
def generate_custom_tflite_model():
    print("Iniciando geração do modelo TFLite personalizado...")

    # Carrega todos os arquivos de áudio por label
    files = {label: tf.io.gfile.glob(os.path.join(DATA_DIR, label, "*.wav")) for label in LABELS}
    for label, file_list in files.items():
        print(f"Encontrados {len(file_list)} arquivos para '{label}'")
    all_files = sum(files.values(), [])

    AUTOTUNE = tf.data.AUTOTUNE  # para otimização de leitura de dados
    file_ds = tf.data.Dataset.from_tensor_slices(all_files)
    waveform_ds = file_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    # Cria dataset aumentado
    augmented_datasets = []
    for i in range(AUGMENT_TIMES):
        augmented = waveform_ds.map(lambda w, l: (augment_waveform(w), l), num_parallel_calls=AUTOTUNE)
        augmented_datasets.append(augmented)

    # Combina dataset original + dataset aumentado
    if augmented_datasets:
        augmented_combined = reduce(lambda ds1, ds2: ds1.concatenate(ds2), augmented_datasets)
        final_ds = waveform_ds.concatenate(augmented_combined)
    else:
        final_ds = waveform_ds

    # Converte waveform em Mel-Spectrograma + label_id
    processed_ds = final_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    # Divisão treino/validação (80/20)
    dataset_size = len(all_files) * (AUGMENT_TIMES + 1)
    train_size = int(0.8 * dataset_size)
    shuffled_ds = processed_ds.shuffle(dataset_size, reshuffle_each_iteration=True)
    train_ds = shuffled_ds.take(train_size)
    val_ds = shuffled_ds.skip(train_size)

    train_ds = train_ds.batch(64).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.batch(64).cache().prefetch(AUTOTUNE)

    # Define o modelo CNN simples
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(124, 64, 1)),  # entrada: Mel-Spectrograma
        tf.keras.layers.Conv2D(32, 3, activation="relu"),  # convolução para extrair padrões
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),  # evita overfitting
        tf.keras.layers.Flatten(),  # transforma 2D em 1D para a Dense
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(LABELS), activation="softmax")  # saída: probabilidades das classes
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Callbacks: paramétricos para parar ou ajustar treino automaticamente
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    # Treina o modelo
    print("Treinando modelo...")
    model.fit(train_ds, epochs=50, callbacks=callbacks_list, validation_data=val_ds)

    # Converte o modelo para TFLite (formato leve para microcontroladores)
    print("Convertendo para TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # ativa quantização automática
    tflite_model = converter.convert()

    # Salva o modelo TFLite
    tflite_path = "custom_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"\nModelo TFLite salvo como {tflite_path}")
    print("Treinamento concluído!")

# ------------------------
# EXECUÇÃO
# ------------------------
if __name__ == '__main__':
    generate_custom_tflite_model()
