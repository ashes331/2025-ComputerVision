import numpy as np
import tensorflow as tf
import keras
from keras import layers 
import keras.ops as ops
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. MNIST 데이터 불러오기 및 전처리
# ---------------------------------------------------------
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# 실수형 변환 및 정규화
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# CNN 입력 형태(28x28x1)로 변환 
x_train = np.reshape(x_train, (-1,28,28,1))
x_test = np.reshape(x_test, (-1,28,28,1))


# ---------------------------------------------------------
# 2. Sampling Layer 구현
#   VAE에서 z = μ + σ * ε 를 계산하는 부분
# ---------------------------------------------------------
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        
        # ε ~ N(0, 1)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        
        # z = μ + exp(σ/2) * ε
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ---------------------------------------------------------
# 3. Encoder 모델
#   입력 이미지 -> 잠복공간(latent vector) z로 압축
# ---------------------------------------------------------
latent_dim = 32  # 잠복 공간 차원

encoder_inputs = keras.Input(shape=(28, 28, 1))

# CNN 기반 특징 추출
x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)

# 잠복 공간의 평균과 분산 계산
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# 재매개변수화 (reparameterization trick)
z = Sampling()([z_mean, z_log_var])

# Encoder 모델 정의
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


# ---------------------------------------------------------
# 4. Decoder 모델
#   잠복공간 -> 이미지 복원
# ---------------------------------------------------------
latent_inputs = keras.Input(shape=(latent_dim,))

x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)

# 출력은 원본 이미지와 동일한 28x28x1
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

# Decoder 모델 정의
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")


# ---------------------------------------------------------
# 5. VAE 모델 정의 (Custom training loop 사용)
#   손실 = 재구성 존실 + KL 발산
# ---------------------------------------------------------
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    # 학습 단계 정의
    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            
            # 재구성 손실 (Binary Cross Entropy)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(x, reconstruction),
                    axis=(1, 2)
                )
            )
            
            # KL 발산 
            kl_loss = -0.5 * ops.mean(
                ops.sum(
                    1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                    axis=1
                )
            )
            
            # 전체 손실 
            loss = reconstruction_loss + kl_loss
            
            # 가중치 업데이트
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            return {"loss": loss}
            
        
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
        

# ---------------------------------------------------------
# 6. 모델 학습
# ---------------------------------------------------------
vae.fit(x_train, epochs=20, batch_size=128)


# ---------------------------------------------------------
# 7. 잠복 공간 보간 (latent interpolation)
#   두 이미지의 z 벡터를 선형 보간하여 생성을 비교
# ---------------------------------------------------------
i = np.random.randint(x_test.shape[0])
j = np.random.randint(x_test.shape[0])

# 두 개 테스트 이미지 선택
x = np.array((x_test[i], x_test[j]))

# 두 이미지의 z 벡터 추출
z_values = encoder.predict(x)[2]
z1, z2 = z_values[0], z_values[1]

# 0~1 사이를 11단계로 나누어 본다
alpha = np.linspace(0, 1, 11)
zz = np.array([(1 - a) * z1 + a * z2 for a in alpha])

# 보간된 z를 이용하여 이미지 생성
gen = decoder.predict(zz)


# ---------------------------------------------------------
# 8. 보간 결과 시각화
# ---------------------------------------------------------
plt.figure(figsize=(20, 4))
for i in range(11):
    plt.subplot(1, 11, i+1)
    plt.imshow(gen[i].reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{alpha[i]:.1f}")
plt.show()


