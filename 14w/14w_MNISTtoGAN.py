import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Fashion-MNIST 데이터 불러오기
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 클래스 8 (Bag)만 선택
x_train = x_train[np.isin(y_train, [8])]

# [-1, 1] 구간으로 정규화 (tanh 출력과 맞추기 위함)
x_train = (x_train.astype("float32") / 255.0) * 2.0 - 1.0

# GAN 입력 형태를 위해 채널 차원 추가
x_train = np.expand_dims(x_train, axis=-1)

# 잠복 공간 차원 정의
zdim = 100

# ---------------------------------------------------------
# 2. 분멸망(Discriminator) 정의
# ---------------------------------------------------------
def make_discriminator(input_shape):
    # 진짜와 가짜 이미지를 구분하는 이진 분류기
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=["accuracy"]
        )
    
    return model


# ---------------------------------------------------------
# 3. 생성망(Generator) 정의
# ---------------------------------------------------------
def make_generator(zdim):
    # 잠복 벡터 z를 입력받아 28x28x1 이미지를 생성하는 모델
    model = Sequential()

    model.add(Dense(7 * 7 * 64, input_dim=zdim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 64)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(LeakyReLU(0.2))

    # 14x14 업샘플링
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))
    
    # 28x28 업샘플링
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))
    
    # 최종 출력 (tanh 사용)
    model.add(Conv2D(1, (3, 3), padding="same", activation="tanh"))

    return model

# ------------------------------------------------------------------
# 4. GAN 모델(G + D) 연결
# ------------------------------------------------------------------

def make_gan(G, D):
    # GAN 학습 시에는 D의 가중치를 업데이트하지 않음
    D.trainable = False

    model = Sequential()
    model.add(G)
    model.add(D)

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
    )
    return model

# ------------------------------------------------------------------
# 5. 진짜·가짜 샘플 생성 함수
# ------------------------------------------------------------------

def generate_real_samples(dataset, n_samples):
    # 진짜 이미지 선택
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[idx]
    y = np.ones((n_samples, 1))   # 진짜는 1
    return x, y


def generate_latent_points(zdim, n_samples):
    # 잠복 공간에서 랜덤 노이즈 샘플링
    return np.random.randn(n_samples, zdim)


def generate_fake_samples(G, zdim, n_samples):
    # G(z)를 통해 가짜 이미지 생성
    z = generate_latent_points(zdim, n_samples)
    x = G.predict(z, verbose=0)
    y = np.zeros((n_samples, 1))   # 가짜는 0
    return x, y


# ------------------------------------------------------------------
# 6. GAN 학습 루프
# ------------------------------------------------------------------
def train(G, D, GAN, dataset, zdim, n_epochs=200, batch_size=128, verbose=1):
    n_batch = dataset.shape[0] // batch_size

    for epoch in range(n_epochs):
        for _ in range(n_batch):

            # 분별망 학습(진짜 이미지)
            x_real, y_real = generate_real_samples(dataset, batch_size // 2)
            d_loss_real, _ = D.train_on_batch(x_real, y_real)

            # 분별망 학습(가짜 이미지)
            x_fake, y_fake = generate_fake_samples(G, zdim, batch_size // 2)
            d_loss_fake, _ = D.train_on_batch(x_fake, y_fake)

            # 생성망 학습(D를 속이도록 학습)
            z = generate_latent_points(zdim, batch_size)
            y_gan = np.ones((batch_size, 1))   # 생성망은 가짜를 진짜처럼 보이게 학습
            g_loss = GAN.train_on_batch(z, y_gan)

        if verbose:
            print(f"Epoch {epoch+1}: D(real)={d_loss_real:.3f}, D(fake)={d_loss_fake:.3f}, G={g_loss:.3f}")

        # 10 epoch마다 생성 결과 시각화
        if (epoch + 1) % 10 == 0:
            x_fake, y_fake = generate_fake_samples(G, zdim, 12)
            plt.figure(figsize=(20, 2))
            plt.suptitle(f"Epoch {epoch+1}")
            for i in range(12):
                plt.subplot(1, 12, i + 1)
                plt.imshow((x_fake[i] + 1) / 2.0, cmap="gray")
                plt.axis("off")
            plt.show()


# ------------------------------------------------------------------
# 7. 모델 생성 및 학습 실행
# ------------------------------------------------------------------
D = make_discriminator((28, 28, 1))
G = make_generator(zdim)
GAN = make_gan(G, D)

train(G, D, GAN, x_train, zdim, verbose=1)



