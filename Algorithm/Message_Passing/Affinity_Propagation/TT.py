import tensorflow as tf
import numpy as np
import time
import os as os
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# GPU 가속 확인
from tensorflow.python.client import device_lib

# 기존 데이터에서 Train & Test data 따로 때어놓기
from sklearn.model_selection import train_test_split

# Message Passing 함수 모아놓은 script
from yyyeeesss import (update_alpha, update_rho,
                              conclude_update,
                              conclude_update_nn,
                              update_similarity)

# print(device_lib.list_local_devices([0]))

# 환경 변수 설정
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

#소숫점 반올림 (5까지)
np.set_printoptions(precision=5)

#--------------------- 초기 변수 할당 --------------------- 

N_NODE = 8  # number of nodes per group
N_ITER = N_NODE*20 # MP iteration
N_DATASET = 10 # number of input sets
INF = 10**10
bLogSumExp = False # if dataset exists
filenames = {
    "w": f"{N_NODE}-by-{N_NODE} - w.csv",
    "w_n": f"{N_NODE}-by-{N_NODE} - w_n.csv",
    "alpha_star": f"{N_NODE}-by-{N_NODE} - alpha_star.csv",
    "rho_star": f"{N_NODE}-by-{N_NODE} - rho_star.csv",
    "mp_result": f"{N_NODE}-by-{N_NODE} - mp_result.csv",
}
FILENAME_NN_WEIGHT = "weights_unsupervised.h5" # saved NN weights
SEED_W = 0

def main():
    (w, w_n, alpha_star, rho_star, mp_res) = fetch_dataset()
    dataset_w = w # for training
    dataset_w_n = w_n # for calculating sum rate
    dataset_alpha_rho_star = np.concatenate((alpha_star,
                                             rho_star),
                                            axis=1)
    dataset_mp_res = mp_res
    
    # print(len(dataset_w)) # 얘는 5
    # print(len(dataset_alpha_rho_star)) # 얘는 50000으로 다름

    n_samples_to_use = N_DATASET
    
    (w_train, w_test, w_n_train, w_n_test,
     alpha_rho_star_train,
     alpha_rho_star_test, mp_res_train, mp_res_test) = train_test_split(
        dataset_w[:n_samples_to_use, :],
        dataset_w_n[:n_samples_to_use, :],
        dataset_alpha_rho_star[:n_samples_to_use, :],
        dataset_mp_res[:n_samples_to_use, :],
        test_size=0.2,
        shuffle=True)
    
    # --------------------- Model Training ---------------------  
     
    try: # if NN weights already exists (여기는 안건들여도 됨)
        model = initialize_model()
        model.load_weights(FILENAME_NN_WEIGHT)
        print("Trained NN weights loaded.")
        print(model.summary())
        
    except: # generating new NN weights
        model = initialize_model()
        print("Training NN.")
        
        # Option 1 
        # learning rate가 너무 작은 것 같음 (시간 소요가 너무 많음)
        # 이훈 교수님의 code에서는 learning_rate = 0.001임
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # adam optimizer
        
        # epoch
        n_epochs = 2
        for epoch in range(n_epochs):
            print(f"Starting epoch {epoch}...")
            for step, w_sample in enumerate(w_train):
               
                # Option 2
                # persistent = True
                with tf.GradientTape( ) as tape:
                    w_in_tmp = -construct_nn_input(w_sample)
                    w_in = np.sqrt(w_in_tmp)
                    alpha_rho = model(w_in, training=True) # NN output
                    alpha_rho_passed = forward_pass(w_sample, alpha_rho) # a single iter of the output
                    
                    # loss 계산
                    loss_value = tf.keras.losses.mean_squared_error(alpha_rho,
                                                                     alpha_rho_passed) # loss: MSE
                    
                # gradient 저장    
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                log_interval = 100
                if step % log_interval == 0:
                    print(f"Epoch {epoch}, step {step}: loss={loss_value}")
                    
        model.save_weights(FILENAME_NN_WEIGHT) # saving NN weights

    run_test_matching(w_test, w_n_test, mp_res_test, model) # validation


def fetch_dataset(): # checking if a dataset available
    bAvailable = check_dataset_availability()
    if not bAvailable: # generating a new dataset
        (w, w_n, alpha_star, rho_star, mp_result) = generate_and_write_dataset()
        print("Dataset generated.")
    else: # reading a existing dataset
        (w, w_n, alpha_star, rho_star, mp_result) = read_dataset()
        print("Dataset loaded.")
    return w, w_n, alpha_star, rho_star, mp_result


def check_dataset_availability(): # checking if a dataset available
    bAvailable = []
    for entry in filenames:
        filename = filenames[entry]
        bAvailable.append(os.path.exists(filename))
    return all(bAvailable)


def generate_and_write_dataset(): # generating and saving a dataset
    w, w_n = generate_dataset_input_awgn() # generating a dataset

    np.savetxt(filenames['w'], w, delimiter=',') # saving a dataset
    np.savetxt(filenames['w_n'], w_n, delimiter=',')
    alpha_star, rho_star, mp_result = generate_dataset_output(w) # MP iteration of w
    np.savetxt(filenames['alpha_star'], alpha_star, delimiter=',')
    np.savetxt(filenames['rho_star'], rho_star, delimiter=',')
    np.savetxt(filenames['mp_result'], mp_result, delimiter=',')
    return w, w_n, alpha_star, rho_star, mp_result


def generate_dataset_input_awgn(): # generating a locational dataset
    dist = np.zeros((N_DATASET, N_NODE ** 2))
    capacity = np.zeros((N_DATASET, N_NODE ** 2))
    
    for k in range(N_DATASET):
        points = np.random.uniform(1,100,(N_NODE,2))
          
        w = update_similarity(points)
        w_capa = w + 1199*np.eye(N_NODE)
        #sqrt_w = np.sqrt(w)
        dist[k] = reshape_to_flat(w)
        pos_dist = np.sqrt(-w_capa)
        pathloss = 1. / np.power(pos_dist, 3)
        capa_tmp = 0.5*np.log2(1 + (10**7)*pathloss / 4 / np.pi)
        capacity[k] = reshape_to_flat(capa_tmp)
    
    return dist, capacity


def generate_dataset_output(w): # a single MP iteration
    
    # alpha_star = np.zeros(np.shape(w))
    # rho_star = np.zeros(np.shape(w))
     
    alpha_star = np.zeros(np.shape(w))
    rho_star = np.zeros(np.shape(w))
    mp_result = np.zeros(np.shape(w))
   
    for i in range(N_DATASET):
        #print('mp_iter:',i)
        w_now = reshape_to_square(w[i])
        alpha_tmp = np.zeros(np.shape(w_now))
        rho_tmp = np.zeros(np.shape(w_now))
        
        arbi = reshape_to_square(w[i])
        
        mp_result_tmp = np.zeros(np.shape(w_now))
        #alpha_tmp = np.zeros((N_NODE,N_NODE))
        #rho_tmp = np.zeros((N_NODE,N_NODE))
        
        # exemplars = -1*np.ones(N_NODE)
        # exemplar1s = INF*np.ones(N_NODE)

        for iter in range(N_ITER):
            rho_tmp = update_rho(
                alpha_tmp, w_now, rho_tmp)
            alpha_tmp = update_alpha(
                alpha_tmp, rho_tmp)
            
        
        # arbi = rho_tmp + alpha_tmp
    
        
        #     for i in range(N_NODE):
        #         cur_exemplar, cur_value = i, -INF
        #         for j in range(N_NODE):
        #             if alpha_tmp[i,j] + rho_tmp[i,j] > cur_value:
        #                 cur_exemplar = j
        #                 cur_value = alpha_tmp[i][j] + rho_tmp[i][j]
        #         exemplars[i] = cur_exemplar
            
        #     for i in range(N_NODE):
        #         if exemplars[i] != exemplar1s[i]:
        #             break

        #     for i in range(N_NODE):
        #         exemplar1s[i] = exemplars[i]
        
        # for i in range(len(exemplars)):
        #     e = np.int64(exemplars[i])
        #     mp_result_tmp[i, e] = 1
        # print(mp_result_tmp)
        
        mp_result_tmp = conclude_update(alpha_tmp, rho_tmp)
        
        alpha_star[i] = reshape_to_flat(alpha_tmp)
        rho_star[i] = reshape_to_flat(rho_tmp)
        mp_result[i] = reshape_to_flat(mp_result_tmp)
        
    return alpha_star, rho_star, mp_result


def read_dataset(): # reading a existing dataset
    w = np.loadtxt(filenames['w'], dtype=float, delimiter=',')
    w_n = np.loadtxt(filenames['w_n'], dtype=float, delimiter=',')
    alpha_star = np.loadtxt(filenames['alpha_star'], dtype=float, delimiter=',')
    rho_star = np.loadtxt(filenames['rho_star'], dtype=float, delimiter=',')
    mp_result = np.loadtxt(filenames['mp_result'], dtype=int, delimiter=',')
    return w, w_n, alpha_star, rho_star, mp_result


def reshape_to_square(flat_array): # 1*N^2 to N*N
    try:
        return np.reshape(flat_array, (N_NODE, N_NODE))
    except Exception as e:
        print(f"ERROR: array reshaping failed: {e}")


def reshape_to_flat(square_array): # N*N to 1*N^2
    try:
        return np.reshape(square_array, N_NODE ** 2)
    except Exception as e:
        print(f"ERROR: array reshaping failed: {e}")


def decompose_dataset(arr, mode): # separating concatenated alpha&rho
    if mode == 'input':
        w, alpha, rho = np.array_split(arr, 3)
        for _ in [w, alpha, rho]:
            w = reshape_to_square(w)
            alpha = reshape_to_square(alpha)
            rho = reshape_to_square(rho)
        return w, alpha, rho
    elif mode == 'output':
        alpha, rho = np.array_split(arr, 2)
        for _ in [alpha, rho]:
            alpha = reshape_to_square(alpha)
            rho = reshape_to_square(rho)
        return alpha, rho
    else:
        pass


def print_dataset_dimensions(arr_list):
    for arr in arr_list:
        print(f"Shape: {np.shape(arr)}")


def initialize_model(): # NN model
    
    # Option 3
    # 이훈 교수님의 코드에서는 layer가 10개, n_hidden이 1000개
    # Batch nomarilization을 사용해보기
    # relu 말고 leakyrelu를 사용해보기
    # leakyrelu말고 selu를 사용해보기
    #model = Sequential()
    
    #inputs = tf.keras.layers.Input(shape=(N_NODE ** 2,))
    
    #model.add(Dense(500,activation = 'relu', input_shape = inputs))
    #model.add(Dense(250,activation = 'relu'))
    #model.add(Dense(120,activation = 'relu'))
    #model.add(Dense(50,activation = 'relu'))
    
    inputs = tf.keras.layers.Input(shape=(N_NODE ** 2,))
    x = tf.keras.layers.Dense(800, activation="relu")(inputs)
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    outputs = tf.keras.layers.Dense(128, name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def forward_pass(w, alpha_rho): # a single MP step
    alpha, rho = decompose_dataset(alpha_rho[0], 'output')
    alpha_next = update_alpha(alpha, rho)
    rho_next = update_rho(alpha, rho, reshape_to_square(w))

    alpha_next = reshape_to_flat(alpha_next)
    rho_next = reshape_to_flat(rho_next)
    alpha_rho_passed = np.concatenate((alpha_next, rho_next))
    return alpha_rho_passed


def construct_nn_input(w): # a proper format for NN input
    w = np.array([w])
    return w


def run_test_matching(w, w_n, mp_res, model): # testing with hungarian&MP&NN
    n_samples = np.size(w, axis=0)

    D_mp_sum_rate = get_D_mp(
        w_n, n_samples, mp_res) # MP result
    mp_mean_sumrate = np.mean(D_mp_sum_rate)
    print(f"MP sum-rate: {mp_mean_sumrate}")
 
    D_nn, D_nn_validity, D_nn_sum_rate \
        = get_D_nn(w, w_n, n_samples, mp_res, model) # NN result
        
        
    ssss = reshape_to_square(D_nn[1])
    print(ssss) 
    
    nn_mean_sumrate = np.mean(D_nn_sum_rate)
    print(f'NN sum-rate: {nn_mean_sumrate}')
    print(f"NN sum-rate rate: {nn_mean_sumrate/mp_mean_sumrate*100}%")
    print_assessment(D_nn_validity, n_samples)

    plt.style.use('seaborn-deep') # plotting a sum rate graph
    bins = np.linspace(0, 35, 36)
    plt.hist([D_mp_sum_rate, D_nn_sum_rate],
             bins, label=["Ising", "UL"])
    plt.legend()
    plt.xlim(0, 35)
    plt.title("Test set evaluations")
    plt.xlabel("sum-rate [bps]")
    plt.ylabel("number of samples")
    plt.savefig("tmp.png")


def get_D_mp(w_n, n_samples, mp_res): # getting MP result
    D_mp_sum_rate = np.array([])
    for i in range(n_samples):
        D_mp_sum_rate = np.append(D_mp_sum_rate, np.sum(np.multiply(mp_res[i], w_n[i])))

    return D_mp_sum_rate


def get_D_nn(w, w_n, n_samples, mp_res, model): # getting NN result
    D_nn = np.zeros((n_samples, N_NODE ** 2), dtype=int)
    D_nn_validity = np.zeros(n_samples, dtype=bool)
    D_nn_sum_rate = np.array([])

    for i in range(n_samples):
        w_sample = construct_nn_input(w[i])
        alpha_rho = model(w_sample).numpy()[0]
        alpha_sample, rho_sample = decompose_dataset(alpha_rho, 'output')
        D_pred = conclude_update_nn(alpha_sample, rho_sample)
        D_nn[i] = reshape_to_flat(D_pred)
        D_nn_validity[i] = check_validity_precisely(mp_res[i], D_nn[i])
        D_nn_sum_rate = np.append(D_nn_sum_rate, np.sum(np.multiply(D_nn[i], w_n[i])))
        if i % 10000 == 0:
            print(f"{i} valid:{D_nn_validity[i]}")

    return D_nn, D_nn_validity, D_nn_sum_rate


def check_validity_precisely(D_mp_single, D_nn_single): # validity b/w MP&NN
    x = (D_mp_single == D_nn_single).all()
    return x


def print_assessment(D_validity, n_samples): # validation of exact dataset matching
    nValid = np.count_nonzero(D_validity)
    print(f"{nValid} out of {n_samples}",
          f"({nValid / n_samples * 100} %)")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print(f"Runtime: {toc - tic}sec.")  
