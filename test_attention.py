import numpy as np

def Calcular_Atencao(Q,K,V):
    # Calcular aproximação de matrizes
    K_transpose = np.transpose(K)
    score =np.matmul(Q,K_transpose) # Multiplicação entre lista Q com K transpose
    
    # Fazendo o Escalonamento 
    raiz_dimensao = np.sqrt(K.shape[-1]) # Raiz quadrada da dimensão
    Escalonamento = score/raiz_dimensao # Valor escalonado
    
    # Normalização dos valores, transforma entre pesos 0 e 1
    softmax = np.exp(Escalonamento - np.max(Escalonamento, axis=-1, keepdims=True)) 
    pesos = softmax / np.sum(softmax, axis=-1, keepdims=True) 

    # Fazendo Ponderação
    resultado_ponderacao = np.matmul(pesos,V) # Multiplição dos pesos com o valor

    print(f'\nO Resultado Final Ponderado: {resultado_ponderacao}')
    

Q = np.array([[1, 2, 2],[1,1,2]], dtype=np.int8)
K = np.array([[2,5,6],[0,8,1]], dtype=np.int8)
V = np.array([[10,50],[20,60]], dtype=np.int8)

Calcular_Atencao(Q,K,V)