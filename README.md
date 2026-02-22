## Atividade LAB-01: Implementação do Scaled Dot-Product Attention

O trabalho implementar o mecanismo de Scaled Dot-Product Attention, apresentado no artigo "Attention Is All You Need" (base da arquitetura Transformer).

#### LINGUAGENS DE PROGRAMAÇÃO
Linguagem Utilizada: Python 3.12.3

#### INSTALAÇÂO NECESSARIA

Dependência necessária: **pip install numpy**

#### INSTRUÇÕES COMO RODA O SCRIPT:

No terminal: **python3 test_attention.py**

#### EXEMPLO DE INPUT UTILIZADO:

Q = [[1,2,2],[1,1,2]]
K = [[2,5,6],[0,8,1]]
V = [[10,50],[20,60]]

#### EXEMPLO DE OUTPUT ESPERADO:

[[10.3035109  50.3035109 ]
 [10.05507332 50.05507332]]


 #### NORMALIZAÇÃO APLICADO ( raiz𝑑k ): 
 
 Antes dos Dados passarem para o softmax, precisar passar no escalonamento divide os produto escalar pela raiz quadrada da dimensão das chaves.
 
