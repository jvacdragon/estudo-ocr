import cv2
import pytesseract
import numpy as np
from imutils .object_detection import non_max_suppression #descarta boxes que nao tem qualidade esperada

config_pytesseract = '--oem 3 --psm 6'
image_path = '02.EAST/src/natural_image.png'

image = cv2.imread(image_path)
cv2.imshow('original', image)

string = pytesseract.image_to_string(image, lang='eng', config=config_pytesseract)
print(string)

detector = "02.EAST/src/frozen_east_text_detection.pb"
largura, altura = 320, 320 #detector EAST só funciona com multiplos de 32
min_confianca = 0.9 #nivel minimo de confiança para q o EAST considere o bloco de texto identificado

print(image.shape)

heigth = image.shape[0]
width = image.shape[1]

proporcao_altura = heigth/float(altura)
proporcao_largura = width/float(largura)

print(proporcao_altura)
print(proporcao_largura)

image_resized = cv2.resize(image, (largura,altura))
print(image.shape)

cv2.imshow("imagem_redimensionada",image_resized)

#definindo camadas da rede neural para o EAST
nomes_camadas = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'] #saidas da arquitetura do detector EAST. Sigmoid é o nome de de ativação da função que vai ajudar a obter as probabilidades de uma região ter ou não texto. Concat 3 é usado para obter o mapa de caracteristicas do objeto, como as coordenadas do bounding box onde é localizado o texto

rede_neural = cv2.dnn.readNet(detector)

#transformar imagem para formato de blob. Primeio parametro é a imagem a ser transformada e segundo parametro é a escala de redimensionamento. Nesse caso é 1.0 porque nao quero redimensionar novamente. Terceiro parametro é a chave com largura e altura da imagem. Quarto parametro é o swapRB como true, indicando a troca de BGR para RGB. Isso acontece porque o opencv trabalha com BGR e para utilizar a estrutura que vai ser usada é necessário RGB. O quinto parametro indica se a imagem será ou nao cortada para caber no tamanho definido
blob = cv2.dnn.blobFromImage(image_resized, 1.0, (largura, altura), swapRB=True, crop=False)

print(blob.shape) #retorna uma estrutura com 4 valores. O primerio é a quantidade de imagens comprimidas em blob, o segundo é a quantidade de canais de cores dos pixels da imagem, o terceiro é a largura da imagem e o quarto é a altura da imagem

rede_neural.setInput(blob)
#scores seriam os niveis de confianca de cada bounding box possivel e geometry são as localizações do texto na imagem. O forward indica que o input usado irá passar por todas as camadas ,utilizadas no parametro, da rede neural até chegar na camada de saida que gera o nivel de confiança e a localizaçao em geometry
scores, geometry = rede_neural.forward(nomes_camadas)

""" print(scores) """
""" print(geometry[0, 0, 0]) """

""" print(scores.shape) """

linhas, colunas = scores.shape[2:4] #pegando a quantidade de linhas e colunas retornadas em scores
caixas = []
confiancas = []



#decodificando valores
def geometry_data(geometry, y):
    xData0 = geometry[0,0,y] #distancia do topo da box até o centro
    xData1 = geometry[0,1,y] #distancia da direita da box até o centro
    xData2 = geometry[0,2,y] #distancia da base da box até o centro
    xData3 = geometry[0,3,y] #distancia da esquerda da box até o centro
    data_angles = geometry[0,4,y] #angulo de inclinação da box
    
    return data_angles, xData0, xData1, xData2, xData3 #retrona o array de cada de cada um desses dados para todas as boxes registradas

def calc_geo(data_angles, xData0, xData1, xData2, xData3):
    (offsetx, offsety) = (x*4, y*4)
    angle = data_angles[x]
    cos = np.cos(angle)
    sin = np.sin(angle)
    h = xData0[x] + xData2[x]
    w = xData1[x] + xData3[x]
    
    endX = int(offsetx + (cos * xData1[x]) + (sin * xData2[x]))
    endY = int(offsety - (sin * xData1[x]) + (cos * xData2[x]))
    
    initX = int(endX - w)
    initY = int(endY - h)
    
    return initX, initY, endX, endY
    
for y in range(0,linhas):
    data_scores = scores[0,0,y] #conseguindo cada valor de confiança por linha do score
    
    data_angles, xData0, xData1,xData2,xData3 = geometry_data(geometry, y)
    
    for x in range(0, colunas):
        if(data_scores[x] >= min_confianca): 
            initX, initY, endX, endY = calc_geo(data_angles, xData0, xData1,xData2,xData3)
            confiancas.append(data_scores[x])
            caixas.append((initX, initY, endX, endY))
            
print(confiancas)
print(caixas)

deteccoes = non_max_suppression(np.array(caixas), probs=confiancas) #selecionando as coordenadas com maiores niveis de confiança. Nesse caso seleciona só uma, pois todas são mmuito parecidas, criando sobreposições, entao é escolhida apenas uma coordenada com o maior nivel de confianca dentre as que tem
print(deteccoes)

#mostrando as bounding boxes na imagem
#mantendo proporção para a imagem original
for (initX, initY, endX, endY) in deteccoes:
    initX = int(initX*proporcao_largura)
    initY = int(initY*proporcao_altura)
    endY = int(endY*proporcao_altura)
    endX = int(endX*proporcao_largura)

    #roi = region of interest
    roi = image[initY:endY, initX:endX]
    #criando retagulo na imagem. Parametros: imagem, pontos de inicio do eixo x e y, pontos de fim do eixo x e y, cor do retangulo em BGR, grossura da linha
    cv2.rectangle(image, (initX, initY), (endX,endY), (0,255,0), 2)
    
cv2.imshow("teste", image)
cv2.imshow("region of interest roi", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()