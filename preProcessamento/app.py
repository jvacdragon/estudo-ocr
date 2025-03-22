import cv2
import numpy as np

image = cv2.imread("preProcessamento/src/teste.jpg")

cv2.imshow('teste', image)

#imagem em escalas de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

#todo thresholding deve ser feito após transformação em escala de cinza

#val(valor do mimiar), thresh (imagem) - Imagem a ser modificada, limiar (pixels com valor menor que ele será transformado em valor 0), valor que pixels acima do parametro irão assumir, tipo de threshold
val, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold", thresh)

#metodo de otsu - cria um limiar de forma automática usando histogramas
val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("otsu", otsu)

#limiarização adaptiva pela média (boa para quando há grande variação de luz na imagem, como em fotos com sombra)
#parametros: imagem, valor maximo que deve ser assumido o limiar, tipo de limiarização adaptiva, tipo de threshold, pixels que serão comparados em volta para realizar a média, constante que irá subtrair (normalmente é 9)
adapt_media = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 9)
cv2.imshow("media_thresh", adapt_media)

livro = cv2.imread("preProcessamento/src/pagina.jpg")
cv2.imshow("livro", livro)

adapt_mean = cv2.adaptiveThreshold(cv2.cvtColor(livro, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
cv2.imshow("mean_adaptive", adapt_mean)

#limiarização gaussiana (utiliza a média tal como o adaptive mean e também usa desvio padrão)
adapt_gauss = cv2.adaptiveThreshold(cv2.cvtColor(livro, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
cv2.imshow("gauss_adaptive", adapt_gauss)

#invertendo cores da imagem (serve para transformar textos claros com fundo escuro em textos escuros com fundo claro)
print(gray) #valor das cores de cada pixel da imagem gray em formato de matriz
#agora se subtrai o valor máximo de pixel do valor dele para inverter a imagem
invert = 255 - gray
cv2.imshow("inverted", invert) #imagem mostrada no negativo
adapt_mean = cv2.adaptiveThreshold(invert, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,9)
cv2.imshow("negative_mean", adapt_mean)

adapt_gauss = cv2.adaptiveThreshold(invert, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 9)
cv2.imshow("negative_gauss", adapt_gauss)

#operacoes morfológicas - erosão (diminui a quantidade de pixels brancos do objeto) e dilatação (aumenta quantidade de pixels brancos do objeto)

cv2.waitKey(0)
cv2.destroyAllWindows()