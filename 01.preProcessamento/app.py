import cv2
import numpy as np
import pytesseract

image = cv2.imread("01.preProcessamento/src/teste.jpg")

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

livro = cv2.imread("01.preProcessamento/src/pagina.jpg")
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
image = cv2.imread("01.preProcessamento/src/morfologica.png")
cv2.imshow("morfologica", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

""" image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 9)
cv2.imshow("image", image) """

#erosao - método erode para erosão, recebe uma matriz de 1s para fazer comparação entre os pixels e definir como será feita essa erosão de imagem. Ao redor de cada pixel é verificado com a matriz definida(no caso uma 3x3),  se um dos pixels nessa janela for identificado como preto, o pixel central se tornará preto também.
erosao = cv2.erode(gray, np.ones((5,5), np.uint8))
cv2.imshow('erosao', erosao)

#dilatacao - funciona de forma semelhante ao método de erosão, mas ao inves de tornar os pixels em preto, ele verifica se há pixels vizinhos brancos e transforma os centrais em brancos
dilatacao = cv2.dilate(gray, np.ones((3,3), np.uint8))
cv2.imshow("dilatacao", dilatacao)

#fazendo uma erosao para eliminar ruido e logo apos uma dilatação, é possível voltar ao estado anterios da imagem mas sem os ruídos existentes antes. Esse seria o método de abertura
erosaoDilatacao = cv2.dilate(erosao, np.ones((5,5), np.uint8))
cv2.imshow("erosaoDilatacao", erosaoDilatacao)

#blur para remoção de ruido
ruido = cv2.imread("01.preProcessamento/src/imageRuido.jpg")
cv2.imshow("original", ruido)

ruidoAmpliado = cv2.resize(ruido, None, fx=1.8, fy=1.8, interpolation=1)
cv2.imshow("ampliado", ruidoAmpliado)

gray = cv2.cvtColor(ruidoAmpliado, cv2.COLOR_BGR2GRAY)

mean_blur = cv2.blur(gray, (5,5))
cv2.imshow("mean_blur", mean_blur)

gaussBlur = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow("gaussBlur", gaussBlur)

#o segundo argumento é o tipo de matriz. Nesse caso é uma 3x3 usada
median_blur = cv2.medianBlur(gray, 3)
cv2.imshow("median_blur", median_blur)

bilateral_blur = cv2.bilateralFilter(ruidoAmpliado, 80,55,45)
cv2.imshow("bilateral_blur", bilateral_blur)

exercicio = cv2.imread("01.preProcessamento/src/exercicio.webp")
cv2.imshow("exercicio", exercicio)

exercicio = cv2.cvtColor(exercicio, cv2.COLOR_BGR2GRAY)
cv2.imshow("exercicio", exercicio)

val, thresh = cv2.threshold(exercicio, 180, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)

invert = 255 - thresh
cv2.imshow("invert", invert)


config_tesseract = '--oem 3 --psm 6'
text = pytesseract.image_to_string(invert, config=config_tesseract, lang="por")
print(text)


cv2.waitKey(0)
cv2.destroyAllWindows()