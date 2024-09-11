from models import ImageModel, FilterModel
from tqdm import tqdm
import numpy as np

import numpy as np

def apply_convolution(image, dimensoes, offset, matriz):
    height, width, _ = image.shape
    m, n = dimensoes
    pad_h, pad_w = m // 2, n // 2
    
    # Expansão por zeros
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    output_image = np.zeros_like(image)
    
    # Aplicação da convolução sobre cada canal R, G, B
    for i in tqdm(range(pad_h, height + pad_h), desc="Aplicando convolução"):
        for j in range(pad_w, width + pad_w):
            for c in range(3):  # 3 canais (R, G, B)
                region = padded_image[i - pad_h: i + pad_h + 1, j - pad_w: j + pad_w + 1, c]
                value = np.sum(region * matriz) + offset
                output_image[i - pad_h, j - pad_w, c] = min(max(int(value), 0), 255)
    
    return output_image

def aplicar_correlacao(imagem, dimensoes, offset, matriz):
    print("Shape da imagem:", imagem.shape)
    altura, largura = imagem.shape[:2]
    
    m, n = dimensoes
    
    # Cria uma nova imagem inicializada como um array de zeros
    # Esta imagem será do mesmo tamanho e tipo que a imagem original
    nova_imagem = np.zeros_like(imagem)

    
    # Itera sobre cada pixel da imagem
    for y in tqdm(range(altura), desc="Aplicando correlação"):
        for x in range(largura):
            # Inicializa a soma dos valores RGB para o novo pixel
            soma_r = soma_g = soma_b = 0
            
            # Itera sobre o filtro/máscara
            for dy in range(-m//2, m//2 + 1):
                for dx in range(-n//2, n//2 + 1):
                    # Calcula as coordenadas do pixel atual na imagem para aplicação do filtro
                    x_filtro, y_filtro = x + dx, y + dy
                    
                    # Verifica se as coordenadas estão dentro dos limites da imagem
                    if 0 <= x_filtro < largura and 0 <= y_filtro < altura:
                        # Obtém o pixel atual na imagem
                        pixel = imagem[y_filtro, x_filtro]
                        # Obtém o coeficiente correspondente do filtro para o pixel atual
                        coeficiente_filtro = matriz[dy + m//2][dx + n//2]
                        
                        # Aplica o filtro e soma os resultados para cada canal de cor
                        soma_r += pixel[0] * coeficiente_filtro
                        soma_g += pixel[1] * coeficiente_filtro
                        soma_b += pixel[2] * coeficiente_filtro
            
            # Após aplicar o filtro em todos os pixels sob o kernel do filtro,
            # ajusta os valores acumulados com base no offset e se certifica de que 
            # permanecem dentro dos limites válidos de um byte (0 a 255)
            nova_imagem[y, x] = [min(255, max(0, soma_r + offset)), 
                                 min(255, max(0, soma_g + offset)), 
                                 min(255, max(0, soma_b + offset))]

    # Retorna a imagem resultante após a aplicação do filtro
    return nova_imagem

def filtro_sobel(imagem, arq):
    filtered_image = apply_convolution(imagem, arq.dimensoes, arq.offset, arq.matriz)

    filtered_image = np.abs(filtered_image)

    filtered_image = (filtered_image / filtered_image.max()) * 255

    return filtered_image

def filtro_gaussiano(imagem, arq):
    matriz = [[elemento / 273 for elemento in linha] for linha in arq.matriz]
    return apply_convolution(imagem, arq.dimensoes, arq.offset, matriz)

def main():
    arq_gauss  =  FilterModel("assets/filtro_gaussiano.txt")
    arq_sobel_vert = FilterModel("assets/filtro_sobel_vertical.txt")
    arq_sobel_hor = FilterModel("assets/filtro_sobel_horizontal.txt")
    print("arq_gauss:", arq_gauss)
    print("arq_gauss:", type(arq_gauss.matriz))
    print("arq_sobel_vert:", arq_sobel_vert)
    print("arq_sobel_hor:", arq_sobel_hor)

    image = ImageModel("assets/testpat.1k.color2.tif")
    image_gauss = filtro_gaussiano(image.array_img, arq_gauss)
    image.atualizar_imagem(image_gauss)
    image.salvar_imagem("outputs/testpat.tif")

    shapes_vert = ImageModel("assets/Shapes.png")
    img_sobel_vert = filtro_sobel(shapes_vert.array_img, arq_sobel_vert)
    shapes_vert.atualizar_imagem(img_sobel_vert)
    shapes_vert.salvar_imagem("outputs/shapes_sobel_vert.png")

    shapes_hor = ImageModel("assets/Shapes.png")
    img_sobel_hor = filtro_sobel(shapes_hor.array_img, arq_sobel_hor)
    shapes_hor.atualizar_imagem(img_sobel_hor)
    shapes_hor.salvar_imagem("outputs/shapes_sobel_hor.png")

if __name__ == "__main__":
    main()
