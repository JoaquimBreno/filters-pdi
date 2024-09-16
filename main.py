from models import ImageModel, FilterModel
from tqdm import tqdm
import numpy as np

import numpy as np


def apply_convolution(image, dimensoes, offset, mask):
    new_img = np.zeros_like(image)

    r_mask, c_mask = dimensoes 
    r_img, c_img = image.shape[:2]

    pivo = [int(np.ceil(r_mask / 2) - 1),
            int(np.ceil(c_mask / 2) - 1)]

    print('Pivo[i, j] =', pivo)

    channels = image.shape[2] if len(image.shape) == 3 else 1

    for i in tqdm(range(pivo[0], r_img - pivo[0]), desc="Aplicando convolução"):
        for j in range(pivo[1], c_img - pivo[1]):
            if channels == 1:
                region = image[i - pivo[0]:i + pivo[0] + 1, j - pivo[1]:j + pivo[1] + 1]
                new_value = np.sum(region * mask)
                new_img[i, j] = np.abs(new_value)
            else:
                for c in range(channels):
                    region = image[i - pivo[0]:i + pivo[0] + 1, j - pivo[1]:j + pivo[1] + 1, c]
                    new_value = np.sum(region * mask)
                    new_img[i, j, c] = np.abs(new_value)

    new_img = new_img.astype(np.int16) + offset
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0

    return new_img.astype(np.uint8)

def filtro_sobel(imagem, arq):
    filtered_image = apply_convolution(
        imagem, arq.dimensoes, arq.offset, arq.matriz)
    
    return filtered_image

def filtro_gaussiano(imagem, arq):
    matriz = [[elemento / 273 for elemento in linha] for linha in arq.matriz]
    return apply_convolution(imagem, arq.dimensoes, arq.offset, matriz)

def expansao_histograma(image):
    if len(image.shape) == 3:  # Caso de imagens coloridas
        resultado = np.zeros_like(image)
        for c in range(image.shape[2]):
            canal = image[:, :, c]
            min_val = np.min(canal)
            max_val = np.max(canal)
            resultado[:, :, c] = np.clip((canal - min_val) / (max_val - min_val) * 255, 0, 255)
    else:  # Caso de imagem em escala de cinza
        min_val = np.min(image)
        max_val = np.max(image)
        resultado = np.clip((image - min_val) / (max_val - min_val) * 255, 0, 255)
    return resultado.astype(np.uint8)

# Função para o filtro pontual triangular baseado no gráfico
def triangular_filter(pixel_value):
    if pixel_value < 128:
        return np.clip((pixel_value / 128) * 255, 0, 255)
    else:
        return np.clip(((255 - pixel_value) / 128) * 255, 0, 255)

# Aplicação do filtro em uma imagem RGB
def apply_triangular_filter(image_array):
    filtered_image = np.zeros_like(image_array)

    for i in tqdm(range(image_array.shape[0]), desc="Applying triangular filter"):
        for j in range(image_array.shape[1]):
            for k in range(3):  # For R, G, B channels
                filtered_image[i, j, k] = triangular_filter(
                    image_array[i, j, k])

    return filtered_image

# Função para converter RGB para YIQ
def rgb_to_yiq(rgb_array):
    yiq_array = np.zeros_like(rgb_array, dtype=float)
    for i in tqdm(range(rgb_array.shape[0]), desc="Converting RGB to YIQ"):
        for j in range(rgb_array.shape[1]):
            r, g, b = rgb_array[i, j]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            i_value = 0.596 * r - 0.275 * g - 0.321 * b
            q = 0.212 * r - 0.523 * g + 0.311 * b
            yiq_array[i, j] = [y, i_value, q]
    return yiq_array

# Função para converter YIQ para RGB
def yiq_to_rgb(yiq_array):
    rgb_array = np.zeros_like(yiq_array, dtype=float)
    for i in tqdm(range(yiq_array.shape[0]), desc="Converting YIQ to RGB"):
        for j in range(yiq_array.shape[1]):
            y, i_value, q = yiq_array[i, j]
            r = np.clip(y + 0.956 * i_value + 0.621 * q, 0, 255)
            g = np.clip(y - 0.272 * i_value - 0.647 * q, 0, 255)
            b = np.clip(y - 1.105 * i_value + 1.702 * q, 0, 255)
            rgb_array[i, j] = [r, g, b]
    return rgb_array.astype(np.uint8)

# Aplicação do filtro pontual apenas na banda Y
def apply_filter_to_y_band(image_array):
    # Converte RGB para YIQ
    yiq_image = rgb_to_yiq(image_array)

    # Aplica o filtro triangular apenas na banda Y (luminância)
    for i in tqdm(range(yiq_image.shape[0]), desc="Applying triangular filter"):
        for j in range(yiq_image.shape[1]):
            yiq_image[i, j, 0] = triangular_filter(
                yiq_image[i, j, 0])  # Filtro apenas na banda Y

    # Converte de volta para RGB
    filtered_rgb_image = yiq_to_rgb(yiq_image)

    return filtered_rgb_image

def main():
    arq_gauss = FilterModel("assets/filtro_gaussiano.txt")
    arq_sobel_vert = FilterModel("assets/filtro_sobel_vertical.txt")
    arq_sobel_hor = FilterModel("assets/filtro_sobel_horizontal.txt")

    image = ImageModel("assets/tigre.jpeg")
    image_gauss = filtro_gaussiano(image.array_img, arq_gauss)
    image.atualizar_imagem(image_gauss)
    image.salvar_imagem("outputs/tigre.jpeg")
    print("Imagem gauss salva com sucesso!")

    shapes_vert = ImageModel("assets/Shapes.png")
    img_sobel_vert = filtro_sobel(shapes_vert.array_img, arq_sobel_vert)
    shapes_vert.atualizar_imagem(img_sobel_vert)
    shapes_vert.salvar_imagem("outputs/shapes_sobel_vert.png")
    print("Imagem shapes_sobel_vert salva com sucesso!")

    shapes_hor = ImageModel("assets/Shapes.png")
    img_sobel_hor = filtro_sobel(shapes_hor.array_img, arq_sobel_hor)
    shapes_hor.atualizar_imagem(img_sobel_hor)
    shapes_hor.salvar_imagem("outputs/shapes_sobel_hor.png")
    print("Imagem shapes_sobel_hor salva com sucesso!")

    shapes_full = ImageModel("assets/Shapes.png")
    shapes_expanded = np.abs(img_sobel_hor.astype(np.int16)) + np.abs(img_sobel_vert.astype(np.int16))
    shapes_expanded = np.clip(shapes_expanded, 0, 255).astype(np.uint8)
    shapes_expanded = expansao_histograma(shapes_expanded)
    shapes_full.atualizar_imagem(shapes_expanded)
    shapes_full.salvar_imagem("outputs/shapes_full.png")
    print("Imagem shapes_full salva com sucesso!")

    cd_hor = ImageModel("assets/cd.png")
    img_sobel_hor = filtro_sobel(cd_hor.array_img, arq_sobel_hor)
    cd_hor.atualizar_imagem(img_sobel_hor)
    cd_hor.salvar_imagem("outputs/cd_sobel_hor.png")
    print("Imagem cd_sobel_hor salva com sucesso!")
    
    #Aplicar a expansão de histograma na imagem combinada
    #expanded_image = expansao_histograma(cd_hor.array_img)
    xadrez_hor = ImageModel("assets/xadrez.png")
    img_sobel_hor = filtro_sobel(xadrez_hor.array_img, arq_sobel_hor)
    xadrez_hor.atualizar_imagem(img_sobel_hor)
    xadrez_hor.salvar_imagem("outputs/xadrez_sobel_hor.png")
    print("Imagem xadrez_sobel_hor salva com sucesso!")
    xadrez_vert = ImageModel("assets/xadrez.png")
    img_sobel_vert = filtro_sobel(xadrez_vert.array_img, arq_sobel_vert)
    xadrez_vert.atualizar_imagem(img_sobel_vert)
    xadrez_vert.salvar_imagem("outputs/xadrez_sobel_vert.png")
    print("Imagem xadrez_sobel_vert salva com sucesso!")

    xadrez_full = ImageModel("assets/xadrez.png")
    xadrez_expanded = np.abs(img_sobel_hor.astype(np.int16)) + np.abs(img_sobel_vert.astype(np.int16))
    xadrez_expanded = np.clip(xadrez_expanded, 0, 255).astype(np.uint8)
    xadrez_expanded = expansao_histograma(xadrez_expanded)
    xadrez_full.atualizar_imagem(xadrez_expanded)
    xadrez_full.salvar_imagem("outputs/xadrez_full.png")

    test_hor = ImageModel("assets/testpat.1k.color2.tif")
    img_sobel_hor = filtro_sobel(test_hor.array_img, arq_sobel_hor)
    test_hor.atualizar_imagem(img_sobel_hor)
    test_hor.salvar_imagem("outputs/testpat_sobel_hor.png")
    print("Imagem testpat_sobel_hor salva com sucesso!")

    test_vert = ImageModel("assets/testpat.1k.color2.tif")
    img_sobel_vert = filtro_sobel(test_vert.array_img, arq_sobel_vert)
    test_vert.atualizar_imagem(img_sobel_vert)
    test_vert.salvar_imagem("outputs/testpat_sobel_vert.png")

    test_full = ImageModel("assets/testpat.1k.color2.tif")
    test_expanded = np.abs(img_sobel_hor.astype(np.int16)) + np.abs(img_sobel_vert.astype(np.int16))
    test_expanded = np.clip(test_expanded, 0, 255).astype(np.uint8)
    test_expanded = expansao_histograma(test_expanded) 
    test_full.atualizar_imagem(test_expanded)
    test_full.salvar_imagem("outputs/testpat_full.png")
    
    #Criar uma instância da classe ImageModel e carregar a imagem
    #Substitua pelo nome da sua imagem
    img_model = ImageModel('assets/testpat.1k.color2.tif')
    # Aplicar o filtro triangular
    filtered_image = apply_triangular_filter(img_model.array_img)
    # Atualizar a imagem no modelo com o novo array de pixels filtrados
    img_model.atualizar_imagem(filtered_image)
    # Salvar a imagem filtrada
    img_model.salvar_imagem('outputs/pontualRGB.jpg')

    # Criar uma instância da classe ImageModel e carregar a imagem
    # Substitua pelo nome da sua imagem
    img_model = ImageModel('assets/testpat.1k.color2.tif')
    # Aplicar o filtro à banda Y (YIQ)
    filtered_image = apply_filter_to_y_band(img_model.array_img)
    # Atualizar a imagem no modelo com o novo array de pixels filtrados
    img_model.atualizar_imagem(filtered_image)
    # Salvar a imagem filtrada
    img_model.salvar_imagem('outputs/pontualYIQ.jpg')


if __name__ == "__main__":
    main()
