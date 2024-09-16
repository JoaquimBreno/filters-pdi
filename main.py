from models import ImageModel, FilterModel
from tqdm import tqdm
import numpy as np

import numpy as np


def apply_convolution(image, dimensoes, offset, mask):
    img = image.copy()
    mask = mask.copy()
    new_img = np.zeros_like(img)

    r_mask, c_mask = dimensoes 
    r_img, c_img = img.shape[:2]

    pivo = [int(np.ceil(r_mask / 2) - 1),
            int(np.ceil(c_mask / 2) - 1)]

    print('Pivo[i, j] =', pivo)

    channels = img.shape[2] if len(img.shape) == 3 else 1

    for i in tqdm(range(pivo[0], r_img - pivo[0]), desc="Aplicando convolução"):
        for j in range(pivo[1], c_img - pivo[1]):
            if channels == 1:
                region = img[i - pivo[0]:i + pivo[0] + 1, j - pivo[1]:j + pivo[1] + 1]
                new_value = np.sum(region * mask)
                new_img[i, j] = np.abs(new_value)
            else:
                for c in range(channels):
                    region = img[i - pivo[0]:i + pivo[0] + 1, j - pivo[1]:j + pivo[1] + 1, c]
                    new_value = np.sum(region * mask)
                    new_img[i, j, c] = np.abs(new_value)

    new_img = new_img.astype(np.int16) + offset
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0

    return new_img.astype(np.uint8)
'''
def apply_convolution(image: np.array, dimensoes: tuple, offset: int, matriz: np.array) -> np.array:
    img = image.copy()
    mask = matriz.copy()
    new_img = np.zeros_like(img)

    r_mask, c_mask = dimensoes

    pivo = [int(np.ceil(r_mask / 2) - 1),
            int(np.ceil(c_mask / 2) - 1)]

    print('Pivo[i, j] =', pivo)

    channels = img.shape[2] if len(img.shape) == 3 else 1

    padded_img_shape = (img.shape[0] + 2 * pivo[0], img.shape[1] + 2 * pivo[1], channels)
    img_padded = np.zeros(padded_img_shape, dtype=img.dtype)

    if channels == 1:
        img_padded[pivo[0]:pivo[0] + img.shape[0], pivo[1]:pivo[1] + img.shape[1], 0] = img
    else:
        img_padded[pivo[0]:pivo[0] + img.shape[0], pivo[1]:pivo[1] + img.shape[1], :] = img

    strides = img_padded.strides
    height, width = img.shape[0], img.shape[1]

    if channels == 1:
        sub_imgs = np.lib.stride_tricks.as_strided(
            img_padded,
            shape=(height, width, r_mask, c_mask),
            strides=(strides[0], strides[1], strides[0], strides[1])
        )

        correlation = np.abs(np.einsum('ijkl, kl->ij', sub_imgs, mask))
        new_img[:, :] = correlation
    else:
        sub_imgs = np.lib.stride_tricks.as_strided(
            img_padded,
            shape=(height, width, r_mask, c_mask, channels),
            strides=(strides[0], strides[1], strides[0], strides[1], strides[2])
        )

        correlation = np.abs(np.einsum('ijklm,kl->ijm', sub_imgs, mask))
        new_img[:, :, :] = correlation

    new_img = new_img.astype(np.int16) + offset
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0

    return new_img.astype(np.uint8)
'''
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
    filtered_image = apply_convolution(
        imagem, arq.dimensoes, arq.offset, arq.matriz)

    filtered_image = np.abs(filtered_image)

    filtered_image = (filtered_image / filtered_image.max()) * 255

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
    #print("arq_gauss:", arq_gauss)
    #print("arq_gauss:", type(arq_gauss.matriz))
    #print("arq_sobel_vert:", arq_sobel_vert)
    #print("arq_sobel_hor:", arq_sobel_hor)
    image = ImageModel("assets/testpat.1k.color2.tif")
    image_gauss = filtro_gaussiano(image.array_img, arq_gauss)
    image.atualizar_imagem(image_gauss)
    image.salvar_imagem("outputs/testpat.tif")
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
    #Criar uma instância da classe ImageModel e carregar a imagem
    #Substitua pelo nome da sua imagem
    img_model = ImageModel('assets/testpat.1k.color2.tif')
    # Aplicar o filtro triangular
    filtered_image = apply_triangular_filter(img_model.array_img)
    # Aplicar o filtro à banda Y (YIQ)
    filtered_image = apply_filter_to_y_band(img_model.array_img)
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
