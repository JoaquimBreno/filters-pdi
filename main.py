from models import ImageModel, FilterModel
from tqdm import tqdm
import numpy as np

import numpy as np

def aplicar_correlacao(imagem, dimensoes, offset, filtro):
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
                        coeficiente_filtro = filtro[dy + m//2][dx + n//2]
                        
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

def main():
    arq  =  FilterModel("assets/filtro_gaussiano.txt")

    image = ImageModel("assets/cd.png")
    array_img = aplicar_correlacao(image.array_img, arq.dimensoes, arq.offset, arq.matriz)
    image.atualizar_imagem(array_img)

    image.salvar_imagem("outputs/cd2.png")

if __name__ == "__main__":
    main()
