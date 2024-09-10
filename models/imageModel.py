import numpy as np
from PIL import Image

class ImageModel:
    def __init__(self, path=None):
        self.img = None  # Será um objeto PIL.Image
        if path:
            self.carregar_imagem(path)
        else:
            self.array_img = None  # Será um array do NumPy

    def carregar_imagem(self, path):
        self.img = Image.open(path).convert('RGB')
        self.array_img = np.array(self.img)

    def salvar_imagem(self, path):
        # Reconverte o array do NumPy para uma imagem PIL e salva
        if self.array_img is not None:
            Image.fromarray(self.array_img.astype('uint8')).save(path)

    def atualizar_imagem(self, novos_pixels):
        self.array_img = novos_pixels