class FilterModel:
    def __init__(self, arquivo_filtro=None):
        self.dimensoes = (0, 0)   # Dimensões do filtro (m, n)
        self.offset = 0           # Offset do filtro
        self.matriz = []          # Matriz de valores do filtro
        
        if arquivo_filtro:
            self.ler_filtro_do_arquivo(arquivo_filtro)

    def ler_filtro_do_arquivo(self, arquivo_filtro):
        with open(arquivo_filtro, 'r') as arquivo:
            self.dimensoes = tuple(map(int, arquivo.readline().split()))  # Lê as dimensões do filtro (m,n)
            self.offset = int(arquivo.readline())  # Lê o offset
            self.matriz = [list(map(float, linha.split())) for linha in arquivo]  # Lê a matriz do filtro
            
    def __str__(self):
        filtro_str = f"Dimensões: {self.dimensoes}\nOffset: {self.offset}\nMatriz:\n"
        for linha in self.matriz:
            filtro_str += ' '.join(map(str, linha)) + "\n"
        return filtro_str