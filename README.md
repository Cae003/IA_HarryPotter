# IA_HarryPotter
Detector de rostos que identifica os personagens em um trecho do filme Harry Potter e a Pedra Filosofal

### Baixar Bibliotecas
Para que o projeto funcione corretamente, é necessário baixar a biblioteca dlib pré-compilada compativel com a versão do python que está usando no seguinte link:
https://github.com/z-mahmud22/Dlib_Windows_Python3.x

Após isso entre dentro do diretório da pasta que está o arquivo ".whl" e de um pip install nele conforme demonstrado na imagem:
![Guia para qual dlib baixar](./assets/Guia.png)

Perceba que a versão que peguei do dlib foi para a verção 3.9 do python, sinalizada por "cp39" no arquivo.

Após isso instale as bibliotecas que estão no "requirements.txt" usando o comando:

```bash
pip install -r requirements.txt
```

E pronto! Já pode rodar o código normalmente!

### Atenção! Tratamento de Possível Erro!
Ao rodar o código, existe a possibilidade de gerar um erro que diz que é necessário instalar a biblioteca “face_recognition_models”, como demonstra a imagem abaixo:
![Guia para Erro](./assets/Guia.png)

Se ele ocorrer, tente instalar a biblioteca. Se não funcionar, será necessário baixar outra versão do python diferente da que você está usando e pegar a “dlib” correspondente conforme mostrado acima. Se estiver usando ambientes virtuais, tente simplesmente criar um ambiente com uma versão diferente do python.