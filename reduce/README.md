# Estrutura do projeto e como utilizar

## Estrutura do projeto

  O projeto `reduce`, possui um arquivo de cabeçalho com algumas definições comuns (`common.h`) e também possui uma implementação de redução utilizando paralelismo em GPU e algoritmo sequencial em CPU. Os respectivos arquivos são `reduce.cu` e `reduce_seq.cu`.

  Para exemplo, foi criado um arquivo com dados de entrada chamado `input.dat` que contém 4 matrizes para redução.

  Um detalhe importante (e também uma limitação), é que esse projeto funciona apenas com um número par de matrizes, devido à estratégia utilizada para redução na GPU.

### Dependências

  O diagrama a seguir, apresenta as dependências entre os principais arquivos do projeto:

```
       main.c --> reduce.h
        | |
        | +-----> reduce_seq.h
        |
        v
      common.h
```

  O código criado para teste de unidade possui uma relação de dependências semelhante.

## Como utilizar

  Ao executar o comando make dentro do diretório do projeto, alguns arquivos binários serão criados dentro do sub-diretório bin, que por sua vez irá conter os seguintes arquivos:

  - main: arquivo principal a ser usado para o cálculo de matrizes
  - bench: executa um benchmark comparando as implementações disponíveis
  - bench_sparse: executa um benchmark utilizando matriz esparsa gerada randomicamente
  - genmatrix: usado para gerar matrizes densas com valores randômicos
  - matrix_test: usado para executar testes de unidade

## Executando o programa principal (main)

  Antes de executar o programa, rode o seguinte comando para que a compilação do projeto seja executada (o arquivo final gerado após a compilação será executado uma vez utilizando o arquivo de entrada de exemplo).

  Ao tentar executar o programa `main` sem nenhum parâmetro, a seguinte saída será apresentada:
  `Usage: reduce <input_file>`

  Portanto, deve ser informado um arquivo (para testes simples, utilizar o arquivo `input.dat`).

## Testes de unidade

  Para executar os testes de unidade, basta executar o seguinte comando:
  `make test`

## Dificuldades encontradas

  Algumas dificuldades foram encontradas durante a criação do projeto conforme mostrado a seguir:
  - Utilização dinâmica de memória compartilhada entre as threads que executam na GPU;
  - Realização da cópia de matriz de 2 dimensões para a GPU;
  - Recuperação dos dados dentro do Kernel.

## Limitações

  - limitado a processar apenas 1024 matrizes;
  - definição de poucas threads por bloco, não aproveitando a capacidade de paralelismo da GPU.
