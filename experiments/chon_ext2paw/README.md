# CHON: revisão e benchmark de `ext2paw`

O experimento usa os **432 materiais** de `data/CHON`, sem incluir os átomos
isolados no conjunto de treinamento. As referências C, H, N e O de
`data/CHON/isolated_atoms` definem a superposição atômica para o alvo residual.
O benchmark conserva a magnetização e os registros PAW completos.

## Execução reproduzível

A partir da raiz do repositório:

```bash
python experiments/chon_ext2paw/prepare.py
python experiments/chon_ext2paw/benchmark.py --epochs 30 --seeds 0
# Confirmação: repetir as configurações finalistas com outra inicialização.
python experiments/chon_ext2paw/benchmark.py --epochs 30 --seeds 0 1 \
  --arms baseline width32 --output experiments/chon_ext2paw/results/confirmation \
  --evaluate-winner
```

O segundo comando de exemplo precisa receber as finalistas efetivamente
identificadas na triagem. `--evaluate-winner` deve ser usado somente depois de
encerrar a seleção: o teste não participa do ranking. `--pilot` usa as primeiras
48 estruturas disponíveis, serve apenas para medir custo e não avalia o teste.

## Protocolo

- Divisão por **estruturas inteiras**, estratificada pelo conjunto de elementos,
  com semente 42: 303 treino, 88 validação e 41 teste. Grupos muito pequenos ficam
  no treino. Isso mede interpolação entre estruturas, não extrapolação para
  elementos novos ou validação por famílias químicas inteiramente excluídas.
- Normalizações dos campos e das ocupações ajustadas somente ao treino.
  A média PAW de referência também é somente de átomos do treino.
- Banda comum do readout determinada pela compatibilidade geométrica dos grids,
  sem ajustar estatísticas dos alvos: **2,239581 Å⁻¹** na resolução 24.
  É uma restrição geométrica conhecida para este domínio, incluindo os grids
  reservados. O modelo não pode exigir uma frequência ausente numa estrutura.
- O melhor checkpoint de cada execução segue sua própria função objetivo.
  O ranking entre configurações usa **o mesmo** critério:
  `L2 relativo da carga física + 3 × perda PAW normalizada e balanceada`.
  Isso permite comparar diferentes pesos PAW sem premiar um peso menor apenas
  porque ele diminuiu a escala do score.
- Erro PAW físico acumulado sobre os átomos, separado também por elemento, conjunto
  de carga/magnetização e L. Erro de densidade médio sobre estruturas.
  Nenhum recorte de valores negativos ou ajuste da contagem eletrônica entra
  nas métricas: essas operações esconderiam parte do erro.
- Mesmos materiais, inicialização e orçamento de épocas entre os controles.
  Uma segunda inicialização deve confirmar as finalistas; erro entre sementes
  não equivale a incerteza de validação cruzada.

Arquivos `split.json`, `audit.json`, configurações resolvidas, histórico, métricas
por estrutura/bloco, CSV e checkpoints tornam o resultado verificável. Os pesos
`.poraque` permanecem locais, conforme o `.gitignore` do projeto. Uma execução
com métricas já completas é retomada somente se configuração, divisão e
fingerprint das referências atômicas coincidirem.

## Auditoria dos dados

Há **13.225 átomos**: H 4.525, C 3.112, N 2.754 e O 2.834. Todos os registros
verificados no cache têm 15 componentes para H e 33 para C/N/O, de acordo com os
projetores da biblioteca POTCAR. Os POTCARs dos átomos isolados são idênticos,
por SHA-256, aos correspondentes usados para construir os potenciais.
Essa compatibilidade não comprova a identidade dos POTCARs originais do Materials
Project, cujos arquivos não acompanham o download.

Uma verificação adicional de geometria, com célula e posições fracionárias
ordenadas e arredondadas a seis casas, não encontrou duplicatas exatas.
Ela não elimina equivalências por outra base de célula ou estruturas muito próximas.

O cache HDF5 ocupa aproximadamente 112 MiB e tem 83 formatos de grid. O truncamento
na resolução 24 deixa 4,06% dos pontos de carga negativos por oscilação espectral.
A norma da magnetização é 5,20% da norma da carga. Portanto, eliminar a
magnetização mudaria o problema, e exigir positividade do campo truncado também
alteraria o alvo. A precisão aqui é a precisão sobre o **campo truncado**, não
sobre todos os modos do CHGCAR nativo.

O CHGCAR guarda tanto o campo quanto ocupações de um centro; os blocos adicionais
de magnetização e o limite `LMAXMIX` também precisam ser preservados para um
restart. [Documentação do VASP](https://vasp.at/wiki/CHGCAR).

## Revisão do caminho de código

| Componente | Papel e avaliação |
| --- | --- |
| `fields/vasp`, `fields/hdf5` | Leitura da densidade, geometria, canais de spin e ocupações; HDF5 preserva os registros sem truncá-los junto com o grid. |
| `data/sources.py`, `data/cache.py` | Descoberta dos cálculos, potencial externo derivado dos POTCARs e cache com proveniência da construção. O benchmark exige POTCARs e não admite fallback Gaussiano. |
| `fields/atomic.py` | Referências independentes e fatores de forma radiais; a superposição entra no alvo residual e é restaurada na predição física. |
| `ml/data.py` | Lotes por formato de grid, padding de átomos e componentes e ajuste das normalizações sobre o treino. Essa divisão é essencial para evitar vazamento de alvos. |
| `ml/fno.py` | Operador escalar periódico, truncamento físico, filtros radiais e condicionamento pela célula. Modos, base radial, largura e profundidade têm efeitos diferentes de capacidade e custo. |
| `ml/paw.py` | Leitura do campo e do latente em torno de cada átomo, expansão de vizinhos e cabeça por elemento com blocos de momento angular. A escala é compartilhada entre componentes M para preservar a transformação tensorial. |
| `ml/training.py` | Otimização conjunta, métricas físicas e restauração do melhor modelo. A perda PAW dá peso semelhante a cada bloco L; o RMS bruto, sozinho, pode esconder erros multipolares e magnéticos. |
| `scripts/poraque_train.py` | Montagem da tarefa e resolução da arquitetura. `model.paw` determina o backbone de `ext2paw`, inclusive os modos usados para resolver a base radial. |
| `scripts/poraque_inference.py` | Usa o operador PAW do bundle para gerar registros quando solicitado. As cabeças são por elemento, e não se deve assumir cobertura de elementos que não foram treinados. |
| `physics`, `calculator.py` | Energia e forças são consumidores das predições; este benchmark não valida sua precisão nem uma execução SCF/restart do VASP. |

Os problemas observáveis corrigidos incluem:

1. Lotes só de H tinham alvos com 15 valores e saída com 33: a perda e o RMS agora
   alinham o comprimento do lote ao da saída, preservando a máscara do padding.
2. O volume no readout era calculado por `linalg.det` no Metal, operação não
   implementada no PyTorch local. O cálculo usa o helper de volume do framework.
3. A indexação complexa do readout falhava no backward do Metal. A cópia dos
   coeficientes agora usa uma representação real com dois componentes,
   mantendo a FFT complexa e o gradiente.
4. Células alongadas podiam produzir grids `2×2×24`, impossíveis para o truncamento
   que descarta Nyquist. O tamanho reduzido mantém ao menos quatro pontos por
   eixo, sem ampliar um eixo originalmente menor.
5. O CLI de inferência exigia `ext2chg` e `chg2tau`, mesmo para um bundle contendo
   só `ext2paw`. Agora usa a saída de densidade do operador PAW e escreve seus
   registros, omitindo TAUCAR e energia quando não existe operador cinético.
   Comparação e figuras também tratam essa ausência. O CLI de treino aceita
   `--task ext2paw` diretamente.

Também foi corrigida a resolução de `g_basis: auto` e seu relato: ambos devem
seguir os modos do backbone PAW, em vez de usar `model.modes` quando os dois
valores são diferentes. Isso é relevante para a comparação de bandas.

## Limitações encontradas

`ext2paw` ainda não aceita o caminho `--kfold` nem fine-tuning do CLI. A confirmação
por sementes deste experimento não substitui essas capacidades. Os agregados
também dão pesos diferentes a estruturas e átomos; por isso o relatório mantém
ambas as métricas.

O cache verifica parâmetros de construção e origem dos POTCARs, mas não calcula
um hash de cada CHGCAR para invalidar automaticamente arquivos alterados no mesmo
caminho. Alterar os dados nativos exige construir um cache novo. A média PAW de
todo o conjunto não é usada neste benchmark.

A expansão geométrica dos vizinhos da cabeça atual não separa o elemento de cada
vizinho. A informação química adicional vem dos campos e da cabeça do elemento
central. Enriquecer essa expansão é uma possível mudança de arquitetura, ainda
sem medição neste experimento.

Na suíte geral sem testes marcados GPU: **2.237 passaram, 101 foram ignorados e
27 excluídos; um falhou**, `test_vis.py::TestLogTickLabels::test_thins_the_minor_labels`.
A falha se reproduz isoladamente e depende de uma expectativa sobre os labels
menores padrão do Matplotlib; não envolve treinamento ou ocupações PAW. Os testes
específicos das correções, inclusive treino/backward de um lote H em CPU e Metal,
passaram.
O teste de inferência de um bundle exclusivamente PAW, junto com os testes de
inferência e padrões do pipeline, também passou: 54 testes, 7 ignorados.

Os pilotos de 48 estruturas mediram 1,89 s por época em CPU e 17,02 s em Metal,
com métricas praticamente iguais. Esses tempos são somente do piloto; o custo
do conjunto completo fica registrado por época nos resultados.
