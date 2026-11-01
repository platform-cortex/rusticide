<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Facade de kernels e guarda de desempenho (Design)

Status: aprovado para planejamento, 2026-11-01. Segue o milestone de desempenho de inferencia
([notas](2026-10-31-rusticide-inference-performance.pt-BR.md), release 0.6.0)
e paga os debitos que ele deixou, do mais importante para o menos. Branch
`chore/kernel-architecture`. Versao em ingles:
[2026-11-01-rusticide-kernel-facade-design.md](2026-11-01-rusticide-kernel-facade-design.md).

## Intencao

A release 0.6.0 deixou o `rusticide run` de 3,6 a 4,3 vezes mais rapido,
mas fez isso contra duas decisoes escritas e deixou o codigo de calculo
espalhado:

- [O design do run](2026-10-22-rusticide-run-design.md) lista kernels de
  matmul quantizado como nao-objetivo e diz que reduzir memoria
  explicitamente nao e objetivo. O codigo agora faz as duas coisas, e os
  documentos dizem o contrario.
- Nada no repositorio mede velocidade. Todos os numeros da 0.6.0 vieram de
  ferramentas descartaveis fora dele, entao uma mudanca de aparencia
  inofensiva pode desfazer o milestone com todos os gates verdes.
- Os kernels moram em tres crates: kernels f32 de slice em
  `rusticide-kernels`, os kernels inteiros por linha (`int_dot`) e o
  quantizador de ativacao (`Q8Input`) em `rusticide-quantize`, e o
  agendador paralelo (`matmul_batch`) em `rusticide-model`. O contrato de
  formato (`BlockFormat`) ganhou um kernel e um tipo de ativacao, entao um
  formato novo ja nao pode ser "um decoder e uma entrada na tabela".

A direcao do dono: um facade de kernels com modulos de kernel plugaveis,
no espirito da arquitetura hexagonal (portas e adaptadores), parando onde
o PyTorch para nos kernels de CPU, e sem `unsafe` neste milestone.

## Objetivos, em ordem

1. **Os documentos dizem a verdade.** O design do run e o design do
   framework descrevem o que a 0.6.0 faz e registram a arquitetura abaixo
   como decisao. Isso vem primeiro porque todo passo seguinte argumenta a
   partir deles.
2. **Um gate local de desempenho com linha de base.** Ele falha quando a
   velocidade cai; o CI so mostra os numeros. Isso vem antes da
   refatoracao porque a refatoracao move exatamente o codigo que ele
   protege.
3. **O facade de kernels.** `rusticide-kernels` e o unico lugar que
   calcula. Quem chama usa as portas dele; um adaptador, `portable` (o
   codigo seguro e autovetorizado de hoje), as implementa, e a costura e
   real o bastante para que um segundo adaptador seja um modulo mais um
   registro, nao uma edicao em algum crate de modelo.
4. **`rusticide-quantize` volta a ser dados.** Um formato de bloco e seu
   nome, type id, tamanho de bloco, bytes por bloco e dequantizador, como
   o design do run especificou. Adicionar um formato e um decoder em
   `rusticide-quantize` mais um kernel registrado em `rusticide-kernels`.

## Nao-objetivos

- `unsafe`, `rusticide-simd`, adaptadores AVX2, NEON ou dotprod, deteccao
  de recursos da CPU em tempo de execucao alem da propria costura de
  selecao. Proximo milestone.
- Medir ou ajustar o servidor Intel.
- Kernels de prefill com tiling, paralelizar o trabalho serial por camada,
  formatos novos (`i2_s`).
- Rotear o operador `matmul` do dispatcher pelo facade. Isso e trabalho do
  SP2; este milestone so torna isso possivel.
- Traduzir para o portugues documentos que este milestone nao cria (ver
  "Documentos").

## Anti-objetivos

- **Mais lento que a 0.6.0.** Qualquer numero de decode ou prefill medido
  no M1 Pro abaixo da linha de base commitada por mais que a tolerancia
  do gate reprova o milestone, por mais limpo que o codigo pareca.
- **Uma chamada indireta por bloco.** A 0.6.0 mediu que uma chamada
  indireta por bloco de 32 elementos escondia os lacos do vetorizador. As
  portas sao chamadas por sequencia de linhas ou por lote, nunca por
  bloco.
- **Um segundo dispatcher.** O facade nao e um registro de operadores
  empacotados. A selecao acontece uma vez, e depois disso uma chamada por
  uma porta e uma chamada de funcao comum.
- **Paridade ou cobertura mais fracas.** `test:reference` passa com o
  NEAR_TIE de hoje (0.25), e a cobertura fica em 100%.

## Arquitetura: dois niveis, como no PyTorch

O PyTorch escolhe implementacoes em dois niveis, e o rusticide copia os
dois:

- **Nivel de operador: o dispatcher** (`rusticide-dispatch`, ja
  construido, modelado nas dispatch keys do `c10`). Ele escolhe um backend
  por chamada de operador e paga uma pilha de argumentos empacotados a
  cada vez. Certo para "este tensor esta na CPU", errado para um laco
  interno. Nao muda aqui.
- **Nivel de kernel: o facade** (`rusticide-kernels`), modelado no
  `DispatchStub` do PyTorch (`DECLARE_DISPATCH` / `REGISTER_DISPATCH` com
  `CPU_CAPABILITY`). Cada porta e um slot de ponteiros de funcao comuns;
  cada adaptador preenche todos os slots; um adaptador e escolhido uma vez
  por processo e guardado. Um futuro kernel de `matmul` despachado chama o
  facade, entao os dois caminhos convergem em vez de divergir.

Em termos hexagonais:

- **Portas** -- uma por familia de kernel: produto escalar, elemento a
  elemento (soma, multiplicacao, SiLU), normalizacao (RMSNorm, softmax),
  embedding rotatorio (as duas variantes), quantizacao de ativacao (os
  esquemas `Q8_0` e `Q8_K` do ggml) e produtos de matriz quantizados (uma
  linha de um formato de bloco contra uma ou varias entradas
  quantizadas).
- **Adaptadores** -- `portable`, o codigo atual, e o unico adaptador de
  producao.
- **Selecao** -- uma funcao que devolve o adaptador do processo, escolhido
  uma vez. Com um unico adaptador de producao ela sempre devolve
  `portable`; a costura existe para que o proximo milestone acrescente uma
  checagem de CPU, nao uma API nova.
- **Agendamento** -- dividir um produto de matrizes no pool de threads (o
  `matmul_batch` de hoje, com suas regioes fundidas e `TASKS_PER_THREAD`)
  faz parte do facade, acima do adaptador: todo adaptador recebe o mesmo
  agendador e so fornece kernels por linha. *Decisao do construtor,
  fechada aqui:* so o agendamento de produtos de matriz se move. O laco de
  atencao paralelo por cabeca e o `rayon::scope` em volta do forward pass
  sao estrutura do proprio modelo e ficam em `rusticide-llama`, chamando
  portas do facade por dentro; `rusticide-model` deixa de depender do
  rayon se nada mais ali o usar.

Direcao das dependencias: `rusticide-kernels` depende de
`rusticide-quantize` (layouts de bloco, leitura de f16) e do rayon;
`rusticide-quantize` nao depende de nenhum dos dois. Os crates de modelo
dependem de `rusticide-kernels` para todo calculo, e de
`rusticide-quantize` so para carregar e decodificar pesos.

O que se move, fechado aqui para a direcao se manter:

- `BlockFormat` perde `dot_q8`, `dot_rows` e `activation`; fica com nome,
  type id, tamanho de bloco, bytes por bloco e o dequantizador.
- `int_dot`, `Q8Input` e `Q8Kind` vao para `rusticide-kernels`, com seus
  testes. O adaptador associa um formato (pelo type id) aos seus kernels
  por linha e ao esquema de ativacao com que eles trabalham; um formato
  para o qual o adaptador nao tem kernel e um erro ao carregar os pesos,
  nao um panic.
- A porta quantizada recebe a identidade do formato e os bytes crus da
  linha, entao nenhum tipo de `rusticide-quantize` nomeia um kernel.

## O gate de desempenho

Duas camadas de medicao:

- **Benchmarks de kernel** nao precisam de modelo: o kernel por linha de
  cada formato quantizado (uma thread, entrada unica e em lote) e a
  quantizacao de ativacao, em blocos sinteticos, reportados em elementos
  por segundo. Rodam em qualquer lugar, CI incluido.
- **Benchmarks de ponta a ponta** precisam de `RUSTICIDE_MODELS`: tokens
  por segundo de decode e de prefill para os dois modelos de referencia,
  com os prompts e tamanhos que as notas da 0.6.0 usaram.

Os benchmarks sao construidos sobre o codigo da 0.6.0, antes que a
refatoracao toque em qualquer coisa, e a primeira linha de base e
capturada desse codigo no M1 Pro: o gate entao mede a refatoracao contra o
que foi lancado, nao contra numeros de ferramentas descartaveis. *Decisao
minha:* a tolerancia sai do ruido medido (a variacao entre execucoes nas
rodadas que o gate faz, com piso de 5%) e fica registrada no arquivo de
linha de base junto com os numeros, entao aperta-la tambem e uma mudanca
revisada. Decode e prefill usam a mesma regra. O gate e rodado a mao. O
formato da linha de base e indexado pela identidade da maquina para que
outras maquinas acrescentem a sua; este milestone commita so a do M1 Pro.

Uma linha de base commitada por maquina guarda os numeros. A task do gate
compara uma execucao nova com a linha de base da maquina atual e falha
numa queda alem da tolerancia; uma segunda task reescreve a linha de base,
e reescreve-la e uma mudanca revisada como qualquer outra. Uma maquina sem
linha de base recebe uma mensagem clara, nao uma aprovacao. O CI roda os
benchmarks de kernel e imprime a tabela no resumo do job, sem nunca falhar
pelos numeros. O gate nao faz parte do `ci:quick`; o AGENTS.md o exige
antes do merge de qualquer mudanca em kernels, agendamento ou forward
pass.

## Documentos

- `2026-10-22-rusticide-run-design.md`: um bloco de emenda (o estilo que o
  documento ja usa) que retira os dois nao-objetivos, descreve o contrato
  de formato que este milestone restaura e aponta para ca.
- `2026-08-21-rusticide-framework-design.md`: a selecao em dois niveis
  como decisao de arquitetura, e `rusticide-simd` como a casa dos futuros
  adaptadores nativos.
- AGENTS.md: plano atual, comandos de benchmark, a regra do gate.
- Este documento e as notas de desempenho da 0.6.0 ganham traducoes
  `.pt-BR.md`, mantidas em sincronia.

## Restricoes

Tudo no AGENTS.md vale: TDD estrito, 100% de cobertura, nada de
`unwrap`/`expect` fora de testes, arquivos com menos de 300 linhas, ASCII,
cabecalhos SPDX, `-D warnings`, MSRV 1.96, integracao so por rebase.

## Deixado para o construtor

- Se o tipo de armazenamento de matriz vai para `rusticide-kernels` ou
  fica em `rusticide-model`, dentro de "O que se move" acima.
- A ferramenta de benchmark e seu formato de saida, desde que o
  `cargo deny` passe e os benchmarks de kernel terminem no CI em poucos
  minutos.
- Como o benchmark de ponta a ponta mede o prefill separado do
  carregamento.
- Como uma maquina e identificada, numero de execucoes, aquecimento e a
  estatistica comparada (uma regra so para decode e prefill).
