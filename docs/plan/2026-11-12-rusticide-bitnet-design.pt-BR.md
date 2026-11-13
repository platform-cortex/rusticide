<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# BitNet b1.58: marco 3 do `rusticide run` (Design)

Status: rascunho, 2026-11-12. Refina o marco 3 do
[design do `rusticide run`](2026-10-22-rusticide-run-design.md), que
continua valendo para o que este marco entrega; este documento registra o
que mudou desde entao e as decisoes tomadas para ele. Branch
`feat/bitnet`.

## Intencao

`rusticide run ggml-model-i2_s.gguf --prompt ...` executa o BitNet b1.58
2B 4T da Microsoft e produz os mesmos tokens gulosos que o bitnet.cpp. E
o ultimo dos tres marcos do `rusticide run` que a spec do framework poe
antes de SP2-SP7, e o dono o escolheu antes do trabalho de desempenho em
attention/RoPE (2026-11-12).

O design do run foi escrito em 2026-10-22. Desde entao os pesos ficam
quantizados em memoria, os produtos de todo formato passam por
`rusticide-kernels` e os adaptadores de kernel sao escolhidos em tempo de
execucao. E a attention que o BitNet precisa compartilhar com o `llama`
vive como `pub(crate)` dentro de `rusticide-llama`. Por isso este marco
primeiro move os blocos de transformer compartilhados para uma crate
propria e depois adiciona o BitNet por cima.

## Sequencia

1. **Fixtures de referencia primeiro** (decisao do autor do design): o
   bitnet.cpp e compilado no iMac Pro e as fixtures sao capturadas antes
   de qualquer codigo mudar de lugar. Elas sao o vai/nao-vai do marco: se
   nem a compilacao legada nem a alternativa (veja abaixo) gerarem
   fixtures, ou se as duas discordarem, o marco para e volta ao dono;
   nenhum outro oraculo e usado.
2. **`rusticide-transformer`** (decisao do dono, 2026-11-12): attention
   com GQA, os estilos de RoPE, a mistura com o KV cache, o feed-forward
   com portao e ativacao plugavel, o RMSNorm por bloco e os auxiliares de
   lote paralelo por token saem de `rusticide-llama`. Uma mudanca pura de
   lugar: veja "Prova da refatoracao" abaixo.
3. **`i2_s`**: o formato em `rusticide-quantize`, a quantizacao das suas
   ativacoes e um kernel portavel em `rusticide-kernels`, seguidos de um
   ponto de controle de velocidade de decode no M1 (veja Objetivos).
4. **`rusticide-bitnet`**: a familia `bitnet-b1.58`, registrada ao lado
   de `llama` e `qwen2`; o `test:reference` ganha o BitNet.
5. Linhas de bench, resultados, docs, release 0.10.0.

## Prova da refatoracao

"Bit-identico" significa logits identicos, nao so tokens identicos
(decisao do autor do design): o `test:reference` verifica tokens e tolera
quase-empates, entao nao prova uma mudanca de lugar. Antes da primeira
mudanca, uma task registra, por prompt de referencia e posicao, um hash
dos bits f32 dos logits do TinyLlama e do Qwen2.5 na maquina em uso
(mantido local, nunca commitado: depende da maquina e precisa dos
arquivos de modelo). Toda task da refatoracao termina conferindo esse
hash. Assim a mudanca nao pode alterar a ordem das somas, fundir lacos
nem reordenar lotes; o que fizer isso e uma mudanca separada, posterior.
O `test:reference` e o `bench:gate` ainda rodam ao fim da refatoracao.

## Objetivos

- O criterio de sucesso do design do run, sem mudanca: para pelo menos
  tres prompts (um nao-ASCII), o prompt tokeniza para os ids do
  bitnet.cpp e os 32 primeiros tokens gulosos coincidem, sob a regra de
  quase-empate (uma divergencia so passa quando a margem top-1/top-2 do
  rusticide e menor que 0.1).
- `rusticide-llama` fica so com o que e especifico de `llama`/`qwen2`:
  configuracao, layout dos pesos, parametros de variante.
  `rusticide-bitnet` fica so com o que e especifico do BitNet:
  configuracao, pesos `i2_s`, sub-norms, ligacao do ReLU ao quadrado.
- `i2_s` e um modulo mais entradas de tabela em `rusticide-quantize` e
  `rusticide-kernels`, como todo outro formato.
- Decode de pelo menos 2 tokens/s no M1 Pro (a meta do design do run).
  Uma meta, nao um piso rigido (decisao do autor do design): e medida
  logo depois que o kernel portavel entra; um resultado abaixo e
  reportado ao dono nesse momento, que decide se um kernel NEON de
  `i2_s` entra neste marco, e fora isso fica registrado nos Resultados
  para o marco de desempenho.

## Nao-objetivos

- Kernels SIMD de `i2_s` (NEON, AVX2). Portavel primeiro; um marco de
  desempenho medido vem depois, como foi com o `llama`. (Decisao do autor
  do design: o risco aqui e a corretude e a refatoracao da crate
  compartilhada.)
- Os formatos de tabela de consulta TL1/TL2 (`BITNET_ARM_TL1`,
  `BITNET_X86_TL2`) e o layout ARM de 64 pesos do `i2_s`. So o layout do
  arquivo publicado e lido.
- Templates de chat. O `run` envia o prompt cru, como nas outras
  familias.
- Igualar os logits do bitnet.cpp bit a bit (KV cache f16, ativacoes f16
  no lm_head, somas float na ordem das threads): a regua sao os tokens,
  nao os logits.

## Anti-objetivos

- Qualquer copia de attention, RoPE, RMSNorm, softmax ou do feed-forward
  em `rusticide-bitnet`. A refatoracao existe para impedir isso.
- `if arch == "bitnet-b1.58"` fora da crate da familia e da sua unica
  linha no registry.
- Uma regressao de `llama`/`qwen2` escondida pela refatoracao: um token
  diferente no `test:reference` ou uma falha no `bench:gate` bloqueia o
  marco.
- Relaxar as regras do AGENTS.md, ou qualquer panic num arquivo GGUF
  hostil (os novos caminhos de tamanho e escala do `i2_s` ganham
  cobertura de fuzz).

## Referencia: qual bitnet.cpp

O reconhecimento de `microsoft/BitNet` (2026-11-12) mostrou que o HEAD
atual (`0b341e5`, submodulo llama.cpp movido para
`isHuangXin/llama.cpp@390c307` em 2026-07-15) monta o FFN do BitNet com
SiLU em vez de ReLU ao quadrado e nao tem caminho NEON para `i2_s`, com
um fallback escalar que le outro layout. O submodulo legado
(`Eddie-Wang1120/llama.cpp` em `1f86f058d`, BitNet `1876a3e`,
2026-01-27; o commit seguinte do BitNet, `69bf64d`, move o submodulo para
uma versao com SiLU) implementa o modelo como publicado. Decisao (autor
do design): as fixtures vem do fork legado, compilado no iMac Pro
(compilado e capturado em 2026-11-12: Apple clang 17, BLAS e Metal
desligados, um patch so de tipo (`const`) que o clang exige; veja o bloco
`reference` da fixture),
carregando o arquivo publicado sem mudanca. A fixture registra os commits
do BitNet e do submodulo. Se essa compilacao nao funcionar, a alternativa
e o BitNet `765741d` (2025-05) no M1; seu caminho NEON le o mesmo layout
de 128 (inferido do codigo, nao testado).

Captura das fixtures: `llama-cli -m <arquivo> -p "<prompt>" -n 32
--temp 0 -t <n> -c 2048 -ngl 0` (o modo conversa vem desligado no fork
legado, BOS prefixado); os ids do prompt e dos tokens gerados saem da
propria ferramenta (quem implementa acha a flag ou o endpoint que os
imprime).

## Fatos do reconhecimento

As citacoes sao do fork legado, salvo indicacao; veja o ledger da task
para as referencias arquivo:linha.

- **Arquivo.** `microsoft/bitnet-b1.58-2B-4T-gguf`,
  `ggml-model-i2_s.gguf`, 1.187.801.280 bytes, SHA-256
  `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`.
  332 tensores: 210 `i2_s` (type id 36), 121 F32, 1 F16 (`token_embd`,
  amarrado: nao existe `output.weight`).
- **Layout do `i2_s` (x86, o arquivo publicado).** Codigos de 2 bits,
  0 -> -1, 1 -> 0, 2 -> +1, 3 sem uso. Blocos de 128 pesos em 32 bytes: o
  byte `k` de um bloco guarda os pesos `k` (bits 7-6), `k+32` (5-4),
  `k+64` (3-2) e `k+96` (1-0). Linhas tem `ne0/4` bytes, contiguas. Uma
  escala f32 por tensor logo apos os dados empacotados, depois 28 bytes
  sem uso: um tensor tem `ne0*ne1/4 + 32` bytes. Verificado nos bytes do
  arquivo (teste de correlacao de layout; todas as 210 escalas em
  [0.744, 4.596]).
- **Quantizacao das ativacoes.** Por linha de token:
  `max = max(|x|)` calculado em f64 a partir de `1e-5`;
  `s = (f32)(127 / max)`; `q = nearest_int(x * s)` (arredondamento para
  o par mais proximo) limitado a [-128, 127]; e tambem `sum(q)`. O
  produto e `(sum(code * q) - sum(q)) / s * w_scale` -- equivalente ao
  produto inteiro de `code - 1` com `q`, exato em i32.
- **Forward, por bloco.** `attn_norm` (RMS) -> produtos `i2_s` separados
  de q, k, v -> RoPE NEOX em q, k (dim 128, base 500000) -> attention,
  GQA 20/5 heads, head dim 128, escala `1/sqrt(128)` -> `attn_sub_norm`
  (RMS) -> `attn_output` -> residual -> `ffn_norm` (RMS) -> gate e up ->
  `relu(gate)^2 * up` -> `ffn_sub_norm` (RMS, largura 6912) ->
  `ffn_down` -> residual. Depois `output_norm` e os logits a partir do
  `token_embd` F16. RMS eps 1e-5. Nenhum tensor de escala opcional neste
  arquivo.
- **Tokenizer.** Modelo `gpt2`, vocabulario do Llama 3 (128256 tokens,
  280147 merges), sem `tokenizer.ggml.pre`: o pre-tokenizer `DEFAULT` do
  llama.cpp, que o rusticide ja escolhe quando a chave falta. BOS 128000
  prefixado; EOS 128001; `<|eot_id|>` 128009 tambem encerra a geracao.

## Arquitetura

- **`rusticide-transformer`** depende de `rusticide-model` (o
  `KvCache`), de `rusticide-kernels` e de `rusticide-io` (os valores de
  metadados GGUF). Guarda os blocos de construcao do decoder,
  parametrizados por pequenas structs de configuracao, e nada que nomeie
  uma familia. `rusticide-llama` e `rusticide-bitnet` dependem dela;
  `rusticide-model` nunca depende.
- **Familias como dados.** Uma familia se descreve como um valor
  `Family` (prefixo das chaves da arquitetura, estilo de RoPE, ativacao,
  se os biases da attention sao obrigatorios, se os blocos tem
  sub-norms). `DecoderConfig::from_source` le e valida os hiperparametros
  compartilhados sob esse prefixo e `DecoderWeights::from_source` carrega
  cada peso pelo nome do tensor, entao um crate de familia e o seu
  `Family` mais um loader: nenhum leitor ou loader e copiado entre
  familias.
- **Produtos atras de uma unica costura.** Um bloco chama "projete estas
  ativacoes por este peso". Para o `llama` o peso e um formato de blocos
  GGUF; para o BitNet e `i2_s`, cujo produto primeiro quantiza as
  ativacoes para int8. Os blocos compartilhados nao podem saber qual; quem
  implementa decide se essa costura e uma trait, um enum de tipos de
  peso ou o registry de kernels existente, desde que a quantizacao das
  ativacoes do BitNet fique fora de `rusticide-transformer`.
- **Sub-norms.** `attn_sub_norm` e `ffn_sub_norm` do BitNet sao ganchos
  opcionais nos caminhos compartilhados de saida da attention e do
  feed-forward (ausentes no `llama`), nao um bloco bifurcado.
- **Contrato de formato do `i2_s`.** O tamanho e por tensor
  (`numel/4 + 32`), nao por bloco, e a escala e por tensor. O contrato de
  formato de `rusticide-quantize` ganha o que isso exige (quem implementa
  decide como), mantendo "um formato e um modulo mais uma entrada de
  tabela".
- **lm_head.** O `token_embd` F16 amarrado passa pelo caminho F16
  existente.

## Restricoes

- O AGENTS.md inteiro: TDD estrito, cobertura de 100%, nada de
  `unwrap`/`expect` em codigo de producao, `unsafe` so em
  `rusticide-alloc`/`rusticide-simd`, arquivos com menos de 300 linhas,
  so ASCII, cabecalhos SPDX, MSRV 1.98.
- O arquivo do modelo nunca e commitado; o `RUSTICIDE_MODELS` ganha o
  arquivo do BitNet nas duas maquinas, e o `test:reference` o nomeia.
- As fixtures commitam so ids de token e procedencia (ferramenta,
  commits, flags, SHA-256 do arquivo), nunca bytes do modelo.
- Os binarios de release continuam portaveis: nada de
  `target-cpu=native`.

## Deixado para quem implementa

- O formato exato da API de `rusticide-transformer` e da costura de
  produtos, dentro das restricoes acima.
- Como o kernel `i2_s` se organiza em `rusticide-kernels` (pontos de
  entrada por linha e por lote como os outros formatos, ou menos se o
  forward do BitNet precisar de menos), e suas linhas de bench.
- Como as fixtures sao capturadas dos binarios legados, desde que cada
  uma registre ferramenta, commits e flags.

## Resultados

Preenchidos pela ultima task do marco.
