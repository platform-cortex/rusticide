<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# BitNet b1.58: marco 3 do `rusticide run` (Design)

Status: concluido, 2026-11-13. Refina o marco 3 do
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

Concluido em 2026-11-13 em `feat/bitnet`.

### Paridade com o bitnet.cpp

`mise run test:reference` contra a fixture capturada do bitnet.cpp
`1876a3e` (32 ids gulosos por prompt, BOS 128000 nos ids do prompt):

- "The capital of France is": 32/32 tokens.
- "def fibonacci(n):": 32/32 tokens.
- O prompt em portugues sobre a capital do Brasil (nao ASCII na
  fixture): 32/32 tokens.
- Mesmo resultado no M1 Pro (adaptador `neon` e
  `RUSTICIDE_KERNELS=portable`) e no iMac Pro (`avx2` e `portable`). Todo
  prompt tokeniza para os ids do bitnet.cpp; nenhum quase-empate foi
  necessario.
- Limiar de quase-empate: o harness usa 0.25 (`NEAR_TIE` em
  `crates/rusticide/tests/reference.rs`, desde `12c2290`, 2026-10-31,
  porque os proprios backends do llama.cpp divergem no Qwen2.5), nao o
  0.1 que os Objetivos deste design e o design do run declaram. O BitNet
  nao usou nenhum quase-empate. O TinyLlama bate 3/3 prompts por inteiro;
  o Qwen2.5 para em quase-empates com margens 0.0764, 0.0276, 0.0300 e
  0.0280, todas abaixo de 0.1 tambem.
- Commit de referencia: o design nomeou primeiro o bitnet.cpp `3b04140`;
  seu submodulo (`355c0c4d1`) monta a FFN com SiLU, entao a referencia e
  `1876a3e` com submodulo `1f86f058d`, o ultimo pin com ReLU ao quadrado
  (corrigido em `aca6c4c`; o texto do plano ainda diz `3b04140` e esta
  superado).

### Verificacao da refatoracao

- Digest de logits: por prompt de referencia, um hash dos bits f32 dos
  logits do TinyLlama e do Qwen2.5 depois de um prefill de uma vez, de um
  prefill em dois pedacos (o segundo em `start > 0`) e de 8 passos de
  decode, gravado no M1 (`neon`) em `f87fd24` antes da primeira mudanca
  de lugar. As tasks 3, 4, 5, 9 e 10 bateram com ele bit a bit; os hashes
  do prefill em pedacos e de uma vez sao iguais em todas as linhas.
- O `test:reference` do TinyLlama e do Qwen2.5 nao mudou com a mudanca de
  lugar.
- `bench:gate` contra as baselines de antes do marco, toda linha
  existente dentro da tolerancia (saida 0): iMac Pro no fim da
  refatoracao (Task 5) e de novo na Task 12 (carga de 1 min 1.95); M1 Pro
  com a maquina quieta na Task 12, antes de regravar qualquer baseline
  (carga de 1 min 1.40, na tomada). Linhas e2e do M1, baseline -> medido
  em tokens/s: TinyLlama prefill 470.73 -> 472.18, decode 105.54 ->
  109.67; Qwen2.5 prefill 993.77 -> 993.07, decode 144.57 -> 145.65.

### Velocidade

Tokens/s das baselines regravadas (mediana dos melhores de tres sessoes).
O bench decodifica 64 tokens depois de "The history of the Roman Empire"
e faz prefill de um paragrafo fixo de 143 tokens do BitNet. Threads: o
padrao do rayon, uma por CPU logica (8 no M1 Pro, 20 no iMac Pro). O
`i2_s` roda o kernel portavel sob todo adaptador.

- M1 Pro (`neon`): BitNet decode 8.65, prefill 10.92. A meta de 2
  tokens/s de decode foi atingida (7.94 no checkpoint da Task 10).
- iMac Pro, Xeon W-2150B (`avx2`): BitNet decode 13.22, prefill
  26.81. Um servico externo no iMac Pro (um job do launchd que reinicia
  um processo node a ate 150% de CPU) perturbou muitas sessoes, entao a
  baseline nao e um `bench:baseline` simples. Nove sessoes rodaram em
  tres tentativas; a baseline e o `rusticide-bench baseline` sobre tres
  delas, escolhidas depois de ver seus resultados:
  - a sessao 1 de um `bench:baseline` em sequencia (carga de 1 min 1.77
    no inicio, sem checagem de CPU); suas sessoes 2 e 3 cairam 25-30% e
    foram descartadas;
  - as sessoes 2 e 3 de uma terceira tentativa, cada uma iniciada com
    carga de 1 min 2.19 mas com o node a 46.5% e 64.1% de CPU (a segunda
    falha a propria checagem "nenhum processo acima de 50%" da
    tentativa); sua sessao 1 foi descartada;
  - as tres sessoes de uma segunda tentativa (so checagem de carga)
    foram descartadas.
  Seis de nove sessoes foram descartadas. Escolher pelo resultado puxa a
  baseline para cima: o prefill do TinyLlama e 403.02, 13% acima da rodada
  do gate da Task 12 (357.07) e 10% acima da baseline anterior (367.66). O
  prefill do Qwen2.5 e 654.10, 16.6% acima da baseline anterior (561.02),
  o maior salto, e em linha com a rodada do gate (658.26). Os kernels de
  linha `Q4_0`, `Q6_K` e `Q8_0` estao 8-15% acima da baseline anterior sem
  mudanca nesses kernels, e em linha com a rodada do gate (G/s, baseline
  contra gate: 6.60 e 6.75, 18.39 e 18.44, 30.10 e 30.01; baseline
  anterior 5.76, 16.97 e 27.61). Os valores sao um teto de maquina quieta
  ainda nao mostrado reproduzivel; um gate futuro no iMac Pro pode falhar
  nessas linhas enquanto o servico rodar.
- bitnet.cpp no iMac Pro (`1876a3e`, o build das fixtures, Apple clang 17,
  so CPU, kernel mad `i2_s` x86), carga de 1 min 1.91 no inicio:
  `llama-bench -m bitnet-b1.58-2b-4t-i2_s.gguf -p 143 -n 64 -t 10,20 -r 5`.
  `-t 10`: pp143 252.46 +- 54.77, tg64 42.24 +- 3.54. `-t 20`: pp143
  230.47 +- 86.10, tg64 29.55 +- 13.34. Uma segunda rodada (cargas de 1, 5
  e 15 min 0.82, 3.06 e 5.12, node a 31.9% de CPU no inicio): `-t 10`
  pp143 204.18 +- 48.55, tg64 42.34 +- 3.04; `-t 20` pp143 192.99 +-
  95.54, tg64 30.56 +- 12.69. O decode e estavel; o prefill varia muito
  nas duas rodadas.
- Contra o melhor do bitnet.cpp (`-t 10`, o maior pp143), o rusticide
  chega a 31% do decode e 11% do prefill dele no iMac Pro.
- Nao ha numero do bitnet.cpp para o M1: o arquivo publicado usa o
  layout `i2_s` x86 de 128 pesos, o caminho ARM do fork legado le outro
  layout, e o caminho ARM do HEAD esta quebrado (ver "Referencia: qual
  bitnet.cpp").
- Linhas de kernel (G pesos/s, uma thread): `I2_S` row 3.73 no M1 Pro e
  8.68 no iMac Pro, contra `Q8_0` row 59.69 e 30.10 com seus kernels SIMD;
  `rows8` nao e mais rapido que `row` (3.73 no M1 Pro, 8.80 no iMac Pro).
  No M1 Pro e por isso que o prefill mal supera o decode (10.92 contra
  8.65); no iMac Pro o prefill e cerca do dobro do decode (26.81 contra
  13.22).
- Pico de memoria no M1 Pro: 4.4 GB (checkpoint da Task 10).

### Proximos passos

- Kernels SIMD `i2_s` (NEON `dotprod`, AVX2) com caminho de varias
  linhas: a distancia para o bitnet.cpp acima e o kernel.
- Atencao e RoPE: o Marco B do design de adaptadores de runtime, o
  proximo passo (ordem do dono, 2026-11-12).
- Alinhar o limiar de quase-empate declarado nos designs (0.1) com o do
  harness (0.25), ou apertar o harness.
- Em aberto (dono, no merge): aceitar a baseline do iMac Pro acima ou
  regrava-la depois que o servico openclaw for parado.
- Minors adiados para a revisao final triar (do ledger das tasks):
  - `Decoder::new` nao confere os formatos de `DecoderWeights` contra a
    config (os loaders conferem).
  - A ordem dos erros para metadados com varias falhas mudou; um teste
    com `1 << 40` assume `usize` de 64 bits; uma contagem zero de tokens
    em `layout.rs` nao e verificada.
  - Docs: o limite de paridade de `round_row_to_i8` (`|x*s| <= 4194303`,
    NaNs padrao); `Q8Input.kind` "set by quantize"; os bytes por linha de
    `Weights::Ternary` sao multiplo de 32; `Q8Kind::Row` tira o maximo
    sobre blocos inteiros (igual para as larguras do BitNet).
  - O `copy_row` de `Ternary` escala o `out` inteiro e repete o laco de
    `apply_trailer`; uma mensagem `InvalidTensor` enganosa para buffer
    curto (inalcancavel pelo `rusticide-io`).
  - O `dequantize_row_i2_s` upstream mapeia o codigo 3 para 0 enquanto
    seu kernel mad da +2; o rusticide da +2 em todo lugar (arquivos reais
    nao tem codigo 3).
  - Duas fontes para o prefixo das chaves de metadados
    (`Family.architecture` e `general.architecture`); o leitor de
    hiperparametros e o loader do transformer so sao testados via `llama`
    e `bitnet`; `LlamaConfig` espelha `DecoderConfig`; nenhum teste do
    `bitnet` carrega um tensor `i2_s` real.
  - `mise run fuzz run <target> -- -max_total_time=60` perde o `--`
    interno.
  - Linhas acima de 80 colunas: `docs/ARCHITECTURE.md:167` e a doc do
    alvo de fuzz `llama_load.rs`.
  - O `rusticide-bench` monta o registro `llama` + `bitnet` ele mesmo, a
    mesma linha do `models::registry()` da fachada.
