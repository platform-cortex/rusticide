<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Milestone de desempenho de inferencia (notas e plano)

Versao em ingles:
[2026-10-31-rusticide-inference-performance.md](2026-10-31-rusticide-inference-performance.md).

Status: concluido no branch `perf/inference`, 2026-11-01. O escopo e apenas
desempenho: tokens por segundo do `rusticide run` na CPU, decodificacao e
prefill do prompt, medidos contra o backend de CPU do llama.cpp nos dois
modelos de referencia, com saida gulosa que permanece no caminho do
llama.cpp. Nenhum codigo `unsafe` foi adicionado: todo kernel e Rust seguro
que o compilador vetoriza.

## Benchmark

Apple M1 Pro (6 nucleos de desempenho e 2 de eficiencia, 32 GiB, macOS
arm64), `cargo build --profile profiling --features cli -p rusticide`,
guloso (`--temperature 0`). Modelos: Qwen2.5 0.5B Instruct `Q4_K_M`
(majoritariamente `Q5_0`, com embeddings e saida em `Q8_0`) e TinyLlama 1.1B
Chat `Q4_K_M` (majoritariamente `Q4_K`, com saida em `Q6_K`).

- Decodificacao: `--prompt "The history of the Roman Empire" -n 64`, a linha
  `tokens/s`, tres execucoes.
- Prefill: um prompt de 140 tokens (Qwen) ou 169 tokens (TinyLlama) com
  `-n 1`, tempo de parede menos a mesma execucao com um prompt de uma palavra
  (carga do modelo).
- llama.cpp 0.8.0 (build do Homebrew): `llama-bench -ngl 0`, `tg64` e
  `pp128`. Aqui o melhor numero de threads de CPU dele e 4 (com 8 threads: 95
  e 68 tokens/s).

## Resultados

Decodificacao, tokens/s:

- Qwen2.5 0.5B: 32,7 no inicio, 117 agora; llama.cpp CPU 144 (81%).
- TinyLlama 1.1B: 19,1 no inicio, 82 agora; llama.cpp CPU 97 (85%).
- Uma thread: Qwen 13,8 -> 42 (llama.cpp 53), TinyLlama 10,3 -> 22,9
  (llama.cpp 29,6).

Memoria residente: Qwen 3,6 GB -> 1,05 GB, TinyLlama 5,4 GB -> 1,36 GB.

Prefill, tokens/s (`pp128` do llama.cpp entre parenteses):

- Qwen2.5 0.5B, 140 tokens: cerca de 114 no inicio, 350 agora (652, 54%);
  um prompt de 600 tokens roda a cerca de 340.
- TinyLlama 1.1B, 169 tokens: cerca de 81 no inicio, 152 agora (238, 64%);
  um prompt de 721 tokens roda a cerca de 136.

## Perfis e correcoes

O profiler e o `sample` no processo em execucao, dobrado com
`inferno-collapse-sample`, como no milestone do tokenizador.

1. Os pesos eram ampliados para f32 na carga: o perfil inicial tinha 34% de
   `dot` f32 e 40% de threads rayon ociosas. `Matrix` agora guarda os blocos
   GGUF como lidos (`ModelSource::quantized_tensor`,
   `GgufFile::quantized_tensor`) e as linhas sao agendadas em poucas faixas
   por thread em vez de uma tarefa cada. A memoria caiu 3,5 vezes; a
   velocidade so se moveu quando os kernels mudaram.
2. Kernels inteiros. A ativacao e quantizada uma vez por produto
   matriz-vetor exatamente como o backend de CPU do llama.cpp faz (ggml
   `Q8_0` para os formatos de 32 elementos, `Q8_K` para K-quants: `Q8Input`,
   `Q8Kind`), e cada bloco multiplica em `i8 x i8 -> i32`.
3. Kernels de linha monomorfizados (`int_dot.rs`). Uma chamada indireta por
   bloco de 32 elementos e fatias de tamanho desconhecido em tempo de
   compilacao escondiam os lacos do vetorizador. Uma funcao por formato sobre
   arrays de tamanho fixo deixa o LLVM emitir `sdot` (a CPU padrao de
   `aarch64-apple-darwin` tem dotprod). Detalhes que importaram:
   - os nibbles sao deslocados para valores com sinal (`q - 8`, corrigidos
     pelas somas do bloco), porque o LLVM so escolhe `sdot` quando os dois
     operandos tem extensao de sinal;
   - os K-quants acumulam o bloco inteiro em `i32` e convertem para float uma
     vez, como o ggml faz, o que tambem aproximou a saida gulosa da do
     llama.cpp;
   - as escalas e minimos de `Q4_K` sao desempacotados quatro por vez em
     palavras `u32` com as mascaras do ggml: a extracao das escalas e os
     termos de minimo/deslocamento custavam mais que os produtos escalares
     (ablacao: 19,6 -> 26,3 e -> 31,2 G elementos/s);
   - os quintos bits de `Q5_0` vem da tabela de expansao de 256 entradas do
     ggml.

   Vazao de um nucleo, G elementos/s: `Q4_K` 12 -> 29, `Q6_K` 14 -> 26,
   `Q5_0` 6 -> 21, `Q8_0` 13 -> 37, `Q4_0` 12 -> 33.
4. Menos regioes paralelas, mais baratas. Uma regiao rayon custa cerca de
   18 us para acordar e juntar de fora do pool e 10 us de uma thread do pool,
   pagos 169 vezes por token no Qwen. `matvec_batch` roda Q, K e V em uma
   regiao e gate e up em outra (7 -> 4 regioes por camada), e
   `Llama::forward` roda dentro de `rayon::scope`. A quantizacao da ativacao
   foi reescrita em buffers preallocados (2,3 -> 0,7 ns por elemento).
5. Prefill em lote (`matmul_batch`, `Model::forward_batch`): o prompt passa
   pelo modelo em blocos de 64 tokens, cada linha de pesos multiplicada
   contra todos os tokens do bloco enquanto esta em L1. `Q5_0`, `Q4_K` e
   `Q6_K` desempacotam uma faixa de blocos de pesos uma vez e a reutilizam
   para todos os tokens (`int_dot/rows.rs`); um unico token mantem os
   kernels fundidos. Os logits sao identicos bit a bit a rodar os tokens um
   por vez, e a decodificacao e o caso de um token do mesmo caminho.
6. A atencao roda sobre (token, cabeca) em paralelo: o lote guarda antes
   todas as chaves e valores, depois o token `t` atende as linhas `0..=start+t`
   do cache. Isso tambem ajuda a decodificacao em contextos maiores
   (TinyLlama com 400 tokens: 73 -> 86 tokens/s).

Tentado e revertido: quatro somas parciais independentes por linha (o
armazenamento `index % 4` quebrou a vetorizacao dos formatos de 32
elementos), um laco `Q4_K` totalmente indexado (o compilador vetorizou o laco
externo com cargas por lane).

## Paridade com a referencia

Os fixtures gulosos foram capturados com `llama-server` no backend Metal. Os
proprios backends de CPU e Metal do llama.cpp ja escolhem tokens diferentes
do Qwen2.5 em ate tres passos em dois dos quatro prompts, e o rusticide agora
segue a aritmetica da CPU, entao os fixtures foram recapturados com `-ngl 0`
(o campo de configuracao diz isso). A margem de quase-empate subiu de 0,1
para 0,25 logits: os dois motores arredondam ativacoes para 8 bits. O
TinyLlama confere os 32 tokens de todo prompt; o Qwen confere ate um
quase-empate em cada um.

## Cobertura

O gate de 100% foi relaxado neste milestone, mas nada precisou disso: todo
kernel e verificado contra a referencia dequantizada em blocos aleatorios com
todos os bits de quantizacao variando (uma primeira versao mantinha o bit 6
zerado e perdeu os bits altos; verificacoes de mutacao em `Q6_K` e `Q5_0`
confirmam a nova), e o caminho em lote e verificado bit a bit contra o
sequencial.

## Nao feito

- Kernels de matriz em tiles para o prefill. Desempacotar uma vez removeu
  apenas cerca de 8% do tempo de `Q4_K`: o produto escalar por token e suas
  reducoes dominam. Alcancar o prefill do llama.cpp exige tiles de registrador
  entre linhas e tokens, como fazem os kernels reempacotados dele. `Q4_0` e
  `Q8_0` ainda nao tem kernel em lote.
- Trabalho serial por camada: quantizacao da ativacao, normas, rope e os
  armazenamentos de chave/valor ainda rodam em uma thread.
- Outros alvos. Os binarios de release sao construidos para a CPU base de cada
  alvo: Linux aarch64 nao tem dotprod e x86_64 nao tem AVX2, entao os kernels
  recorrem a instrucoes mais lentas ali. O servidor iMac Intel so tem a
  velocidade total com um build local com
  `RUSTFLAGS="-C target-cpu=native"`; o despacho em tempo de execucao
  exigiria `unsafe`.
- Um pool de threads com espera ativa: o custo restante por regiao e o
  despertar do rayon. Um pool seguro exige todos os buffers compartilhados
  como atomicos; um com `unsafe` pertence a `docs/UNSAFE.md`, uma decisao de
  politica.

## Onde os kernels vivem agora

Depois do milestone da facade de kernels
([design](2026-11-01-rusticide-kernel-facade-design.pt-BR.md)) o codigo
descrito acima passou para `rusticide-kernels`: os kernels de linha estao em
`portable/int_dot*`, o quantizador de ativacao (`Q8Input`, `Q8Kind`) em
`q8_input.rs` e o escalonador (`matmul_batch`, `matvec_batch`) em `matmul.rs`.
`rusticide-quantize` mantem apenas os layouts de bloco e os dequantizadores, e
`rusticide-model` nao depende mais de rayon.
