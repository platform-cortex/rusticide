<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Adaptador de kernels NEON (Design)

Status: concluido, 2026-11-02. Segue o
[kernel facade](2026-11-01-rusticide-kernel-facade-design.pt-BR.md)
(release 0.7.0) e o patch de endurecimento do GGUF (release 0.7.1).
Branch `perf/prefill`.

## Intencao

O prefill do prompt roda a cerca de metade da velocidade de CPU do
llama.cpp no M1 Pro: Qwen2.5 0.5B cerca de 350 tokens/s contra 650-735,
TinyLlama cerca de 148 contra 280-320 (llama.cpp 0.8.0
`llama-bench -ngl 0`, melhor numero de threads; o perfil dele nao mostra
BLAS, o trabalho e de kernels de CPU). Um perfil de prefill do TinyLlama
coloca 62% do tempo nos kernels de matriz quantizados, cerca de 16% na
atencao e cerca de 14% em espera de threads; o trabalho serial por camada
e pequeno sozinho, mas deixa os outros workers parados.

Tres spikes descartaveis (2026-11-01) mostraram que a diferenca nao fecha
em Rust seguro:

- Os kernels do llama.cpp (`ggml_gemm_q4_K_8x4_q8_K`) intercalam varias
  linhas de peso para que um `sdot` acumule uma lane por linha, sem
  reducao horizontal. Escrito como Rust seguro escalar, esse layout vira
  multiplicacoes alargadas: 5.2 G multiplicacoes-somas/s contra 31.5 do
  formato atual.
- Ladrilhar linhas x tokens no layout atual ajuda produtos `i8` simples em
  ate 42%, de forma instavel, e nada quando entra a aritmetica real de
  bloco `Q4_K`: o kernel atual de uma linha contra muitos tokens ja e o
  formato mais rapido.

O dono escolheu o adaptador nativo para o qual o facade foi feito:
intrinsecos NEON atras de wrappers seguros em `rusticide-simd`, e um
adaptador `neon` em `rusticide-kernels`, compilado em todo alvo que
garante NEON e dotprod.

## Sequencia

O milestone entra em tres partes, cada uma atras do `bench:gate`:

1. **Trabalho paralelo por token** (Rust seguro). As secoes seriais por
   camada (quantizacao Q8 das ativacoes, `to_token_major`, RoPE,
   `rms_norm`, copias) custam cerca de 15-17% do prefill com sete workers
   parados. Cada uma e independente por token e roda no pool.
2. **Amostragem nos sobreviventes do top-k** (Rust seguro). O
   `apply_top_p` ordena o vocabulario inteiro a cada token (cerca de 15%
   do decode do Qwen com amostragem ligada); depois do top-k, temperatura,
   softmax, top-p e o sorteio rodam sobre os k pares (indice, logit)
   mantidos, e os logits nao sao copiados.
3. **O adaptador `neon`**, o resto deste documento.

As partes 1 e 2 vem primeiro porque sao baratas, e porque depois da parte
3 a fatia serial cresce: com kernels 1.5x mais rapidos ela seria cerca de
22% do prefill.

## Objetivos

1. **Kernels de matriz na velocidade do llama.cpp no M1 Pro**: os kernels
   de prefill `Q4_K`, `Q6_K`, `Q5_0` e `Q8_0` alcancam a vazao por thread
   dos kernels reempacotados do llama.cpp (cerca de 51 G
   multiplicacoes-somas/s por thread, contra 33 hoje). Como os kernels sao
   62% do prefill, isso sozinho leva o prefill de cerca de 50% para cerca
   de 65% do llama.cpp; *decisao minha:* 65% do melhor `pp128` do
   llama.cpp e o piso ponta a ponta depois da parte 3, as partes 1 e 2
   somam a isso, e a atencao (proximo milestone) fica fora do escopo.
2. **Decode nao mais lento**, e mais rapido onde os mesmos kernels se
   aplicam.
3. **Todo binario de release continua rodando em qualquer lugar**: o
   adaptador `neon` so e compilado para alvos que garantem NEON e dotprod
   em tempo de build; em todo o resto roda o `portable`, como hoje.
4. **`unsafe` fica dentro de `rusticide-simd`** sob a politica existente
   (`docs/UNSAFE.md`): API publica segura, um comentario `// SAFETY:` por
   bloco, justificativa no README, entrada no `docs/UNSAFE.md`, Miri no CI
   com fallbacks escalares sob `cfg(miri)`.

## Nao-objetivos

- AVX2, AVX-512 ou qualquer adaptador x86 (o servidor Intel): proximo
  milestone, na mesma costura.
- SVE, i8mm (`smmla`) ou outras extensoes ARM alem de dotprod.
- Atencao ou exp mais rapidas, e tabelas de RoPE, exceto onde uma mudanca
  de kernel as traga de graca.
- Compartilhar os pesos com o buffer do arquivo em vez de copia-los.
- GPU ou Metal.

## Anti-objetivos

- **`unsafe` em qualquer lugar fora de `rusticide-simd`.**
  `rusticide-kernels` e todo crate de modelo mantem
  `unsafe_code = "deny"`.
- **Um panic ou comportamento indefinido alcancavel pela entrada de quem
  chama.** A API segura torna leituras fora dos limites irrepresentaveis
  (referencias a arrays de tamanho fixo), e a unica chamada para codigo
  compilado para uma feature se apoia no alvo de build garantir essa
  feature.
- **Paridade mais fraca.** `mise run test:reference` passa com o NEAR_TIE
  atual (0.25) com o adaptador `neon` ativo, e dentro de um adaptador um
  prefill em lote continua identico bit a bit a rodar os tokens um por
  um.
- **Um `portable` mais lento.** O gate continua protegendo o caminho de
  fallback.

## Arquitetura

- **`rusticide-simd`** (crate novo, `unsafe` permitido): os kernels NEON,
  compilados com `#[target_feature(enable = "neon,dotprod")]`, atras de
  pontos de entrada seguros que recebem arrays de blocos de tamanho fixo e
  slices de saida. Seu unico `unsafe` e a chamada para cada kernel
  compilado com a feature (ver "Achados"). Em outros alvos o modulo nao
  existe. Sob `cfg(miri)` cada kernel tem um corpo escalar com os mesmos
  resultados inteiros, para que seus testes rodem no Miri.
- **O adaptador e dono do layout dos pesos.** O adaptador `neon`
  reempacota cada matriz quantizada uma vez no load para o seu layout
  intercalado (varias linhas lado a lado, como os formatos reempacotados
  do llama.cpp), entao o facade precisa de uma porta nova: preparar uma
  matriz para o adaptador ativo. A preparacao do `portable` e a
  identidade. O decode usa um kernel matriz-vetor no mesmo layout, entao
  ha uma copia dos pesos, nao duas. Hoje o `Matrix::load` ja copia cada
  tensor do buffer do arquivo inteiro (`data.to_vec()`); o reempacotamento
  escreve nessa copia (um grupo intercalado de N linhas tem exatamente os
  bytes de N blocos, como no `block_q4_Kx8` do llama.cpp), entao nao
  acrescenta memoria.
- **Os formatos vem de um arquivo nao confiavel.** Uma matriz cujo numero
  de linhas nao e multiplo do grupo de intercalacao, ou cujos bytes nao
  fecham em blocos inteiros, fica sem preparar e roda no `portable`; a
  preparacao nunca da panic e nunca le alem de um tensor.
- **Formatos**: os quatro que os modelos de referencia usam, em ordem de
  tempo gasto: `Q4_K` e `Q6_K` (TinyLlama), depois `Q5_0` e `Q8_0`
  (Qwen2.5). Um formato sem kernel no adaptador `neon` cai no kernel
  `portable` para aquela matriz.
- **Selecao**: `active()` escolhe `neon` onde ele estiver compilado, senao
  `portable`. *Decisao minha:* uma variavel de ambiente
  (`RUSTICIDE_KERNELS=portable`) forca o `portable`, para que o gate e os
  testes de referencia comparem adaptadores numa maquina e um usuario
  possa descartar o adaptador ao investigar um problema. Qualquer outro
  valor e reportado e ignorado, nunca um panic.
- **Equivalencia**: as somas inteiras de `neon` e `portable` sao exatas e
  devem bater; a ordem de acumulacao em float pode diferir, entao os
  testes entre adaptadores comparam com a referencia dequantizada dentro
  de uma tolerancia declarada, e a paridade de referencia e a checagem
  ponta a ponta.

## Achados que moldam o design

Sondas no toolchain fixado (2026-11-01):

- Intrinsecos NEON sao seguros de chamar dentro de uma funcao marcada com
  `#[target_feature(enable = "neon,dotprod")]`, e o `bytemuck` (feature
  `aarch64_simd`) converte `[i8; 16]` em `int8x16_t` sem `unsafe` no nosso
  codigo. Entao os kernels podem ser Rust seguro comum compilado para
  essas features, e nenhuma leitura por ponteiro e necessaria.
- Chamar uma funcao dessas a partir de codigo sem o atributo exige um
  bloco `unsafe`, mesmo quando o build liga as features globalmente. Essa
  chamada e o unico `unsafe` de que o design precisa, e ela fica em
  `rusticide-simd`, cujos pontos de entrada seguros sao os proprios
  kernels: `rusticide-simd` guarda os kernels NEON (nao so wrappers de
  vetor, que nao fariam inline atraves da fronteira), e o adaptador
  `neon` em `rusticide-kernels` os chama de forma segura.
- *Decisao minha:* o argumento de solidez e uma garantia de build, nao
  deteccao em tempo de execucao: o modulo so compila sob
  `cfg(all(target_arch = "aarch64", target_feature = "dotprod"))`, que o
  `aarch64-apple-darwin` satisfaz por padrao e os alvos Linux aarch64 nao.
  Os binarios de release Linux aarch64 ficam no `portable` neste
  milestone; deteccao em tempo de execucao ali e uma decisao posterior e
  separada.
- `vdotq_s32` e similares so sao estaveis a partir do Rust 1.98. O dono
  definiu a regra de que o MSRV e o maior entre o alvo do rust-skills
  (1.96) e a menor versao que uma feature entregue exige, entao este
  milestone sobe o MSRV para 1.98 (registrado no AGENTS.md). Sem sonda de
  versao do compilador nem `build.rs`: todo compilador suportado compila
  o adaptador `neon`.
- O Miri nao executa esses intrinsecos, e o bloco `unsafe` e uma chamada,
  nao um acesso a memoria. *Decisao minha:* o job do Miri interpreta os
  testes do `rusticide-simd` para `aarch64-apple-darwin` (para que o
  modulo seja compilado) com os corpos escalares de `cfg(miri)`, num
  nightly datado e fixado como na tarefa de cobertura, e vira um gate no
  `cicd.yml`, como o `docs/UNSAFE.md` exige da primeira mudanca com
  `unsafe` (hoje o Miri so roda no `nightly.yml`).
- O CI ganha um job `arm` no runner Linux arm64 do GitHub
  (`ubuntu-24.04-arm`, gratuito para este repositorio publico; nucleos
  classe Neoverse com dotprod) que roda `mise run ci:quick`, com o gate de
  cobertura, com `RUSTFLAGS="-D warnings -C target-feature=+dotprod"` para
  que o modulo `neon` seja compilado. O `RUSTFLAGS` substitui o
  `-D warnings` do `.cargo/config.toml` em vez de somar, entao o job o
  repete. O flag existe so nesse job de teste, nunca na configuracao de
  build ou de release versionada. O job x86_64 `gates` continua cobrindo
  o `portable`; o `mise run ci:quick` do M1 cobre os dois localmente. O
  job novo entra nos checks obrigatorios do ruleset da `main`.
- O M1 Pro tem 6 nucleos de performance e 2 de eficiencia, e o pool roda
  8 threads; o llama.cpp e mais rapido com 6. *Decisao minha:* o numero de
  threads e medido de novo depois da parte 1, nao mudado as cegas.

## O gate

O `bench:gate` mede o adaptador ativo. Espera-se que cada parte melhore
as linhas e2e alem da tolerancia; depois de cada uma, a baseline do M1 e
regravada (uma mudanca revisada) para travar o ganho. As metricas de
kernel mantem as chaves e medem os kernels do adaptador ativo.
`RUSTICIDE_KERNELS=portable mise run bench:gate` precisa continuar
passando contra a baseline gravada depois da parte 2, provando que o
fallback nao regrediu.

## Restricoes

Tudo no AGENTS.md vale (TDD estrito, 100% de cobertura, arquivos com
menos de 300 linhas, ASCII, cabecalhos SPDX, `-D warnings`, MSRV 1.98 pela
regra acima). Cobertura e Miri: o tarpaulin mede o build `neon` no M1 e no
job `arm` do CI; o Miri roda os corpos de `cfg(miri)` no CI; runners
x86_64 compilam sem o adaptador e cobrem o `portable`.

## Deixado para quem constroi

- A intercalacao exata (linhas por grupo, bytes por lane), o formato do
  ladrilho e o layout das ativacoes, seguindo os kernels reempacotados do
  llama.cpp onde couberem.
- O conjunto de operacoes de wrapper no `rusticide-simd` e seus nomes.
- A assinatura da porta de preparacao e onde ficam os dados empacotados.
- Os valores de tolerancia entre adaptadores.
- Se `Q5_0`/`Q8_0` chegam na mesma estrutura ou num kernel ladrilhado
  mais simples, desde que o Qwen2.5 cumpra o objetivo 1.
- Como a parte 1 divide o trabalho serial no pool (por token, norma e
  quantizacao fundidas, ou dobrado na proxima regiao paralela).

## Resultados

Parte 1, Apple M1 Pro (8 nucleos), tokens/s, `PARALLEL_MIN_TOKENS` = 2
(inalterado). Antes e a baseline anterior; depois e a baseline regravada
(mediana de tres sessoes).

- Prefill do Qwen2.5: 348,48 antes, 426,03 depois (+22%).
- Decode do Qwen2.5: 131,01 antes, 132,60 depois (dentro da tolerancia).
- Prefill do TinyLlama: 148,00 antes, 166,82 depois (+13%).
- Decode do TinyLlama: 96,85 antes, 96,36 depois (dentro da tolerancia).
- Os numeros de contagem de threads vem de execucoes separadas de
  `mise run bench` e nao sao comparaveis com as linhas do baseline acima.
- Contagem de threads, `RAYON_NUM_THREADS=6` (melhor): prefill do Qwen2.5
  405,53, decode 117,98; prefill do TinyLlama 154,23, decode 80,04.
- Contagem de threads, `RAYON_NUM_THREADS=8` (melhor): prefill do Qwen2.5
  422,72, decode 117,10; prefill do TinyLlama 159,25, decode 89,63.
- Decisao: manter o padrao (8 threads); 8 threads e mais rapido ou igual
  em todas as linhas, exceto o decode do Qwen2.5 (117,98 contra 117,10,
  menos de 1%).

Parte 2, Apple M1 Pro (8 nucleos), tokens/s. O decode com amostragem e
`rusticide run` no Qwen2.5 com `--temperature 0.8 --seed 1 --max-tokens
256`, tres execucoes alternando antes (commit 6e56270) e depois.

- Decode com amostragem do Qwen2.5: 100,08, 105,03, 105,56 antes; 117,25,
  117,51, 117,44 depois (cerca de +12%).
- Baseline regravado: prefill do Qwen2.5 426,03 -> 424,68, decode 132,60 ->
  130,38; prefill do TinyLlama 166,82 -> 167,55, decode 96,36 -> 96,25. O
  bench guloso nao exercita a amostragem, entao sao variacoes de ruido.

Parte 3, Apple M1 Pro (8 nucleos), adaptador NEON (dotprod, escolhido em
tempo de build em aarch64-apple-darwin). Antes e o baseline da parte 2
(kernels portateis); depois e o baseline regravado. As metricas de kernel
sao G/s: pesos (multiplicacoes-acumulacoes) por segundo em uma thread, como
impressas pelo gate.

- Kernel Q4_K: row 29,43 -> 35,10, rows8 29,94 -> 90,25.
- Kernel Q6_K: row 25,94 -> 28,43, rows8 26,81 -> 79,99.
- Kernel Q5_0: row 20,28 -> 29,24, rows8 24,37 -> 81,48.
- Kernel Q8_0: row 35,04 -> 59,53, rows8 35,43 -> 97,18.
- Prefill do Qwen2.5: 424,68 antes, 1016,21 depois (+139%); decode 130,38
  antes, 146,21 depois (+12%).
- Prefill do TinyLlama: 167,55 antes, 468,82 depois (+180%); decode 96,25
  antes, 102,82 depois (+7%).
- O fallback portatil (`RUSTICIDE_KERNELS=portable`) passa o gate contra o
  baseline da parte 2: sem regressao.
- llama.cpp (`llama-bench -ngl 0 -t 6,8 -p 128,512 -n 0 -r 5`, build
  8642541b2, backends BLAS+Metal). A referencia e `-t 6`: `-t 8` e bem mais
  lento e ruidoso neste M1 Pro 6P+2E porque duas threads caem nos nucleos de
  eficiencia. TinyLlama pp128 318,20 +- 4,62 (t8 273,26), pp512 281,36 (t6).
  Qwen2.5-0.5B pp128 599,26 +- 25,91 (t8 340,41), pp512 722,82 +- 17,44 (t8
  475,82). Uma medicao anterior de execucao unica (286,71 e 640,46) era
  ruidosa e foi substituida.
- O prompt de prefill do nosso bench e um paragrafo fixo de cerca de 150
  tokens, entre pp128 e pp512, entao ambos sao dados. TinyLlama: 468,82 /
  318,20 = 147% do pp128 (o pp512 e menor que o pp128, entao 147% e o valor
  conservador). Qwen2.5: 1016,21 / 599,26 = 170% do pp128, 1016,21 / 722,82
  = 141% do pp512. Destaque (conservador): 141% no Qwen2.5 e 147% no
  TinyLlama. O Objetivo 1 (pelo menos 65%) foi atingido.
- Proximo passo: especializacao de decode com N=1 para o Q4_K.
- Proximo passo: os builds de release Linux aarch64 ainda rodam portateis
  (deteccao de dotprod em tempo de execucao fica para uma decisao futura).
- Proximo passo: tabelas de attention e RoPE sao o proximo marco.
