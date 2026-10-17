# Reflexo Estética · Landing e cadastro · conteúdo, regras e estados

Versão 1.0 · 10 de outubro de 2026

**O que este documento é.** O inventário de conteúdo da landing do vertical e do fluxo de cadastro:
o que cada superfície precisa dizer, o que não pode dizer, quais campos existem, quais regras valem
e quais estados cada tela tem, com o texto atual de cada um. Fonte: o código em produção nesta data,
e as decisões registradas no Norte v0.60 e no Produto v1.0, que vão anexos.

**O que este documento não é.** Instrução de layout, ordem visual, hierarquia ou estilo. Isso é do
design system. Onde aparece "texto atual", é material de partida: pode ser reescrito, desde que
respeite as regras do capítulo 1.

---

## 1. Regras que valem nas duas superfícies

**Marca.** O cliente vê só "Reflexo Estética" (e "Reflexo" como forma curta). Nenhuma outra empresa
ou marca aparece.

**Posicionamento.** O Reflexo se vende como captação de pacientes com peso técnico e em
conformidade, com a profissional no controle. Não se vende como "simulação com IA": gerar imagem é
commodity, e a conversa que começa por ela termina em preço. O que diferencia é a revisão
obrigatória com assinatura, o registro de tudo e o acompanhamento do caso.

**Vocabulário.**
- Usar: simulação, projeção aproximada, expectativa, revisão, aprovação, assinatura, profissional
  habilitada(o).
- Nunca: previsão, resultado garantido, "antes e depois" ou "revelação" sobre simulação,
  transformação, diagnóstico, indicação de tratamento, decisão clínica.
- A garantia que se pode dar é "a identidade da pessoa é preservada". Não usar "fora da área nada
  muda" (promessa encerrada).

**Proibido em qualquer texto.**
- Promessa ou garantia de resultado.
- Dizer que a revisão humana elimina o erro da IA. Ela reduz, e não se afirma mais que isso.
- Dose, ml, volume ou preço de procedimento.
- Depoimento de paciente sobre resultado de procedimento.
- Sensacionalismo, apelo à autoestima, concurso ou sorteio.
- A frase "não existe equivalente" ou qualquer variação.
- Vender o plano estrutural (o raciocínio de "o que se estrutura para a queixa") como promessa: ele
  hoje existe só para um cliente, atrás de chave de liberação.

**Imagens.**
- Rosto de banco de imagem licenciado ou de pessoa com autorização de uso de imagem assinada.
  Nunca foto de paciente, nunca captura de tela do ambiente real com dado de alguém.
- Simulação só aparece na versão sintética da demonstração (rosto sintético, imagem fixa), sempre
  com a marca de IA visível na própria imagem.
- Antes e depois com foto real não é usado.

**Tratamento.** Dr. ou Dra., conforme quem revisa. Texto genérico usa "a profissional" ou
"profissional habilitada(o)".

---

## 2. Landing (`estetica.reflexo.health`)

### 2.1 Para quem e para quê

**Quem lê:** a profissional que faz procedimento estético visível e a clínica onde ela atua, antes
de falar com alguém. Ela pesquisa antes de responder, então a landing é fator de credibilidade
pré-conversa.

**O que a landing precisa levar a fazer:**
1. **Agendar uma demonstração** (formulário de contato seguido da agenda, seção 2.2 item 9). É a
   ação principal hoje.
2. **Ver a demonstração aberta** sem cadastro (item 8).
3. **Contratar**, indo para o cadastro (capítulo 3).
4. **Entrar**, para quem já é cliente (vai para `app.reflexo.health/entrar`).

### 2.2 Conteúdo que precisa existir

1. **Promessa.** Antes de decidir, a paciente quer se ver no resultado. Ela envia a própria foto pelo
   link da clínica e recebe uma projeção aproximada no rosto dela, que só chega depois que uma
   profissional habilitada revisa e assina.

2. **Para quem serve, com a condição junto.** Três habilitações nomeadas, cada uma com o registro que
   a sustenta: médica com registro no CRM; cirurgiã-dentista com especialidade em harmonização
   orofacial registrada no CRO; enfermeira esteta com especialização registrada no COREN. A lista
   declara que são as habilitações atendidas hoje, com outras em avaliação. Esteticista não entra na
   comunicação (pode usar o produto, mas não é público da landing).

3. **Como funciona**, na ordem real: a paciente acessa o link da clínica, se identifica, responde à
   triagem, consente e envia as fotos; o Reflexo gera a simulação; a profissional revisa e aprova,
   pede nova simulação ou encerra; a paciente recebe a projeção identificada como simulação, com nome
   e registro de quem aprovou; a clínica segue acompanhando o caso até o agendamento.

4. **A revisão obrigatória.** Nada chega à paciente sem aprovação de profissional habilitada para
   aquele serviço, e cada entrega sai com nome e registro de quem assinou. A fila de revisão é o
   produto na cabeça do comprador.

5. **O argumento do registro.** Texto travado em três frases, nesta ordem, e a terceira não pode
   sair: a conversa sobre expectativa acontece de qualquer jeito e hoje não deixa registro; a
   jurisprudência trata a obrigação como de resultado, e cabe à profissional demonstrar que agiu
   corretamente; o Reflexo não promete o desfecho de disputa nenhuma. O registro exportado descreve
   o que reúne e não afirma valor de prova.

6. **O caso não termina na imagem.** Histórico de cada pessoa e caso, estado de atendimento
   (solicitou, recebeu, precisa de retorno, agendou, concluído), links diferentes por canal
   (Instagram, Google, campanha, indicação) com a origem de cada pedido, e saída dos dados. O Reflexo
   não substitui sistema de gestão nem prontuário.

7. **Planos, preço e calculadora** (tabela no capítulo 4).
   - A calculadora pede o ticket médio e o volume esperado antes de mostrar os planos, para o preço
     chegar ancorado na conta da própria profissional.
   - Ela calcula ponto de equilíbrio: quantos procedimentos pagam a mensalidade. Não projeta
     conversão e não usa a palavra "retorno".
   - Mostra o plano indicado pelo volume. Rede de clínicas fica fora da tabela, como conversa.
   - Regra de preço de fundador explicada (capítulo 4).

8. **Demonstração aberta** (`/demo`). O fluxo inteiro, como paciente e como profissional, com
   simulações fixas de rosto sintético e marca de IA. Sem upload de foto, sem cadastro, não cobra.
   Declara o que a entrega real tem a mais: a etapa educativa obrigatória que a paciente percorre.

9. **Agendar demonstração.** Formulário, e depois de enviado, a agenda (Cal.com) já preenchida com os
   dados. Campos atuais, todos obrigatórios:
   - Nome
   - Nome da clínica
   - WhatsApp
   - E-mail
   - Quantos profissionais habilitados atuam hoje na clínica? (1 · 2 a 3 · 4 a 10 · Mais de 10)
   - Sua clínica já recebe pacientes interessados em procedimentos de estética facial com
     frequência? (Sim, diariamente · Sim, algumas vezes por semana · Algumas vezes por mês · Ainda
     estamos estruturando essa demanda)
   - Em quanto tempo você pretende implementar uma solução como a Reflexo na sua clínica? (Agora ·
     Nos próximos 30 dias · Nos próximos 3 meses · Estou apenas pesquisando)
   - Aviso: "Ao enviar, você concorda que seus dados serão utilizados para contato comercial,
     conforme a Política de Privacidade."
   - Estados: enviando ("Enviando..."), erro ("Não foi possível enviar agora. Tente novamente em
     instantes."), enviado com a agenda aberta.

10. **Perguntas frequentes** (texto atual na seção 2.3).

11. **Rodapé.** Política de privacidade e termos (vivem em `reflexo.health`), `contato@reflexo.health`,
    `privacidade@reflexo.health`, Instagram e Facebook da marca, link "Entrar".

12. **Cookies.** Banner de consentimento com recusa por padrão: aceitar e recusar com o mesmo peso,
    nada pré-marcado. Medição de campanha só roda depois do aceite.

### 2.3 Texto atual da landing (no ar, entregue pela agência)

Material de partida. Os itens 2, 5, 7 e 8 da seção 2.2 não existem no texto atual.

**Título da aba:** Reflexo Estética · Simulação visual personalizada, com revisão profissional.
**Frase de rodapé:** Simulação personalizada, sempre revisada por profissional habilitado.
**Menu:** Como funciona · Funcionalidades · Perguntas frequentes · Contato · Entrar · Agendar demo.

**Abertura**
- Título: Antes de decidir, sua paciente quer se ver no resultado.
- Texto: Ela envia a própria foto e recebe uma projeção aproximada no rosto dela, revisada por um
  profissional habilitado antes de chegar até ela.
- Ação: Agende uma demonstração.
- Apoio: Depois da entrega, sua equipe acompanha o caso até o agendamento.
- Cartão ilustrativo: "Simulação pronta" · "Prévia personalizada da paciente disponível para
  revisão." · "Revisada por profissional habilitado".

**Por dentro do sistema** · Veja a Reflexo por dentro
- Telas reais do portal: a simulação no rosto da paciente, a revisão antes da entrega e o histórico
  de cada caso.
- Simulação: a foto original e a projeção lado a lado. O profissional aprova, pede nova simulação ou
  encerra o caso.
- Fila de revisão: casos aguardando revisão, do mais antigo ao mais novo. O profissional assume o caso
  e decide antes de qualquer entrega.
- Casos: cada caso com procedimento, origem e estado: na fila, em revisão, aguardando foto nova ou
  encerrado.
- Atenção: "telas reais" só com dado fictício e simulação sintética.

**Mais do que gerar uma imagem**
- Cada simulação carrega revisão profissional, histórico do caso e a origem de onde veio o lead.
- Revisão profissional obrigatória: antes de qualquer entrega, a simulação passa por um profissional
  habilitado. A IA gera; quem aprova é o profissional.
- Controle total sobre cada caso: o profissional pode aprovar, solicitar uma nova geração, pedir
  outra foto ou encerrar o caso. A decisão final é sempre dele.
- Acompanhamento comercial: veja quem solicitou, já recebeu, precisa de follow-up, agendou ou teve o
  caso concluído, com o histórico daquela oportunidade dentro da operação.
- Origem e organização: crie links diferentes por canal (Instagram, Google, campanhas, indicações) e
  acompanhe de onde vêm as solicitações de simulação.

**Resultado de outra pessoa não responde: "E em mim?"**
- A referência pode ser de outra pessoa, mas a dúvida continua sendo sobre o próprio rosto. A Reflexo
  responde com uma projeção no rosto dela.
- Personalizada: feita a partir da foto da própria paciente.
- Aproximada: uma projeção, não uma promessa de resultado.
- Revisada por profissional habilitado antes da entrega.

**Da dúvida da paciente ao acompanhamento do caso** (cinco passos)
1. A paciente acessa o link da clínica: informa os dados, responde à triagem, concede os
   consentimentos e envia a própria foto.
2. A Reflexo gera a simulação: uma projeção aproximada é criada para o procedimento de interesse.
3. O profissional revisa: a simulação passa obrigatoriamente por um profissional habilitado antes de
   qualquer entrega.
4. A paciente recebe: depois da aprovação, a projeção é entregue identificada como simulação e
   vinculada ao profissional responsável.
5. A clínica continua acompanhando: histórico, status, follow-up, agendamento e evolução da
   oportunidade seguem na plataforma.

**Comparativo do processo**
- Sem a Reflexo: a clínica explica o procedimento; mostra referências de outros casos; a paciente
  tenta imaginar como aquilo poderia ficar nela.
- Com a Reflexo: a paciente envia a própria foto; recebe uma projeção aproximada, revisada por um
  profissional antes da entrega; a clínica acompanha o caso depois da simulação.
- Atenção: comparativo de processo, nunca de imagem.

**Perguntas frequentes**
- A Reflexo mostra exatamente como a paciente vai ficar? Não. A Reflexo trabalha com projeções
  aproximadas, que não representam garantia ou previsão exata de resultado.
- A simulação é enviada automaticamente para a paciente? Não. Antes da entrega, a imagem precisa
  passar pela revisão de um profissional habilitado.
- A IA decide qual procedimento a paciente deve realizar? Não. A Reflexo não substitui avaliação,
  orientação ou decisão profissional.
- A Reflexo serve apenas para gerar simulações? Não. A plataforma também organiza o caso e permite
  acompanhar histórico, origem, follow-up, agendamento e evolução da oportunidade.
- A Reflexo substitui o sistema de gestão ou prontuário da clínica? Não. O foco está na experiência
  de simulação, revisão profissional e acompanhamento daquela jornada.
- Existe teste grátis? Sim. Após a demonstração, quando fizer sentido para a sua clínica, é possível
  liberar um período de teste.

**Chamada final**
- Sua paciente já está tentando imaginar o resultado.
- Na demonstração, você vê a simulação no rosto dela, a revisão profissional e o acompanhamento do
  caso.

**Contato**
- Veja a Reflexo funcionando na prática. · Agendar minha demonstração.

---

## 3. Cadastro (`app.reflexo.health/cadastro`)

### 3.0 Como se entra e como se sai

- **Entradas:** a ação "contratar" da landing; e o link com código (`/cadastro?cupom=CODIGO`), usado
  para cortesia de parceiro, que já chega com o código preenchido no passo 3.
- **Caminhos de saída:** contratar e pagar; ou acessar sem cobrança por código de cortesia. Hoje não
  existe a opção "testar grátis" escolhida pela própria pessoa: período sem cobrança entra por
  código.
- **Ordem:** Passo 1 Profissional responsável → Passo 2 A clínica → Confirmação do e-mail → Passo 3
  Como você quer começar → (pago) Revise e confirme → Dados do cartão → Aprovado ou recusado; (código)
  Conta criada.
- **Regras de navegação:** voltar (botão ou gesto do navegador) volta um passo sem apagar o que foi
  digitado. Botão de avançar fica bloqueado enquanto houver campo com erro, e diz por quê ("corrija
  os campos marcados para continuar"). Um campo corrigido limpa o próprio erro.
- **Indicador de passo:** "criar conta · passo N de 3" e "criar conta · verificação".
- **Título da aba:** Criar conta, Reflexo.

### 3.1 Passo 1 · Profissional responsável

- Título: Profissional responsável.
- Texto: Quem assina as aprovações da clínica. Cada simulação entregue à paciente sai com nome e
  registro de quem aprovou.

**Campos**

| Campo | Regra | Mensagem de erro |
|---|---|---|
| Nome completo | obrigatório | Informe o seu nome completo. |
| E-mail (exemplo: nome@exemplo.com) | formato com domínio completo | Confira o e-mail. Falta o domínio completo, como nome@clinica.com.br. |
| Senha | mínimo 8 caracteres e uma letra maiúscula | A senha precisa de pelo menos 8 caracteres e uma letra maiúscula. |
| Tipo de profissional | escolha única: Médica(o) · Dentista · Biomédica(o) · Enfermeira(o) · Fisioterapeuta · Farmacêutica(o) · Esteticista | |
| Conselho | vem preenchido pelo tipo (CRM, CRO, CRBM, COREN, CREFITO, CRF) e pode ser trocado | |
| UF do conselho | obrigatória; nunca herda a UF da clínica | Selecione a UF do conselho. |
| Número (apoio: número completo do registro, com o dígito) | 4 a 8 dígitos, sufixo de letra opcional (ex.: 123456-F) | Informe um número de registro válido (4 a 8 dígitos). |
| Telefone (exemplo: (11) 98220-4310) | com DDD | Informe um telefone com DDD. |

**Variante esteticista.** Os campos de conselho, UF e número deixam de existir (não ficam
desabilitados). No lugar:
- Aviso: Esteticista não tem conselho profissional: os campos de conselho não existem aqui. No lugar,
  declare a sua formação e anexe o diploma ou certificado. O catálogo disponível é limitado à sua
  categoria, e a assinatura das entregas sai com o nome e a categoria.
- Curso (exemplo: Tecnólogo em Estética) · "Informe o curso."
- Instituição (exemplo: SENAC) · "Informe a instituição."
- Ano de conclusão (4 dígitos) · "Informe o ano de conclusão (4 dígitos)."
- Diploma ou certificado: anexar arquivo, PDF ou imagem (jpg, png). Apoio: "o diploma é registro de
  responsabilidade, não credencial verificada". Erro: "Anexe o diploma ou certificado."
- Declaração (caixa de marcar, obrigatória): Declaro estar habilitada(o) a executar os procedimentos
  que vou simular, conforme as regras da minha profissão. A verificação de competência é da minha
  categoria, não do Reflexo. Erro: "Confirme a declaração de enquadramento para continuar."

**Variante conselho sem verificação automática** (CRF e CREFITO em qualquer UF; COREN fora de SP e
PR). Aparece depois que a UF é escolhida. Mantém conselho, UF e número, e acrescenta:
- Aviso: O {conselho} ainda não tem verificação automática no Reflexo. A sua habilitação vem do
  documento de formação e da declaração abaixo; o número do conselho fica no registro.
- Diploma, certificado ou registro: anexar arquivo. Apoio: "a habilitação vem do documento, não do
  número do conselho".
- Declaração: Declaro estar habilitada(o) a executar os procedimentos que vou simular, conforme as
  regras do meu conselho ou da legislação da minha profissão. A verificação de competência é da minha
  categoria, não do Reflexo.

**Bloco de apoio "O que fica pronto agora":** portal com a sua fila de revisão · primeiro link de
captação · catálogo de serviços para ativar · assinatura com nome e conselho.

**Ações:** Continuar · "Já tem conta? Entrar".

**Estados:** base; base esteticista; base sem cobertura; com erros nos campos; e-mail já cadastrado
(janela abaixo).

**Janela "e-mail já cadastrado"** (aparece quando o e-mail já tem conta, ao enviar o código ou ao
concluir):
- Mostra o e-mail, só leitura.
- Texto: **Este e-mail já tem conta no Reflexo.** Entre com a sua senha ou recupere o acesso.
- Ações: Entrar · Recuperar senha · "ou use outro e-mail para esta conta" · fechar.

### 3.2 Passo 2 · A clínica

- Título: A clínica.
- Texto: O nome que a paciente vê na entrega, o documento e o endereço de quem contrata (para a nota
  fiscal) e o WhatsApp que recebe os agendamentos.

| Campo | Regra | Mensagem de erro |
|---|---|---|
| Nome da clínica | obrigatório; é o nome que a paciente vê | Informe o nome da clínica. |
| CNPJ ou CPF | um campo só, o tipo se deduz pelo tamanho; dígito verificador conferido | Informe um CNPJ ou CPF válido. |
| CEP | 8 dígitos; ao completar, preenche logradouro, bairro, cidade e estado, que continuam editáveis; se a consulta falhar, nada aparece e a pessoa preenche | Informe um CEP com 8 dígitos. |
| Logradouro (exemplo: Rua, avenida) | obrigatório | Informe o logradouro. |
| Número | obrigatório | Informe o número. |
| Complemento (opcional; exemplo: Sala, andar) | | |
| Bairro | obrigatório | Informe o bairro. |
| Cidade | obrigatória | Informe a cidade. |
| Estado | lista das 27 UFs | |
| WhatsApp da clínica (apoio: recebe agendamentos e devolve entregas) | com DDD | Informe o WhatsApp da clínica com DDD. |

- Apoio do documento: Identifica quem contrata e garante um teste por clínica. É o mesmo documento da
  clínica, e um campo só resolve: quando não houver CNPJ, o CPF entra no lugar.
- Declaração (caixa de marcar, opcional hoje): Declaro ser a profissional responsável técnica desta
  clínica. O meu nome e conselho passam a assinar as aprovações.
- Bloco de apoio "O que vem depois": passo 2, a clínica e o documento · verificação, o código de 6
  dígitos no seu e-mail · passo 3, como começar. Nota: "o documento é pedido uma vez só: ele não volta
  a aparecer no pagamento".
- Ações: Continuar (enquanto envia: "Enviando o código") · Voltar. Apoio: "o e-mail de verificação sai
  depois deste passo".
- Erros de envio do código: "Muitas tentativas em pouco tempo. Tente de novo em alguns minutos." ·
  "Não conseguimos enviar o código agora. Tente de novo." · "Não conseguimos enviar o código agora.
  Confira a sua conexão."

### 3.3 Confirmação do e-mail

- Título: Confirme o seu e-mail.
- Texto: A conta da {clínica} está criada. Enviamos um código de 6 dígitos para {e-mail}.
- Código em 6 casas: o foco entra na primeira, cada dígito avança sozinho, apagar volta, colar
  distribui os seis.
- Ação: Confirmar e continuar (habilita com os 6 dígitos; enquanto confere: "Confirmando").
- "Não recebeu?" · "Reenviar em 0:47" (contagem de 60 segundos) e depois "Reenviar código".
- "Trocar endereço de e-mail" (volta ao passo 1).
- Bloco de apoio "O código": validade 15 minutos · reenvio a qualquer momento · depois, passo 3.
  Nota: "o código vale uma vez e expira em 15 minutos · reenviar invalida o anterior".

**Estados**
- Base.
- Incorreto: as casas marcam erro e esvaziam. "Código incorreto. Confira os dígitos ou reenvie. N
  tentativas restantes."
- Expirado (também tentativas esgotadas ou sem código válido): as casas esvaziam e desabilitam; a
  ação principal vira "Reenviar código" (enquanto envia: "Enviando"). Alerta: **Este código
  expirou.** Reenvie para receber um novo. O anterior deixa de valer.
- Reenviado: aviso "Código novo enviado para {e-mail}." com "o código anterior deixou de valer", e a
  contagem recomeça.

### 3.4 Passo 3 · Como você quer começar

- Título: Como você quer começar?

**Configuração (sem código de cortesia)**
- Plano, escolha entre dois: **Individual** ("você assina tudo") e **Equipe** ("duas ou mais assinam ·
  1 recepção incluída").
- Degrau de cota, de um em um: mostra "{cota} simulações aprovadas · R$ {preço} /mês". Individual vai
  de Base a 5x; Equipe de Base a 20x. Trocar para Individual com degrau acima de 5x volta para 5x.
- Só Equipe: "Profissionais que assinam", contador a partir de 1 (a primeira está incluída; cada
  adicional soma R$ 149,90).
- Só Individual: recepção como linha opcional, "recepção · + adicionar · R$ 99,90/mês", e quando
  adicionada, "retirar".
- Resumo: plano e degrau com o preço; assentos adicionais (Equipe); cota do degrau; desconto, quando
  houver código de desconto; total mensal. Equipe: "1 recepção incluída · recepção adicional R$ 99,90
  cada".
- Código: "Tem um código?" abre um campo ("seu código") com "Aplicar". Estados do código:
  - válido, cortesia: "válido · cortesia liberada · sem cobrança" (leva à tela de cortesia abaixo);
  - válido, desconto: "válido · {N}% de desconto na primeira mensalidade" (o valor recorrente segue
    cheio);
  - inválido: "código inválido ou não é deste e-mail";
  - aplicado: o campo mostra o código fixo com um botão de confirmado que, clicado, remove.
- Ação: Continuar para a revisão · apoio: "você revisa o pedido antes de pagar".

**Revisão antes do pagamento**
- Linhas: plano e degrau · cota · recepção ("nenhuma adicionada · opcional, R$ 99,90 cada" ou o valor)
  · contratante ({clínica} · CNPJ ou CPF {número}) · termos de uso (versão, aceite junto do pagamento)
  · desconto, quando houver · total mensal.
- "Ajustar plano e degrau" (volta à configuração).
- Ação: Continuar para o pagamento · apoio: "pagamento e aceite dos termos a seguir". Enquanto envia:
  "Criando a conta".

**Com código de cortesia válido.** Plano, degrau, assentos e recepção somem. A tela mostra só o que o
código concede:
- Sua cortesia: acesso, cortesia · sem cartão; simulações aprovadas, {N} por mês; até, {data};
  contratante; termos de uso, licença · sem cobrança; total R$ 0,00/mês.
- Ação: Acessar grátis · apoio: "sem cartão · sua cortesia já está liberada".
- Nunca pede cartão.

**Bloco de apoio:** "O que fica pronto agora" (mesmos quatro itens do passo 1) e o progresso:
"passo 1 profissional ✓ · passo 2 clínica ✓ · passo 3 plano, revisão e contratação".

**Erros ao concluir:** "Este cupom de cortesia não vale para este e-mail, ou já foi usado." · "Muitas
tentativas em pouco tempo. Tente de novo em alguns minutos." · "Não conseguimos concluir o cadastro
agora. Confira os dados e tente de novo." · "Não conseguimos concluir o cadastro agora. Confira a sua
conexão." · e-mail já cadastrado (janela da seção 3.1).

### 3.5 Revise e confirme

- Indicador: contratação · revise e confirme. Título: Revise e confirme.
- Resumo: {plano} · degrau {X} · R$ {total}/mês · "{N} profissionais e {N} recepções · {cota}
  simulações aprovadas por mês, da clínica inteira".
- Linhas: degrau com a cota e o preço; cada assento profissional adicional ("assento profissional 2",
  R$ 149,90); cada recepção ("recepção 1", R$ 99,90); total mensal; com desconto, "desconto na 1ª
  mensalidade ({N}%)" e "primeira cobrança".
- "Entenda o preço" (abre e fecha):
  - Acesso: o plano inclui a cota do degrau e um assento profissional; assento adicional custa
    R$ 149,90 e cada recepção R$ 99,90.
  - Cota: são as simulações aprovadas incluídas por mês, contadas para a clínica inteira.
  - Excedente: acima da cota, cada simulação aprovada a mais custa R$ 9,90, só com o pacote extra
    ligado.
  - Preço de fundador: assine agora e este valor fica seu enquanto a assinatura viver.
  - Nota: "pacote extra desligado · ative depois em Cota".
- "Ajustar plano e degrau".
- **Forma de pagamento**, duas opções: **Pix Automático** ("sem cartão"; é a recomendada: melhor para
  a clínica, sem estorno contestado; hoje em ativação) e **Cartão de crédito** ("no mesmo cartão").
  Boleto não existe no plano mensal.
- **Autorização da cobrança:**
  - no cartão, caixa de marcar obrigatória: Você autoriza a cobrança recorrente de R$ {total} por mês
    no cartão, a partir de {data}, até que cancele. O cancelamento pode ser feito a qualquer momento
    nas configurações, pela mesma via desta contratação. Link: Termos de cobrança.
  - no Pix Automático, sem caixa: "no Pix Automático você autoriza a recorrência no aplicativo do banco
    ao concluir; não há aceite de cobrança a marcar aqui."
- **Contrato de serviço:** "Inclui a cláusula de tratamento de dados de pacientes." Link "Abrir o
  documento completo". O texto final está em revisão jurídica: o espaço reserva o lugar dele.
- Aceite (obrigatório): Li e aceito os termos de uso e o contrato de serviço, incluindo a cláusula de
  tratamento de dados de pacientes.
- Bloco "O pedido": plano · degrau e cota · assentos · recepção · total/mês. Nota: "a autorização de
  cobrança recorrente fica junto do botão de pagar".
- Ação: Confirmar e pagar. Bloqueada até: "aceite os termos e marque a autorização para liberar a
  confirmação" (cartão) ou "aceite os termos para liberar a confirmação" (Pix). Apoio: "processado
  pela Asaas · cartão de crédito ou Pix Automático".

### 3.6 Dados do cartão

- Indicador: contratação · dados do cartão. Título: Dados do cartão.
- Texto: {plano} · degrau {X} · R$ {total} por mês, com renovação automática.

| Campo | Mensagem de erro |
|---|---|
| Número do cartão (agrupado de 4 em 4) | Número do cartão inválido. |
| Validade (MM/AA ou MM/AAAA) | Validade no passado ou inválida. |
| CVV | CVV incompleto. |
| Nome impresso no cartão | Informe o nome impresso no cartão. |

- Caixa "O cartão é de outra pessoa". Marcada, pede: Nome do titular · CPF ou CNPJ do titular · CEP ·
  Número · Telefone do titular, com os erros "Informe o nome do titular." · "Informe um CPF ou CNPJ
  válido." · "Informe um CEP com 8 dígitos." · "Informe o número do endereço." · "Informe um telefone
  com DDD."
- Bloco "A captura": processador, Asaas · dados, seguem direto para o Asaas · recorrência, no mesmo
  cartão. Nota: "Pagamento processado pelo Asaas. O Reflexo não guarda o número do cartão nem o CVV, só
  a bandeira e os quatro últimos dígitos."
- Ação: Pagar R$ {primeira cobrança} · "Voltar à revisão".
- Erros ao pagar:
  - registro no conselho recusado: O seu registro no conselho não foi confirmado, então o pagamento não
    pode seguir. Confira a sigla, o número e a UF, e refaça o cadastro.
  - registro em verificação: Estamos verificando o seu registro no conselho. Isso leva alguns
    instantes; aguarde e toque em Pagar de novo.
  - genéricos: Não conseguimos processar o pagamento agora. Confira os dados e tente de novo. · Não
    conseguimos processar o pagamento agora. Confira a sua conexão.

### 3.7 Pagamento aprovado e pagamento recusado

**Aprovado**
- Indicador: contratação · pagamento aprovado. Título: Pagamento aprovado.
- Texto: {N} assentos ativos. O recibo foi para o seu e-mail e a nota fiscal sai em nome da clínica.
- Ação: Ativar o portal em 6 passos. Apoio: "processado pela Asaas · fatura mensal no mesmo cartão".

**Recusado**
- Indicador: contratação · pagamento recusado. Título: O pagamento não passou.
- Texto: {motivo da operadora, quando houver}. Nada foi cobrado. Tente outro cartão ou o Pix
  Automático. Apoio: "o pedido fica guardado; nenhum dado se perde".
- Bloco "O que fazer": pedido, fica guardado · caminho 1, trocar cartão · caminho 2, Pix Automático.
- Ações: Trocar cartão · Usar Pix Automático.

### 3.8 Conta criada (por código)

- Título: Conta criada. Sua cortesia vai até {data}. (sem data: "Conta criada. Sua cortesia está
  ativa.")
- Texto: Antes de tudo, um combinado de uso responsável: a Reflexo gera uma projeção ilustrativa, e
  nada chega à paciente sem você revisar e assinar. A simulação apoia a conversa, não promete
  resultado.
- "A sua porta permanente": o link da clínica (`https://{identificador}.reflexo.health/`), com o
  apoio "Publique este link na bio, no site e na ficha do Google. Toda paciente que abrir cai no seu
  funil."
- Ação: Entrar no portal.

### 3.9 Fora do cadastro (não desenhar aqui)

- A ativação (seis passos dentro do portal: ativar serviços, completar o perfil, ponto de partida,
  aceite de uso responsável, publicar o link, WhatsApp de contato). O funil termina na contratação.
- A conversão no fim da cortesia ou do teste (tela própria, com o uso real à vista).
- Os e-mails (código de verificação, recibo, fim de cortesia).
- A verificação do registro no conselho roda em segundo plano depois do cadastro. Cadastrar e revisar
  não esperam por ela; a primeira aprovação espera.

---

## 4. Planos e preço (preço de fundador)

**Dois planos, o mesmo produto.**
- **Individual:** uma profissional, que é a responsável técnica e aprova tudo. Recepção não vem
  incluída; entra como adicional opcional. Papéis, equipe e governança não aparecem.
- **Equipe:** duas ou mais profissionais assinando, com papéis, permissões, convites e uma recepção
  incluída.

**A cota conta simulações aprovadas, nunca gerações.** Gerar de novo até aprovar não consome. É da
clínica inteira.

| Degrau | Aprovadas por mês | Mensal (fundador) | Planos |
|---|---|---|---|
| Base | 80 | R$ 390 | Individual e Equipe |
| 2x | 160 | R$ 590 | Individual e Equipe |
| 5x | 400 | R$ 1.290 | Individual e Equipe |
| 10x | 800 | R$ 2.590 | só Equipe |
| 20x | 1.600 | R$ 4.490 | só Equipe |

- Assento profissional adicional: R$ 149,90 por mês. Recepção: R$ 99,90 por mês cada.
- O 5x do Individual carrega o aviso de teto físico: é o limite do que uma profissional revisa.
- Excedente: R$ 9,90 por simulação aprovada acima da cota, só com o pacote extra ligado pela clínica,
  nunca pré-marcado.
- Ciclo: só mensal.
- Preço de fundador: quem assina agora mantém o preço enquanto ficar. A tabela cheia vale só para
  cliente novo, e a virada é anunciada com 30 dias de antecedência na landing.
- A Assistente (módulo de atendimento) não entra no lançamento e não aparece como oferta.

---

*Documento de apoio ao desenho. Direção no Norte, estado de cada capacidade no Produto, e acima dos
dois o código.*
