from transformers import pipeline

question = "Onde fica?"
context = "Estamos localizados em Alto Paraíso de Goiás, no estado de Goiás, Brasil, com as coordenadas: -14.256856277485067, -47.516158131804325. Localizada na região da Chapada dos Veadeiros, estamos a 20 minutos de carro do centro da cidade e próximos das melhores cachoeiras e atrações da região, incluindo Catarata dos Couros (20 minutos), Complexo do Macaquinhos (30 minutos), Vale da Lua (40 minutos) e Parque Nacional Chapada dos Veadeiros (50 minutos). Nosso rancho fica a 200 quilômetros de Brasília e a 400 quilômetros de Goiânia. Por questões de segurança e bem-estar dos nossos animais, não permitimos a entrada de animais de estimação em nossa propriedade. Já tivemos incidentes no passado em que animais de estimação atacaram e feriram nossos animais, e, por isso, decidimos proibir sua entrada. No entanto, ficaríamos felizes em recebê-lo sem seu animal de estimação, pois temos muitos animais diferentes para você conhecer e interagir durante sua estadia. Para chegar à nossa localização a partir do centro de Alto Paraíso, são 12 quilômetros de estrada asfaltada bem conservada, seguidos por 3 quilômetros de estrada de terra também bem conservada, que é facilmente acessível para qualquer tipo de veículo. Nosso rancho está dentro de uma fazenda maior chamada Flor da Mata, que é totalmente cercada e possui uma única entrada, protegida por um portão com cadeado, para o qual somente os moradores e hóspedes têm a senha. A entrada é monitorada por câmeras 24 horas por dia, e também temos câmeras dentro do nosso rancho. Cada chalé está equipado com uma chave que fica sempre com o hóspede, e ninguém entra no seu chalé até o seu check-out, a menos que nos chame para manutenção ou assistência. Atualmente, não temos uma piscina, mas está nos nossos planos para o futuro próximo. Alguns dos nossos chalés possuem uma banheira de hidromassagem para 2 pessoas. Além disso, dentro da fazenda onde fica nosso rancho, há um córrego com uma cachoeira privada. Fica a 3 quilômetros do rancho e é acessível de carro até a beira do rio, embora haja uma parte um pouco íngreme. Se você tiver dúvidas sobre a capacidade do seu carro de passar por essa área, pode estacionar na parte superior e descer a pé (~1,2 quilômetros). Nossas diárias variam de acordo com a data, o modelo do chalé e o número de pessoas. As tarifas começam em R$499 para um casal durante a semana e chegam a R$1499 para quatro pessoas durante feriados. Para obter o preço exato, por favor, visite o nosso motor de reservas: Link do Motor de Reservas. Temos uma estadia mínima de 2 noites, exceto durante feriados, quando a estadia mínima é de 3 noites. Exigimos o pagamento integral antecipado para garantir a sua reserva e aceitamos pagamento com cartão de crédito à vista ou Pix com 5% de desconto."

question_answerer = pipeline("question-answering", model="Ryan20/sqoin_qa_model_first")

answer = question_answerer(question=question, context=context)["answer"]
index = question_answerer(question=question, context=context)["start"]

print(f"\nQuestion : {question}")

print(f"\nAnswer : {answer}")

print(f"\nIndex : {index}")


