from transformers import  AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def preprocess_function(examples):
    questions = [qa["question"] for qa in examples["data"][0][0]["paragraphs"][0]["qas"]]

    contexts = [examples["data"][0][0]["paragraphs"][0]["context"]] * len(questions)
    
    answers = [qa["answers"] for qa in examples["data"][0][0]["paragraphs"][0]["qas"]]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []


    for i, offset in enumerate(offset_mapping):
        answer = answers[i][0]
        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)  # Corrected

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

    # for i, offset in enumerate(offset_mapping):
    #     answer = answers[i]
    #     start_char = answer["answer_start"][0]
    #     end_char = answer["answer_start"][0] + len(answer["text"][0])

    #     sequence_ids = inputs.sequence_ids(i)
        

    # inputs["start_positions"] = start_positions
    # inputs["end_positions"] = end_positions
    # return inputs














# def preprocess_function(examples):
#     questions = [q.strip() for q in examples["question"]]
#     inputs = tokenizer(
#         questions,
#         examples["context"],
#         max_length=1000,
#         truncation="only_second",
#         return_offsets_mapping=True,
#         padding="max_length",
#     )

#     offset_mapping = inputs.pop("offset_mapping")
#     answers = examples["answers"]
#     start_positions = []
#     end_positions = []

#     for i, offset in enumerate(offset_mapping):
#         answer = answers[i]
#         start_char = answer["answer_start"][0]

#         # print(f"\n start chat  {start_char}")
#         # end_char = answer["answer_start"][0] + len(answer["text"][0])
#         end_char = answer["answer_start"][0] + len(answer["text"][0])

#         # print(f"\n end char  {end_char}")


#         sequence_ids = inputs.sequence_ids(i)
        

#         # # Find the start and end of the context
#         # idx = 0
#         # while sequence_ids[idx] != 1:
#         #     idx += 1
#         # context_start = idx
#         # while sequence_ids[idx] == 1:
#         #     idx += 1
#         # context_end = idx - 1

#         # # If the answer is not fully inside the context, label it (0, 0)
#         # if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
#         #     start_positions.append(0)
#         #     end_positions.append(0)
#         # else:
#         #     # Otherwise it's the start and end token positions
#         #     idx = context_start
#         #     while idx <= context_end and offset[idx][0] <= start_char:
#         #         idx += 1
#         #     start_positions.append(idx - 1)

#         #     idx = context_end
#         #     while idx >= context_start and offset[idx][1] >= end_char:
#         #         idx -= 1
#         #     end_positions.append(idx + 1)

#     inputs["start_positions"] = start_positions
#     inputs["end_positions"] = end_positions
#     return inputs