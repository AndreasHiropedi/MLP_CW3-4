import pandas as pd
import torch
import time

from transformers import MarianMTModel, MarianTokenizer


def back_translate(text, source_lang="en", target_lang="es"):
    # Initialize the tokenizer and model for translation to the target language
    tokenizer_to = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')
    model_to = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}').to(torch.device("mps"))

    # Prepare the text and move it to MPS
    encoded_text_to = tokenizer_to(text, return_tensors="pt", padding=True).to(torch.device("mps"))
    translated_to = model_to.generate(**encoded_text_to)
    text_to = tokenizer_to.batch_decode(translated_to, skip_special_tokens=True)[0]

    # Repeat for back translation
    tokenizer_back = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}')
    model_back = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}').to(torch.device("mps"))

    encoded_text_back = tokenizer_back(text_to, return_tensors="pt", padding=True).to(torch.device("mps"))
    translated_back = model_back.generate(**encoded_text_back)
    text_back = tokenizer_back.batch_decode(translated_back, skip_special_tokens=True)[0]

    return text_back


def augment_dataframe_for_class(df, class_name, class_value, target_count, text_column='post'):
    # Filter data for the specific class
    class_data = df[df[class_name] == class_value]

    amount_of_data = len(class_data)

    print(f'Augmenting {class_value} data ...')

    augmented_rows = []
    rows_count = 0
    while len(augmented_rows) < target_count - len(class_data) or rows_count <= amount_of_data:
        for _, row in class_data.iterrows():
            if len(augmented_rows) >= target_count - len(class_data):
                break
            rows_count += 1
            original_text = row[text_column]
            back_translated_text = back_translate(original_text)
            if class_data['post'].isin([back_translated_text]).any():
                continue
            new_row = row.copy()
            new_row[text_column] = back_translated_text
            augmented_rows.append(new_row)

    print(f'Finished {class_value} data augmentation!')

    return pd.DataFrame(augmented_rows)


# Capture the start time
start_time = time.time()

# Load the dataset
data = pd.read_csv('new_augmented_data.tsv', delimiter='\t')

augmented_other_data = augment_dataframe_for_class(data, 'implicit_class', 'other', 1100)
augmented_explicit_data = augment_dataframe_for_class(data, 'implicit_or_explicit', 'explicit_hate', 6300)

final_df = pd.concat([data, augmented_explicit_data, augmented_other_data], ignore_index=True)

final_df.to_csv('final_augmented_dataset.tsv', sep='\t', index=False)

# Place this line at the statement you're interested in
elapsed_time = time.time() - start_time

# Calculate hours, minutes, and seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

# Format the time as a string
formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

print(f"It took {formatted_time} (hh:mm:ss) for generating augmented data.")